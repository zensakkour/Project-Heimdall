"""
Detection post-processing utilities.
"""
from __future__ import annotations

from math import atan2
from typing import Iterable, List, Tuple

from .types import Detection


def filter_detections(
    detections: Iterable[Detection],
    min_confidence: float,
    nms_iou: float,
    max_detections: int,
    nms_mode: str = "obb",
    min_area_px: float = 16.0,
    class_agnostic_nms: bool = False,
) -> List[Detection]:
    normalized: List[Detection] = []
    for det in detections:
        if det.confidence < min_confidence:
            continue
        obb = _normalize_obb(det.obb, min_area_px=min_area_px)
        if obb is None:
            continue
        heading = det.heading_deg if det.heading_deg is not None else _heading_from_obb(obb)
        normalized.append(
            Detection(
                label=det.label,
                confidence=det.confidence,
                obb=obb,
                heading_deg=heading,
                shadow_azimuth_deg=det.shadow_azimuth_deg,
                shadow_length_ratio=det.shadow_length_ratio,
            )
        )
    kept = normalized
    mode = str(nms_mode).strip().lower()
    if mode == "aabb":
        kept = _nms_aabb(kept, nms_iou, class_agnostic=class_agnostic_nms)
    else:
        kept = _nms_obb(kept, nms_iou, class_agnostic=class_agnostic_nms)
    if max_detections > 0:
        kept = kept[:max_detections]
    return kept


def _valid_obb(
    obb: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    min_area_px: float,
) -> bool:
    if len(obb) != 4:
        return False
    area = _polygon_area(obb)
    return area >= max(1e-3, float(min_area_px))


def _normalize_obb(
    obb: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    min_area_px: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] | None:
    if not _valid_obb(obb, min_area_px=min_area_px):
        return None
    cx = sum(p[0] for p in obb) / 4.0
    cy = sum(p[1] for p in obb) / 4.0
    points = sorted(obb, key=lambda p: atan2(p[1] - cy, p[0] - cx), reverse=True)
    # Rotate so the first point is top-left-ish (min y, then min x)
    min_index = min(range(4), key=lambda i: (points[i][1], points[i][0]))
    ordered = points[min_index:] + points[:min_index]
    return tuple((float(x), float(y)) for x, y in ordered)


def _heading_from_obb(
    obb: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
) -> float:
    p0, p1, p2, p3 = obb
    edge1 = (p1[0] - p0[0], p1[1] - p0[1])
    edge2 = (p2[0] - p1[0], p2[1] - p1[1])
    len1 = (edge1[0] ** 2 + edge1[1] ** 2) ** 0.5
    len2 = (edge2[0] ** 2 + edge2[1] ** 2) ** 0.5
    vec = edge1 if len1 >= len2 else edge2
    angle = atan2(vec[1], vec[0])
    deg = (angle * 180.0 / 3.141592653589793) % 360.0
    return deg


def _polygon_area(points: Tuple[Tuple[float, float], ...]) -> float:
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def _signed_polygon_area(points: List[Tuple[float, float]]) -> float:
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return area * 0.5


def _ensure_ccw(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if _signed_polygon_area(points) < 0.0:
        return list(reversed(points))
    return points


def _cross(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _line_intersection(
    s: Tuple[float, float],
    e: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> Tuple[float, float]:
    dx1 = e[0] - s[0]
    dy1 = e[1] - s[1]
    dx2 = b[0] - a[0]
    dy2 = b[1] - a[1]
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-9:
        return e
    t = ((a[0] - s[0]) * dy2 - (a[1] - s[1]) * dx2) / denom
    return s[0] + t * dx1, s[1] + t * dy1


def _clip_convex_polygon(
    subject: List[Tuple[float, float]],
    clipper: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    output = subject[:]
    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        input_list = output[:]
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            e_inside = _cross(a, b, e) >= 0.0
            s_inside = _cross(a, b, s) >= 0.0
            if e_inside:
                if not s_inside:
                    output.append(_line_intersection(s, e, a, b))
                output.append(e)
            elif s_inside:
                output.append(_line_intersection(s, e, a, b))
            s = e
    return output


def _iou_obb(
    a: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    b: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> float:
    poly_a = _ensure_ccw(list(a))
    poly_b = _ensure_ccw(list(b))
    area_a = abs(_signed_polygon_area(poly_a))
    area_b = abs(_signed_polygon_area(poly_b))
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    inter_poly = _clip_convex_polygon(poly_a, poly_b)
    if len(inter_poly) < 3:
        return 0.0
    inter_area = abs(_signed_polygon_area(inter_poly))
    if inter_area <= 0.0:
        return 0.0
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _aabb_from_obb(obb: Tuple[Tuple[float, float], ...]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in obb]
    ys = [p[1] for p in obb]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _nms_aabb(detections: List[Detection], iou_threshold: float, class_agnostic: bool = False) -> List[Detection]:
    if not detections or iou_threshold <= 0.0:
        return list(detections)
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    while sorted_dets:
        current = sorted_dets.pop(0)
        kept.append(current)
        current_aabb = _aabb_from_obb(current.obb)
        remaining: List[Detection] = []
        for det in sorted_dets:
            if not class_agnostic and det.label != current.label:
                remaining.append(det)
                continue
            if _iou(current_aabb, _aabb_from_obb(det.obb)) <= iou_threshold:
                remaining.append(det)
        sorted_dets = remaining
    return kept


def _nms_obb(detections: List[Detection], iou_threshold: float, class_agnostic: bool = False) -> List[Detection]:
    if not detections or iou_threshold <= 0.0:
        return list(detections)
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    while sorted_dets:
        current = sorted_dets.pop(0)
        kept.append(current)
        remaining: List[Detection] = []
        for det in sorted_dets:
            if not class_agnostic and det.label != current.label:
                remaining.append(det)
                continue
            if _iou_obb(current.obb, det.obb) <= iou_threshold:
                remaining.append(det)
        sorted_dets = remaining
    return kept


