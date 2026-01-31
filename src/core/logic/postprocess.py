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
) -> List[Detection]:
    normalized: List[Detection] = []
    for det in detections:
        if det.confidence < min_confidence:
            continue
        obb = _normalize_obb(det.obb)
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
    kept = _nms_aabb(kept, nms_iou)
    if max_detections > 0:
        kept = kept[:max_detections]
    return kept


def _valid_obb(obb: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]) -> bool:
    if len(obb) != 4:
        return False
    area = _polygon_area(obb)
    return area > 1e-3


def _normalize_obb(
    obb: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] | None:
    if not _valid_obb(obb):
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


def _nms_aabb(detections: List[Detection], iou_threshold: float) -> List[Detection]:
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
            if _iou(current_aabb, _aabb_from_obb(det.obb)) <= iou_threshold:
                remaining.append(det)
        sorted_dets = remaining
    return kept


