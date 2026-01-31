"""
Lightweight shadow extraction stub.
Assumes imagery is north-up; output azimuth is treated as world azimuth.
"""
from __future__ import annotations

from math import atan2
from typing import Iterable, List, Optional, Tuple

from PIL import Image

from .types import Detection


def enrich_detections_with_shadows(image_path: str, detections: Iterable[Detection]) -> List[Detection]:
    enriched: List[Detection] = []
    for det in detections:
        if det.shadow_azimuth_deg is not None and det.shadow_length_ratio is not None:
            enriched.append(det)
            continue
        estimate = estimate_shadow_from_image(image_path, det.obb)
        if estimate is None:
            enriched.append(det)
            continue
        azimuth_deg, length_ratio = estimate
        enriched.append(
            Detection(
                label=det.label,
                confidence=det.confidence,
                obb=det.obb,
                heading_deg=det.heading_deg,
                shadow_azimuth_deg=det.shadow_azimuth_deg
                if det.shadow_azimuth_deg is not None
                else azimuth_deg,
                shadow_length_ratio=det.shadow_length_ratio
                if det.shadow_length_ratio is not None
                else length_ratio,
            )
        )
    return enriched


def estimate_shadow_from_image(
    image_path: str,
    obb: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    threshold: int = 60,
    min_pixels: int = 20,
) -> Optional[Tuple[float, float]]:
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            width, height = gray.size
            aabb = _aabb_from_obb(obb)
            expand = 2.0
            x1 = max(0, int(aabb[0] - expand * (aabb[2] - aabb[0])))
            y1 = max(0, int(aabb[1] - expand * (aabb[3] - aabb[1])))
            x2 = min(width, int(aabb[2] + expand * (aabb[2] - aabb[0])))
            y2 = min(height, int(aabb[3] + expand * (aabb[3] - aabb[1])))
            if x2 <= x1 or y2 <= y1:
                return None

            region = gray.crop((x1, y1, x2, y2))
            pixels = region.load()
            dark_points = []
            for y in range(region.height):
                for x in range(region.width):
                    if pixels[x, y] <= threshold:
                        dark_points.append((x + x1, y + y1))
            if len(dark_points) < min_pixels:
                return None

            cx, cy = _centroid(obb)
            dx, dy = _centroid(dark_points)
            vx = dx - cx
            vy = dy - cy
            if vx == 0 and vy == 0:
                return None
            azimuth = (atan2(vy, vx) * 180.0 / 3.141592653589793) % 360.0

            width_obb, height_obb = _obb_dims(obb)
            base = max(width_obb, height_obb, 1e-3)
            length_ratio = ((vx**2 + vy**2) ** 0.5) / base
            return azimuth, length_ratio
    except Exception:
        return None


def _centroid(points: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    pts = list(points)
    if not pts:
        return 0.0, 0.0
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    return sx / len(pts), sy / len(pts)


def _aabb_from_obb(
    obb: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in obb]
    ys = [p[1] for p in obb]
    return min(xs), min(ys), max(xs), max(ys)


def _obb_dims(
    obb: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
) -> Tuple[float, float]:
    p0, p1, p2, _ = obb
    edge1 = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
    edge2 = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return edge1, edge2


