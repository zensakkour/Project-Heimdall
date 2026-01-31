"""
Sidecar detector with optional classic fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.core.logic.postprocess import filter_detections
from src.core.logic.types import Detection

from .classic import ClassicDetector


class SidecarDetector:
    def __init__(
        self,
        min_confidence: float = 0.25,
        nms_iou: float = 0.5,
        max_detections: int = 100,
        use_classic: bool = False,
    ) -> None:
        self.min_confidence = min_confidence
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.use_classic = use_classic
        self.classic = ClassicDetector() if use_classic else None

    def predict(self, image_path: str) -> List[Detection]:
        detections = _load_sidecar_detections(image_path)
        if not detections and self.classic is not None:
            detections = self.classic.predict(image_path)
        return filter_detections(
            detections,
            min_confidence=self.min_confidence,
            nms_iou=self.nms_iou,
            max_detections=self.max_detections,
        )


def _load_sidecar_detections(image_path: str) -> List[Detection]:
    path = Path(image_path)
    candidates = [
        Path(str(path) + ".detections.json"),
        path.with_suffix(".detections.json"),
    ]
    sidecar = next((p for p in candidates if p.exists()), None)
    if sidecar is None:
        return []

    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(raw, dict):
        raw = raw.get("detections", [])
    if not isinstance(raw, list):
        return []

    detections: List[Detection] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        confidence = item.get("confidence")
        obb = item.get("obb")
        if not isinstance(label, str) or not isinstance(confidence, (int, float)):
            continue
        if not isinstance(obb, list) or len(obb) != 4:
            continue
        points = []
        valid = True
        for point in obb:
            if (
                not isinstance(point, (list, tuple))
                or len(point) != 2
                or not all(isinstance(v, (int, float)) for v in point)
            ):
                valid = False
                break
            points.append((float(point[0]), float(point[1])))
        if not valid:
            continue
        detections.append(
            Detection(
                label=label,
                confidence=float(confidence),
                obb=tuple(points),  # type: ignore[arg-type]
                heading_deg=_maybe_float(item.get("heading_deg")),
                shadow_azimuth_deg=_maybe_float(item.get("shadow_azimuth_deg")),
                shadow_length_ratio=_maybe_float(item.get("shadow_length_ratio")),
            )
        )
    return detections


def _maybe_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


