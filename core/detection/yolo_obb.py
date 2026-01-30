"""
YOLOv11-OBB detection adapter stub.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from core.logic.types import Detection


from core.logic.postprocess import filter_detections


class YoloObbDetector:
    def __init__(
        self,
        weights_path: str,
        min_confidence: float = 0.25,
        nms_iou: float = 0.5,
        max_detections: int = 100,
    ) -> None:
        self.weights_path = weights_path
        self.min_confidence = min_confidence
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        # TODO: load model weights when integrating real model.

    def predict(self, image_path: str) -> List[Detection]:
        # TODO: run inference and return detections.
        detections = _load_sidecar_detections(image_path)
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
        heading_deg = item.get("heading_deg")
        if not isinstance(heading_deg, (int, float)):
            heading_deg = None
        shadow_azimuth_deg = item.get("shadow_azimuth_deg")
        if not isinstance(shadow_azimuth_deg, (int, float)):
            shadow_azimuth_deg = None
        shadow_length_ratio = item.get("shadow_length_ratio")
        if not isinstance(shadow_length_ratio, (int, float)):
            shadow_length_ratio = None
        detections.append(
            Detection(
                label=label,
                confidence=float(confidence),
                obb=tuple(points),  # type: ignore[arg-type]
                heading_deg=None if heading_deg is None else float(heading_deg),
                shadow_azimuth_deg=None if shadow_azimuth_deg is None else float(shadow_azimuth_deg),
                shadow_length_ratio=None
                if shadow_length_ratio is None
                else float(shadow_length_ratio),
            )
        )
    return detections
