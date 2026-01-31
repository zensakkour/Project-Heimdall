"""
Detector factory.
"""
from __future__ import annotations

from typing import Optional

from src.core.logic.config import DetectorConfig

from .base import Detector
from .classic import ClassicDetector
from .sidecar_detector import SidecarDetector
from .ultralytics_obb import UltralyticsObbDetector


def create_detector(cfg: Optional[DetectorConfig]) -> Optional[Detector]:
    if cfg is None:
        return None
    if cfg.weights_path:
        return UltralyticsObbDetector(
            cfg.weights_path,
            min_confidence=cfg.min_confidence,
            iou=cfg.nms_iou,
            max_detections=cfg.max_detections,
            imgsz=cfg.imgsz,
        )
    if cfg.use_sidecar:
        return SidecarDetector(
            min_confidence=cfg.min_confidence,
            nms_iou=cfg.nms_iou,
            max_detections=cfg.max_detections,
            use_classic=cfg.use_classic,
        )
    if cfg.use_classic:
        return ClassicDetector()
    return None


