"""
Detector factory.
"""
from __future__ import annotations

from typing import Optional

from src.core.logic.config import DetectorConfig

from .base import Detector
from .classic import ClassicDetector
from .rfdetr_detector import RFDetrDetector
from .sidecar_detector import SidecarDetector
from .ultralytics_obb import UltralyticsObbDetector


def create_detector(cfg: Optional[DetectorConfig]) -> Optional[tuple[Detector, str]]:
    if cfg is None:
        return None
    backend = str(getattr(cfg, "backend", "rfdetr") or "rfdetr").strip().lower()
    if backend in {"rfdetr", "rf_detr", "rf-detr"}:
        try:
            from .rfdetr_detector import RFDetrDetector
            return RFDetrDetector(
                model_size=cfg.rfdetr_model_size,
                min_confidence=cfg.min_confidence,
                iou=cfg.nms_iou,
                nms_mode="aabb" if cfg.nms_mode == "obb" else cfg.nms_mode,
                max_detections=cfg.max_detections,
                min_area_px=cfg.min_area_px,
                class_agnostic_nms=cfg.class_agnostic_nms,
            ), "rfdetr"
        except ImportError:
            if cfg.use_sidecar:
                from .sidecar_detector import SidecarDetector
                return SidecarDetector(
                    min_confidence=cfg.min_confidence,
                    nms_iou=cfg.nms_iou,
                    nms_mode=cfg.nms_mode,
                    max_detections=cfg.max_detections,
                    min_area_px=cfg.min_area_px,
                    class_agnostic_nms=cfg.class_agnostic_nms,
                    use_classic=cfg.use_classic,
                ), "sidecar"
            if cfg.use_classic:
                from .classic import ClassicDetector
                return ClassicDetector(), "classic"
            raise
    if cfg.weights_path:
        from .ultralytics_obb import UltralyticsObbDetector
        return UltralyticsObbDetector(
            cfg.weights_path,
            min_confidence=cfg.min_confidence,
            iou=cfg.nms_iou,
            nms_mode=cfg.nms_mode,
            max_detections=cfg.max_detections,
            min_area_px=cfg.min_area_px,
            class_agnostic_nms=cfg.class_agnostic_nms,
            use_tta=cfg.use_tta,
            imgsz=cfg.imgsz,
        ), "ultralytics_obb"
    if cfg.use_sidecar:
        return SidecarDetector(
            min_confidence=cfg.min_confidence,
            nms_iou=cfg.nms_iou,
            nms_mode=cfg.nms_mode,
            max_detections=cfg.max_detections,
            min_area_px=cfg.min_area_px,
            class_agnostic_nms=cfg.class_agnostic_nms,
            use_classic=cfg.use_classic,
        ), "sidecar"
    if cfg.use_classic:
        return ClassicDetector(), "classic"
    return None
