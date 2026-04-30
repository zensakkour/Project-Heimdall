"""
Optional RF-DETR detector adapter.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from src.core.logic.postprocess import filter_detections
from src.core.logic.types import Detection


_MODEL_CLASSES: Mapping[str, str] = {
    "nano": "RFDETRNano",
    "n": "RFDETRNano",
    "small": "RFDETRSmall",
    "s": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "m": "RFDETRMedium",
    "large": "RFDETRLarge",
    "l": "RFDETRLarge",
    "xlarge": "RFDETRXLarge",
    "xl": "RFDETRXLarge",
    "2xlarge": "RFDETR2XLarge",
    "2xl": "RFDETR2XLarge",
}


class RFDetrDetector:
    def __init__(
        self,
        model_size: str = "medium",
        min_confidence: float = 0.25,
        iou: float = 0.5,
        nms_mode: str = "aabb",
        max_detections: int = 100,
        min_area_px: float = 16.0,
        class_agnostic_nms: bool = False,
    ) -> None:
        try:
            import rfdetr  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency-specific
            raise ImportError(
                "RF-DETR backend requested but package 'rfdetr' is not installed. "
                "Install with: pip install rfdetr"
            ) from exc

        size_key = str(model_size or "medium").strip().lower().replace("-", "")
        class_name = _MODEL_CLASSES.get(size_key)
        if class_name is None:
            supported = ", ".join(sorted(_MODEL_CLASSES))
            raise ValueError(f"Unsupported RF-DETR model size '{model_size}'. Supported: {supported}")
        model_cls = getattr(rfdetr, class_name)

        self.model = model_cls()
        self.min_confidence = min_confidence
        self.iou = iou
        self.nms_mode = nms_mode
        self.max_detections = max_detections
        self.min_area_px = min_area_px
        self.class_agnostic_nms = class_agnostic_nms
        self.class_names = _load_coco_class_names()

    def predict(self, image_path: str) -> List[Detection]:
        raw = self.model.predict(image_path, threshold=self.min_confidence)
        xyxy = getattr(raw, "xyxy", None)
        confidence = getattr(raw, "confidence", None)
        class_id = getattr(raw, "class_id", None)
        if xyxy is None or confidence is None:
            return []

        detections: List[Detection] = []
        for idx, box in enumerate(_to_list(xyxy)):
            if len(box) < 4:
                continue
            score = _safe_float(_index_or_none(confidence, idx), default=0.0)
            cls = _index_or_none(class_id, idx)
            label = self._label_for(cls)
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            detections.append(
                Detection(
                    label=label,
                    confidence=score,
                    obb=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
                )
            )

        return filter_detections(
            detections,
            min_confidence=self.min_confidence,
            nms_iou=self.iou,
            nms_mode=self.nms_mode,
            max_detections=self.max_detections,
            min_area_px=self.min_area_px,
            class_agnostic_nms=self.class_agnostic_nms,
        )

    def _label_for(self, class_id: object) -> str:
        if class_id is None:
            return "object"
        try:
            idx = int(class_id)
        except Exception:
            return str(class_id)
        return self.class_names.get(idx, str(idx))


def _load_coco_class_names() -> Dict[int, str]:
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES  # type: ignore
    except Exception:
        return {}
    if isinstance(COCO_CLASSES, dict):
        return {int(k): str(v) for k, v in COCO_CLASSES.items()}
    if isinstance(COCO_CLASSES, (list, tuple)):
        return {idx: str(label) for idx, label in enumerate(COCO_CLASSES)}
    return {}


def _to_list(value: object) -> list:
    try:
        return value.cpu().numpy().tolist()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return value.tolist()  # type: ignore[attr-defined]
    except Exception:
        pass
    return list(value)  # type: ignore[arg-type]


def _index_or_none(value: object, idx: int) -> Optional[object]:
    if value is None:
        return None
    try:
        item = value[idx]  # type: ignore[index]
    except Exception:
        return None
    try:
        return item.item()  # type: ignore[attr-defined]
    except Exception:
        return item


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default
