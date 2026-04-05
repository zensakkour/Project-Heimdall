"""
Ultralytics YOLO OBB detector adapter.
"""
from __future__ import annotations

from typing import List, Optional

from src.core.logic.postprocess import filter_detections
from src.core.logic.types import Detection


class UltralyticsObbDetector:
    def __init__(
        self,
        weights_path: str,
        min_confidence: float = 0.25,
        iou: float = 0.5,
        max_detections: int = 100,
        min_area_px: float = 16.0,
        class_agnostic_nms: bool = False,
        use_tta: bool = False,
        imgsz: int = 1280,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.min_confidence = min_confidence
        self.iou = iou
        self.max_detections = max_detections
        self.min_area_px = min_area_px
        self.class_agnostic_nms = class_agnostic_nms
        self.use_tta = use_tta
        self.imgsz = imgsz

    def predict(self, image_path: str) -> List[Detection]:
        results = self.model.predict(
            source=image_path,
            conf=self.min_confidence,
            iou=self.iou,
            max_det=self.max_detections,
            agnostic_nms=self.class_agnostic_nms,
            augment=self.use_tta,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []
        res = results[0]
        if res.obb is None:
            return []

        obb = res.obb
        xyxyxyxy = _to_numpy(getattr(obb, "xyxyxyxy", None))
        conf = _to_numpy(getattr(obb, "conf", None))
        cls = _to_numpy(getattr(obb, "cls", None))
        names = getattr(res, "names", None) or {}

        detections: List[Detection] = []
        if xyxyxyxy is None or conf is None or cls is None:
            return detections

        for i in range(len(conf)):
            points = xyxyxyxy[i].reshape(4, 2).tolist()
            label = names.get(int(cls[i]), str(int(cls[i])))
            detections.append(
                Detection(
                    label=label,
                    confidence=float(conf[i]),
                    obb=(
                        (points[0][0], points[0][1]),
                        (points[1][0], points[1][1]),
                        (points[2][0], points[2][1]),
                        (points[3][0], points[3][1]),
                    ),
                )
            )
        return filter_detections(
            detections,
            min_confidence=self.min_confidence,
            nms_iou=self.iou,
            max_detections=self.max_detections,
            min_area_px=self.min_area_px,
            class_agnostic_nms=self.class_agnostic_nms,
        )


def _to_numpy(value: Optional[object]):
    if value is None:
        return None
    try:
        return value.cpu().numpy()
    except Exception:
        return None


