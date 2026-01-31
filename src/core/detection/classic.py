"""
Simple classical detector for dark blobs (no ML).
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.core.logic.types import Detection


class ClassicDetector:
    def __init__(
        self,
        threshold: int = 60,
        min_area: int = 80,
        max_area: int | None = None,
    ) -> None:
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area

    def predict(self, image_path: str) -> List[Detection]:
        image = cv2.imread(image_path)
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        img_area = float(image.shape[0] * image.shape[1])
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            obb = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
            confidence = min(1.0, max(0.05, area / max(1.0, img_area * 0.02)))
            detections.append(Detection(label="blob", confidence=confidence, obb=obb))
        return detections


