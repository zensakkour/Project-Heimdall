"""
Simple classical detector for dark blobs (no ML).
"""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

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
        if cv2 is not None:
            image = cv2.imread(image_path)
            if image is None:
                return []
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return self._from_contours(contours, image.shape[0], image.shape[1])

        try:
            with Image.open(image_path) as img:
                gray = np.asarray(img.convert("L"), dtype=np.uint8)
        except Exception:
            return []
        mask = gray < int(self.threshold)
        return self._from_connected_components(mask)

    def _from_contours(self, contours, height: int, width: int) -> List[Detection]:
        detections: List[Detection] = []
        img_area = float(height * width)
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

    def _from_connected_components(self, mask: np.ndarray) -> List[Detection]:
        if mask.ndim != 2:
            return []
        h, w = mask.shape
        detections: List[Detection] = []
        visited = np.zeros((h, w), dtype=np.uint8)
        img_area = float(h * w)
        for y in range(h):
            for x in range(w):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = 1
                area = 0
                min_x = max_x = x
                min_y = max_y = y
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    if cx < min_x:
                        min_x = cx
                    if cx > max_x:
                        max_x = cx
                    if cy < min_y:
                        min_y = cy
                    if cy > max_y:
                        max_y = cy
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if not mask[ny, nx] or visited[ny, nx]:
                            continue
                        visited[ny, nx] = 1
                        stack.append((ny, nx))

                if area < self.min_area:
                    continue
                if self.max_area is not None and area > self.max_area:
                    continue
                x0 = float(min_x)
                y0 = float(min_y)
                x1 = float(max_x + 1)
                y1 = float(max_y + 1)
                obb = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
                confidence = min(1.0, max(0.05, float(area) / max(1.0, img_area * 0.02)))
                detections.append(Detection(label="blob", confidence=confidence, obb=obb))
        return detections


