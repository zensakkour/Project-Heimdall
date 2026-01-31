"""
Classic detector tests.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from src.core.detection.classic import ClassicDetector


def test_classic_detector_finds_dark_blob() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "img.png"
        img = Image.new("RGB", (120, 120), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)
        draw.rectangle([60, 50, 90, 80], fill=(10, 10, 10))
        img.save(path)

        detector = ClassicDetector(threshold=60, min_area=20)
        detections = detector.predict(str(path))
        assert len(detections) >= 1


