"""
Shadow extraction stub tests.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from src.core.logic.shadow_extract import estimate_shadow_from_image


def test_shadow_extractor_detects_dark_blob() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "img.png"
        img = Image.new("RGB", (100, 100), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        # Draw dark region to the right of the OBB center
        draw.rectangle([70, 45, 90, 55], fill=(0, 0, 0))
        img.save(path)

        obb = ((40, 40), (60, 40), (60, 60), (40, 60))
        result = estimate_shadow_from_image(str(path), obb)
        assert result is not None
        azimuth, length_ratio = result
        assert 300.0 <= azimuth or azimuth <= 60.0
        assert length_ratio > 0.5


