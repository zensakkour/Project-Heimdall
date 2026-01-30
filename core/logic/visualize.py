"""
Visualization helpers for detections.
"""
from __future__ import annotations

from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .types import Detection


def draw_detections(image_path: str, detections: Iterable[Detection]) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font()

    for det in detections:
        points = det.obb
        draw.polygon(points, outline=(0, 255, 200), width=2)
        label = f"{det.label} {det.confidence:.2f}"
        x, y = points[0]
        text_pos = (x + 2, y + 2)
        draw.text(text_pos, label, fill=(255, 255, 255), font=font)
    return img


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 14)
    except Exception:
        return ImageFont.load_default()
