"""
Generate sample image/video and sidecar JSONs for the live UI.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"


def make_image() -> Path:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    image_path = SAMPLES / "sample.jpg"
    img = Image.new("RGB", (640, 360), color=(30, 40, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle([220, 140, 340, 200], outline=(0, 255, 200), width=3)
    draw.rectangle([400, 220, 520, 300], outline=(0, 255, 200), width=3)
    draw.rectangle([360, 140, 420, 200], fill=(5, 5, 5))
    img.save(image_path)

    det = {
        "detections": [
            {
                "label": "vehicle",
                "confidence": 0.92,
                "obb": [[220, 140], [340, 140], [340, 200], [220, 200]],
                "heading_deg": 90.0,
                "shadow_azimuth_deg": 210.0,
                "shadow_length_ratio": 1.8,
            },
            {
                "label": "truck",
                "confidence": 0.81,
                "obb": [[400, 220], [520, 220], [520, 300], [400, 300]],
                "heading_deg": 120.0,
                "shadow_azimuth_deg": 230.0,
                "shadow_length_ratio": 2.2,
            },
        ]
    }
    (SAMPLES / "sample.jpg.detections.json").write_text(json.dumps(det, indent=2), encoding="utf-8")

    geo = {
        "latitude": 35.0,
        "longitude": -120.0,
        "confidence": 0.74,
        "uncertainty_m": 120.0,
        "landmarks": ["sample"],
    }
    (SAMPLES / "sample.geo.json").write_text(json.dumps(geo, indent=2), encoding="utf-8")
    return image_path


def make_video() -> Path:
    video_path = SAMPLES / "sample.mp4"
    width, height = 640, 360
    fps = 12
    frames = 48
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (25, 30, 40)
        x = 80 + i * 6
        y = 160 + int(20 * np.sin(i / 6))
        cv2.rectangle(frame, (x, y), (x + 80, y + 40), (0, 255, 200), 2)
        cv2.rectangle(frame, (x + 90, y + 10), (x + 130, y + 30), (5, 5, 5), -1)
        writer.write(frame)
    writer.release()
    return video_path


def main() -> int:
    img = make_image()
    vid = make_video()
    print(f"Generated: {img}")
    print(f"Generated: {vid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


