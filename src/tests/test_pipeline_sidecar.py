"""
Pipeline sidecar smoke test.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from src.core.detection import SidecarDetector
from src.core.geo import GeoLocator
from src.core.logic.pipeline import HeimdallPipeline


def test_pipeline_with_sidecars() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "sample.jpg"
        Image.new("RGB", (32, 32), color=(0, 0, 0)).save(image_path)

        detections = [
            {
                "label": "vehicle",
                "confidence": 0.9,
                "obb": [[0, 0], [10, 0], [10, 5], [0, 5]],
                "shadow_azimuth_deg": 200.0,
                "shadow_length_ratio": 2.0,
            }
        ]
        (root / "sample.jpg.detections.json").write_text(
            json.dumps({"detections": detections}), encoding="utf-8"
        )
        (root / "sample.geo.json").write_text(
            json.dumps(
                {
                    "latitude": 35.0,
                    "longitude": -120.0,
                    "confidence": 0.7,
                    "uncertainty_m": 80.0,
                    "landmarks": ["sidecar"],
                }
            ),
            encoding="utf-8",
        )

        detector = SidecarDetector()
        geolocator = GeoLocator(use_sidecar=True, use_exif=False)
        pipeline = HeimdallPipeline(detector=detector, geolocator=geolocator)

        result = pipeline.run(str(image_path))
        assert len(result.detections) == 1
        assert result.geo is not None
        assert result.geo.uncertainty_m == 80.0


