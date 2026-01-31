"""
Tests for geo candidate sidecar loading.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.core.geo.geoclip_provider import GeoCLIPProvider


def test_sidecar_candidates_list() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "sample.jpg"
        image_path.write_text("", encoding="utf-8")

        sidecar = root / "sample.geo.json"
        sidecar.write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "latitude": 35.0,
                            "longitude": -120.0,
                            "retrieval_score": 0.9,
                            "match_id": "tile-1",
                        },
                        {
                            "latitude": 36.0,
                            "longitude": -121.0,
                            "retrieval_score": 0.7,
                            "match_id": "tile-2",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        provider = GeoCLIPProvider(use_sidecar=True, use_exif=False, top_n=5)
        candidates = provider.candidates(str(image_path))

        assert len(candidates) == 2
        assert candidates[0].match_id == "tile-1"
        assert candidates[1].match_id == "tile-2"


def test_sidecar_single_candidate() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_path = root / "sample.jpg"
        image_path.write_text("", encoding="utf-8")

        sidecar = root / "sample.geoloc.json"
        sidecar.write_text(
            json.dumps(
                {
                    "latitude": 10.0,
                    "longitude": 20.0,
                    "retrieval_score": 0.4,
                    "match_id": "single",
                }
            ),
            encoding="utf-8",
        )

        provider = GeoCLIPProvider(use_sidecar=True, use_exif=False, top_n=5)
        candidates = provider.candidates(str(image_path))

        assert len(candidates) == 1
        assert candidates[0].match_id == "single"


