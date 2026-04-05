from __future__ import annotations

import json
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from src.core.logic.config import (
    DetectorConfig,
    FusionConfig,
    GeoConfig,
    HeimdallConfig,
    ScoreConfig,
    VerificationConfig,
)
from src.tools import ui_server


def _png_bytes(width: int = 320, height: int = 200) -> bytes:
    img = Image.new("RGB", (width, height), color=(30, 44, 52))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _test_config() -> HeimdallConfig:
    return HeimdallConfig(
        detector=DetectorConfig(
            weights_path=None,
            min_confidence=0.1,
            nms_iou=0.5,
            max_detections=20,
            min_area_px=4.0,
            class_agnostic_nms=False,
            use_tta=False,
            use_sidecar=True,
            use_classic=False,
            imgsz=640,
        ),
        geolocator=GeoConfig(
            model_path=None,
            model_id=None,
            model_cache_dir=None,
            encoder_name=None,
            use_sidecar=True,
            use_exif=False,
            top_n=5,
            geospot_score_scale=1.0,
            retrieval_index_path=None,
            retrieval_model_id=None,
            retrieval_top_k=10,
            retrieval_min_score=0.1,
            candidate_dedupe_radius_m=150.0,
            candidate_max_results=20,
        ),
        fusion=FusionConfig(
            retrieval_temperature=0.2,
            retrieval_score_norm="none",
            source_prior_retrieval=1.0,
            source_prior_geoclip=1.0,
            source_prior_exif=1.0,
            use_spatial_consensus=True,
            spatial_sigma_km=2.0,
            spatial_consensus_weight=1.0,
            shadow_sigma_deg=20.0,
            terrain_sigma=100.0,
            use_shadow=False,
            use_terrain=False,
            credible_mass=0.9,
            min_credible_candidates=2,
            top_k=5,
        ),
        score=ScoreConfig(),
        verification=VerificationConfig(
            use_shadow=False,
            use_shadow_length=False,
            use_shadow_heading=False,
        ),
    )


def test_app_launch_routes() -> None:
    client = TestClient(ui_server.app)
    root = client.get("/")
    assert root.status_code == 200
    analysis = client.get("/analysis", follow_redirects=False)
    assert analysis.status_code in {307, 308}


def test_analyze_image_runs_pipeline_with_sidecars(monkeypatch) -> None:
    cfg = _test_config()
    monkeypatch.setattr(ui_server, "_load_config_from_env", lambda _profile=None: cfg)

    client = TestClient(ui_server.app)
    image_bytes = _png_bytes()
    det_sidecar = {
        "detections": [
            {
                "label": "vehicle",
                "confidence": 0.91,
                "obb": [[40, 40], [120, 40], [120, 95], [40, 95]],
                "shadow_azimuth_deg": 122.0,
                "shadow_length_ratio": 1.1,
            }
        ]
    }
    geo_sidecar = {
        "latitude": 48.8566,
        "longitude": 2.3522,
        "confidence": 0.71,
        "uncertainty_m": 120.0,
        "landmarks": ["sidecar-geo"],
        "candidates": [
            {
                "latitude": 48.8566,
                "longitude": 2.3522,
                "retrieval_score": 0.85,
                "match_id": "cand-1",
            },
            {
                "latitude": 48.8571,
                "longitude": 2.3530,
                "retrieval_score": 0.72,
                "match_id": "cand-2",
            },
        ],
    }

    files = {
        "image": ("sample.png", image_bytes, "image/png"),
        "det_json": ("sample.detections.json", json.dumps(det_sidecar).encode("utf-8"), "application/json"),
        "geo_json": ("sample.geo.json", json.dumps(geo_sidecar).encode("utf-8"), "application/json"),
    }
    res = client.post("/analyze/image", files=files)
    assert res.status_code == 200

    payload = res.json()
    assert payload["safe_demo"] is False
    assert payload["result"]["detections"]
    assert payload["result"]["detections"][0]["label"] == "vehicle"
    assert payload["result"]["geo"] is not None
    assert payload["result"]["geo"]["latitude"] == 48.8566
    assert payload["result"]["fusion"] is not None
    assert payload["result"]["fusion"]["candidates"]
    assert payload["geo_debug"]["safe_demo"] is False
