from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.tools.fit_fusion_priors import (
    _apply_priors_to_config,
    _estimate_source_reliability,
    _recommended_priors,
    _recommended_retrieval_source_priors,
)


def test_recommended_priors_reflect_source_reliability() -> None:
    rows = [
        {
            "image": "a.jpg",
            "result": {
                "fusion": {
                    "candidates": [
                        {"candidate": {"latitude": 10.0, "longitude": 10.0, "match_id": "retrieval:x"}, "posterior_weight": 0.8},
                        {"candidate": {"latitude": 10.0, "longitude": 10.0, "match_id": "geoclip"}, "posterior_weight": 0.7},
                    ]
                }
            },
        },
        {
            "image": "b.jpg",
            "result": {
                "fusion": {
                    "candidates": [
                        {"candidate": {"latitude": 20.0, "longitude": 20.0, "match_id": "retrieval:y"}, "posterior_weight": 0.8},
                        {"candidate": {"latitude": 0.0, "longitude": 0.0, "match_id": "geoclip"}, "posterior_weight": 0.7},
                    ]
                }
            },
        },
    ]
    gt = {
        "a.jpg": type("GT", (), {"latitude": 10.0, "longitude": 10.0})(),
        "b.jpg": type("GT", (), {"latitude": 20.0, "longitude": 20.0})(),
    }

    stats = _estimate_source_reliability(rows, gt, radius_km=1.0)
    priors = _recommended_priors(stats, smoothing=0.0)

    assert stats["retrieval"]["hits"] == 2
    assert stats["geoclip"]["hits"] == 1
    assert priors["source_prior_retrieval"] > priors["source_prior_geoclip"]


def test_estimate_source_reliability_tracks_retrieval_by_source() -> None:
    rows = [
        {
            "image": "a.jpg",
            "result": {
                "fusion": {
                    "candidates": [
                        {
                            "candidate": {
                                "latitude": 10.0,
                                "longitude": 10.0,
                                "match_id": "retrieval:spacenet_paris_clip:a",
                            },
                            "posterior_weight": 0.8,
                        },
                        {
                            "candidate": {
                                "latitude": 0.0,
                                "longitude": 0.0,
                                "match_id": "retrieval:open_geo_clip:a",
                            },
                            "posterior_weight": 0.7,
                        },
                    ]
                }
            },
        },
        {
            "image": "b.jpg",
            "result": {
                "fusion": {
                    "candidates": [
                        {
                            "candidate": {
                                "latitude": 20.0,
                                "longitude": 20.0,
                                "match_id": "retrieval:spacenet_paris_clip:b",
                            },
                            "posterior_weight": 0.9,
                        },
                        {
                            "candidate": {
                                "latitude": 1.0,
                                "longitude": 1.0,
                                "match_id": "retrieval:open_geo_clip:b",
                            },
                            "posterior_weight": 0.7,
                        },
                    ]
                }
            },
        },
    ]
    gt = {
        "a.jpg": type("GT", (), {"latitude": 10.0, "longitude": 10.0})(),
        "b.jpg": type("GT", (), {"latitude": 20.0, "longitude": 20.0})(),
    }
    stats = _estimate_source_reliability(rows, gt, radius_km=1.0)
    by_source = stats["retrieval_by_source"]
    assert by_source["retrieval:spacenet_paris_clip"]["hits"] == 2
    assert by_source["retrieval:open_geo_clip"]["hits"] == 0


def test_recommended_retrieval_source_priors_prioritize_more_reliable_source() -> None:
    source_stats = {
        "retrieval:spacenet_paris_clip": {"count": 8, "hits": 7},
        "retrieval:open_geo_clip": {"count": 8, "hits": 2},
        "retrieval:tiny_source": {"count": 1, "hits": 1},
    }
    priors = _recommended_retrieval_source_priors(
        source_stats,
        global_retrieval_prior=1.0,
        smoothing=0.0,
        min_count=3,
    )
    assert "retrieval:tiny_source" not in priors
    assert priors["retrieval:spacenet_paris_clip"] > priors["retrieval:open_geo_clip"]


def test_apply_priors_to_config_writes_global_and_per_source_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "cfg.json"
        config_path.write_text(json.dumps({"fusion": {"source_prior_retrieval": 1.0}}), encoding="utf-8")
        _apply_priors_to_config(
            config_path,
            {
                "source_prior_retrieval": 1.4,
                "source_prior_geoclip": 0.9,
                "source_prior_exif": 0.8,
            },
            {
                "retrieval:open_geo_clip": 0.75,
                "retrieval:spacenet_paris_clip": 1.45,
            },
        )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        fusion = payload["fusion"]
        assert fusion["source_prior_retrieval"] == 1.4
        assert fusion["source_prior_geoclip"] == 0.9
        assert fusion["source_prior_exif"] == 0.8
        assert fusion["source_prior_retrieval_by_source"] == {
            "retrieval:open_geo_clip": 0.75,
            "retrieval:spacenet_paris_clip": 1.45,
        }
