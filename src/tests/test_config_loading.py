"""
Tests for config loading defaults/overrides.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.core.logic.config import load_config


def _write_config(tmpdir: str, payload: dict) -> Path:
    path = Path(tmpdir) / "cfg.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_config_spatial_consensus_defaults() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_config(tmpdir, {})
        cfg = load_config(str(path))

        assert cfg.fusion.use_spatial_consensus is True
        assert cfg.fusion.spatial_sigma_km == 2.0
        assert cfg.fusion.spatial_consensus_weight == 1.0
        assert cfg.fusion.source_prior_retrieval == 1.0
        assert cfg.fusion.source_prior_geoclip == 1.0
        assert cfg.fusion.source_prior_exif == 1.0
        assert cfg.fusion.credible_mass == 0.9
        assert cfg.fusion.min_credible_candidates == 2
        assert cfg.fusion.use_top_cluster_for_stats is True
        assert cfg.fusion.credible_cluster_radius_km == 500.0
        assert cfg.fusion.min_credible_cluster_weight == 0.35
        assert cfg.detector.min_area_px == 16.0
        assert cfg.detector.class_agnostic_nms is False
        assert cfg.detector.use_tta is False


def test_load_config_spatial_consensus_overrides() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_config(
            tmpdir,
            {
                "fusion": {
                    "use_spatial_consensus": False,
                    "spatial_sigma_km": 12.5,
                    "spatial_consensus_weight": 2.5,
                    "source_prior_retrieval": 0.95,
                    "source_prior_geoclip": 1.1,
                    "source_prior_exif": 0.8,
                    "credible_mass": 0.8,
                    "min_credible_candidates": 3,
                    "use_top_cluster_for_stats": False,
                    "credible_cluster_radius_km": 220.0,
                    "min_credible_cluster_weight": 0.5,
                },
                "detector": {
                    "min_area_px": 20.0,
                    "class_agnostic_nms": True,
                    "use_tta": True,
                }
            },
        )
        cfg = load_config(str(path))

        assert cfg.fusion.use_spatial_consensus is False
        assert cfg.fusion.spatial_sigma_km == 12.5
        assert cfg.fusion.spatial_consensus_weight == 2.5
        assert cfg.fusion.source_prior_retrieval == 0.95
        assert cfg.fusion.source_prior_geoclip == 1.1
        assert cfg.fusion.source_prior_exif == 0.8
        assert cfg.fusion.credible_mass == 0.8
        assert cfg.fusion.min_credible_candidates == 3
        assert cfg.fusion.use_top_cluster_for_stats is False
        assert cfg.fusion.credible_cluster_radius_km == 220.0
        assert cfg.fusion.min_credible_cluster_weight == 0.5
        assert cfg.detector.min_area_px == 20.0
        assert cfg.detector.class_agnostic_nms is True
        assert cfg.detector.use_tta is True
