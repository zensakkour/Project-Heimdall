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


def test_load_config_spatial_consensus_overrides() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_config(
            tmpdir,
            {
                "fusion": {
                    "use_spatial_consensus": False,
                    "spatial_sigma_km": 12.5,
                    "spatial_consensus_weight": 2.5,
                }
            },
        )
        cfg = load_config(str(path))

        assert cfg.fusion.use_spatial_consensus is False
        assert cfg.fusion.spatial_sigma_km == 12.5
        assert cfg.fusion.spatial_consensus_weight == 2.5
