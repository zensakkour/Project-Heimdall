from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.tools.run_geo_eval import (
    infer_dataset_scope,
    load_profile_scope,
    normalize_scope,
    validate_scope_alignment,
)


def test_normalize_scope_maps_expected_values() -> None:
    assert normalize_scope("paris") == "PARIS"
    assert normalize_scope("PARIS_TEST") == "PARIS"
    assert normalize_scope("open_geo") == "US"
    assert normalize_scope("usa") == "US"
    assert normalize_scope("GLOBAL") == "GLOBAL"
    assert normalize_scope("") == ""


def test_load_profile_scope_prefers_json_field() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "cfg.json"
        cfg_path.write_text(json.dumps({"profile_scope": "paris"}), encoding="utf-8")
        assert load_profile_scope(str(cfg_path)) == "PARIS"


def test_load_profile_scope_falls_back_to_filename() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "open_geo.json"
        cfg_path.write_text(json.dumps({"geolocator": {}}), encoding="utf-8")
        assert load_profile_scope(str(cfg_path)) == "US"


def test_infer_dataset_scope_from_paths() -> None:
    paris = infer_dataset_scope(
        Path("data/spacenet_paris_test/chips"),
        Path("data/spacenet_paris_test/metadata.csv"),
    )
    us = infer_dataset_scope(
        Path("data/open_geo/images"),
        Path("data/open_geo/metadata.csv"),
    )
    unknown = infer_dataset_scope(Path("data/custom/images"), Path("data/custom/meta.csv"))
    assert paris == "PARIS"
    assert us == "US"
    assert unknown == "UNKNOWN"


def test_validate_scope_alignment_raises_without_override() -> None:
    with pytest.raises(ValueError, match="scope mismatch"):
        validate_scope_alignment("US", "PARIS", allow_scope_mismatch=False)


def test_validate_scope_alignment_returns_warning_with_override() -> None:
    warning = validate_scope_alignment("US", "PARIS", allow_scope_mismatch=True)
    assert warning is not None
    assert "profile_scope='US'" in warning
    assert "dataset_scope='PARIS'" in warning


def test_validate_scope_alignment_accepts_match() -> None:
    assert validate_scope_alignment("PARIS", "PARIS", allow_scope_mismatch=False) is None

