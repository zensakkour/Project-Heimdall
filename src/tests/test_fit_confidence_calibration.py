from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.tools.fit_confidence_calibration import (
    _apply_calibration_to_config,
    _build_fusion_calibration_patch,
    calibrate_probability,
    find_best_logit_calibration,
)
from src.tools.eval_metrics import compute_nll


def test_calibrate_probability_bounds() -> None:
    assert 0.0 <= calibrate_probability(0.9, 1.0, 0.0) <= 1.0
    assert 0.0 <= calibrate_probability(0.1, 1.5, -0.2) <= 1.0


def test_find_best_calibration_improves_nll() -> None:
    confidences = [0.95, 0.9, 0.85, 0.8, 0.9, 0.95]
    correctness = [1, 1, 1, 0, 0, 0]
    raw_nll = compute_nll(confidences, correctness)
    best = find_best_logit_calibration(
        confidences,
        correctness,
        scale_values=[0.5, 1.0, 1.5],
        bias_values=[-0.8, -0.4, 0.0],
    )
    assert best["nll"] <= raw_nll


def test_build_fusion_calibration_patch_enforces_monotonic_thresholds() -> None:
    patch = _build_fusion_calibration_patch(1.1, -0.2, high_threshold=0.55, medium_threshold=0.80)
    assert patch["confidence_high_threshold"] == 0.55
    assert patch["confidence_medium_threshold"] == 0.55


def test_apply_calibration_to_config_writes_patch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "cfg.json"
        config_path.write_text(json.dumps({"fusion": {"confidence_high_threshold": 0.70}}), encoding="utf-8")
        patch = _build_fusion_calibration_patch(1.3, -0.1, high_threshold=0.76, medium_threshold=0.50)
        _apply_calibration_to_config(config_path, patch)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        fusion = payload["fusion"]
        assert fusion["confidence_calibration_logit_scale"] == 1.3
        assert fusion["confidence_calibration_logit_bias"] == -0.1
        assert fusion["confidence_high_threshold"] == 0.76
        assert fusion["confidence_medium_threshold"] == 0.5
