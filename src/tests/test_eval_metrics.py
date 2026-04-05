"""
Tests for evaluation metric helpers.
"""
from __future__ import annotations

from src.tools.eval_metrics import (
    compute_brier,
    compute_ece,
    compute_nll,
    extract_confidence_tier,
    extract_top1_cross_source_support,
)


def test_brier_zero_for_perfect_predictions() -> None:
    confidences = [0.99, 0.98, 0.01, 0.02]
    correctness = [1, 1, 0, 0]
    assert compute_brier(confidences, correctness) < 0.001


def test_nll_penalizes_overconfident_errors() -> None:
    good = compute_nll([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    bad = compute_nll([0.99, 0.99, 0.99, 0.99], [1, 1, 0, 0])
    assert good < bad


def test_ece_zero_when_calibrated_per_bin() -> None:
    confidences = [0.2, 0.2, 0.8, 0.8]
    correctness = [0, 0, 1, 1]
    ece = compute_ece(confidences, correctness, bins=5)
    assert 0.0 <= ece < 0.21


def test_extract_fusion_confidence_diagnostics() -> None:
    row = {
        "result": {
            "fusion": {
                "confidence_tier": "High",
                "top1_cross_source_support": 1.2,
            }
        }
    }
    assert extract_confidence_tier(row) == "high"
    assert extract_top1_cross_source_support(row) == 1.0
