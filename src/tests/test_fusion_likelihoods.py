"""
Tests for fusion likelihood helpers.
"""
from __future__ import annotations

import math

from src.core.logic.likelihoods import gaussian_likelihood, von_mises_likelihood


def test_gaussian_likelihood_decreases_with_residual() -> None:
    near = gaussian_likelihood(1.0, sigma=5.0)
    far = gaussian_likelihood(10.0, sigma=5.0)
    assert near > far


def test_von_mises_prefers_zero_angle() -> None:
    peak = von_mises_likelihood(0.0, kappa=2.0)
    opposite = von_mises_likelihood(math.pi, kappa=2.0)
    assert peak > opposite


