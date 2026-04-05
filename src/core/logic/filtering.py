"""
Bayesian filtering stub for temporal fusion.
"""
from __future__ import annotations

from typing import List

from .types import FusionResult, UncertaintyEllipse


def update_posterior(prior: FusionResult, current: FusionResult) -> FusionResult:
    combined_candidates = prior.candidates + current.candidates
    combined_candidates.sort(key=lambda item: item.posterior_weight, reverse=True)
    top = combined_candidates[: max(len(current.candidates), 1)]

    if not top:
        return current

    total_weight = sum(max(0.0, item.posterior_weight) for item in top)
    if total_weight <= 0.0:
        total_weight = 1.0
    mean_lat = sum(item.candidate.latitude * item.posterior_weight for item in top) / total_weight
    mean_lon = sum(item.candidate.longitude * item.posterior_weight for item in top) / total_weight

    return FusionResult(
        candidates=top,
        mean_latitude=mean_lat,
        mean_longitude=mean_lon,
        covariance_m=current.covariance_m,
        ellipse=UncertaintyEllipse(
            major_axis_m=current.ellipse.major_axis_m,
            minor_axis_m=current.ellipse.minor_axis_m,
            orientation_deg=current.ellipse.orientation_deg,
        ),
        uncertainty_radius_m=current.uncertainty_radius_m,
        normalized_entropy=current.normalized_entropy,
        effective_candidate_count=current.effective_candidate_count,
        top1_posterior=current.top1_posterior,
        top2_margin=current.top2_margin,
        confidence_tier=current.confidence_tier,
        ambiguous=current.ambiguous,
        credible_set_size=current.credible_set_size,
    )


