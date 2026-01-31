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

    mean_lat = sum(item.candidate.latitude * item.posterior_weight for item in top)
    mean_lon = sum(item.candidate.longitude * item.posterior_weight for item in top)

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
    )


