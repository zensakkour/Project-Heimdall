from __future__ import annotations

from src.core.logic.filtering import update_posterior
from src.core.logic.types import Evidence, FusionCandidate, FusionResult, GeoCandidate, UncertaintyEllipse


def _make_result(points: list[tuple[float, float, float, str]], uncertainty_m: float) -> FusionResult:
    fused: list[FusionCandidate] = []
    for lat, lon, weight, match_id in points:
        cand = GeoCandidate(latitude=lat, longitude=lon, retrieval_score=weight, match_id=match_id)
        evidence = Evidence(
            retrieval_score=weight,
            shadow_residual_deg=None,
            terrain_residual=None,
            likelihoods={"retrieval": max(0.0, weight)},
            posterior_weight=weight,
            explanation="test",
        )
        fused.append(FusionCandidate(candidate=cand, posterior_weight=weight, evidence=evidence))
    top1 = max((w for _, _, w, _ in points), default=0.0)
    return FusionResult(
        candidates=fused,
        mean_latitude=points[0][0] if points else 0.0,
        mean_longitude=points[0][1] if points else 0.0,
        covariance_m=((0.0, 0.0), (0.0, 0.0)),
        ellipse=UncertaintyEllipse(major_axis_m=uncertainty_m, minor_axis_m=uncertainty_m, orientation_deg=0.0),
        uncertainty_radius_m=uncertainty_m,
        normalized_entropy=0.5,
        effective_candidate_count=max(1.0, float(len(points))),
        top1_posterior=top1,
        calibrated_top1_posterior=top1,
        top2_margin=0.0,
        confidence_tier="medium",
        ambiguous=False,
        credible_set_size=len(points),
    )


def test_update_posterior_reweights_toward_temporally_consistent_candidate() -> None:
    prior = _make_result(
        [
            (48.8566, 2.3522, 0.80, "paris"),
            (35.6764, 139.6500, 0.20, "tokyo"),
        ],
        uncertainty_m=2000.0,
    )
    current = _make_result(
        [
            (35.6764, 139.6500, 0.52, "tokyo"),
            (48.8566, 2.3522, 0.48, "paris"),
        ],
        uncertainty_m=2000.0,
    )

    updated = update_posterior(prior, current)
    assert updated.candidates[0].candidate.match_id == "paris"


def test_update_posterior_can_reduce_uncertainty_when_frames_agree() -> None:
    prior = _make_result([(48.8566, 2.3522, 1.0, "paris")], uncertainty_m=2000.0)
    current = _make_result([(48.8566, 2.3522, 1.0, "paris")], uncertainty_m=3000.0)

    updated = update_posterior(prior, current)
    assert updated.uncertainty_radius_m < current.uncertainty_radius_m


def test_update_posterior_no_candidates_passthrough() -> None:
    prior = _make_result([(48.8566, 2.3522, 1.0, "paris")], uncertainty_m=2000.0)
    current = _make_result([], uncertainty_m=3000.0)

    updated = update_posterior(prior, current)
    assert updated == current
