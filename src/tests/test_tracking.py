from __future__ import annotations

from src.core.logic.tracking import TrackState, associate_track
from src.core.logic.types import Evidence, FusionCandidate, FusionResult, GeoCandidate, UncertaintyEllipse


def _fusion_result(lat: float, lon: float, uncertainty_m: float) -> FusionResult:
    candidate = GeoCandidate(latitude=lat, longitude=lon, retrieval_score=1.0, match_id="test")
    evidence = Evidence(
        retrieval_score=1.0,
        shadow_residual_deg=None,
        terrain_residual=None,
        likelihoods={"retrieval": 1.0},
        posterior_weight=1.0,
        explanation="test",
    )
    fused_candidate = FusionCandidate(candidate=candidate, posterior_weight=1.0, evidence=evidence)
    return FusionResult(
        candidates=[fused_candidate],
        mean_latitude=lat,
        mean_longitude=lon,
        covariance_m=((0.0, 0.0), (0.0, 0.0)),
        ellipse=UncertaintyEllipse(major_axis_m=uncertainty_m, minor_axis_m=uncertainty_m, orientation_deg=0.0),
        uncertainty_radius_m=uncertainty_m,
        normalized_entropy=0.0,
        effective_candidate_count=1.0,
        top1_posterior=1.0,
        calibrated_top1_posterior=1.0,
        top2_margin=1.0,
        confidence_tier="high",
        ambiguous=False,
        credible_set_size=1,
    )


def test_associate_track_respects_base_gate_for_low_uncertainty() -> None:
    prior = _fusion_result(0.0, 0.0, uncertainty_m=100.0)
    current = _fusion_result(0.0, 0.06, uncertainty_m=100.0)  # ~6.7 km at equator.
    tracks = [TrackState(track_id="t1", fused_history=[prior])]
    assert associate_track(tracks, current, max_distance_m=5000.0) is None


def test_associate_track_expands_gate_for_high_uncertainty() -> None:
    prior = _fusion_result(0.0, 0.0, uncertainty_m=4000.0)
    current = _fusion_result(0.0, 0.06, uncertainty_m=4000.0)  # ~6.7 km at equator.
    tracks = [TrackState(track_id="t1", fused_history=[prior])]
    assert associate_track(tracks, current, max_distance_m=5000.0) is not None


def test_associate_track_handles_dateline_crossing() -> None:
    prior = _fusion_result(0.0, 179.9, uncertainty_m=100.0)
    current = _fusion_result(0.0, -179.9, uncertainty_m=100.0)
    tracks = [TrackState(track_id="t1", fused_history=[prior])]
    assert associate_track(tracks, current, max_distance_m=30000.0) is not None
