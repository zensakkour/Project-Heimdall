"""
Serialization tests for geo confidence tier.
"""
from __future__ import annotations

from src.core.logic.serialize import assessment_to_dict
from src.core.logic.types import (
    Assessment,
    Evidence,
    FusionCandidate,
    FusionResult,
    GeoCandidate,
    GeoEstimate,
    UncertaintyEllipse,
)


def test_confidence_tier_serialization() -> None:
    geo = GeoEstimate(latitude=0.0, longitude=0.0, confidence=0.8, landmarks=None, uncertainty_m=30.0)
    assessment = Assessment(detections=[], geo=geo, verification=None, score=0.5)
    payload = assessment_to_dict(assessment)
    assert payload["geo"]["confidence_tier"] == "high"


def test_fusion_diagnostics_serialization() -> None:
    candidate = GeoCandidate(latitude=10.0, longitude=20.0, retrieval_score=0.7, match_id="x")
    fusion = FusionResult(
        candidates=[
            FusionCandidate(
                candidate=candidate,
                posterior_weight=1.0,
                evidence=Evidence(
                    retrieval_score=0.7,
                    shadow_residual_deg=None,
                    terrain_residual=None,
                    likelihoods={"retrieval": 0.7},
                    posterior_weight=1.0,
                    explanation="unit test",
                ),
            )
        ],
        mean_latitude=10.0,
        mean_longitude=20.0,
        covariance_m=((1.0, 0.0), (0.0, 1.0)),
        ellipse=UncertaintyEllipse(major_axis_m=2.0, minor_axis_m=1.0, orientation_deg=0.0),
        uncertainty_radius_m=2.0,
        normalized_entropy=0.1,
        effective_candidate_count=1.2,
        top1_posterior=0.9,
        top2_margin=0.8,
        confidence_tier="high",
        ambiguous=False,
        credible_set_size=1,
    )
    assessment = Assessment(detections=[], geo=None, verification=None, score=0.1, fusion=fusion)
    payload = assessment_to_dict(assessment)

    assert payload["fusion"]["confidence_tier"] == "high"
    assert payload["fusion"]["ambiguous"] is False
    assert payload["fusion"]["top1_posterior"] == 0.9
    assert payload["fusion"]["credible_set_size"] == 1


