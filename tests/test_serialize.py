"""
Serialization tests for geo confidence tier.
"""
from __future__ import annotations

from core.logic.serialize import assessment_to_dict
from core.logic.types import Assessment, GeoEstimate


def test_confidence_tier_serialization() -> None:
    geo = GeoEstimate(latitude=0.0, longitude=0.0, confidence=0.8, landmarks=None, uncertainty_m=30.0)
    assessment = Assessment(detections=[], geo=geo, verification=None, score=0.5)
    payload = assessment_to_dict(assessment)
    assert payload["geo"]["confidence_tier"] == "high"
