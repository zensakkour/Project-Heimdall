"""
Score computation sanity checks.
"""
from __future__ import annotations

from src.core.logic.score import compute_score
from src.core.logic.types import Detection, GeoEstimate, Verification


def _box(x1: float, y1: float, x2: float, y2: float):
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def test_score_with_geo_and_verification() -> None:
    detections = [Detection(label="tank", confidence=0.8, obb=_box(0, 0, 10, 10))]
    geo = GeoEstimate(
        latitude=1.0,
        longitude=2.0,
        confidence=0.6,
        landmarks=None,
        uncertainty_m=100.0,
    )
    verification = Verification(shadow_ok=True, topo_ok=False, notes=None)
    score = compute_score(detections, geo, verification)
    assert 0.0 <= score <= 1.0
    assert score > 0.5


