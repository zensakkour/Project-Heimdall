"""
Tests for geo fusion robustness improvements.
"""
from __future__ import annotations

import math

from src.core.logic.config import FusionConfig
from src.core.logic.fusion import fuse_candidates
from src.core.logic.types import GeoCandidate


def test_fusion_handles_dateline_longitudes() -> None:
    candidates = [
        GeoCandidate(latitude=10.0, longitude=179.8, retrieval_score=0.90, match_id="a"),
        GeoCandidate(latitude=10.0, longitude=-179.8, retrieval_score=0.89, match_id="b"),
    ]
    cfg = FusionConfig(
        retrieval_temperature=0.4,
        retrieval_score_norm="none",
        use_shadow=False,
        use_terrain=False,
        top_k=2,
    )

    result = fuse_candidates("missing.jpg", candidates, detections=[], config=cfg)

    assert result is not None
    assert abs(result.mean_longitude) > 170.0


def test_fusion_zscore_sigmoid_constant_scores_remains_stable() -> None:
    candidates = [
        GeoCandidate(latitude=35.0, longitude=2.0, retrieval_score=0.5, match_id="retrieval:1"),
        GeoCandidate(latitude=35.1, longitude=2.2, retrieval_score=0.5, match_id="retrieval:2"),
        GeoCandidate(latitude=35.2, longitude=2.4, retrieval_score=0.5, match_id="retrieval:3"),
    ]
    cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="zscore_sigmoid",
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )

    result = fuse_candidates("missing.jpg", candidates, detections=[], config=cfg)

    assert result is not None
    assert len(result.candidates) == 3
    weights = [item.posterior_weight for item in result.candidates]
    assert all(math.isfinite(w) for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-6


def test_fusion_rank_exp_mode_prefers_higher_ranked_candidates() -> None:
    candidates = [
        GeoCandidate(latitude=0.0, longitude=0.0, retrieval_score=0.1, match_id="low"),
        GeoCandidate(latitude=1.0, longitude=1.0, retrieval_score=0.4, match_id="high"),
        GeoCandidate(latitude=2.0, longitude=2.0, retrieval_score=0.2, match_id="mid"),
    ]
    cfg = FusionConfig(
        retrieval_temperature=0.5,
        retrieval_score_norm="rank_exp",
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )

    result = fuse_candidates("missing.jpg", candidates, detections=[], config=cfg)

    assert result is not None
    assert result.candidates[0].candidate.match_id == "high"
