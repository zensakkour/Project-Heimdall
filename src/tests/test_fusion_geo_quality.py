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


def test_spatial_consensus_can_downrank_isolated_outlier() -> None:
    candidates = [
        GeoCandidate(latitude=34.0522, longitude=-118.2437, retrieval_score=0.80, match_id="outlier"),
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.72, match_id="cluster_a"),
        GeoCandidate(latitude=48.8572, longitude=2.3530, retrieval_score=0.70, match_id="cluster_b"),
    ]
    base_cfg = FusionConfig(
        retrieval_temperature=0.30,
        retrieval_score_norm="none",
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    with_consensus_cfg = FusionConfig(
        retrieval_temperature=0.30,
        retrieval_score_norm="none",
        use_spatial_consensus=True,
        spatial_sigma_km=8.0,
        spatial_consensus_weight=6.0,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    without_consensus_cfg = FusionConfig(
        retrieval_temperature=0.30,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )

    no_consensus = fuse_candidates("missing.jpg", candidates, detections=[], config=without_consensus_cfg)
    with_consensus = fuse_candidates("missing.jpg", candidates, detections=[], config=with_consensus_cfg)
    base = fuse_candidates("missing.jpg", candidates, detections=[], config=base_cfg)

    assert no_consensus is not None
    assert with_consensus is not None
    assert base is not None
    assert no_consensus.candidates[0].candidate.match_id == "outlier"
    assert base.candidates[0].candidate.match_id == "outlier"
    assert with_consensus.candidates[0].candidate.match_id in {"cluster_a", "cluster_b"}


def test_spatial_consensus_zero_weight_behaves_like_disabled() -> None:
    candidates = [
        GeoCandidate(latitude=10.0, longitude=10.0, retrieval_score=0.60, match_id="a"),
        GeoCandidate(latitude=10.1, longitude=10.1, retrieval_score=0.59, match_id="b"),
        GeoCandidate(latitude=42.0, longitude=-7.0, retrieval_score=0.61, match_id="c"),
    ]
    disabled_cfg = FusionConfig(
        retrieval_temperature=0.4,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    zero_weight_cfg = FusionConfig(
        retrieval_temperature=0.4,
        retrieval_score_norm="none",
        use_spatial_consensus=True,
        spatial_sigma_km=5.0,
        spatial_consensus_weight=0.0,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )

    disabled = fuse_candidates("missing.jpg", candidates, detections=[], config=disabled_cfg)
    zero_weight = fuse_candidates("missing.jpg", candidates, detections=[], config=zero_weight_cfg)

    assert disabled is not None
    assert zero_weight is not None
    assert [item.candidate.match_id for item in disabled.candidates] == [
        item.candidate.match_id for item in zero_weight.candidates
    ]


def test_source_priors_can_shift_ranking() -> None:
    candidates = [
        GeoCandidate(latitude=35.0, longitude=10.0, retrieval_score=0.70, match_id="retrieval:tile-1"),
        GeoCandidate(latitude=35.01, longitude=10.01, retrieval_score=0.69, match_id="geoclip"),
    ]
    neutral_cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="none",
        source_prior_retrieval=1.0,
        source_prior_geoclip=1.0,
        source_prior_exif=1.0,
        use_spatial_consensus=False,
        use_shadow=False,
        use_terrain=False,
        top_k=2,
    )
    geoclip_boost_cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="none",
        source_prior_retrieval=0.6,
        source_prior_geoclip=1.8,
        source_prior_exif=1.0,
        use_spatial_consensus=False,
        use_shadow=False,
        use_terrain=False,
        top_k=2,
    )

    neutral = fuse_candidates("missing.jpg", candidates, detections=[], config=neutral_cfg)
    boosted = fuse_candidates("missing.jpg", candidates, detections=[], config=geoclip_boost_cfg)

    assert neutral is not None
    assert boosted is not None
    assert neutral.candidates[0].candidate.match_id == "retrieval:tile-1"
    assert boosted.candidates[0].candidate.match_id == "geoclip"


def test_fusion_outputs_confidence_diagnostics() -> None:
    candidates = [
        GeoCandidate(latitude=0.0, longitude=0.0, retrieval_score=0.95, match_id="a"),
        GeoCandidate(latitude=0.1, longitude=0.1, retrieval_score=0.20, match_id="b"),
        GeoCandidate(latitude=0.2, longitude=0.2, retrieval_score=0.10, match_id="c"),
    ]
    cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    result = fuse_candidates("missing.jpg", candidates, detections=[], config=cfg)
    assert result is not None
    assert 0.0 <= result.normalized_entropy <= 1.0
    assert result.effective_candidate_count >= 1.0
    assert 0.0 <= result.top1_posterior <= 1.0
    assert 0.0 <= result.top2_margin <= 1.0
    assert result.confidence_tier in {"low", "medium", "high"}
    assert isinstance(result.ambiguous, bool)


def test_credible_set_stats_reduce_far_tail_bias() -> None:
    candidates = [
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.95, match_id="core_a"),
        GeoCandidate(latitude=48.8570, longitude=2.3530, retrieval_score=0.85, match_id="core_b"),
        GeoCandidate(latitude=35.6764, longitude=139.6500, retrieval_score=0.30, match_id="tail"),
    ]
    full_cfg = FusionConfig(
        retrieval_temperature=0.4,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        credible_mass=0.999,
        min_credible_candidates=3,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    credible_cfg = FusionConfig(
        retrieval_temperature=0.4,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        credible_mass=0.75,
        min_credible_candidates=2,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )

    full = fuse_candidates("missing.jpg", candidates, detections=[], config=full_cfg)
    credible = fuse_candidates("missing.jpg", candidates, detections=[], config=credible_cfg)
    assert full is not None
    assert credible is not None

    assert credible.credible_set_size <= full.credible_set_size
    # Credible-set statistics should stay closer to the dominant Paris cluster.
    assert abs(credible.mean_latitude - 48.8568) < abs(full.mean_latitude - 48.8568)


def test_top_cluster_stats_prevent_midpoint_between_modes() -> None:
    candidates = [
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.91, match_id="paris_a"),
        GeoCandidate(latitude=35.6764, longitude=139.6500, retrieval_score=0.90, match_id="tokyo"),
        GeoCandidate(latitude=48.8570, longitude=2.3530, retrieval_score=0.89, match_id="paris_b"),
    ]
    no_cluster_cfg = FusionConfig(
        retrieval_temperature=0.6,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        credible_mass=0.99,
        min_credible_candidates=2,
        use_top_cluster_for_stats=False,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    cluster_cfg = FusionConfig(
        retrieval_temperature=0.6,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        credible_mass=0.99,
        min_credible_candidates=2,
        use_top_cluster_for_stats=True,
        credible_cluster_radius_km=250.0,
        min_credible_cluster_weight=0.25,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )

    no_cluster = fuse_candidates("missing.jpg", candidates, detections=[], config=no_cluster_cfg)
    cluster = fuse_candidates("missing.jpg", candidates, detections=[], config=cluster_cfg)
    assert no_cluster is not None
    assert cluster is not None

    paris_lat = 48.8568
    assert abs(cluster.mean_latitude - paris_lat) < abs(no_cluster.mean_latitude - paris_lat)
    assert cluster.credible_set_size <= no_cluster.credible_set_size
