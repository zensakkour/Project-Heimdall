"""
Tests for geo fusion robustness improvements.
"""
from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.core.logic.config import FusionConfig
from src.core.logic.fusion import fuse_candidates
from src.core.logic.types import Detection, GeoCandidate


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


def test_retrieval_source_priors_can_shift_multi_index_ranking() -> None:
    candidates = [
        GeoCandidate(
            latitude=48.8566,
            longitude=2.3522,
            retrieval_score=0.68,
            match_id="retrieval:spacenet_paris_clip:tile-1",
        ),
        GeoCandidate(
            latitude=48.8567,
            longitude=2.3523,
            retrieval_score=0.70,
            match_id="retrieval:open_geo_clip:tile-2",
        ),
    ]
    neutral_cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="none",
        source_prior_retrieval=1.0,
        source_prior_retrieval_by_source=None,
        use_spatial_consensus=False,
        use_shadow=False,
        use_terrain=False,
        top_k=2,
    )
    dataset_prior_cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="none",
        source_prior_retrieval=1.0,
        source_prior_retrieval_by_source={
            "retrieval:spacenet_paris_clip": 1.8,
            "retrieval:open_geo_clip": 0.55,
        },
        use_spatial_consensus=False,
        use_shadow=False,
        use_terrain=False,
        top_k=2,
    )
    neutral = fuse_candidates("missing.jpg", candidates, detections=[], config=neutral_cfg)
    tuned = fuse_candidates("missing.jpg", candidates, detections=[], config=dataset_prior_cfg)
    assert neutral is not None
    assert tuned is not None
    assert neutral.candidates[0].candidate.match_id == "retrieval:open_geo_clip:tile-2"
    assert tuned.candidates[0].candidate.match_id == "retrieval:spacenet_paris_clip:tile-1"


def test_cross_source_agreement_legacy_retrieval_ids_still_single_source_family() -> None:
    candidates = [
        GeoCandidate(latitude=40.0, longitude=-74.0, retrieval_score=0.72, match_id="retrieval:a"),
        GeoCandidate(latitude=41.0, longitude=-73.0, retrieval_score=0.68, match_id="retrieval:b"),
        GeoCandidate(latitude=39.5, longitude=-75.0, retrieval_score=0.51, match_id="retrieval:c"),
    ]
    disabled_cfg = FusionConfig(
        retrieval_temperature=0.35,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    enabled_cfg = FusionConfig(
        retrieval_temperature=0.35,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=True,
        cross_source_sigma_km=10.0,
        cross_source_weight=4.0,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    disabled = fuse_candidates("missing.jpg", candidates, detections=[], config=disabled_cfg)
    enabled = fuse_candidates("missing.jpg", candidates, detections=[], config=enabled_cfg)
    assert disabled is not None
    assert enabled is not None
    assert [item.candidate.match_id for item in disabled.candidates] == [
        item.candidate.match_id for item in enabled.candidates
    ]


def test_cross_source_agreement_can_downrank_source_isolated_outlier() -> None:
    candidates = [
        GeoCandidate(latitude=34.0522, longitude=-118.2437, retrieval_score=0.84, match_id="retrieval:outlier"),
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.78, match_id="retrieval:paris"),
        GeoCandidate(latitude=48.8570, longitude=2.3530, retrieval_score=0.76, match_id="geoclip"),
    ]
    without_cross_cfg = FusionConfig(
        retrieval_temperature=0.25,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    with_cross_cfg = FusionConfig(
        retrieval_temperature=0.25,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=True,
        cross_source_sigma_km=8.0,
        cross_source_weight=5.0,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )

    no_cross = fuse_candidates("missing.jpg", candidates, detections=[], config=without_cross_cfg)
    with_cross = fuse_candidates("missing.jpg", candidates, detections=[], config=with_cross_cfg)
    assert no_cross is not None
    assert with_cross is not None
    assert no_cross.candidates[0].candidate.match_id == "retrieval:outlier"
    assert with_cross.candidates[0].candidate.match_id in {"retrieval:paris", "geoclip"}


def test_cross_source_agreement_noop_with_single_source_family() -> None:
    candidates = [
        GeoCandidate(latitude=40.0, longitude=-74.0, retrieval_score=0.72, match_id="retrieval:a"),
        GeoCandidate(latitude=41.0, longitude=-73.0, retrieval_score=0.68, match_id="retrieval:b"),
        GeoCandidate(latitude=39.5, longitude=-75.0, retrieval_score=0.51, match_id="retrieval:c"),
    ]
    disabled_cfg = FusionConfig(
        retrieval_temperature=0.35,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    enabled_cfg = FusionConfig(
        retrieval_temperature=0.35,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=True,
        cross_source_sigma_km=10.0,
        cross_source_weight=4.0,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )

    disabled = fuse_candidates("missing.jpg", candidates, detections=[], config=disabled_cfg)
    enabled = fuse_candidates("missing.jpg", candidates, detections=[], config=enabled_cfg)
    assert disabled is not None
    assert enabled is not None
    assert [item.candidate.match_id for item in disabled.candidates] == [
        item.candidate.match_id for item in enabled.candidates
    ]


def test_plausibility_rerank_can_prefer_supported_cluster() -> None:
    candidates = [
        GeoCandidate(latitude=34.0522, longitude=-118.2437, retrieval_score=0.86, match_id="retrieval:outlier"),
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.80, match_id="retrieval:paris_a"),
        GeoCandidate(latitude=48.8570, longitude=2.3530, retrieval_score=0.79, match_id="retrieval:paris_b"),
    ]
    base_cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=False,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    rerank_cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=True,
        plausibility_radius_km=40.0,
        plausibility_weight=6.0,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )

    base = fuse_candidates("missing.jpg", candidates, detections=[], config=base_cfg)
    reranked = fuse_candidates("missing.jpg", candidates, detections=[], config=rerank_cfg)
    assert base is not None
    assert reranked is not None
    assert base.candidates[0].candidate.match_id == "retrieval:outlier"
    assert reranked.candidates[0].candidate.match_id in {"retrieval:paris_a", "retrieval:paris_b"}


def test_plausibility_rerank_zero_weight_matches_disabled() -> None:
    candidates = [
        GeoCandidate(latitude=10.0, longitude=10.0, retrieval_score=0.62, match_id="a"),
        GeoCandidate(latitude=10.05, longitude=10.05, retrieval_score=0.61, match_id="b"),
        GeoCandidate(latitude=40.0, longitude=-72.0, retrieval_score=0.60, match_id="c"),
    ]
    disabled_cfg = FusionConfig(
        retrieval_temperature=0.3,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=False,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    zero_weight_cfg = FusionConfig(
        retrieval_temperature=0.3,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=True,
        plausibility_radius_km=50.0,
        plausibility_weight=0.0,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    disabled = fuse_candidates("missing.jpg", candidates, detections=[], config=disabled_cfg)
    zero = fuse_candidates("missing.jpg", candidates, detections=[], config=zero_weight_cfg)
    assert disabled is not None
    assert zero is not None
    assert [item.candidate.match_id for item in disabled.candidates] == [
        item.candidate.match_id for item in zero.candidates
    ]


def test_adaptive_outlier_guard_can_downrank_isolated_outlier() -> None:
    candidates = [
        GeoCandidate(latitude=34.0522, longitude=-118.2437, retrieval_score=0.89, match_id="retrieval:outlier"),
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.83, match_id="retrieval:paris_a"),
        GeoCandidate(latitude=48.8570, longitude=2.3530, retrieval_score=0.82, match_id="retrieval:paris_b"),
    ]
    base_cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=False,
        use_adaptive_outlier_guard=False,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    guarded_cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=False,
        use_adaptive_outlier_guard=True,
        outlier_guard_strength=8.0,
        outlier_guard_min_scale_km=40.0,
        outlier_guard_mad_scale=2.5,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    base = fuse_candidates("missing.jpg", candidates, detections=[], config=base_cfg)
    guarded = fuse_candidates("missing.jpg", candidates, detections=[], config=guarded_cfg)
    assert base is not None
    assert guarded is not None
    assert base.candidates[0].candidate.match_id == "retrieval:outlier"
    assert guarded.candidates[0].candidate.match_id in {"retrieval:paris_a", "retrieval:paris_b"}


def test_adaptive_outlier_guard_zero_strength_matches_disabled() -> None:
    candidates = [
        GeoCandidate(latitude=10.0, longitude=10.0, retrieval_score=0.70, match_id="a"),
        GeoCandidate(latitude=10.1, longitude=10.1, retrieval_score=0.68, match_id="b"),
        GeoCandidate(latitude=40.0, longitude=-75.0, retrieval_score=0.74, match_id="c"),
    ]
    disabled_cfg = FusionConfig(
        retrieval_temperature=0.3,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=False,
        use_adaptive_outlier_guard=False,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    zero_strength_cfg = FusionConfig(
        retrieval_temperature=0.3,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=False,
        use_adaptive_outlier_guard=True,
        outlier_guard_strength=0.0,
        outlier_guard_min_scale_km=30.0,
        outlier_guard_mad_scale=2.0,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    disabled = fuse_candidates("missing.jpg", candidates, detections=[], config=disabled_cfg)
    zero = fuse_candidates("missing.jpg", candidates, detections=[], config=zero_strength_cfg)
    assert disabled is not None
    assert zero is not None
    assert [item.candidate.match_id for item in disabled.candidates] == [
        item.candidate.match_id for item in zero.candidates
    ]


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
    assert 0.0 <= result.calibrated_top1_posterior <= 1.0
    assert 0.0 <= result.top1_cross_source_support <= 1.0
    assert 0.0 <= result.top2_margin <= 1.0
    assert result.confidence_tier in {"low", "medium", "high"}
    assert isinstance(result.ambiguous, bool)


def test_calibration_knobs_can_reduce_confidence_tier() -> None:
    candidates = [
        GeoCandidate(latitude=0.0, longitude=0.0, retrieval_score=0.9, match_id="a"),
        GeoCandidate(latitude=0.1, longitude=0.1, retrieval_score=0.3, match_id="b"),
    ]
    base_cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=False,
        use_shadow=False,
        use_terrain=False,
        top_k=2,
    )
    calibrated_cfg = FusionConfig(
        retrieval_temperature=0.2,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=False,
        confidence_calibration_logit_scale=0.4,
        confidence_calibration_logit_bias=-0.6,
        confidence_high_threshold=0.85,
        confidence_medium_threshold=0.65,
        use_shadow=False,
        use_terrain=False,
        top_k=2,
    )
    base = fuse_candidates("missing.jpg", candidates, detections=[], config=base_cfg)
    calibrated = fuse_candidates("missing.jpg", candidates, detections=[], config=calibrated_cfg)
    assert base is not None
    assert calibrated is not None
    assert calibrated.calibrated_top1_posterior <= base.calibrated_top1_posterior
    tier_rank = {"low": 0, "medium": 1, "high": 2}
    assert tier_rank[calibrated.confidence_tier] <= tier_rank[base.confidence_tier]


def test_uncertainty_caps_can_downgrade_confidence_tier() -> None:
    candidates = [
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.80, match_id="a"),
        GeoCandidate(latitude=-33.8688, longitude=151.2093, retrieval_score=0.70, match_id="b"),
    ]
    no_cap_cfg = FusionConfig(
        retrieval_temperature=0.4,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=False,
        confidence_high_threshold=0.72,
        confidence_medium_threshold=0.46,
        confidence_high_max_uncertainty_m=None,
        confidence_medium_max_uncertainty_m=None,
        use_shadow=False,
        use_terrain=False,
        top_k=2,
    )
    capped_cfg = FusionConfig(
        retrieval_temperature=0.4,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_plausibility_rerank=False,
        confidence_high_threshold=0.72,
        confidence_medium_threshold=0.46,
        confidence_high_max_uncertainty_m=200_000.0,
        confidence_medium_max_uncertainty_m=500_000.0,
        use_shadow=False,
        use_terrain=False,
        top_k=2,
    )
    no_cap = fuse_candidates("missing.jpg", candidates, detections=[], config=no_cap_cfg)
    capped = fuse_candidates("missing.jpg", candidates, detections=[], config=capped_cfg)
    assert no_cap is not None
    assert capped is not None
    assert no_cap.confidence_tier == "medium"
    assert capped.confidence_tier == "low"


def test_cross_source_support_caps_can_downgrade_confidence_tier() -> None:
    candidates = [
        GeoCandidate(latitude=34.0522, longitude=-118.2437, retrieval_score=0.95, match_id="retrieval:outlier"),
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.55, match_id="retrieval:paris"),
        GeoCandidate(latitude=48.8570, longitude=2.3530, retrieval_score=0.54, match_id="geoclip"),
    ]
    no_cap_cfg = FusionConfig(
        retrieval_temperature=0.1,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=True,
        cross_source_sigma_km=8.0,
        cross_source_weight=0.0,
        use_plausibility_rerank=False,
        confidence_high_threshold=0.72,
        confidence_medium_threshold=0.46,
        confidence_high_min_cross_source_support=None,
        confidence_medium_min_cross_source_support=None,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    capped_cfg = FusionConfig(
        retrieval_temperature=0.1,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=True,
        cross_source_sigma_km=8.0,
        cross_source_weight=0.0,
        use_plausibility_rerank=False,
        confidence_high_threshold=0.72,
        confidence_medium_threshold=0.46,
        confidence_high_min_cross_source_support=0.30,
        confidence_medium_min_cross_source_support=0.12,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )
    no_cap = fuse_candidates("missing.jpg", candidates, detections=[], config=no_cap_cfg)
    capped = fuse_candidates("missing.jpg", candidates, detections=[], config=capped_cfg)
    assert no_cap is not None
    assert capped is not None
    assert no_cap.candidates[0].candidate.match_id == "retrieval:outlier"
    assert capped.top1_cross_source_support < 0.30
    tier_rank = {"low": 0, "medium": 1, "high": 2}
    assert tier_rank[capped.confidence_tier] < tier_rank[no_cap.confidence_tier]


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


def test_top_cluster_stats_choose_densest_cluster_not_top1_anchor() -> None:
    candidates = [
        GeoCandidate(latitude=34.0522, longitude=-118.2437, retrieval_score=0.910, match_id="outlier_top1"),
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.900, match_id="paris_a"),
        GeoCandidate(latitude=48.8570, longitude=2.3530, retrieval_score=0.895, match_id="paris_b"),
    ]
    no_cluster_cfg = FusionConfig(
        retrieval_temperature=1.0,
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
        retrieval_temperature=1.0,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        credible_mass=0.99,
        min_credible_candidates=2,
        use_top_cluster_for_stats=True,
        credible_cluster_radius_km=200.0,
        min_credible_cluster_weight=0.30,
        use_shadow=False,
        use_terrain=False,
        top_k=3,
    )

    no_cluster = fuse_candidates("missing.jpg", candidates, detections=[], config=no_cluster_cfg)
    cluster = fuse_candidates("missing.jpg", candidates, detections=[], config=cluster_cfg)
    assert no_cluster is not None
    assert cluster is not None

    paris_lat = 48.8568
    # Cluster-aware stats should snap toward the denser Paris pair even if top-1 is an outlier.
    assert abs(cluster.mean_latitude - paris_lat) < abs(no_cluster.mean_latitude - paris_lat)
    assert cluster.credible_set_size == 2


def test_sidecar_capture_time_enables_shadow_reranking() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "query.jpg"
        Path(str(image_path) + ".meta.json").write_text(
            json.dumps({"captured_at": "2026-06-01T12:00:00Z"}),
            encoding="utf-8",
        )

        candidates = [
            GeoCandidate(latitude=34.0522, longitude=-118.2437, retrieval_score=0.80, match_id="la"),
            GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.72, match_id="paris"),
        ]
        detections = [
            Detection(
                label="building",
                confidence=0.9,
                obb=((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)),
                shadow_azimuth_deg=6.0,
            )
        ]
        without_shadow_cfg = FusionConfig(
            retrieval_temperature=0.3,
            retrieval_score_norm="none",
            use_spatial_consensus=False,
            use_cross_source_agreement=False,
            use_shadow=False,
            use_terrain=False,
            top_k=2,
        )
        with_shadow_cfg = FusionConfig(
            retrieval_temperature=0.3,
            retrieval_score_norm="none",
            use_spatial_consensus=False,
            use_cross_source_agreement=False,
            use_shadow=True,
            shadow_sigma_deg=18.0,
            use_terrain=False,
            top_k=2,
        )

        baseline = fuse_candidates(str(image_path), candidates, detections=detections, config=without_shadow_cfg)
        reranked = fuse_candidates(str(image_path), candidates, detections=detections, config=with_shadow_cfg)

        assert baseline is not None
        assert reranked is not None
        assert baseline.candidates[0].candidate.match_id == "la"
        assert reranked.candidates[0].candidate.match_id == "paris"
        assert reranked.candidates[0].evidence.shadow_residual_deg is not None


def test_explicit_capture_time_enables_shadow_reranking_without_sidecar() -> None:
    candidates = [
        GeoCandidate(latitude=34.0522, longitude=-118.2437, retrieval_score=0.80, match_id="la"),
        GeoCandidate(latitude=48.8566, longitude=2.3522, retrieval_score=0.72, match_id="paris"),
    ]
    detections = [
        Detection(
            label="building",
            confidence=0.9,
            obb=((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)),
            shadow_azimuth_deg=6.0,
        )
    ]
    cfg = FusionConfig(
        retrieval_temperature=0.3,
        retrieval_score_norm="none",
        use_spatial_consensus=False,
        use_cross_source_agreement=False,
        use_shadow=True,
        shadow_sigma_deg=18.0,
        use_terrain=False,
        top_k=2,
    )

    result = fuse_candidates(
        "missing.jpg",
        candidates,
        detections=detections,
        config=cfg,
        capture_time=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert result is not None
    assert result.candidates[0].candidate.match_id == "paris"
