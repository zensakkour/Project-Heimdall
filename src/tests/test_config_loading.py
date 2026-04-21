"""
Tests for config loading defaults/overrides.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.core.logic.config import load_config


def _write_config(tmpdir: str, payload: dict) -> Path:
    path = Path(tmpdir) / "cfg.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_config_spatial_consensus_defaults() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_config(tmpdir, {})
        cfg = load_config(str(path))

        assert cfg.fusion.use_spatial_consensus is True
        assert cfg.fusion.spatial_sigma_km == 2.0
        assert cfg.fusion.spatial_consensus_weight == 1.0
        assert cfg.fusion.use_cross_source_agreement is True
        assert cfg.fusion.cross_source_sigma_km == 15.0
        assert cfg.fusion.cross_source_weight == 1.0
        assert cfg.fusion.use_plausibility_rerank is False
        assert cfg.fusion.plausibility_radius_km == 200.0
        assert cfg.fusion.plausibility_weight == 1.0
        assert cfg.fusion.use_adaptive_outlier_guard is False
        assert cfg.fusion.outlier_guard_strength == 1.0
        assert cfg.fusion.outlier_guard_min_scale_km == 120.0
        assert cfg.fusion.outlier_guard_mad_scale == 3.0
        assert cfg.fusion.confidence_calibration_logit_scale == 1.0
        assert cfg.fusion.confidence_calibration_logit_bias == 0.0
        assert cfg.fusion.confidence_high_threshold == 0.70
        assert cfg.fusion.confidence_medium_threshold == 0.45
        assert cfg.fusion.confidence_high_min_cross_source_support == 0.30
        assert cfg.fusion.confidence_medium_min_cross_source_support == 0.10
        assert cfg.fusion.confidence_high_max_uncertainty_m == 500_000.0
        assert cfg.fusion.confidence_medium_max_uncertainty_m == 2_000_000.0
        assert cfg.fusion.source_prior_retrieval == 1.0
        assert cfg.fusion.source_prior_retrieval_by_source is None
        assert cfg.fusion.source_prior_geoclip == 1.0
        assert cfg.fusion.source_prior_exif == 1.0
        assert cfg.fusion.credible_mass == 0.9
        assert cfg.fusion.min_credible_candidates == 2
        assert cfg.fusion.use_top_cluster_for_stats is True
        assert cfg.fusion.credible_cluster_radius_km == 500.0
        assert cfg.fusion.min_credible_cluster_weight == 0.35
        assert cfg.detector.min_area_px == 16.0
        assert cfg.detector.class_agnostic_nms is False
        assert cfg.detector.use_tta is False
        assert cfg.detector.nms_mode == "obb"
        assert cfg.geolocator.retrieval_diversity_radius_km == 0.0
        assert cfg.geolocator.retrieval_diversity_lambda == 1.0
        assert cfg.geolocator.retrieval_diversity_min_keep == 1
        assert cfg.geolocator.retrieval_index_paths == ()
        assert cfg.geolocator.retrieval_index_weights == ()
        assert cfg.geolocator.retrieval_index_model_ids == ()
        assert cfg.geolocator.retrieval_projection_path is None
        assert cfg.geolocator.retrieval_per_index_top_k == 0
        assert cfg.geolocator.retrieval_index_score_norm == "auto"
        assert cfg.geolocator.retrieval_source_fusion_mode == "weighted_score"
        assert cfg.geolocator.retrieval_source_balance_beta == 0.0
        assert cfg.geolocator.retrieval_min_keep_topk == 0
        assert cfg.geolocator.candidate_source_balance_beta == 0.0
        assert cfg.geolocator.retrieval_locality_radius_km == 0.0
        assert cfg.geolocator.retrieval_locality_weight == 0.0
        assert cfg.geolocator.retrieval_consensus_top_n == 0
        assert cfg.geolocator.retrieval_consensus_radius_km == 0.0
        assert cfg.geolocator.retrieval_consensus_score_power == 1.0
        assert cfg.geolocator.retrieval_query_tta_degrees == (0.0,)
        assert cfg.geolocator.retrieval_query_tta_modes == ("rgb",)
        assert cfg.geolocator.retrieval_query_tta_scales == (1.0,)
        assert cfg.geolocator.retrieval_query_tta_auto_modality is False
        assert cfg.geolocator.retrieval_query_tta_reduce == "mean"
        assert cfg.geolocator.retrieval_query_expansion_top_n == 0
        assert cfg.geolocator.retrieval_query_expansion_beta == 0.0
        assert cfg.geolocator.retrieval_query_expansion_alpha == 0.5
        assert cfg.geolocator.retrieval_tta_agreement_top_n == 0
        assert cfg.geolocator.retrieval_tta_agreement_weight == 0.0
        assert cfg.geolocator.retrieval_local_match_top_n == 0
        assert cfg.geolocator.retrieval_local_match_weight == 0.0
        assert cfg.geolocator.retrieval_local_match_ratio == 0.8
        assert cfg.geolocator.retrieval_local_match_max_features == 1200
        assert cfg.geolocator.retrieval_graph_rerank_top_n == 0
        assert cfg.geolocator.retrieval_graph_rerank_sigma_km == 3.0
        assert cfg.geolocator.retrieval_graph_rerank_score_alpha == 0.4
        assert cfg.geolocator.retrieval_graph_rerank_support_beta == 1.0
        assert cfg.geolocator.retrieval_graph_rerank_center_radius_km == 0.0
        assert cfg.geolocator.retrieval_kde_refine_top_n == 0
        assert cfg.geolocator.retrieval_kde_refine_sigma_km == 2.0
        assert cfg.geolocator.retrieval_kde_refine_score_power == 1.0
        assert cfg.geolocator.retrieval_kde_refine_margin_threshold == 0.0
        assert cfg.geolocator.retrieval_kde_refine_switch_radius_km == 0.0
        assert cfg.geolocator.retrieval_kde_refine_max_iters == 8
        assert cfg.geolocator.retrieval_kde_refine_adaptive_mass == 0.0


def test_load_config_spatial_consensus_overrides() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_config(
            tmpdir,
            {
                "fusion": {
                    "use_spatial_consensus": False,
                    "spatial_sigma_km": 12.5,
                    "spatial_consensus_weight": 2.5,
                    "use_cross_source_agreement": False,
                    "cross_source_sigma_km": 33.0,
                    "cross_source_weight": 3.0,
                    "use_plausibility_rerank": True,
                    "plausibility_radius_km": 95.0,
                    "plausibility_weight": 2.2,
                    "use_adaptive_outlier_guard": True,
                    "outlier_guard_strength": 1.7,
                    "outlier_guard_min_scale_km": 80.0,
                    "outlier_guard_mad_scale": 2.2,
                    "confidence_calibration_logit_scale": 0.8,
                    "confidence_calibration_logit_bias": -0.1,
                    "confidence_high_threshold": 0.75,
                    "confidence_medium_threshold": 0.5,
                    "confidence_high_min_cross_source_support": 0.42,
                    "confidence_medium_min_cross_source_support": 0.21,
                    "confidence_high_max_uncertainty_m": 250_000.0,
                    "confidence_medium_max_uncertainty_m": 1_250_000.0,
                    "source_prior_retrieval": 0.95,
                    "source_prior_retrieval_by_source": {
                        "retrieval:spacenet_paris_clip": 1.25,
                        "open_geo_clip": 0.75,
                        "": 9.9,
                        "bad": "x",
                    },
                    "source_prior_geoclip": 1.1,
                    "source_prior_exif": 0.8,
                    "credible_mass": 0.8,
                    "min_credible_candidates": 3,
                    "use_top_cluster_for_stats": False,
                    "credible_cluster_radius_km": 220.0,
                    "min_credible_cluster_weight": 0.5,
                },
                "detector": {
                    "min_area_px": 20.0,
                    "class_agnostic_nms": True,
                    "use_tta": True,
                    "nms_mode": "aabb",
                },
                "geolocator": {
                    "retrieval_index_paths": ["a.npz", "b.npz", "a.npz"],
                    "retrieval_index_weights": [1.0, 0.8],
                    "retrieval_index_model_ids": ["openai/clip-vit-large-patch14", "google/siglip-base-patch16-224"],
                    "retrieval_projection_path": "runs/projections/paris_v1.npz",
                    "retrieval_per_index_top_k": 5,
                    "retrieval_index_score_norm": "zscore_sigmoid",
                    "retrieval_source_fusion_mode": "rrf",
                    "retrieval_source_balance_beta": 0.6,
                    "candidate_source_balance_beta": 0.25,
                    "retrieval_diversity_radius_km": 3.5,
                    "retrieval_diversity_lambda": 0.77,
                    "retrieval_diversity_min_keep": 4,
                    "retrieval_min_keep_topk": 2,
                    "retrieval_locality_radius_km": 80.0,
                    "retrieval_locality_weight": 1.4,
                    "retrieval_consensus_top_n": 18,
                    "retrieval_consensus_radius_km": 3.2,
                    "retrieval_consensus_score_power": 1.3,
                    "retrieval_query_tta_degrees": [0.0, 90.0, 180.0],
                    "retrieval_query_tta_modes": ["rgb", "gray", "rgb"],
                    "retrieval_query_tta_scales": [1.0, 0.8, 1.0],
                    "retrieval_query_tta_auto_modality": True,
                    "retrieval_query_tta_reduce": "rrf",
                    "retrieval_query_expansion_top_n": 12,
                    "retrieval_query_expansion_beta": 0.4,
                    "retrieval_query_expansion_alpha": 0.35,
                    "retrieval_tta_agreement_top_n": 8,
                    "retrieval_tta_agreement_weight": 0.3,
                    "retrieval_local_match_top_n": 10,
                    "retrieval_local_match_weight": 0.45,
                    "retrieval_local_match_ratio": 0.78,
                    "retrieval_local_match_max_features": 1800,
                    "retrieval_graph_rerank_top_n": 30,
                    "retrieval_graph_rerank_sigma_km": 2.8,
                    "retrieval_graph_rerank_score_alpha": 0.35,
                    "retrieval_graph_rerank_support_beta": 1.4,
                    "retrieval_graph_rerank_center_radius_km": 2.6,
                    "retrieval_kde_refine_top_n": 24,
                    "retrieval_kde_refine_sigma_km": 2.2,
                    "retrieval_kde_refine_score_power": 1.25,
                    "retrieval_kde_refine_margin_threshold": 0.01,
                    "retrieval_kde_refine_switch_radius_km": 3.5,
                    "retrieval_kde_refine_max_iters": 11,
                    "retrieval_kde_refine_adaptive_mass": 0.7,
                },
            },
        )
        cfg = load_config(str(path))

        assert cfg.fusion.use_spatial_consensus is False
        assert cfg.fusion.spatial_sigma_km == 12.5
        assert cfg.fusion.spatial_consensus_weight == 2.5
        assert cfg.fusion.use_cross_source_agreement is False
        assert cfg.fusion.cross_source_sigma_km == 33.0
        assert cfg.fusion.cross_source_weight == 3.0
        assert cfg.fusion.use_plausibility_rerank is True
        assert cfg.fusion.plausibility_radius_km == 95.0
        assert cfg.fusion.plausibility_weight == 2.2
        assert cfg.fusion.use_adaptive_outlier_guard is True
        assert cfg.fusion.outlier_guard_strength == 1.7
        assert cfg.fusion.outlier_guard_min_scale_km == 80.0
        assert cfg.fusion.outlier_guard_mad_scale == 2.2
        assert cfg.fusion.confidence_calibration_logit_scale == 0.8
        assert cfg.fusion.confidence_calibration_logit_bias == -0.1
        assert cfg.fusion.confidence_high_threshold == 0.75
        assert cfg.fusion.confidence_medium_threshold == 0.5
        assert cfg.fusion.confidence_high_min_cross_source_support == 0.42
        assert cfg.fusion.confidence_medium_min_cross_source_support == 0.21
        assert cfg.fusion.confidence_high_max_uncertainty_m == 250_000.0
        assert cfg.fusion.confidence_medium_max_uncertainty_m == 1_250_000.0
        assert cfg.fusion.source_prior_retrieval == 0.95
        assert cfg.fusion.source_prior_retrieval_by_source == {
            "retrieval:spacenet_paris_clip": 1.25,
            "open_geo_clip": 0.75,
        }
        assert cfg.fusion.source_prior_geoclip == 1.1
        assert cfg.fusion.source_prior_exif == 0.8
        assert cfg.fusion.credible_mass == 0.8
        assert cfg.fusion.min_credible_candidates == 3
        assert cfg.fusion.use_top_cluster_for_stats is False
        assert cfg.fusion.credible_cluster_radius_km == 220.0
        assert cfg.fusion.min_credible_cluster_weight == 0.5
        assert cfg.detector.min_area_px == 20.0
        assert cfg.detector.class_agnostic_nms is True
        assert cfg.detector.use_tta is True
        assert cfg.detector.nms_mode == "aabb"
        assert cfg.geolocator.retrieval_index_paths == ("a.npz", "b.npz")
        assert cfg.geolocator.retrieval_index_weights == (1.0, 0.8)
        assert cfg.geolocator.retrieval_index_model_ids == (
            "openai/clip-vit-large-patch14",
            "google/siglip-base-patch16-224",
        )
        assert cfg.geolocator.retrieval_projection_path == "runs/projections/paris_v1.npz"
        assert cfg.geolocator.retrieval_per_index_top_k == 5
        assert cfg.geolocator.retrieval_index_score_norm == "zscore_sigmoid"
        assert cfg.geolocator.retrieval_source_fusion_mode == "rrf"
        assert cfg.geolocator.retrieval_source_balance_beta == 0.6
        assert cfg.geolocator.candidate_source_balance_beta == 0.25
        assert cfg.geolocator.retrieval_diversity_radius_km == 3.5
        assert cfg.geolocator.retrieval_diversity_lambda == 0.77
        assert cfg.geolocator.retrieval_diversity_min_keep == 4
        assert cfg.geolocator.retrieval_min_keep_topk == 2
        assert cfg.geolocator.retrieval_locality_radius_km == 80.0
        assert cfg.geolocator.retrieval_locality_weight == 1.4
        assert cfg.geolocator.retrieval_consensus_top_n == 18
        assert cfg.geolocator.retrieval_consensus_radius_km == 3.2
        assert cfg.geolocator.retrieval_consensus_score_power == 1.3
        assert cfg.geolocator.retrieval_query_tta_degrees == (0.0, 90.0, 180.0)
        assert cfg.geolocator.retrieval_query_tta_modes == ("rgb", "gray")
        assert cfg.geolocator.retrieval_query_tta_scales == (1.0, 0.8, 1.0)
        assert cfg.geolocator.retrieval_query_tta_auto_modality is True
        assert cfg.geolocator.retrieval_query_tta_reduce == "rrf"
        assert cfg.geolocator.retrieval_query_expansion_top_n == 12
        assert cfg.geolocator.retrieval_query_expansion_beta == 0.4
        assert cfg.geolocator.retrieval_query_expansion_alpha == 0.35
        assert cfg.geolocator.retrieval_tta_agreement_top_n == 8
        assert cfg.geolocator.retrieval_tta_agreement_weight == 0.3
        assert cfg.geolocator.retrieval_local_match_top_n == 10
        assert cfg.geolocator.retrieval_local_match_weight == 0.45
        assert cfg.geolocator.retrieval_local_match_ratio == 0.78
        assert cfg.geolocator.retrieval_local_match_max_features == 1800
        assert cfg.geolocator.retrieval_graph_rerank_top_n == 30
        assert cfg.geolocator.retrieval_graph_rerank_sigma_km == 2.8
        assert cfg.geolocator.retrieval_graph_rerank_score_alpha == 0.35
        assert cfg.geolocator.retrieval_graph_rerank_support_beta == 1.4
        assert cfg.geolocator.retrieval_graph_rerank_center_radius_km == 2.6
        assert cfg.geolocator.retrieval_kde_refine_top_n == 24
        assert cfg.geolocator.retrieval_kde_refine_sigma_km == 2.2
        assert cfg.geolocator.retrieval_kde_refine_score_power == 1.25
        assert cfg.geolocator.retrieval_kde_refine_margin_threshold == 0.01
        assert cfg.geolocator.retrieval_kde_refine_switch_radius_km == 3.5
        assert cfg.geolocator.retrieval_kde_refine_max_iters == 11
        assert cfg.geolocator.retrieval_kde_refine_adaptive_mass == 0.7


def test_load_config_invalid_retrieval_index_score_norm_falls_back_to_auto() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_config(
            tmpdir,
            {
                "geolocator": {
                    "retrieval_index_score_norm": "not_a_mode",
                }
            },
        )
        cfg = load_config(str(path))
        assert cfg.geolocator.retrieval_index_score_norm == "auto"


def test_load_config_invalid_retrieval_source_fusion_mode_falls_back_to_weighted_score() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_config(
            tmpdir,
            {
                "geolocator": {
                    "retrieval_source_fusion_mode": "not_a_mode",
                }
            },
        )
        cfg = load_config(str(path))
        assert cfg.geolocator.retrieval_source_fusion_mode == "weighted_score"


def test_load_config_accepts_median_tta_reduce() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_config(
            tmpdir,
            {
                "geolocator": {
                    "retrieval_query_tta_reduce": "median",
                }
            },
        )
        cfg = load_config(str(path))
        assert cfg.geolocator.retrieval_query_tta_reduce == "median"
