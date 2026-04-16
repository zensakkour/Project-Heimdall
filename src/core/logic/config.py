"""
Config loading for pipeline components.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DetectorConfig:
    weights_path: Optional[str] = None
    min_confidence: float = 0.25
    nms_iou: float = 0.5
    nms_mode: str = "obb"
    max_detections: int = 100
    min_area_px: float = 16.0
    class_agnostic_nms: bool = False
    use_tta: bool = False
    use_sidecar: bool = True
    use_classic: bool = False
    imgsz: int = 1280


@dataclass(frozen=True)
class GeoConfig:
    model_path: Optional[str] = None
    model_id: Optional[str] = None
    model_cache_dir: Optional[str] = None
    encoder_name: Optional[str] = None
    use_sidecar: bool = True
    use_exif: bool = True
    top_n: int = 5
    geospot_score_scale: float = 1.0
    retrieval_index_path: Optional[str] = None
    retrieval_index_paths: Tuple[str, ...] = ()
    retrieval_index_weights: Tuple[float, ...] = ()
    retrieval_index_model_ids: Tuple[str, ...] = ()
    retrieval_model_id: Optional[str] = None
    retrieval_top_k: int = 10
    retrieval_per_index_top_k: int = 0
    retrieval_index_score_norm: str = "auto"
    retrieval_source_fusion_mode: str = "weighted_score"
    retrieval_source_balance_beta: float = 0.0
    retrieval_min_score: float = 0.2
    retrieval_min_keep_topk: int = 0
    retrieval_diversity_radius_km: float = 0.0
    retrieval_diversity_lambda: float = 1.0
    retrieval_diversity_min_keep: int = 1
    retrieval_locality_radius_km: float = 0.0
    retrieval_locality_weight: float = 0.0
    retrieval_consensus_top_n: int = 0
    retrieval_consensus_radius_km: float = 0.0
    retrieval_consensus_score_power: float = 1.0
    retrieval_query_tta_degrees: Tuple[float, ...] = (0.0,)
    retrieval_query_tta_modes: Tuple[str, ...] = ("rgb",)
    retrieval_query_tta_auto_modality: bool = False
    retrieval_query_tta_reduce: str = "mean"
    retrieval_query_expansion_top_n: int = 0
    retrieval_query_expansion_beta: float = 0.0
    retrieval_query_expansion_alpha: float = 0.5
    retrieval_local_match_top_n: int = 0
    retrieval_local_match_weight: float = 0.0
    retrieval_local_match_ratio: float = 0.8
    retrieval_local_match_max_features: int = 1200
    retrieval_graph_rerank_top_n: int = 0
    retrieval_graph_rerank_sigma_km: float = 3.0
    retrieval_graph_rerank_score_alpha: float = 0.4
    retrieval_graph_rerank_support_beta: float = 1.0
    retrieval_graph_rerank_center_radius_km: float = 0.0
    retrieval_kde_refine_top_n: int = 0
    retrieval_kde_refine_sigma_km: float = 2.0
    retrieval_kde_refine_score_power: float = 1.0
    retrieval_kde_refine_margin_threshold: float = 0.0
    retrieval_kde_refine_switch_radius_km: float = 0.0
    retrieval_kde_refine_max_iters: int = 8
    candidate_dedupe_radius_m: float = 300.0
    candidate_source_balance_beta: float = 0.0
    candidate_max_results: int = 80


@dataclass(frozen=True)
class VerificationConfig:
    use_shadow: bool = True
    use_shadow_length: bool = True
    use_shadow_heading: bool = True


@dataclass(frozen=True)
class FusionConfig:
    retrieval_temperature: float = 0.2
    retrieval_score_norm: str = "none"
    source_prior_retrieval: float = 1.0
    source_prior_retrieval_by_source: Optional[Dict[str, float]] = None
    source_prior_geoclip: float = 1.0
    source_prior_exif: float = 1.0
    use_spatial_consensus: bool = True
    spatial_sigma_km: float = 2.0
    spatial_consensus_weight: float = 1.0
    use_cross_source_agreement: bool = True
    cross_source_sigma_km: float = 15.0
    cross_source_weight: float = 1.0
    use_plausibility_rerank: bool = False
    plausibility_radius_km: float = 200.0
    plausibility_weight: float = 1.0
    use_adaptive_outlier_guard: bool = False
    outlier_guard_strength: float = 1.0
    outlier_guard_min_scale_km: float = 120.0
    outlier_guard_mad_scale: float = 3.0
    confidence_calibration_logit_scale: float = 1.0
    confidence_calibration_logit_bias: float = 0.0
    confidence_high_threshold: float = 0.70
    confidence_medium_threshold: float = 0.45
    confidence_high_min_cross_source_support: Optional[float] = 0.30
    confidence_medium_min_cross_source_support: Optional[float] = 0.10
    confidence_high_max_uncertainty_m: Optional[float] = 500_000.0
    confidence_medium_max_uncertainty_m: Optional[float] = 2_000_000.0
    shadow_sigma_deg: float = 20.0
    terrain_sigma: float = 100.0
    use_shadow: bool = True
    use_terrain: bool = False
    credible_mass: float = 0.9
    min_credible_candidates: int = 2
    use_top_cluster_for_stats: bool = True
    credible_cluster_radius_km: float = 500.0
    min_credible_cluster_weight: float = 0.35
    top_k: int = 5


@dataclass(frozen=True)
class ScoreConfig:
    detection_weight: float = 0.4
    geo_weight: float = 0.4
    shadow_weight: float = 0.1
    topo_weight: float = 0.1
    detection_top_k: int = 3


@dataclass(frozen=True)
class HeimdallConfig:
    detector: DetectorConfig
    geolocator: GeoConfig
    fusion: FusionConfig
    score: ScoreConfig
    verification: VerificationConfig


def has_retrieval_index(geo: GeoConfig) -> bool:
    return bool(geo.retrieval_index_path) or bool(geo.retrieval_index_paths)


def load_config(path: str) -> HeimdallConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    det = raw.get("detector", {})
    geo = raw.get("geolocator", {})
    fusion = raw.get("fusion", {})
    score = raw.get("score", {})
    ver = raw.get("verification", {})
    return HeimdallConfig(
        detector=DetectorConfig(
            weights_path=det.get("weights_path"),
            min_confidence=det.get("min_confidence", 0.25),
            nms_iou=det.get("nms_iou", 0.5),
            nms_mode=det.get("nms_mode", "obb"),
            max_detections=det.get("max_detections", 100),
            min_area_px=det.get("min_area_px", 16.0),
            class_agnostic_nms=det.get("class_agnostic_nms", False),
            use_tta=det.get("use_tta", False),
            use_sidecar=det.get("use_sidecar", True),
            use_classic=det.get("use_classic", False),
            imgsz=det.get("imgsz", 1280),
        ),
        geolocator=GeoConfig(
            model_path=geo.get("model_path"),
            model_id=geo.get("model_id"),
            model_cache_dir=geo.get("model_cache_dir"),
            encoder_name=geo.get("encoder_name"),
            use_sidecar=geo.get("use_sidecar", True),
            use_exif=geo.get("use_exif", True),
            top_n=geo.get("top_n", 5),
            geospot_score_scale=geo.get("geospot_score_scale", 1.0),
            retrieval_index_path=geo.get("retrieval_index_path"),
            retrieval_index_paths=_parse_str_tuple(geo.get("retrieval_index_paths", [])),
            retrieval_index_weights=_parse_float_tuple_allow_empty(geo.get("retrieval_index_weights", [])),
            retrieval_index_model_ids=_parse_str_tuple(geo.get("retrieval_index_model_ids", [])),
            retrieval_model_id=geo.get("retrieval_model_id"),
            retrieval_top_k=geo.get("retrieval_top_k", 10),
            retrieval_per_index_top_k=geo.get("retrieval_per_index_top_k", 0),
            retrieval_index_score_norm=_parse_index_score_norm(geo.get("retrieval_index_score_norm", "auto")),
            retrieval_source_fusion_mode=_parse_source_fusion_mode(
                geo.get("retrieval_source_fusion_mode", "weighted_score")
            ),
            retrieval_source_balance_beta=geo.get("retrieval_source_balance_beta", 0.0),
            retrieval_min_score=geo.get("retrieval_min_score", 0.2),
            retrieval_min_keep_topk=geo.get("retrieval_min_keep_topk", 0),
            retrieval_diversity_radius_km=geo.get("retrieval_diversity_radius_km", 0.0),
            retrieval_diversity_lambda=geo.get("retrieval_diversity_lambda", 1.0),
            retrieval_diversity_min_keep=geo.get("retrieval_diversity_min_keep", 1),
            retrieval_locality_radius_km=geo.get("retrieval_locality_radius_km", 0.0),
            retrieval_locality_weight=geo.get("retrieval_locality_weight", 0.0),
            retrieval_consensus_top_n=geo.get("retrieval_consensus_top_n", 0),
            retrieval_consensus_radius_km=geo.get("retrieval_consensus_radius_km", 0.0),
            retrieval_consensus_score_power=geo.get("retrieval_consensus_score_power", 1.0),
            retrieval_query_tta_degrees=_parse_float_tuple(geo.get("retrieval_query_tta_degrees", [0.0])),
            retrieval_query_tta_modes=_parse_tta_modes(geo.get("retrieval_query_tta_modes", ["rgb"])),
            retrieval_query_tta_auto_modality=geo.get("retrieval_query_tta_auto_modality", False),
            retrieval_query_tta_reduce=_parse_tta_reduce(geo.get("retrieval_query_tta_reduce", "mean")),
            retrieval_query_expansion_top_n=geo.get("retrieval_query_expansion_top_n", 0),
            retrieval_query_expansion_beta=geo.get("retrieval_query_expansion_beta", 0.0),
            retrieval_query_expansion_alpha=geo.get("retrieval_query_expansion_alpha", 0.5),
            retrieval_local_match_top_n=geo.get("retrieval_local_match_top_n", 0),
            retrieval_local_match_weight=geo.get("retrieval_local_match_weight", 0.0),
            retrieval_local_match_ratio=geo.get("retrieval_local_match_ratio", 0.8),
            retrieval_local_match_max_features=geo.get("retrieval_local_match_max_features", 1200),
            retrieval_graph_rerank_top_n=geo.get("retrieval_graph_rerank_top_n", 0),
            retrieval_graph_rerank_sigma_km=geo.get("retrieval_graph_rerank_sigma_km", 3.0),
            retrieval_graph_rerank_score_alpha=geo.get("retrieval_graph_rerank_score_alpha", 0.4),
            retrieval_graph_rerank_support_beta=geo.get("retrieval_graph_rerank_support_beta", 1.0),
            retrieval_graph_rerank_center_radius_km=geo.get("retrieval_graph_rerank_center_radius_km", 0.0),
            retrieval_kde_refine_top_n=geo.get("retrieval_kde_refine_top_n", 0),
            retrieval_kde_refine_sigma_km=geo.get("retrieval_kde_refine_sigma_km", 2.0),
            retrieval_kde_refine_score_power=geo.get("retrieval_kde_refine_score_power", 1.0),
            retrieval_kde_refine_margin_threshold=geo.get("retrieval_kde_refine_margin_threshold", 0.0),
            retrieval_kde_refine_switch_radius_km=geo.get("retrieval_kde_refine_switch_radius_km", 0.0),
            retrieval_kde_refine_max_iters=geo.get("retrieval_kde_refine_max_iters", 8),
            candidate_dedupe_radius_m=geo.get("candidate_dedupe_radius_m", 300.0),
            candidate_source_balance_beta=geo.get("candidate_source_balance_beta", 0.0),
            candidate_max_results=geo.get("candidate_max_results", 80),
        ),
        fusion=FusionConfig(
            retrieval_temperature=fusion.get("retrieval_temperature", 0.2),
            retrieval_score_norm=fusion.get("retrieval_score_norm", "none"),
            source_prior_retrieval=fusion.get("source_prior_retrieval", 1.0),
            source_prior_retrieval_by_source=_parse_float_dict(
                fusion.get("source_prior_retrieval_by_source", {})
            ),
            source_prior_geoclip=fusion.get("source_prior_geoclip", 1.0),
            source_prior_exif=fusion.get("source_prior_exif", 1.0),
            use_spatial_consensus=fusion.get("use_spatial_consensus", True),
            spatial_sigma_km=fusion.get("spatial_sigma_km", 2.0),
            spatial_consensus_weight=fusion.get("spatial_consensus_weight", 1.0),
            use_cross_source_agreement=fusion.get("use_cross_source_agreement", True),
            cross_source_sigma_km=fusion.get("cross_source_sigma_km", 15.0),
            cross_source_weight=fusion.get("cross_source_weight", 1.0),
            use_plausibility_rerank=fusion.get("use_plausibility_rerank", False),
            plausibility_radius_km=fusion.get("plausibility_radius_km", 200.0),
            plausibility_weight=fusion.get("plausibility_weight", 1.0),
            use_adaptive_outlier_guard=fusion.get("use_adaptive_outlier_guard", False),
            outlier_guard_strength=fusion.get("outlier_guard_strength", 1.0),
            outlier_guard_min_scale_km=fusion.get("outlier_guard_min_scale_km", 120.0),
            outlier_guard_mad_scale=fusion.get("outlier_guard_mad_scale", 3.0),
            confidence_calibration_logit_scale=fusion.get("confidence_calibration_logit_scale", 1.0),
            confidence_calibration_logit_bias=fusion.get("confidence_calibration_logit_bias", 0.0),
            confidence_high_threshold=fusion.get("confidence_high_threshold", 0.70),
            confidence_medium_threshold=fusion.get("confidence_medium_threshold", 0.45),
            confidence_high_min_cross_source_support=fusion.get("confidence_high_min_cross_source_support", 0.30),
            confidence_medium_min_cross_source_support=fusion.get("confidence_medium_min_cross_source_support", 0.10),
            confidence_high_max_uncertainty_m=fusion.get("confidence_high_max_uncertainty_m", 500_000.0),
            confidence_medium_max_uncertainty_m=fusion.get("confidence_medium_max_uncertainty_m", 2_000_000.0),
            shadow_sigma_deg=fusion.get("shadow_sigma_deg", 20.0),
            terrain_sigma=fusion.get("terrain_sigma", 100.0),
            use_shadow=fusion.get("use_shadow", True),
            use_terrain=fusion.get("use_terrain", False),
            credible_mass=fusion.get("credible_mass", 0.9),
            min_credible_candidates=fusion.get("min_credible_candidates", 2),
            use_top_cluster_for_stats=fusion.get("use_top_cluster_for_stats", True),
            credible_cluster_radius_km=fusion.get("credible_cluster_radius_km", 500.0),
            min_credible_cluster_weight=fusion.get("min_credible_cluster_weight", 0.35),
            top_k=fusion.get("top_k", 5),
        ),
        score=ScoreConfig(
            detection_weight=score.get("detection_weight", 0.4),
            geo_weight=score.get("geo_weight", 0.4),
            shadow_weight=score.get("shadow_weight", 0.1),
            topo_weight=score.get("topo_weight", 0.1),
            detection_top_k=score.get("detection_top_k", 3),
        ),
        verification=VerificationConfig(
            use_shadow=ver.get("use_shadow", True),
            use_shadow_length=ver.get("use_shadow_length", True),
            use_shadow_heading=ver.get("use_shadow_heading", True),
        ),
    )


def _parse_float_tuple(raw) -> Tuple[float, ...]:
    if not isinstance(raw, (list, tuple)):
        return (0.0,)
    out = []
    for value in raw:
        if isinstance(value, (int, float)):
            val = float(value)
            if val == val and val not in (float("inf"), float("-inf")):
                out.append(val)
    if not out:
        return (0.0,)
    return tuple(out)


def _parse_float_tuple_allow_empty(raw) -> Tuple[float, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    out = []
    for value in raw:
        if isinstance(value, (int, float)):
            val = float(value)
            if val == val and val not in (float("inf"), float("-inf")):
                out.append(val)
    return tuple(out)


def _parse_str_tuple(raw) -> Tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    out = []
    seen = set()
    for value in raw:
        text = str(value).strip() if value is not None else ""
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def _parse_float_dict(raw) -> Optional[Dict[str, float]]:
    if not isinstance(raw, dict):
        return None
    out: Dict[str, float] = {}
    for key, value in raw.items():
        name = str(key).strip() if key is not None else ""
        if not name:
            continue
        if not isinstance(value, (int, float)):
            continue
        val = float(value)
        if val != val or val in (float("inf"), float("-inf")):
            continue
        out[name] = val
    return out or None


def _parse_tta_reduce(raw: object) -> str:
    mode = str(raw).strip().lower()
    if mode not in {"mean", "median", "max", "rrf"}:
        return "mean"
    return mode


def _parse_tta_modes(raw) -> Tuple[str, ...]:
    allowed = {"rgb", "gray", "equalize", "edge"}
    if not isinstance(raw, (list, tuple)):
        return ("rgb",)
    out = []
    seen = set()
    for item in raw:
        mode = str(item).strip().lower() if item is not None else ""
        if mode not in allowed or mode in seen:
            continue
        seen.add(mode)
        out.append(mode)
    if not out:
        return ("rgb",)
    return tuple(out)


def _parse_index_score_norm(raw: object) -> str:
    mode = str(raw).strip().lower()
    if mode not in {"auto", "none", "minmax", "zscore_sigmoid", "rank_exp"}:
        return "auto"
    return mode


def _parse_source_fusion_mode(raw: object) -> str:
    mode = str(raw).strip().lower()
    if mode not in {"weighted_score", "rrf"}:
        return "weighted_score"
    return mode
