"""
Tune retrieval precision with lightweight candidate post-processing sweeps.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, List, Optional

from src.core.geo.retrieval_provider import (
    GeoRetrievalProvider,
    _apply_consensus_refinement,
    _apply_graph_support_rerank,
    _apply_kde_mode_refinement,
    _apply_locality_rerank,
    _select_diverse_geo_candidates,
    _select_source_balanced_candidates,
)
from src.core.logic.config import HeimdallConfig, load_config
from src.core.logic.types import GeoCandidate
from src.tools.run_geo_eval import haversine_km, load_metadata_records, resolve_image_path

_VALID_TTA_REDUCE = {"mean", "median", "max", "rrf"}
_VALID_RANK_OBJECTIVE = {
    "balanced",
    "median_km",
    "mean_km",
    "p90_km",
    "within_1km_pct",
    "within_2km_pct",
    "within_5km_pct",
    "within_10km_pct",
}


@dataclass(frozen=True)
class RetrievalSample:
    image: str
    gt_latitude: float
    gt_longitude: float
    candidates: List[GeoCandidate]


def _parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_tta_reduce_list(raw: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in raw.split(","):
        mode = item.strip().lower()
        if not mode or mode not in _VALID_TTA_REDUCE or mode in seen:
            continue
        seen.add(mode)
        out.append(mode)
    return out


def _parse_rank_objective(raw: str) -> str:
    mode = str(raw).strip().lower()
    if mode not in _VALID_RANK_OBJECTIVE:
        return "balanced"
    return mode


def _safe_inf(value: Optional[float]) -> float:
    return math.inf if value is None else float(value)


def _safe_neg(value: Optional[float]) -> float:
    return -float(value) if value is not None else 0.0


def _result_sort_key(row: dict, objective: str) -> tuple:
    mode = _parse_rank_objective(objective)
    # Always prioritize non-null predictions first.
    nulls = int(row.get("null_predictions", 0) or 0)
    if mode == "within_1km_pct":
        return (nulls, _safe_neg(row.get("within_1km_pct")), _safe_inf(row.get("median_km")), _safe_inf(row.get("mean_km")))
    if mode == "within_2km_pct":
        return (
            nulls,
            _safe_neg(row.get("within_2km_pct")),
            _safe_neg(row.get("within_1km_pct")),
            _safe_inf(row.get("median_km")),
            _safe_inf(row.get("mean_km")),
        )
    if mode == "within_5km_pct":
        return (nulls, _safe_neg(row.get("within_5km_pct")), _safe_inf(row.get("median_km")), _safe_inf(row.get("mean_km")))
    if mode == "within_10km_pct":
        return (
            nulls,
            _safe_neg(row.get("within_10km_pct")),
            _safe_neg(row.get("within_5km_pct")),
            _safe_inf(row.get("median_km")),
            _safe_inf(row.get("mean_km")),
        )
    if mode == "mean_km":
        return (nulls, _safe_inf(row.get("mean_km")), _safe_inf(row.get("median_km")), _safe_neg(row.get("within_5km_pct")))
    if mode == "p90_km":
        return (
            nulls,
            _safe_inf(row.get("p90_km")),
            _safe_inf(row.get("median_km")),
            _safe_inf(row.get("mean_km")),
            _safe_neg(row.get("within_5km_pct")),
        )
    if mode == "median_km":
        return (nulls, _safe_inf(row.get("median_km")), _safe_inf(row.get("mean_km")), _safe_neg(row.get("within_5km_pct")))
    # balanced (default): robust distance first, then coarse recall.
    return (
        nulls,
        _safe_inf(row.get("median_km")),
        _safe_inf(row.get("mean_km")),
        _safe_inf(row.get("p90_km")),
        _safe_neg(row.get("within_1km_pct")),
        _safe_neg(row.get("within_5km_pct")),
    )


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    k = max(0, min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return float(sorted_vals[k])


def _pct_within(vals: List[float], km: float) -> float:
    if not vals:
        return 0.0
    count = sum(1 for value in vals if value <= km)
    return 100.0 * count / len(vals)


def _postprocess_candidates(
    raw_candidates: Iterable[GeoCandidate],
    *,
    top_k: int,
    min_score: float,
    min_keep_topk: int,
    diversity_radius_km: float,
    diversity_lambda: float,
    diversity_min_keep: int,
    locality_radius_km: float,
    locality_weight: float,
    source_balance_beta: float,
    graph_rerank_top_n: int = 0,
    graph_rerank_sigma_km: float = 3.0,
    graph_rerank_score_alpha: float = 0.4,
    graph_rerank_support_beta: float = 1.0,
    graph_rerank_center_radius_km: float = 0.0,
    consensus_top_n: int = 0,
    consensus_radius_km: float = 0.0,
    consensus_score_power: float = 1.0,
    kde_refine_top_n: int = 0,
    kde_refine_sigma_km: float = 2.0,
    kde_refine_score_power: float = 1.0,
    kde_refine_margin_threshold: float = 0.0,
    kde_refine_switch_radius_km: float = 0.0,
    kde_refine_max_iters: int = 8,
    kde_refine_adaptive_mass: float = 0.0,
) -> List[GeoCandidate]:
    ordered_raw = sorted(list(raw_candidates), key=lambda cand: float(cand.retrieval_score), reverse=True)
    filtered = [cand for cand in ordered_raw if float(cand.retrieval_score) >= float(min_score)]
    localized = _apply_locality_rerank(
        filtered,
        radius_km=max(0.0, float(locality_radius_km)),
        weight=max(0.0, float(locality_weight)),
    )
    balanced = _select_source_balanced_candidates(
        localized,
        top_k=max(1, int(top_k)),
        balance_beta=max(0.0, float(source_balance_beta)),
    )
    graph_ranked = _apply_graph_support_rerank(
        balanced,
        top_n=max(0, int(graph_rerank_top_n)),
        sigma_km=max(0.1, float(graph_rerank_sigma_km)),
        score_alpha=max(0.0, float(graph_rerank_score_alpha)),
        support_beta=max(0.0, float(graph_rerank_support_beta)),
        center_radius_km=max(0.0, float(graph_rerank_center_radius_km)),
    )
    selected = _select_diverse_geo_candidates(
        graph_ranked,
        top_k=max(1, int(top_k)),
        radius_km=max(0.0, float(diversity_radius_km)),
        diversity_lambda=min(1.0, max(0.0, float(diversity_lambda))),
        min_keep=max(0, int(diversity_min_keep)),
    )
    selected = _apply_consensus_refinement(
        selected,
        top_n=max(0, int(consensus_top_n)),
        radius_km=max(0.0, float(consensus_radius_km)),
        score_power=max(0.0, float(consensus_score_power)),
    )
    selected = _apply_kde_mode_refinement(
        selected,
        top_n=max(0, int(kde_refine_top_n)),
        sigma_km=max(0.1, float(kde_refine_sigma_km)),
        score_power=max(0.0, float(kde_refine_score_power)),
        margin_threshold=max(0.0, float(kde_refine_margin_threshold)),
        switch_radius_km=max(0.0, float(kde_refine_switch_radius_km)),
        max_iters=max(1, int(kde_refine_max_iters)),
        adaptive_mass=max(0.0, min(1.0, float(kde_refine_adaptive_mass))),
    )
    required = min(max(0, int(min_keep_topk)), max(1, int(top_k)))
    if len(selected) < required:
        existing = {(cand.latitude, cand.longitude, cand.match_id) for cand in selected}
        for cand in ordered_raw:
            key = (cand.latitude, cand.longitude, cand.match_id)
            if key in existing:
                continue
            selected.append(cand)
            existing.add(key)
            if len(selected) >= required:
                break
        selected.sort(key=lambda item: item.retrieval_score, reverse=True)
        selected = selected[: max(1, int(top_k))]
    return selected


def _evaluate_samples(samples: Iterable[RetrievalSample], **kwargs) -> dict:
    distances: List[float] = []
    null_predictions = 0
    for sample in samples:
        candidates = _postprocess_candidates(sample.candidates, **kwargs)
        if not candidates:
            null_predictions += 1
            continue
        best = candidates[0]
        dist = haversine_km(sample.gt_latitude, sample.gt_longitude, best.latitude, best.longitude)
        distances.append(float(dist))
    distances.sort()
    evaluated = len(distances)
    payload = {
        "evaluated": evaluated,
        "null_predictions": null_predictions,
        "mean_km": (sum(distances) / evaluated) if evaluated else None,
        "median_km": (distances[evaluated // 2]) if evaluated else None,
        "p90_km": _percentile(distances, 90),
        "within_1km_pct": _pct_within(distances, 1.0),
        "within_2km_pct": _pct_within(distances, 2.0),
        "within_5km_pct": _pct_within(distances, 5.0),
        "within_10km_pct": _pct_within(distances, 10.0),
        "within_50km_pct": _pct_within(distances, 50.0),
    }
    return payload


def _collect_raw_samples(
    cfg: HeimdallConfig,
    *,
    images_dir: Path,
    records: List[dict],
    max_top_k: int,
    min_score_floor: float,
    retrieval_index_path: Optional[str],
    query_tta_reduce: str,
    structure_rerank_top_n: int,
    structure_rerank_weight: float,
) -> tuple[List[RetrievalSample], int]:
    index_path = retrieval_index_path or cfg.geolocator.retrieval_index_path
    provider = GeoRetrievalProvider(
        index_path=index_path,
        index_paths=cfg.geolocator.retrieval_index_paths,
        index_weights=cfg.geolocator.retrieval_index_weights,
        index_model_ids=cfg.geolocator.retrieval_index_model_ids,
        index_projection_paths=cfg.geolocator.retrieval_index_projection_paths,
        model_id=cfg.geolocator.retrieval_model_id or "openai/clip-vit-large-patch14",
        projection_path=cfg.geolocator.retrieval_projection_path,
        top_k=max(1, int(max_top_k)),
        per_index_top_k=cfg.geolocator.retrieval_per_index_top_k,
        index_score_norm=cfg.geolocator.retrieval_index_score_norm,
        source_fusion_mode=cfg.geolocator.retrieval_source_fusion_mode,
        min_score=max(0.0, float(min_score_floor)),
        min_keep_topk=cfg.geolocator.retrieval_min_keep_topk,
        # Keep raw sampling unbiased; sweep source balancing in post-processing.
        source_balance_beta=0.0,
        diversity_radius_km=0.0,
        diversity_lambda=1.0,
        diversity_min_keep=1,
        locality_radius_km=0.0,
        locality_weight=0.0,
        query_tta_degrees=cfg.geolocator.retrieval_query_tta_degrees,
        query_tta_modes=cfg.geolocator.retrieval_query_tta_modes,
        query_tta_scales=cfg.geolocator.retrieval_query_tta_scales,
        query_tta_auto_modality=cfg.geolocator.retrieval_query_tta_auto_modality,
        query_tta_reduce=query_tta_reduce,
        query_expansion_top_n=cfg.geolocator.retrieval_query_expansion_top_n,
        query_expansion_beta=cfg.geolocator.retrieval_query_expansion_beta,
        query_expansion_alpha=cfg.geolocator.retrieval_query_expansion_alpha,
        tta_agreement_top_n=cfg.geolocator.retrieval_tta_agreement_top_n,
        tta_agreement_weight=cfg.geolocator.retrieval_tta_agreement_weight,
        local_match_top_n=cfg.geolocator.retrieval_local_match_top_n,
        local_match_weight=cfg.geolocator.retrieval_local_match_weight,
        local_match_ratio=cfg.geolocator.retrieval_local_match_ratio,
        local_match_max_features=cfg.geolocator.retrieval_local_match_max_features,
        structure_rerank_top_n=structure_rerank_top_n,
        structure_rerank_weight=structure_rerank_weight,
        graph_rerank_top_n=cfg.geolocator.retrieval_graph_rerank_top_n,
        graph_rerank_sigma_km=cfg.geolocator.retrieval_graph_rerank_sigma_km,
        graph_rerank_score_alpha=cfg.geolocator.retrieval_graph_rerank_score_alpha,
        graph_rerank_support_beta=cfg.geolocator.retrieval_graph_rerank_support_beta,
        graph_rerank_center_radius_km=cfg.geolocator.retrieval_graph_rerank_center_radius_km,
        kde_refine_top_n=cfg.geolocator.retrieval_kde_refine_top_n,
        kde_refine_sigma_km=cfg.geolocator.retrieval_kde_refine_sigma_km,
        kde_refine_score_power=cfg.geolocator.retrieval_kde_refine_score_power,
        kde_refine_margin_threshold=cfg.geolocator.retrieval_kde_refine_margin_threshold,
        kde_refine_switch_radius_km=cfg.geolocator.retrieval_kde_refine_switch_radius_km,
        kde_refine_max_iters=cfg.geolocator.retrieval_kde_refine_max_iters,
        kde_refine_adaptive_mass=cfg.geolocator.retrieval_kde_refine_adaptive_mass,
        geo_prior_mode=cfg.geolocator.retrieval_geo_prior_mode,
        geo_prior_bbox=cfg.geolocator.retrieval_geo_prior_bbox,
        geo_prior_sigma_km=cfg.geolocator.retrieval_geo_prior_sigma_km,
        geo_prior_min_keep=cfg.geolocator.retrieval_geo_prior_min_keep,
    )
    samples: List[RetrievalSample] = []
    missing_files = 0
    for item in records:
        rel_path = str(item["path"])
        image_path = Path(rel_path) if Path(rel_path).is_absolute() else resolve_image_path(images_dir, rel_path)
        if not image_path.exists():
            missing_files += 1
            continue
        candidates = provider.candidates(str(image_path))
        samples.append(
            RetrievalSample(
                image=str(image_path),
                gt_latitude=float(item["latitude"]),
                gt_longitude=float(item["longitude"]),
                candidates=list(candidates),
            )
        )
    return samples, missing_files


def _write_best_to_config(config_path: Path, best: dict) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    geo = payload.setdefault("geolocator", {})
    geo["retrieval_top_k"] = int(best["retrieval_top_k"])
    geo["retrieval_min_score"] = float(best["retrieval_min_score"])
    geo["retrieval_min_keep_topk"] = int(best["retrieval_min_keep_topk"])
    geo["retrieval_diversity_radius_km"] = float(best["retrieval_diversity_radius_km"])
    geo["retrieval_diversity_lambda"] = float(best["retrieval_diversity_lambda"])
    geo["retrieval_diversity_min_keep"] = int(best["retrieval_diversity_min_keep"])
    geo["retrieval_locality_radius_km"] = float(best["retrieval_locality_radius_km"])
    geo["retrieval_locality_weight"] = float(best["retrieval_locality_weight"])
    geo["retrieval_source_balance_beta"] = float(best["retrieval_source_balance_beta"])
    if "retrieval_structure_rerank_top_n" in best:
        geo["retrieval_structure_rerank_top_n"] = int(best["retrieval_structure_rerank_top_n"])
    if "retrieval_structure_rerank_weight" in best:
        geo["retrieval_structure_rerank_weight"] = float(best["retrieval_structure_rerank_weight"])
    if "retrieval_graph_rerank_top_n" in best:
        geo["retrieval_graph_rerank_top_n"] = int(best["retrieval_graph_rerank_top_n"])
        geo["retrieval_graph_rerank_sigma_km"] = float(best["retrieval_graph_rerank_sigma_km"])
        geo["retrieval_graph_rerank_score_alpha"] = float(best["retrieval_graph_rerank_score_alpha"])
        geo["retrieval_graph_rerank_support_beta"] = float(best["retrieval_graph_rerank_support_beta"])
        geo["retrieval_graph_rerank_center_radius_km"] = float(best["retrieval_graph_rerank_center_radius_km"])
    if "retrieval_consensus_top_n" in best:
        geo["retrieval_consensus_top_n"] = int(best["retrieval_consensus_top_n"])
        geo["retrieval_consensus_radius_km"] = float(best["retrieval_consensus_radius_km"])
        geo["retrieval_consensus_score_power"] = float(best["retrieval_consensus_score_power"])
    if "retrieval_kde_refine_top_n" in best:
        geo["retrieval_kde_refine_top_n"] = int(best["retrieval_kde_refine_top_n"])
        geo["retrieval_kde_refine_sigma_km"] = float(best["retrieval_kde_refine_sigma_km"])
        geo["retrieval_kde_refine_score_power"] = float(best["retrieval_kde_refine_score_power"])
        geo["retrieval_kde_refine_margin_threshold"] = float(best["retrieval_kde_refine_margin_threshold"])
        geo["retrieval_kde_refine_switch_radius_km"] = float(best["retrieval_kde_refine_switch_radius_km"])
        geo["retrieval_kde_refine_max_iters"] = int(best["retrieval_kde_refine_max_iters"])
        geo["retrieval_kde_refine_adaptive_mass"] = float(best["retrieval_kde_refine_adaptive_mass"])
    if best.get("retrieval_query_tta_reduce"):
        geo["retrieval_query_tta_reduce"] = str(best["retrieval_query_tta_reduce"])
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tune retrieval post-processing precision.")
    parser.add_argument("--config", default="src/config/paris_test.json")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="runs/tune_retrieval_geo.json")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-index-path", default="", help="Optional index override path.")
    parser.add_argument("--retrieval-topk", default="25,50,80")
    parser.add_argument("--retrieval-min-score", default="0.05,0.10,0.15")
    parser.add_argument("--retrieval-min-keep-topk", default="0,1,3")
    parser.add_argument("--retrieval-diversity-radius-km", default="0.0,1.0,2.0")
    parser.add_argument("--retrieval-diversity-lambda", default="1.0,0.9,0.8")
    parser.add_argument("--retrieval-diversity-min-keep", default="1,3,5")
    parser.add_argument("--retrieval-locality-radius-km", default="0.0,25.0,60.0")
    parser.add_argument("--retrieval-locality-weight", default="0.0,0.8,1.2,1.8")
    parser.add_argument("--retrieval-source-balance-beta", default="0.0,0.35,0.7")
    parser.add_argument("--retrieval-graph-rerank-top-n", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-graph-rerank-sigma-km", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-graph-rerank-score-alpha", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-graph-rerank-support-beta", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-graph-rerank-center-radius-km", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-consensus-top-n", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-consensus-radius-km", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-consensus-score-power", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-kde-refine-top-n", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-kde-refine-sigma-km", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-kde-refine-score-power", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-kde-refine-margin-threshold", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-kde-refine-switch-radius-km", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-kde-refine-max-iters", default="", help="Comma list; empty uses config value.")
    parser.add_argument("--retrieval-kde-refine-adaptive-mass", default="", help="Comma list; empty uses config value.")
    parser.add_argument(
        "--retrieval-structure-rerank-top-n",
        default="",
        help="Comma list of structure-rerank top_n values to evaluate. Empty uses config value.",
    )
    parser.add_argument(
        "--retrieval-structure-rerank-weight",
        default="",
        help="Comma list of structure-rerank blend weights to evaluate. Empty uses config value.",
    )
    parser.add_argument(
        "--retrieval-query-tta-reduce",
        default="",
        help="Comma list of TTA reduction modes to evaluate (mean,median,max,rrf). Empty uses config mode.",
    )
    parser.add_argument(
        "--rank-objective",
        default="balanced",
        help="How to rank tuning results: balanced, median_km, mean_km, p90_km, within_1km_pct, within_2km_pct, within_5km_pct, within_10km_pct.",
    )
    parser.add_argument("--apply-best-config", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    images_dir = Path(args.images_dir)
    metadata = Path(args.metadata)
    records = load_metadata_records(metadata, images_dir=images_dir)
    random.Random(args.seed).shuffle(records)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    retrieval_topk = _parse_int_list(args.retrieval_topk)
    retrieval_min_score = _parse_float_list(args.retrieval_min_score)
    retrieval_min_keep_topk = _parse_int_list(args.retrieval_min_keep_topk)
    retrieval_diversity_radius = _parse_float_list(args.retrieval_diversity_radius_km)
    retrieval_diversity_lambda = _parse_float_list(args.retrieval_diversity_lambda)
    retrieval_diversity_min_keep = _parse_int_list(args.retrieval_diversity_min_keep)
    retrieval_locality_radius = _parse_float_list(args.retrieval_locality_radius_km)
    retrieval_locality_weight = _parse_float_list(args.retrieval_locality_weight)
    retrieval_source_balance = _parse_float_list(args.retrieval_source_balance_beta)
    retrieval_graph_rerank_top_n = _parse_int_list(args.retrieval_graph_rerank_top_n)
    retrieval_graph_rerank_sigma_km = _parse_float_list(args.retrieval_graph_rerank_sigma_km)
    retrieval_graph_rerank_score_alpha = _parse_float_list(args.retrieval_graph_rerank_score_alpha)
    retrieval_graph_rerank_support_beta = _parse_float_list(args.retrieval_graph_rerank_support_beta)
    retrieval_graph_rerank_center_radius_km = _parse_float_list(args.retrieval_graph_rerank_center_radius_km)
    retrieval_consensus_top_n = _parse_int_list(args.retrieval_consensus_top_n)
    retrieval_consensus_radius_km = _parse_float_list(args.retrieval_consensus_radius_km)
    retrieval_consensus_score_power = _parse_float_list(args.retrieval_consensus_score_power)
    retrieval_kde_refine_top_n = _parse_int_list(args.retrieval_kde_refine_top_n)
    retrieval_kde_refine_sigma_km = _parse_float_list(args.retrieval_kde_refine_sigma_km)
    retrieval_kde_refine_score_power = _parse_float_list(args.retrieval_kde_refine_score_power)
    retrieval_kde_refine_margin_threshold = _parse_float_list(args.retrieval_kde_refine_margin_threshold)
    retrieval_kde_refine_switch_radius_km = _parse_float_list(args.retrieval_kde_refine_switch_radius_km)
    retrieval_kde_refine_max_iters = _parse_int_list(args.retrieval_kde_refine_max_iters)
    retrieval_kde_refine_adaptive_mass = _parse_float_list(args.retrieval_kde_refine_adaptive_mass)
    retrieval_structure_rerank_top_n = _parse_int_list(args.retrieval_structure_rerank_top_n)
    retrieval_structure_rerank_weight = _parse_float_list(args.retrieval_structure_rerank_weight)
    tta_reduce_modes = _parse_tta_reduce_list(str(args.retrieval_query_tta_reduce))
    rank_objective = _parse_rank_objective(str(args.rank_objective))
    if not tta_reduce_modes:
        tta_reduce_modes = [str(cfg.geolocator.retrieval_query_tta_reduce).lower()]
    if not retrieval_structure_rerank_top_n:
        retrieval_structure_rerank_top_n = [int(cfg.geolocator.retrieval_structure_rerank_top_n)]
    if not retrieval_structure_rerank_weight:
        retrieval_structure_rerank_weight = [float(cfg.geolocator.retrieval_structure_rerank_weight)]
    if not retrieval_graph_rerank_top_n:
        retrieval_graph_rerank_top_n = [int(cfg.geolocator.retrieval_graph_rerank_top_n)]
    if not retrieval_graph_rerank_sigma_km:
        retrieval_graph_rerank_sigma_km = [float(cfg.geolocator.retrieval_graph_rerank_sigma_km)]
    if not retrieval_graph_rerank_score_alpha:
        retrieval_graph_rerank_score_alpha = [float(cfg.geolocator.retrieval_graph_rerank_score_alpha)]
    if not retrieval_graph_rerank_support_beta:
        retrieval_graph_rerank_support_beta = [float(cfg.geolocator.retrieval_graph_rerank_support_beta)]
    if not retrieval_graph_rerank_center_radius_km:
        retrieval_graph_rerank_center_radius_km = [float(cfg.geolocator.retrieval_graph_rerank_center_radius_km)]
    if not retrieval_consensus_top_n:
        retrieval_consensus_top_n = [int(cfg.geolocator.retrieval_consensus_top_n)]
    if not retrieval_consensus_radius_km:
        retrieval_consensus_radius_km = [float(cfg.geolocator.retrieval_consensus_radius_km)]
    if not retrieval_consensus_score_power:
        retrieval_consensus_score_power = [float(cfg.geolocator.retrieval_consensus_score_power)]
    if not retrieval_kde_refine_top_n:
        retrieval_kde_refine_top_n = [int(cfg.geolocator.retrieval_kde_refine_top_n)]
    if not retrieval_kde_refine_sigma_km:
        retrieval_kde_refine_sigma_km = [float(cfg.geolocator.retrieval_kde_refine_sigma_km)]
    if not retrieval_kde_refine_score_power:
        retrieval_kde_refine_score_power = [float(cfg.geolocator.retrieval_kde_refine_score_power)]
    if not retrieval_kde_refine_margin_threshold:
        retrieval_kde_refine_margin_threshold = [float(cfg.geolocator.retrieval_kde_refine_margin_threshold)]
    if not retrieval_kde_refine_switch_radius_km:
        retrieval_kde_refine_switch_radius_km = [float(cfg.geolocator.retrieval_kde_refine_switch_radius_km)]
    if not retrieval_kde_refine_max_iters:
        retrieval_kde_refine_max_iters = [int(cfg.geolocator.retrieval_kde_refine_max_iters)]
    if not retrieval_kde_refine_adaptive_mass:
        retrieval_kde_refine_adaptive_mass = [float(cfg.geolocator.retrieval_kde_refine_adaptive_mass)]

    max_top_k = max(retrieval_topk) if retrieval_topk else cfg.geolocator.retrieval_top_k
    min_score_floor = min(retrieval_min_score) if retrieval_min_score else cfg.geolocator.retrieval_min_score
    index_override = args.retrieval_index_path.strip() or None

    samples_by_mode: dict[tuple[str, int, float], List[RetrievalSample]] = {}
    missing_files = 0
    for tta_mode in tta_reduce_modes:
        for structure_top_n in retrieval_structure_rerank_top_n:
            for structure_weight in retrieval_structure_rerank_weight:
                samples, missing = _collect_raw_samples(
                    cfg,
                    images_dir=images_dir,
                    records=records,
                    max_top_k=max_top_k,
                    min_score_floor=min_score_floor,
                    retrieval_index_path=index_override,
                    query_tta_reduce=tta_mode,
                    structure_rerank_top_n=int(structure_top_n),
                    structure_rerank_weight=float(structure_weight),
                )
                samples_by_mode[(tta_mode, int(structure_top_n), float(structure_weight))] = samples
                missing_files = max(missing_files, int(missing))

    results = []
    mode_grid = product(tta_reduce_modes, retrieval_structure_rerank_top_n, retrieval_structure_rerank_weight)
    param_grid_axes = (
        retrieval_topk,
        retrieval_min_score,
        retrieval_min_keep_topk,
        retrieval_diversity_radius,
        retrieval_diversity_lambda,
        retrieval_diversity_min_keep,
        retrieval_locality_radius,
        retrieval_locality_weight,
        retrieval_source_balance,
        retrieval_graph_rerank_top_n,
        retrieval_graph_rerank_sigma_km,
        retrieval_graph_rerank_score_alpha,
        retrieval_graph_rerank_support_beta,
        retrieval_graph_rerank_center_radius_km,
        retrieval_consensus_top_n,
        retrieval_consensus_radius_km,
        retrieval_consensus_score_power,
        retrieval_kde_refine_top_n,
        retrieval_kde_refine_sigma_km,
        retrieval_kde_refine_score_power,
        retrieval_kde_refine_margin_threshold,
        retrieval_kde_refine_switch_radius_km,
        retrieval_kde_refine_max_iters,
        retrieval_kde_refine_adaptive_mass,
    )
    for tta_mode, structure_top_n, structure_weight in mode_grid:
        samples = samples_by_mode.get((tta_mode, int(structure_top_n), float(structure_weight)), [])
        for (
            top_k,
            min_score,
            min_keep_topk,
            diversity_radius,
            diversity_lambda,
            diversity_min_keep,
            locality_radius,
            locality_weight,
            source_balance_beta,
            graph_top_n,
            graph_sigma,
            graph_alpha,
            graph_beta,
            graph_center_radius,
            consensus_top_n,
            consensus_radius,
            consensus_power,
            kde_top_n,
            kde_sigma,
            kde_power,
            kde_margin,
            kde_switch_radius,
            kde_iters,
            kde_mass,
        ) in product(*param_grid_axes):
            metrics = _evaluate_samples(
                samples,
                top_k=top_k,
                min_score=min_score,
                min_keep_topk=min_keep_topk,
                diversity_radius_km=diversity_radius,
                diversity_lambda=diversity_lambda,
                diversity_min_keep=diversity_min_keep,
                locality_radius_km=locality_radius,
                locality_weight=locality_weight,
                source_balance_beta=source_balance_beta,
                graph_rerank_top_n=graph_top_n,
                graph_rerank_sigma_km=graph_sigma,
                graph_rerank_score_alpha=graph_alpha,
                graph_rerank_support_beta=graph_beta,
                graph_rerank_center_radius_km=graph_center_radius,
                consensus_top_n=consensus_top_n,
                consensus_radius_km=consensus_radius,
                consensus_score_power=consensus_power,
                kde_refine_top_n=kde_top_n,
                kde_refine_sigma_km=kde_sigma,
                kde_refine_score_power=kde_power,
                kde_refine_margin_threshold=kde_margin,
                kde_refine_switch_radius_km=kde_switch_radius,
                kde_refine_max_iters=kde_iters,
                kde_refine_adaptive_mass=kde_mass,
            )
            results.append(
                {
                    "retrieval_query_tta_reduce": tta_mode,
                    "retrieval_structure_rerank_top_n": int(structure_top_n),
                    "retrieval_structure_rerank_weight": float(structure_weight),
                    "retrieval_top_k": top_k,
                    "retrieval_min_score": min_score,
                    "retrieval_min_keep_topk": min_keep_topk,
                    "retrieval_diversity_radius_km": diversity_radius,
                    "retrieval_diversity_lambda": diversity_lambda,
                    "retrieval_diversity_min_keep": diversity_min_keep,
                    "retrieval_locality_radius_km": locality_radius,
                    "retrieval_locality_weight": locality_weight,
                    "retrieval_source_balance_beta": source_balance_beta,
                    "retrieval_graph_rerank_top_n": int(graph_top_n),
                    "retrieval_graph_rerank_sigma_km": float(graph_sigma),
                    "retrieval_graph_rerank_score_alpha": float(graph_alpha),
                    "retrieval_graph_rerank_support_beta": float(graph_beta),
                    "retrieval_graph_rerank_center_radius_km": float(graph_center_radius),
                    "retrieval_consensus_top_n": int(consensus_top_n),
                    "retrieval_consensus_radius_km": float(consensus_radius),
                    "retrieval_consensus_score_power": float(consensus_power),
                    "retrieval_kde_refine_top_n": int(kde_top_n),
                    "retrieval_kde_refine_sigma_km": float(kde_sigma),
                    "retrieval_kde_refine_score_power": float(kde_power),
                    "retrieval_kde_refine_margin_threshold": float(kde_margin),
                    "retrieval_kde_refine_switch_radius_km": float(kde_switch_radius),
                    "retrieval_kde_refine_max_iters": int(kde_iters),
                    "retrieval_kde_refine_adaptive_mass": float(kde_mass),
                    **metrics,
                }
            )

    results.sort(key=lambda row: _result_sort_key(row, rank_objective))
    best = results[0] if results else None

    payload = {
        "config": args.config,
        "images_dir": str(images_dir),
        "metadata": str(metadata),
        "index_path": index_override or cfg.geolocator.retrieval_index_path,
        "index_paths": list(cfg.geolocator.retrieval_index_paths),
        "index_weights": list(cfg.geolocator.retrieval_index_weights),
        "index_model_ids": list(cfg.geolocator.retrieval_index_model_ids),
        "index_projection_paths": list(cfg.geolocator.retrieval_index_projection_paths),
        "retrieval_per_index_top_k": cfg.geolocator.retrieval_per_index_top_k,
        "retrieval_index_score_norm": cfg.geolocator.retrieval_index_score_norm,
        "retrieval_source_fusion_mode": cfg.geolocator.retrieval_source_fusion_mode,
        "retrieval_source_balance_beta": cfg.geolocator.retrieval_source_balance_beta,
        "candidate_source_balance_beta": cfg.geolocator.candidate_source_balance_beta,
        "retrieval_query_tta_degrees": list(cfg.geolocator.retrieval_query_tta_degrees),
        "retrieval_query_tta_modes": list(cfg.geolocator.retrieval_query_tta_modes),
        "retrieval_query_tta_scales": list(cfg.geolocator.retrieval_query_tta_scales),
        "retrieval_query_tta_auto_modality": cfg.geolocator.retrieval_query_tta_auto_modality,
        "retrieval_query_tta_reduce": cfg.geolocator.retrieval_query_tta_reduce,
        "retrieval_structure_rerank_top_n_search": retrieval_structure_rerank_top_n,
        "retrieval_structure_rerank_weight_search": retrieval_structure_rerank_weight,
        "retrieval_graph_rerank_top_n_search": retrieval_graph_rerank_top_n,
        "retrieval_graph_rerank_sigma_km_search": retrieval_graph_rerank_sigma_km,
        "retrieval_graph_rerank_score_alpha_search": retrieval_graph_rerank_score_alpha,
        "retrieval_graph_rerank_support_beta_search": retrieval_graph_rerank_support_beta,
        "retrieval_graph_rerank_center_radius_km_search": retrieval_graph_rerank_center_radius_km,
        "retrieval_consensus_top_n_search": retrieval_consensus_top_n,
        "retrieval_consensus_radius_km_search": retrieval_consensus_radius_km,
        "retrieval_consensus_score_power_search": retrieval_consensus_score_power,
        "retrieval_kde_refine_top_n_search": retrieval_kde_refine_top_n,
        "retrieval_kde_refine_sigma_km_search": retrieval_kde_refine_sigma_km,
        "retrieval_kde_refine_score_power_search": retrieval_kde_refine_score_power,
        "retrieval_kde_refine_margin_threshold_search": retrieval_kde_refine_margin_threshold,
        "retrieval_kde_refine_switch_radius_km_search": retrieval_kde_refine_switch_radius_km,
        "retrieval_kde_refine_max_iters_search": retrieval_kde_refine_max_iters,
        "retrieval_kde_refine_adaptive_mass_search": retrieval_kde_refine_adaptive_mass,
        "retrieval_query_expansion_top_n": cfg.geolocator.retrieval_query_expansion_top_n,
        "retrieval_query_expansion_beta": cfg.geolocator.retrieval_query_expansion_beta,
        "retrieval_query_expansion_alpha": cfg.geolocator.retrieval_query_expansion_alpha,
        "retrieval_tta_agreement_top_n": cfg.geolocator.retrieval_tta_agreement_top_n,
        "retrieval_tta_agreement_weight": cfg.geolocator.retrieval_tta_agreement_weight,
        "retrieval_local_match_top_n": cfg.geolocator.retrieval_local_match_top_n,
        "retrieval_local_match_weight": cfg.geolocator.retrieval_local_match_weight,
        "retrieval_local_match_ratio": cfg.geolocator.retrieval_local_match_ratio,
        "retrieval_local_match_max_features": cfg.geolocator.retrieval_local_match_max_features,
        "retrieval_structure_rerank_top_n": cfg.geolocator.retrieval_structure_rerank_top_n,
        "retrieval_structure_rerank_weight": cfg.geolocator.retrieval_structure_rerank_weight,
        "retrieval_graph_rerank_top_n": cfg.geolocator.retrieval_graph_rerank_top_n,
        "retrieval_graph_rerank_sigma_km": cfg.geolocator.retrieval_graph_rerank_sigma_km,
        "retrieval_graph_rerank_score_alpha": cfg.geolocator.retrieval_graph_rerank_score_alpha,
        "retrieval_graph_rerank_support_beta": cfg.geolocator.retrieval_graph_rerank_support_beta,
        "retrieval_graph_rerank_center_radius_km": cfg.geolocator.retrieval_graph_rerank_center_radius_km,
        "retrieval_kde_refine_top_n": cfg.geolocator.retrieval_kde_refine_top_n,
        "retrieval_kde_refine_sigma_km": cfg.geolocator.retrieval_kde_refine_sigma_km,
        "retrieval_kde_refine_score_power": cfg.geolocator.retrieval_kde_refine_score_power,
        "retrieval_kde_refine_margin_threshold": cfg.geolocator.retrieval_kde_refine_margin_threshold,
        "retrieval_kde_refine_switch_radius_km": cfg.geolocator.retrieval_kde_refine_switch_radius_km,
        "retrieval_kde_refine_max_iters": cfg.geolocator.retrieval_kde_refine_max_iters,
        "retrieval_kde_refine_adaptive_mass": cfg.geolocator.retrieval_kde_refine_adaptive_mass,
        "retrieval_geo_prior_mode": cfg.geolocator.retrieval_geo_prior_mode,
        "retrieval_geo_prior_bbox": list(cfg.geolocator.retrieval_geo_prior_bbox),
        "retrieval_geo_prior_sigma_km": cfg.geolocator.retrieval_geo_prior_sigma_km,
        "retrieval_geo_prior_min_keep": cfg.geolocator.retrieval_geo_prior_min_keep,
        "retrieval_query_tta_reduce_search": tta_reduce_modes,
        "rank_objective": rank_objective,
        "retrieval_min_keep_topk_default": cfg.geolocator.retrieval_min_keep_topk,
        "limit": args.limit,
        "missing_files": missing_files,
        "samples_with_candidates": {
            f"{mode}|structure_top_n={top_n}|structure_weight={float(weight):.3f}": len(
                samples_by_mode.get((mode, int(top_n), float(weight)), [])
            )
            for mode in tta_reduce_modes
            for top_n in retrieval_structure_rerank_top_n
            for weight in retrieval_structure_rerank_weight
        },
        "search_space_size": len(results),
        "best": best,
        "top10": results[:10],
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.apply_best_config and best is not None:
        _write_best_to_config(Path(args.config), best)
    print(f"Wrote {out_path}")
    if best is not None:
        print(
            "Best tta_reduce={mode} structure_top_n={top_n} structure_weight={weight:.2f} "
            "median_km={median:.3f}, mean_km={mean:.3f}, within_5km={w5:.2f}%".format(
                mode=str(best.get("retrieval_query_tta_reduce", "unknown")),
                top_n=int(best.get("retrieval_structure_rerank_top_n", 0) or 0),
                weight=float(best.get("retrieval_structure_rerank_weight", 0.0) or 0.0),
                median=float(best["median_km"]) if best["median_km"] is not None else float("nan"),
                mean=float(best["mean_km"]) if best["mean_km"] is not None else float("nan"),
                w5=float(best["within_5km_pct"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
