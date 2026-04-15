"""
Tune retrieval precision with lightweight candidate post-processing sweeps.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from src.core.geo.retrieval_provider import (
    GeoRetrievalProvider,
    _apply_locality_rerank,
    _select_diverse_geo_candidates,
    _select_source_balanced_candidates,
)
from src.core.logic.config import HeimdallConfig, load_config
from src.core.logic.types import GeoCandidate
from src.tools.run_geo_eval import haversine_km, resolve_image_path

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
    selected = _select_diverse_geo_candidates(
        balanced,
        top_k=max(1, int(top_k)),
        radius_km=max(0.0, float(diversity_radius_km)),
        diversity_lambda=min(1.0, max(0.0, float(diversity_lambda))),
        min_keep=max(0, int(diversity_min_keep)),
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
) -> tuple[List[RetrievalSample], int]:
    index_path = retrieval_index_path or cfg.geolocator.retrieval_index_path
    provider = GeoRetrievalProvider(
        index_path=index_path,
        index_paths=cfg.geolocator.retrieval_index_paths,
        index_weights=cfg.geolocator.retrieval_index_weights,
        index_model_ids=cfg.geolocator.retrieval_index_model_ids,
        model_id=cfg.geolocator.retrieval_model_id or "openai/clip-vit-large-patch14",
        top_k=max(1, int(max_top_k)),
        per_index_top_k=cfg.geolocator.retrieval_per_index_top_k,
        index_score_norm=cfg.geolocator.retrieval_index_score_norm,
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
        query_tta_reduce=query_tta_reduce,
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
    if best.get("retrieval_query_tta_reduce"):
        geo["retrieval_query_tta_reduce"] = str(best["retrieval_query_tta_reduce"])
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    import pandas as pd

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
    df = pd.read_csv(metadata)
    if not {"path", "latitude", "longitude"}.issubset(df.columns):
        raise ValueError("metadata must include columns: path, latitude, longitude")
    records = df[["path", "latitude", "longitude"]].to_dict("records")
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
    tta_reduce_modes = _parse_tta_reduce_list(str(args.retrieval_query_tta_reduce))
    rank_objective = _parse_rank_objective(str(args.rank_objective))
    if not tta_reduce_modes:
        tta_reduce_modes = [str(cfg.geolocator.retrieval_query_tta_reduce).lower()]

    max_top_k = max(retrieval_topk) if retrieval_topk else cfg.geolocator.retrieval_top_k
    min_score_floor = min(retrieval_min_score) if retrieval_min_score else cfg.geolocator.retrieval_min_score
    index_override = args.retrieval_index_path.strip() or None

    samples_by_tta_mode: dict[str, List[RetrievalSample]] = {}
    missing_files = 0
    for tta_mode in tta_reduce_modes:
        samples, missing = _collect_raw_samples(
            cfg,
            images_dir=images_dir,
            records=records,
            max_top_k=max_top_k,
            min_score_floor=min_score_floor,
            retrieval_index_path=index_override,
            query_tta_reduce=tta_mode,
        )
        samples_by_tta_mode[tta_mode] = samples
        missing_files = max(missing_files, int(missing))

    results = []
    for tta_mode in tta_reduce_modes:
        samples = samples_by_tta_mode.get(tta_mode, [])
        for top_k in retrieval_topk:
            for min_score in retrieval_min_score:
                for min_keep_topk in retrieval_min_keep_topk:
                    for diversity_radius in retrieval_diversity_radius:
                        for diversity_lambda in retrieval_diversity_lambda:
                            for diversity_min_keep in retrieval_diversity_min_keep:
                                for locality_radius in retrieval_locality_radius:
                                    for locality_weight in retrieval_locality_weight:
                                        for source_balance_beta in retrieval_source_balance:
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
                                            )
                                            results.append(
                                                {
                                                    "retrieval_query_tta_reduce": tta_mode,
                                                    "retrieval_top_k": top_k,
                                                    "retrieval_min_score": min_score,
                                                    "retrieval_min_keep_topk": min_keep_topk,
                                                    "retrieval_diversity_radius_km": diversity_radius,
                                                    "retrieval_diversity_lambda": diversity_lambda,
                                                    "retrieval_diversity_min_keep": diversity_min_keep,
                                                    "retrieval_locality_radius_km": locality_radius,
                                                    "retrieval_locality_weight": locality_weight,
                                                    "retrieval_source_balance_beta": source_balance_beta,
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
        "retrieval_per_index_top_k": cfg.geolocator.retrieval_per_index_top_k,
        "retrieval_index_score_norm": cfg.geolocator.retrieval_index_score_norm,
        "retrieval_source_balance_beta": cfg.geolocator.retrieval_source_balance_beta,
        "candidate_source_balance_beta": cfg.geolocator.candidate_source_balance_beta,
        "retrieval_query_tta_degrees": list(cfg.geolocator.retrieval_query_tta_degrees),
        "retrieval_query_tta_reduce": cfg.geolocator.retrieval_query_tta_reduce,
        "retrieval_query_tta_reduce_search": tta_reduce_modes,
        "rank_objective": rank_objective,
        "retrieval_min_keep_topk_default": cfg.geolocator.retrieval_min_keep_topk,
        "limit": args.limit,
        "missing_files": missing_files,
        "samples_with_candidates": {mode: len(samples_by_tta_mode.get(mode, [])) for mode in tta_reduce_modes},
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
            "Best tta_reduce={mode} median_km={median:.3f}, mean_km={mean:.3f}, within_5km={w5:.2f}%".format(
                mode=str(best.get("retrieval_query_tta_reduce", "unknown")),
                median=float(best["median_km"]) if best["median_km"] is not None else float("nan"),
                mean=float(best["mean_km"]) if best["mean_km"] is not None else float("nan"),
                w5=float(best["within_5km_pct"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
