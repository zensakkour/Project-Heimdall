"""
Grid search for geo fusion parameters using the geo eval dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from src.core.logic.config import HeimdallConfig, load_config
from src.tools.run_geo_eval import build_pipeline, haversine_km, predict_latlon, resolve_image_path


def _parse_float_list(raw: str) -> List[float]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(x) for x in items]


def _parse_int_list(raw: str) -> List[int]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(x) for x in items]


def _parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_bool_list(raw: str) -> List[bool]:
    out: List[bool] = []
    for item in _parse_str_list(raw):
        key = item.lower()
        if key in {"1", "true", "yes", "on"}:
            out.append(True)
        elif key in {"0", "false", "no", "off"}:
            out.append(False)
        else:
            raise ValueError(f"invalid boolean list value: {item}")
    return out


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    k = max(0, min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return float(sorted_vals[k])


def _pct_within(vals: List[float], km: float) -> float:
    if not vals:
        return 0.0
    count = sum(1 for v in vals if v <= km)
    return 100.0 * count / len(vals)


def _evaluate(cfg: HeimdallConfig, images_dir: Path, records: List[dict]) -> dict:
    pipeline = build_pipeline(cfg)
    distances = []
    missing = 0
    null_pred = 0

    for item in records:
        rel_path = str(item["path"])
        image_path = (
            Path(rel_path)
            if Path(rel_path).is_absolute()
            else resolve_image_path(images_dir, rel_path)
        )
        if not image_path.exists():
            missing += 1
            continue
        result = pipeline.run(str(image_path))
        pred = predict_latlon(result)
        if pred is None:
            null_pred += 1
            continue
        gt_lat = float(item["latitude"])
        gt_lon = float(item["longitude"])
        distances.append(haversine_km(gt_lat, gt_lon, pred[0], pred[1]))

    distances.sort()
    evaluated = len(distances)
    mean_km = float(sum(distances) / evaluated) if evaluated else None
    median_km = float(distances[evaluated // 2]) if evaluated else None
    p90_km = _percentile(distances, 90)

    return {
        "evaluated": evaluated,
        "missing": missing,
        "null_predictions": null_pred,
        "mean_km": mean_km,
        "median_km": median_km,
        "p90_km": p90_km,
        "within_1km_pct": _pct_within(distances, 1.0),
        "within_5km_pct": _pct_within(distances, 5.0),
        "within_10km_pct": _pct_within(distances, 10.0),
        "within_50km_pct": _pct_within(distances, 50.0),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Grid search for geo fusion parameters.")
    parser.add_argument("--config", default="src/config/paris_test.json", help="Base config.")
    parser.add_argument("--images-dir", required=True, help="Images directory.")
    parser.add_argument("--metadata", required=True, help="Metadata CSV (path, latitude, longitude).")
    parser.add_argument("--output", default="runs/tune_geo_fusion.json", help="Output JSON.")
    parser.add_argument("--limit", type=int, default=200, help="Limit number of samples.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--geospot-scales", default="0.0,0.05,0.1", help="Comma list.")
    parser.add_argument("--retrieval-temps", default="0.05,0.08,0.12", help="Comma list.")
    parser.add_argument("--retrieval-topk", default="25,50", help="Comma list.")
    parser.add_argument("--fusion-topk", default="10,25", help="Comma list.")
    parser.add_argument("--retrieval-diversity-radius-km", default="0.0", help="Comma list.")
    parser.add_argument("--retrieval-diversity-lambda", default="1.0", help="Comma list.")
    parser.add_argument("--retrieval-diversity-min-keep", default="1", help="Comma list.")
    parser.add_argument("--retrieval-norms", default="none,zscore_sigmoid,minmax,rank_exp", help="Comma list.")
    parser.add_argument("--spatial-consensus", default="true,false", help="Comma list of booleans.")
    parser.add_argument("--spatial-sigmas-km", default="1.0,2.0,5.0", help="Comma list.")
    parser.add_argument("--spatial-weights", default="0.5,1.0,2.0", help="Comma list.")
    parser.add_argument("--cross-source-weights", default="1.0", help="Comma list.")
    parser.add_argument("--plausibility-rerank", default="true", help="Comma list of booleans.")
    parser.add_argument("--plausibility-radius-km", default="200.0", help="Comma list.")
    parser.add_argument("--plausibility-weights", default="1.0", help="Comma list.")
    parser.add_argument("--outlier-guard", default="true", help="Comma list of booleans.")
    parser.add_argument("--outlier-guard-strengths", default="1.0", help="Comma list.")
    parser.add_argument("--outlier-guard-min-scale-km", default="120.0", help="Comma list.")
    parser.add_argument("--outlier-guard-mad-scale", default="3.0", help="Comma list.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    images_dir = Path(args.images_dir)
    metadata_path = Path(args.metadata)
    df = pd.read_csv(metadata_path)
    if not {"path", "latitude", "longitude"}.issubset(df.columns):
        raise ValueError("metadata must include columns: path, latitude, longitude")

    records = df[["path", "latitude", "longitude"]].to_dict("records")
    random.Random(args.seed).shuffle(records)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    geospot_scales = _parse_float_list(args.geospot_scales)
    retrieval_temps = _parse_float_list(args.retrieval_temps)
    retrieval_topk = _parse_int_list(args.retrieval_topk)
    fusion_topk = _parse_int_list(args.fusion_topk)
    retrieval_diversity_radius = _parse_float_list(args.retrieval_diversity_radius_km)
    retrieval_diversity_lambda = _parse_float_list(args.retrieval_diversity_lambda)
    retrieval_diversity_min_keep = _parse_int_list(args.retrieval_diversity_min_keep)
    retrieval_norms = _parse_str_list(args.retrieval_norms)
    spatial_consensus_flags = _parse_bool_list(args.spatial_consensus)
    spatial_sigmas_km = _parse_float_list(args.spatial_sigmas_km)
    spatial_weights = _parse_float_list(args.spatial_weights)
    cross_source_weights = _parse_float_list(args.cross_source_weights)
    plausibility_flags = _parse_bool_list(args.plausibility_rerank)
    plausibility_radius_km = _parse_float_list(args.plausibility_radius_km)
    plausibility_weights = _parse_float_list(args.plausibility_weights)
    outlier_guard_flags = _parse_bool_list(args.outlier_guard)
    outlier_guard_strengths = _parse_float_list(args.outlier_guard_strengths)
    outlier_guard_min_scale_km = _parse_float_list(args.outlier_guard_min_scale_km)
    outlier_guard_mad_scale = _parse_float_list(args.outlier_guard_mad_scale)

    results = []
    for scale in geospot_scales:
        for temp in retrieval_temps:
            for topk in retrieval_topk:
                for ftopk in fusion_topk:
                    for diversity_radius in retrieval_diversity_radius:
                        for diversity_lambda in retrieval_diversity_lambda:
                            for diversity_min_keep in retrieval_diversity_min_keep:
                                for norm in retrieval_norms:
                                    for use_spatial in spatial_consensus_flags:
                                        sigma_values = spatial_sigmas_km if use_spatial else [cfg.fusion.spatial_sigma_km]
                                        spatial_weight_values = spatial_weights if use_spatial else [0.0]
                                        for spatial_sigma in sigma_values:
                                            for spatial_weight in spatial_weight_values:
                                                for cross_weight in cross_source_weights:
                                                    for use_plausibility in plausibility_flags:
                                                        plausibility_radius_values = (
                                                            plausibility_radius_km if use_plausibility else [cfg.fusion.plausibility_radius_km]
                                                        )
                                                        plausibility_weight_values = (
                                                            plausibility_weights if use_plausibility else [0.0]
                                                        )
                                                        for plaus_radius in plausibility_radius_values:
                                                            for plaus_weight in plausibility_weight_values:
                                                                for use_outlier_guard in outlier_guard_flags:
                                                                    outlier_strength_values = (
                                                                        outlier_guard_strengths if use_outlier_guard else [0.0]
                                                                    )
                                                                    outlier_min_scale_values = (
                                                                        outlier_guard_min_scale_km
                                                                        if use_outlier_guard
                                                                        else [cfg.fusion.outlier_guard_min_scale_km]
                                                                    )
                                                                    outlier_mad_values = (
                                                                        outlier_guard_mad_scale
                                                                        if use_outlier_guard
                                                                        else [cfg.fusion.outlier_guard_mad_scale]
                                                                    )
                                                                    for outlier_strength in outlier_strength_values:
                                                                        for outlier_min_scale in outlier_min_scale_values:
                                                                            for outlier_mad in outlier_mad_values:
                                                                                tuned_geo = replace(
                                                                                    cfg.geolocator,
                                                                                    geospot_score_scale=scale,
                                                                                    retrieval_top_k=topk,
                                                                                    retrieval_diversity_radius_km=diversity_radius,
                                                                                    retrieval_diversity_lambda=diversity_lambda,
                                                                                    retrieval_diversity_min_keep=diversity_min_keep,
                                                                                )
                                                                                tuned_fusion = replace(
                                                                                    cfg.fusion,
                                                                                    retrieval_temperature=temp,
                                                                                    retrieval_score_norm=norm,
                                                                                    use_spatial_consensus=use_spatial,
                                                                                    spatial_sigma_km=spatial_sigma,
                                                                                    spatial_consensus_weight=spatial_weight,
                                                                                    cross_source_weight=cross_weight,
                                                                                    use_plausibility_rerank=use_plausibility,
                                                                                    plausibility_radius_km=plaus_radius,
                                                                                    plausibility_weight=plaus_weight,
                                                                                    use_adaptive_outlier_guard=use_outlier_guard,
                                                                                    outlier_guard_strength=outlier_strength,
                                                                                    outlier_guard_min_scale_km=outlier_min_scale,
                                                                                    outlier_guard_mad_scale=outlier_mad,
                                                                                    top_k=ftopk,
                                                                                )
                                                                                tuned_cfg = replace(
                                                                                    cfg, geolocator=tuned_geo, fusion=tuned_fusion
                                                                                )
                                                                                metrics = _evaluate(tuned_cfg, images_dir, records)
                                                                                results.append(
                                                                                    {
                                                                                        "geospot_score_scale": scale,
                                                                                        "retrieval_temperature": temp,
                                                                                        "retrieval_top_k": topk,
                                                                                        "retrieval_diversity_radius_km": diversity_radius,
                                                                                        "retrieval_diversity_lambda": diversity_lambda,
                                                                                        "retrieval_diversity_min_keep": diversity_min_keep,
                                                                                        "fusion_top_k": ftopk,
                                                                                        "retrieval_score_norm": norm,
                                                                                        "use_spatial_consensus": use_spatial,
                                                                                        "spatial_sigma_km": spatial_sigma,
                                                                                        "spatial_consensus_weight": spatial_weight,
                                                                                        "cross_source_weight": cross_weight,
                                                                                        "use_plausibility_rerank": use_plausibility,
                                                                                        "plausibility_radius_km": plaus_radius,
                                                                                        "plausibility_weight": plaus_weight,
                                                                                        "use_adaptive_outlier_guard": use_outlier_guard,
                                                                                        "outlier_guard_strength": outlier_strength,
                                                                                        "outlier_guard_min_scale_km": outlier_min_scale,
                                                                                        "outlier_guard_mad_scale": outlier_mad,
                                                                                        **metrics,
                                                                                    }
                                                                                )

    results.sort(key=lambda r: math.inf if r["median_km"] is None else r["median_km"])
    payload = {
        "config": args.config,
        "images_dir": str(images_dir),
        "metadata": str(metadata_path),
        "limit": args.limit,
        "results": results,
        "best": results[0] if results else None,
        "top5": results[:5],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
