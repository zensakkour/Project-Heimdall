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
    parser.add_argument("--retrieval-norms", default="none,zscore_sigmoid,minmax,rank_exp", help="Comma list.")
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
    retrieval_norms = _parse_str_list(args.retrieval_norms)

    results = []
    for scale in geospot_scales:
        for temp in retrieval_temps:
            for topk in retrieval_topk:
                for ftopk in fusion_topk:
                    for norm in retrieval_norms:
                        tuned_geo = replace(
                            cfg.geolocator,
                            geospot_score_scale=scale,
                            retrieval_top_k=topk,
                        )
                        tuned_fusion = replace(
                            cfg.fusion,
                            retrieval_temperature=temp,
                            retrieval_score_norm=norm,
                            top_k=ftopk,
                        )
                        tuned_cfg = replace(cfg, geolocator=tuned_geo, fusion=tuned_fusion)
                        metrics = _evaluate(tuned_cfg, images_dir, records)
                        results.append(
                            {
                                "geospot_score_scale": scale,
                                "retrieval_temperature": temp,
                                "retrieval_top_k": topk,
                                "fusion_top_k": ftopk,
                                "retrieval_score_norm": norm,
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
