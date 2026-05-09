"""
Train a lightweight candidate reranker from retrieval candidates and ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from src.core.logic.candidate_rerank import (
    FEATURE_NAMES,
    CandidateRerankModel,
    candidate_feature_matrix,
    candidate_rerank_likelihoods,
    model_to_json,
)
from src.core.logic.config import load_config
from src.core.logic.types import GeoCandidate
from src.tools.run_geo_eval import (
    build_retrieval_provider,
    capture_time_from_record,
    haversine_km,
    load_metadata_records,
    resolve_image_path,
)


def _target_score(dist_km: float, sigma_km: float) -> float:
    sigma = max(0.1, float(sigma_km))
    return math.exp(-0.5 * (float(dist_km) / sigma) ** 2)


def _collect_rows(
    *,
    cfg_path: Path,
    images_dir: Path,
    metadata_path: Path,
    query_metadata_path: Optional[Path],
    limit: int,
    seed: int,
    target_sigma_km: float,
) -> tuple[list[list[float]], list[float], list[float], dict]:
    cfg = load_config(str(cfg_path))
    provider_cfg = replace(
        cfg,
        geolocator=replace(
            cfg.geolocator,
            retrieval_top_k=max(25, cfg.geolocator.retrieval_top_k),
            retrieval_min_keep_topk=max(10, cfg.geolocator.retrieval_min_keep_topk),
        ),
    )
    provider = build_retrieval_provider(provider_cfg)
    if provider is None:
        raise ValueError("config has no retrieval index")

    records = load_metadata_records(metadata_path, query_metadata_path=query_metadata_path, images_dir=images_dir)
    random.Random(seed).shuffle(records)
    if limit > 0:
        records = records[:limit]

    rows: list[list[float]] = []
    targets: list[float] = []
    sample_best_distances: list[float] = []
    null_candidates = 0
    missing_files = 0
    for item in records:
        rel_path = str(item["path"])
        image_path = Path(rel_path) if Path(rel_path).is_absolute() else resolve_image_path(images_dir, rel_path)
        if not image_path.exists():
            missing_files += 1
            continue
        candidates = provider.candidates(str(image_path))
        if not candidates:
            null_candidates += 1
            continue
        _ = capture_time_from_record(item)  # keeps metadata enrichment exercised for future feature extensions
        gt_lat = float(item["latitude"])
        gt_lon = float(item["longitude"])
        dists = [haversine_km(gt_lat, gt_lon, cand.latitude, cand.longitude) for cand in candidates]
        sample_best_distances.append(min(dists))
        rows.extend(candidate_feature_matrix(candidates, FEATURE_NAMES))
        targets.extend(_target_score(dist, target_sigma_km) for dist in dists)

    stats = {
        "metadata": str(metadata_path),
        "images_dir": str(images_dir),
        "records_seen": len(records),
        "missing_files": missing_files,
        "null_candidates": null_candidates,
        "candidate_rows": len(rows),
        "oracle_mean_km": float(sum(sample_best_distances) / len(sample_best_distances))
        if sample_best_distances
        else None,
    }
    return rows, targets, sample_best_distances, stats


def _fit_ridge(rows: list[list[float]], targets: list[float], ridge: float) -> CandidateRerankModel:
    if not rows:
        raise ValueError("no training rows collected")
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    xz = (x - means) / scales
    design = np.concatenate([np.ones((xz.shape[0], 1), dtype=np.float64), xz], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64) * max(0.0, float(ridge))
    reg[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return CandidateRerankModel(
        feature_names=tuple(FEATURE_NAMES),
        weights=tuple(float(v) for v in beta[1:]),
        intercept=float(beta[0]),
        means=tuple(float(v) for v in means),
        scales=tuple(float(v) for v in scales),
        activation="linear_clamp",
        output_floor=0.03,
        output_ceiling=1.0,
    )


def _combined_prediction(candidates: list[GeoCandidate], likes: list[float], weight: float, temperature: float) -> GeoCandidate:
    best = candidates[0]
    best_score = -math.inf
    for cand, like in zip(candidates, likes):
        unit = _to_unit_interval(cand.retrieval_score)
        logit = math.log(max(1e-6, unit) / max(1e-6, 1.0 - unit)) / max(1e-3, temperature)
        score = logit + max(0.0, weight) * math.log(max(1e-6, like))
        if score > best_score:
            best_score = score
            best = cand
    return best


def _evaluate_model(
    *,
    cfg_path: Path,
    images_dir: Path,
    metadata_path: Path,
    query_metadata_path: Optional[Path],
    model: CandidateRerankModel,
    limit: int,
    seed: int,
    weight: float,
    temperature: float,
) -> dict:
    cfg = load_config(str(cfg_path))
    provider = build_retrieval_provider(cfg)
    if provider is None:
        raise ValueError("config has no retrieval index")

    records = load_metadata_records(metadata_path, query_metadata_path=query_metadata_path, images_dir=images_dir)
    random.Random(seed).shuffle(records)
    if limit > 0:
        records = records[:limit]

    base_distances: list[float] = []
    model_distances: list[float] = []
    oracle_distances: list[float] = []
    null_candidates = 0
    for item in records:
        rel_path = str(item["path"])
        image_path = Path(rel_path) if Path(rel_path).is_absolute() else resolve_image_path(images_dir, rel_path)
        if not image_path.exists():
            continue
        candidates = provider.candidates(str(image_path))
        if not candidates:
            null_candidates += 1
            continue
        gt_lat = float(item["latitude"])
        gt_lon = float(item["longitude"])
        likes = candidate_rerank_likelihoods(candidates, model)
        selected = _combined_prediction(candidates, likes, weight=weight, temperature=temperature)
        distances = [haversine_km(gt_lat, gt_lon, cand.latitude, cand.longitude) for cand in candidates]
        base_distances.append(distances[0])
        model_distances.append(haversine_km(gt_lat, gt_lon, selected.latitude, selected.longitude))
        oracle_distances.append(min(distances))
    return {
        "evaluated": len(model_distances),
        "null_candidates": null_candidates,
        "base": _metrics(base_distances),
        "reranked": _metrics(model_distances),
        "oracle": _metrics(oracle_distances),
    }


def _metrics(values: list[float]) -> dict:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {
            "mean_km": None,
            "median_km": None,
            "p90_km": None,
            "within_1km_pct": 0.0,
            "within_2km_pct": 0.0,
            "within_5km_pct": 0.0,
            "within_10km_pct": 0.0,
        }
    n = len(vals)
    return {
        "mean_km": float(sum(vals) / n),
        "median_km": float(vals[n // 2]),
        "p90_km": float(vals[max(0, min(n - 1, int(round(0.9 * (n - 1)))))]),
        "within_1km_pct": _pct(vals, 1.0),
        "within_2km_pct": _pct(vals, 2.0),
        "within_5km_pct": _pct(vals, 5.0),
        "within_10km_pct": _pct(vals, 10.0),
    }


def _pct(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return 100.0 * sum(1 for value in values if value <= threshold) / len(values)


def _to_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    if 0.0 <= value <= 1.0:
        return value
    return 1.0 / (1.0 + math.exp(-value))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight geo candidate reranker.")
    parser.add_argument("--config", default="src/config/paris.json")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--metadata", required=True, help="Training CSV.")
    parser.add_argument("--query-metadata", default="")
    parser.add_argument("--eval-metadata", default="", help="Optional held-out CSV.")
    parser.add_argument("--eval-query-metadata", default="")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--eval-limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-sigma-km", type=float, default=2.0)
    parser.add_argument("--ridge", type=float, default=5.0)
    parser.add_argument("--fusion-weight", type=float, default=1.2)
    parser.add_argument("--temperature", type=float, default=0.28)
    parser.add_argument("--output", default="runs/candidate_reranker.json")
    parser.add_argument("--report-output", default="")
    args = parser.parse_args(argv)

    rows, targets, _oracle, train_stats = _collect_rows(
        cfg_path=Path(args.config),
        images_dir=Path(args.images_dir),
        metadata_path=Path(args.metadata),
        query_metadata_path=Path(args.query_metadata) if args.query_metadata else None,
        limit=args.limit,
        seed=args.seed,
        target_sigma_km=args.target_sigma_km,
    )
    model = _fit_ridge(rows, targets, ridge=args.ridge)
    eval_path = Path(args.eval_metadata) if args.eval_metadata else Path(args.metadata)
    eval_query_path = Path(args.eval_query_metadata) if args.eval_query_metadata else (
        Path(args.query_metadata) if args.query_metadata else None
    )
    report = {
        "train": train_stats,
        "target_sigma_km": args.target_sigma_km,
        "ridge": args.ridge,
        "fusion_weight": args.fusion_weight,
        "temperature": args.temperature,
        "eval": _evaluate_model(
            cfg_path=Path(args.config),
            images_dir=Path(args.images_dir),
            metadata_path=eval_path,
            query_metadata_path=eval_query_path,
            model=model,
            limit=args.eval_limit,
            seed=args.seed,
            weight=args.fusion_weight,
            temperature=args.temperature,
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model_to_json(model, {"training_report": report}), indent=2), encoding="utf-8")
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
