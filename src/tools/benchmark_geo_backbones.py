"""
Benchmark geo retrieval backbones by rebuilding indices per model and evaluating top-1 geo error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from src.core.geo import GeoRetrievalProvider
from src.tools.build_geo_index import build_index
from src.tools.run_geo_eval import haversine_km, resolve_image_path
from src.tools import train_retrieval_projection as projection_tools


MODEL_PRESETS: dict[str, list[str]] = {
    # Backward-compatible default pair.
    "legacy_clip_siglip": [
        "openai/clip-vit-large-patch14",
        "google/siglip-base-patch16-224",
    ],
    # Fast preset intended for single-GPU desktop iteration.
    "aerial_rtx5060_fast": [
        "openai/clip-vit-large-patch14",
        "google/siglip-base-patch16-224",
    ],
    # Higher-capacity aerial retrieval candidates (slower/heavier).
    "aerial_rtx5060_precise": [
        "google/siglip-so400m-patch14-384",
        "google/siglip-base-patch16-224",
        "openai/clip-vit-large-patch14",
    ],
    # Broader exploratory set for research sweeps.
    "aerial_research": [
        "google/siglip-so400m-patch14-384",
        "google/siglip-base-patch16-224",
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-large-patch14-336",
    ],
}

RANK_OBJECTIVES = {
    "mean_km",
    "median_km",
    "p90_km",
    "within_1km_pct",
    "within_2km_pct",
    "within_5km_pct",
    "within_10km_pct",
    "within_50km_pct",
}

HIGHER_IS_BETTER = {
    "within_1km_pct",
    "within_2km_pct",
    "within_5km_pct",
    "within_10km_pct",
    "within_50km_pct",
}


def _parse_model_ids(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _resolve_model_ids(raw_model_ids: str, model_preset: str) -> List[str]:
    explicit = _parse_model_ids(raw_model_ids)
    if explicit:
        return explicit
    chosen = str(model_preset).strip()
    if chosen not in MODEL_PRESETS:
        raise ValueError(f"unknown_model_preset:{chosen}")
    return list(MODEL_PRESETS[chosen])


def _parse_float_list(raw: str) -> List[float]:
    out = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        out.append(float(text))
    return out


def _parse_mode_list(raw: str) -> List[str]:
    allowed = {"rgb", "gray", "equalize", "edge"}
    out: List[str] = []
    seen = set()
    for item in str(raw).split(","):
        mode = item.strip().lower()
        if mode not in allowed or mode in seen:
            continue
        seen.add(mode)
        out.append(mode)
    return out or ["rgb"]


def _slug(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return clean or "model"


def _load_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = row.get("path") or row.get("image")
            lat = row.get("latitude") or row.get("lat")
            lon = row.get("longitude") or row.get("lon")
            if not rel or lat is None or lon is None:
                continue
            try:
                rows.append({"path": str(rel), "latitude": float(lat), "longitude": float(lon)})
            except Exception:
                continue
    return rows


def _sample_rows(rows: List[dict], limit: int, seed: int) -> List[dict]:
    data = list(rows)
    random.Random(seed).shuffle(data)
    if limit > 0:
        data = data[:limit]
    return data


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return float(sorted_vals[idx])


def _pct_within(vals: List[float], km: float) -> float:
    if not vals:
        return 0.0
    count = sum(1 for value in vals if value <= km)
    return 100.0 * float(count) / float(len(vals))


def _safe_float(value: object) -> Optional[float]:
    try:
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return num


def _error_text(exc: Exception) -> str:
    text = str(exc).strip().replace("\n", " ")
    if not text:
        text = exc.__class__.__name__
    if len(text) > 240:
        text = text[:237] + "..."
    return text


def _sort_rows_for_objective(rows: Sequence[dict], objective: str) -> List[dict]:
    mode = str(objective).strip()
    if mode not in RANK_OBJECTIVES:
        mode = "median_km"

    valid_rows = [row for row in rows if _safe_float(row.get(mode)) is not None]
    if not valid_rows:
        return []

    if mode in HIGHER_IS_BETTER:
        return sorted(
            valid_rows,
            key=lambda row: (
                -float(_safe_float(row.get(mode)) or 0.0),
                float(_safe_float(row.get("median_km")) or float("inf")),
                str(row.get("model_id") or ""),
            ),
        )

    return sorted(
        valid_rows,
        key=lambda row: (
            float(_safe_float(row.get(mode)) or float("inf")),
            -float(_safe_float(row.get("within_2km_pct")) or 0.0),
            str(row.get("model_id") or ""),
        ),
    )


def _resolve_projection_device(device: str) -> str:
    raw = str(device).strip().lower()
    if raw == "auto":
        if getattr(projection_tools, "torch", None) is not None and projection_tools.torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if raw.startswith("cuda") and getattr(projection_tools, "torch", None) is not None:
        if not projection_tools.torch.cuda.is_available():
            return "cpu"
    return raw or "cpu"


def _collect_reference_paths(triplets: Sequence[dict]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for row in triplets:
        for item in row.get("positives", []):
            path = str((item or {}).get("path") or "").strip()
            if not path:
                continue
            key = projection_tools._as_posix(path)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        for item in row.get("hard_negatives", []):
            path = str((item or {}).get("path") or "").strip()
            if not path:
                continue
            key = projection_tools._as_posix(path)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


def _filter_rows_by_paths(rows: Sequence[dict], include_paths: Sequence[str]) -> List[dict]:
    wanted = {projection_tools._as_posix(path) for path in include_paths if str(path).strip()}
    if not wanted:
        return []
    filtered: List[dict] = []
    for row in rows:
        key = projection_tools._as_posix(str(row.get("path") or row.get("image") or "").strip())
        if key and key in wanted:
            filtered.append(dict(row))
    return filtered


def _fit_projection_for_index(
    *,
    model_id: str,
    raw_index_path: Path,
    triplet_path: Path,
    projection_images_dir: Path,
    projection_path: Path,
    projection_report_path: Path,
    max_triplets: int,
    output_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    margin: float,
    temperature: float,
    ce_weight: float,
    orth_weight: float,
    sample_weight_mode: str,
    sample_weight_power: float,
    sample_weight_max: float,
    seed: int,
    device: str,
) -> dict:
    resolved_device = _resolve_projection_device(device)
    triplets = projection_tools._load_triplets(triplet_path, max_triplets=max(0, int(max_triplets)))
    if not triplets:
        raise ValueError("projection_triplets_empty_or_invalid")
    requested_paths = projection_tools._collect_requested_paths(triplets)
    by_exact, by_name, _ = projection_tools._load_index_embeddings(raw_index_path)
    embed_written, embed_missing = projection_tools._embed_missing(
        requested_paths=requested_paths,
        by_exact=by_exact,
        by_name=by_name,
        model_id=model_id,
        images_dir=projection_images_dir,
        device=resolved_device,
    )
    missing_summary = projection_tools._summarize_missing_paths_by_role(
        triplets,
        by_exact=by_exact,
        by_name=by_name,
    )
    embeddings, rows, dataset_stats = projection_tools._build_training_records(
        triplets,
        by_exact=by_exact,
        by_name=by_name,
        sample_weight_mode=sample_weight_mode,
        sample_weight_power=float(sample_weight_power),
        sample_weight_max=float(sample_weight_max),
    )
    if not rows:
        raise ValueError(
            projection_tools._format_no_valid_training_records_error(
                triplets_loaded=len(triplets),
                requested_unique_paths=len(requested_paths),
                dataset_stats=dataset_stats,
                missing_summary=missing_summary,
                embedding_index=str(raw_index_path),
                images_dir=str(projection_images_dir),
            )
        )

    weight, bias, train_report = projection_tools.train_projection(
        embeddings=embeddings,
        rows=rows,
        output_dim=int(output_dim),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        margin=float(margin),
        temperature=float(temperature),
        ce_weight=float(ce_weight),
        orth_weight=float(orth_weight),
        sample_weight_mode=str(sample_weight_mode),
        sample_weight_power=float(sample_weight_power),
        sample_weight_max=float(sample_weight_max),
        seed=int(seed),
        device=resolved_device,
    )

    projection_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        projection_path,
        matrix=np.asarray(weight, dtype=np.float32),
        bias=np.asarray(bias, dtype=np.float32),
        model_id=np.asarray(str(model_id), dtype=np.str_),
        source_index=np.asarray(str(raw_index_path).replace("\\", "/"), dtype=np.str_),
        triplets_path=np.asarray(str(triplet_path).replace("\\", "/"), dtype=np.str_),
    )
    projection_payload = {
        "model_id": str(model_id),
        "raw_index_path": str(raw_index_path),
        "projection_path": str(projection_path),
        "triplets_path": str(triplet_path),
        "triplets_loaded": len(triplets),
        "triplets_used": len(rows),
        "requested_unique_paths": len(requested_paths),
        "embed_written": int(embed_written),
        "embed_missing": int(embed_missing),
        "missing_summary": missing_summary,
        "dataset_stats": dataset_stats,
        "train_report": train_report,
    }
    projection_report_path.parent.mkdir(parents=True, exist_ok=True)
    projection_report_path.write_text(json.dumps(projection_payload, indent=2), encoding="utf-8")
    return projection_payload


def _evaluate_top1(
    provider: GeoRetrievalProvider,
    rows: List[dict],
    images_dir: Path,
) -> dict:
    distances: List[float] = []
    missing = 0
    null_pred = 0
    score_values: List[float] = []
    errors: dict[str, int] = {}

    for row in rows:
        rel = str(row["path"])
        image_path = Path(rel) if Path(rel).is_absolute() else resolve_image_path(images_dir, rel)
        if not image_path.exists():
            missing += 1
            continue
        candidates = provider.candidates(str(image_path))
        if not candidates:
            null_pred += 1
            key = provider.last_error or "no_candidates"
            errors[key] = int(errors.get(key, 0)) + 1
            continue
        top = candidates[0]
        dist = haversine_km(float(row["latitude"]), float(row["longitude"]), top.latitude, top.longitude)
        distances.append(float(dist))
        score_values.append(float(top.retrieval_score))

    distances.sort()
    evaluated = len(distances)
    return {
        "evaluated": evaluated,
        "missing_files": missing,
        "null_predictions": null_pred,
        "provider_errors": errors,
        "retrieval_score_mean": (sum(score_values) / len(score_values)) if score_values else None,
        "retrieval_score_min": min(score_values) if score_values else None,
        "retrieval_score_max": max(score_values) if score_values else None,
        "mean_km": (sum(distances) / evaluated) if evaluated else None,
        "median_km": float(distances[evaluated // 2]) if evaluated else None,
        "p90_km": _percentile(distances, 90),
        "within_1km_pct": _pct_within(distances, 1.0),
        "within_2km_pct": _pct_within(distances, 2.0),
        "within_5km_pct": _pct_within(distances, 5.0),
        "within_10km_pct": _pct_within(distances, 10.0),
        "within_50km_pct": _pct_within(distances, 50.0),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark geo retrieval backbones with per-model index rebuild.")
    parser.add_argument("--train-images-dir", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--eval-images-dir", required=True)
    parser.add_argument("--eval-metadata", required=True)
    parser.add_argument(
        "--model-ids",
        default="",
        help="Comma-separated HF model ids. When empty, --model-preset is used.",
    )
    parser.add_argument(
        "--model-preset",
        default="aerial_rtx5060_fast",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Curated model set for aerial retrieval benchmarking.",
    )
    parser.add_argument(
        "--rank-objective",
        default="median_km",
        choices=sorted(RANK_OBJECTIVES),
        help="Metric used to choose best_model in output.",
    )
    parser.add_argument("--train-limit", type=int, default=600)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-top-k", type=int, default=50)
    parser.add_argument("--retrieval-min-score", type=float, default=0.1)
    parser.add_argument("--retrieval-min-keep-topk", type=int, default=2)
    parser.add_argument("--query-tta-degrees", default="0,90,180,270")
    parser.add_argument("--query-tta-modes", default="rgb")
    parser.add_argument("--query-tta-scales", default="1.0")
    parser.add_argument("--query-tta-auto-modality", action="store_true")
    parser.add_argument("--query-tta-reduce", default="max")
    parser.add_argument("--query-expansion-top-n", type=int, default=0)
    parser.add_argument("--query-expansion-beta", type=float, default=0.0)
    parser.add_argument("--query-expansion-alpha", type=float, default=0.5)
    parser.add_argument("--tta-agreement-top-n", type=int, default=0)
    parser.add_argument("--tta-agreement-weight", type=float, default=0.0)
    parser.add_argument("--local-match-top-n", type=int, default=0)
    parser.add_argument("--local-match-weight", type=float, default=0.0)
    parser.add_argument("--local-match-ratio", type=float, default=0.8)
    parser.add_argument("--local-match-max-features", type=int, default=1200)
    parser.add_argument("--graph-rerank-top-n", type=int, default=0)
    parser.add_argument("--graph-rerank-sigma-km", type=float, default=3.0)
    parser.add_argument("--graph-rerank-score-alpha", type=float, default=0.4)
    parser.add_argument("--graph-rerank-support-beta", type=float, default=1.0)
    parser.add_argument("--graph-rerank-center-radius-km", type=float, default=0.0)
    parser.add_argument("--kde-refine-top-n", type=int, default=0)
    parser.add_argument("--kde-refine-sigma-km", type=float, default=2.0)
    parser.add_argument("--kde-refine-score-power", type=float, default=1.0)
    parser.add_argument("--kde-refine-margin-threshold", type=float, default=0.0)
    parser.add_argument("--kde-refine-switch-radius-km", type=float, default=0.0)
    parser.add_argument("--kde-refine-max-iters", type=int, default=8)
    parser.add_argument("--kde-refine-adaptive-mass", type=float, default=0.0)
    parser.add_argument("--projection-triplets", default="", help="Optional hard-negative triplet JSONL for per-model projection training.")
    parser.add_argument(
        "--projection-images-dir",
        default="",
        help="Optional image root for triplet query embedding backfill. Defaults to --eval-images-dir.",
    )
    parser.add_argument(
        "--projection-reference-images-dir",
        default="",
        help="Optional image root for projection reference embeddings. Defaults to --train-images-dir.",
    )
    parser.add_argument(
        "--projection-reference-metadata",
        default="",
        help="Optional metadata for projection reference embeddings. Defaults to --train-metadata.",
    )
    parser.add_argument("--projection-max-triplets", type=int, default=0)
    parser.add_argument("--projection-output-dim", type=int, default=0)
    parser.add_argument("--projection-epochs", type=int, default=8)
    parser.add_argument("--projection-batch-size", type=int, default=16)
    parser.add_argument("--projection-learning-rate", type=float, default=3e-4)
    parser.add_argument("--projection-weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-margin", type=float, default=0.08)
    parser.add_argument("--projection-temperature", type=float, default=0.07)
    parser.add_argument("--projection-ce-weight", type=float, default=0.3)
    parser.add_argument("--projection-orth-weight", type=float, default=0.002)
    parser.add_argument(
        "--projection-sample-weight-mode",
        default="triplet_weight",
        choices=["auto", "uniform", "triplet_weight"],
    )
    parser.add_argument("--projection-sample-weight-power", type=float, default=1.0)
    parser.add_argument("--projection-sample-weight-max", type=float, default=3.0)
    parser.add_argument("--projection-device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--output", default="runs/backbone_bench/backbone_benchmark.json")
    parser.add_argument("--reuse-indices", action="store_true")
    args = parser.parse_args(argv)

    train_images_dir = Path(args.train_images_dir)
    train_meta_path = Path(args.train_metadata)
    eval_images_dir = Path(args.eval_images_dir)
    eval_meta_path = Path(args.eval_metadata)
    if not train_meta_path.exists():
        raise FileNotFoundError(f"train_metadata_not_found:{train_meta_path}")
    if not eval_meta_path.exists():
        raise FileNotFoundError(f"eval_metadata_not_found:{eval_meta_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_dir = output_path.parent / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    projection_dir = output_path.parent / "projections"
    projection_requested = bool(str(args.projection_triplets).strip())
    projection_triplet_path = Path(str(args.projection_triplets).strip()) if projection_requested else None
    if projection_requested and (projection_triplet_path is None or not projection_triplet_path.exists()):
        raise FileNotFoundError(f"projection_triplets_not_found:{projection_triplet_path}")
    projection_images_dir = Path(str(args.projection_images_dir).strip()) if str(args.projection_images_dir).strip() else eval_images_dir
    projection_reference_images_dir = (
        Path(str(args.projection_reference_images_dir).strip())
        if str(args.projection_reference_images_dir).strip()
        else train_images_dir
    )
    projection_reference_meta_path = (
        Path(str(args.projection_reference_metadata).strip())
        if str(args.projection_reference_metadata).strip()
        else train_meta_path
    )
    if projection_requested and not projection_reference_meta_path.exists():
        raise FileNotFoundError(f"projection_reference_metadata_not_found:{projection_reference_meta_path}")

    model_ids = _resolve_model_ids(args.model_ids, args.model_preset)
    if not model_ids:
        raise ValueError("no_model_ids")
    tta_degrees = _parse_float_list(args.query_tta_degrees)
    if not tta_degrees:
        tta_degrees = [0.0]
    tta_modes = _parse_mode_list(args.query_tta_modes)
    tta_scales = _parse_float_list(args.query_tta_scales)
    if not tta_scales:
        tta_scales = [1.0]

    train_rows_all = _load_rows(train_meta_path)
    eval_rows_all = _load_rows(eval_meta_path)
    train_rows = _sample_rows(train_rows_all, int(args.train_limit), int(args.seed))
    eval_rows = _sample_rows(eval_rows_all, int(args.eval_limit), int(args.seed))
    projection_triplets = (
        projection_tools._load_triplets(projection_triplet_path, max_triplets=max(0, int(args.projection_max_triplets)))
        if projection_requested and projection_triplet_path is not None
        else []
    )
    projection_reference_rows_all = _load_rows(projection_reference_meta_path) if projection_requested else []
    projection_reference_rows = (
        _filter_rows_by_paths(projection_reference_rows_all, _collect_reference_paths(projection_triplets))
        if projection_requested
        else []
    )

    results = []
    for model_id in model_ids:
        slug = _slug(model_id)
        raw_index_path = index_dir / f"{slug}.npz"
        projected_index_path = index_dir / f"{slug}_projected.npz"
        projection_support_index_path = index_dir / f"{slug}_projection_support.npz"
        projection_path = projection_dir / f"{slug}.projection.npz"
        projection_report_path = projection_dir / f"{slug}.projection.report.json"
        index_path = raw_index_path
        query_projection_path: Optional[str] = None
        build_sec = 0.0
        projection_train_sec = 0.0
        projection_index_sec = 0.0
        projection_support_build_sec = 0.0
        projection_summary = None

        try:
            build_start = time.perf_counter()
            if not args.reuse_indices or not raw_index_path.exists():
                count = build_index(
                    images=[],
                    meta=train_rows,
                    output=raw_index_path,
                    model_id=model_id,
                    root_dir=train_images_dir.parent,
                    images_dir=train_images_dir,
                )
                if count <= 0:
                    results.append(
                        {
                            "model_id": model_id,
                            "status": "failed",
                            "details": "no_embeddings_written",
                            "index_path": str(raw_index_path),
                        }
                    )
                    continue
            build_sec = float(time.perf_counter() - build_start)
        except Exception as exc:
            results.append(
                {
                    "model_id": model_id,
                    "status": "failed",
                    "details": f"build_failed:{_error_text(exc)}",
                    "index_path": str(raw_index_path),
                    "build_seconds": build_sec,
                }
            )
            continue

        if projection_requested and projection_triplet_path is not None:
            try:
                if not projection_triplets:
                    raise ValueError("projection_triplets_empty_or_invalid")
                if not projection_reference_rows:
                    raise ValueError("projection_reference_rows_empty_for_triplets")
                projection_support_build_start = time.perf_counter()
                if not args.reuse_indices or not projection_support_index_path.exists():
                    projection_support_count = build_index(
                        images=[],
                        meta=projection_reference_rows,
                        output=projection_support_index_path,
                        model_id=model_id,
                        root_dir=projection_reference_images_dir.parent,
                        images_dir=projection_reference_images_dir,
                    )
                    if projection_support_count <= 0:
                        raise ValueError("projection_support_index_no_embeddings_written")
                projection_support_build_sec = float(time.perf_counter() - projection_support_build_start)
                projection_train_start = time.perf_counter()
                if not args.reuse_indices or not projection_path.exists() or not projection_report_path.exists():
                    projection_summary = _fit_projection_for_index(
                        model_id=model_id,
                        raw_index_path=projection_support_index_path,
                        triplet_path=projection_triplet_path,
                        projection_images_dir=projection_images_dir,
                        projection_path=projection_path,
                        projection_report_path=projection_report_path,
                        max_triplets=int(args.projection_max_triplets),
                        output_dim=int(args.projection_output_dim),
                        epochs=int(args.projection_epochs),
                        batch_size=int(args.projection_batch_size),
                        learning_rate=float(args.projection_learning_rate),
                        weight_decay=float(args.projection_weight_decay),
                        margin=float(args.projection_margin),
                        temperature=float(args.projection_temperature),
                        ce_weight=float(args.projection_ce_weight),
                        orth_weight=float(args.projection_orth_weight),
                        sample_weight_mode=str(args.projection_sample_weight_mode),
                        sample_weight_power=float(args.projection_sample_weight_power),
                        sample_weight_max=float(args.projection_sample_weight_max),
                        seed=int(args.seed),
                        device=str(args.projection_device),
                    )
                projection_train_sec = float(time.perf_counter() - projection_train_start)

                projection_index_start = time.perf_counter()
                if not args.reuse_indices or not projected_index_path.exists():
                    projected_count = build_index(
                        images=[],
                        meta=train_rows,
                        output=projected_index_path,
                        model_id=model_id,
                        root_dir=train_images_dir.parent,
                        images_dir=train_images_dir,
                        projection_path=str(projection_path),
                    )
                    if projected_count <= 0:
                        raise ValueError("projected_index_no_embeddings_written")
                projection_index_sec = float(time.perf_counter() - projection_index_start)
                index_path = projected_index_path
                query_projection_path = str(projection_path)
            except Exception as exc:
                results.append(
                    {
                        "model_id": model_id,
                        "status": "failed",
                        "details": f"projection_failed:{_error_text(exc)}",
                        "index_path": str(raw_index_path),
                        "raw_index_path": str(raw_index_path),
                        "projection_support_index_path": str(projection_support_index_path),
                        "projection_path": str(projection_path),
                        "projection_report_path": str(projection_report_path),
                        "build_seconds": build_sec,
                        "projection_support_build_seconds": projection_support_build_sec,
                        "projection_train_seconds": projection_train_sec,
                        "projection_index_seconds": projection_index_sec,
                    }
                )
                continue

        try:
            eval_start = time.perf_counter()
            provider = GeoRetrievalProvider(
                index_path=str(index_path),
                model_id=model_id,
                projection_path=query_projection_path,
                top_k=int(args.retrieval_top_k),
                min_score=float(args.retrieval_min_score),
                min_keep_topk=int(args.retrieval_min_keep_topk),
                query_tta_degrees=tta_degrees,
                query_tta_modes=tta_modes,
                query_tta_scales=tta_scales,
                query_tta_auto_modality=bool(args.query_tta_auto_modality),
                query_tta_reduce=str(args.query_tta_reduce),
                query_expansion_top_n=int(args.query_expansion_top_n),
                query_expansion_beta=float(args.query_expansion_beta),
                query_expansion_alpha=float(args.query_expansion_alpha),
                tta_agreement_top_n=int(args.tta_agreement_top_n),
                tta_agreement_weight=float(args.tta_agreement_weight),
                local_match_top_n=int(args.local_match_top_n),
                local_match_weight=float(args.local_match_weight),
                local_match_ratio=float(args.local_match_ratio),
                local_match_max_features=int(args.local_match_max_features),
                graph_rerank_top_n=int(args.graph_rerank_top_n),
                graph_rerank_sigma_km=float(args.graph_rerank_sigma_km),
                graph_rerank_score_alpha=float(args.graph_rerank_score_alpha),
                graph_rerank_support_beta=float(args.graph_rerank_support_beta),
                graph_rerank_center_radius_km=float(args.graph_rerank_center_radius_km),
                kde_refine_top_n=int(args.kde_refine_top_n),
                kde_refine_sigma_km=float(args.kde_refine_sigma_km),
                kde_refine_score_power=float(args.kde_refine_score_power),
                kde_refine_margin_threshold=float(args.kde_refine_margin_threshold),
                kde_refine_switch_radius_km=float(args.kde_refine_switch_radius_km),
                kde_refine_max_iters=int(args.kde_refine_max_iters),
                kde_refine_adaptive_mass=float(args.kde_refine_adaptive_mass),
            )
            metrics = _evaluate_top1(provider=provider, rows=eval_rows, images_dir=eval_images_dir)
            eval_sec = float(time.perf_counter() - eval_start)
            results.append(
                {
                    "model_id": model_id,
                    "status": "ok",
                    "index_path": str(index_path),
                    "raw_index_path": str(raw_index_path),
                    "projection_support_index_path": (str(projection_support_index_path) if query_projection_path else None),
                    "projection_path": query_projection_path,
                    "projection_report_path": (str(projection_report_path) if query_projection_path else None),
                    "projection_enabled": bool(query_projection_path),
                    "build_seconds": build_sec,
                    "projection_support_build_seconds": projection_support_build_sec,
                    "projection_train_seconds": projection_train_sec,
                    "projection_index_seconds": projection_index_sec,
                    "eval_seconds": eval_sec,
                    "projection_triplets_used": (
                        int(projection_summary.get("triplets_used")) if isinstance(projection_summary, dict) else None
                    ),
                    **metrics,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "model_id": model_id,
                    "status": "failed",
                    "details": f"eval_failed:{_error_text(exc)}",
                    "index_path": str(index_path),
                    "raw_index_path": str(raw_index_path),
                    "projection_support_index_path": (str(projection_support_index_path) if query_projection_path else None),
                    "projection_path": query_projection_path,
                    "build_seconds": build_sec,
                    "projection_support_build_seconds": projection_support_build_sec,
                    "projection_train_seconds": projection_train_sec,
                    "projection_index_seconds": projection_index_sec,
                }
            )

    ok_rows = [row for row in results if row.get("status") == "ok"]
    ranked_by_objective = _sort_rows_for_objective(ok_rows, str(args.rank_objective))
    ranked_by_median = _sort_rows_for_objective(ok_rows, "median_km")

    payload = {
        "train_images_dir": str(train_images_dir),
        "train_metadata": str(train_meta_path),
        "eval_images_dir": str(eval_images_dir),
        "eval_metadata": str(eval_meta_path),
        "train_limit": int(args.train_limit),
        "eval_limit": int(args.eval_limit),
        "seed": int(args.seed),
        "model_preset": str(args.model_preset),
        "model_ids": list(model_ids),
        "rank_objective": str(args.rank_objective),
        "projection_triplets": (str(projection_triplet_path) if projection_triplet_path is not None else None),
        "projection_images_dir": (str(projection_images_dir) if projection_requested else None),
        "projection_reference_images_dir": (str(projection_reference_images_dir) if projection_requested else None),
        "projection_reference_metadata": (str(projection_reference_meta_path) if projection_requested else None),
        "projection_max_triplets": int(args.projection_max_triplets),
        "projection_output_dim": int(args.projection_output_dim),
        "projection_epochs": int(args.projection_epochs),
        "projection_batch_size": int(args.projection_batch_size),
        "projection_learning_rate": float(args.projection_learning_rate),
        "projection_weight_decay": float(args.projection_weight_decay),
        "projection_margin": float(args.projection_margin),
        "projection_temperature": float(args.projection_temperature),
        "projection_ce_weight": float(args.projection_ce_weight),
        "projection_orth_weight": float(args.projection_orth_weight),
        "projection_sample_weight_mode": str(args.projection_sample_weight_mode),
        "projection_sample_weight_power": float(args.projection_sample_weight_power),
        "projection_sample_weight_max": float(args.projection_sample_weight_max),
        "projection_device": str(args.projection_device),
        "query_tta_degrees": tta_degrees,
        "query_tta_modes": tta_modes,
        "query_tta_scales": tta_scales,
        "query_tta_auto_modality": bool(args.query_tta_auto_modality),
        "query_expansion_top_n": int(args.query_expansion_top_n),
        "query_expansion_beta": float(args.query_expansion_beta),
        "query_expansion_alpha": float(args.query_expansion_alpha),
        "tta_agreement_top_n": int(args.tta_agreement_top_n),
        "tta_agreement_weight": float(args.tta_agreement_weight),
        "local_match_top_n": int(args.local_match_top_n),
        "local_match_weight": float(args.local_match_weight),
        "local_match_ratio": float(args.local_match_ratio),
        "local_match_max_features": int(args.local_match_max_features),
        "graph_rerank_top_n": int(args.graph_rerank_top_n),
        "graph_rerank_sigma_km": float(args.graph_rerank_sigma_km),
        "graph_rerank_score_alpha": float(args.graph_rerank_score_alpha),
        "graph_rerank_support_beta": float(args.graph_rerank_support_beta),
        "graph_rerank_center_radius_km": float(args.graph_rerank_center_radius_km),
        "kde_refine_top_n": int(args.kde_refine_top_n),
        "kde_refine_sigma_km": float(args.kde_refine_sigma_km),
        "kde_refine_score_power": float(args.kde_refine_score_power),
        "kde_refine_margin_threshold": float(args.kde_refine_margin_threshold),
        "kde_refine_switch_radius_km": float(args.kde_refine_switch_radius_km),
        "kde_refine_max_iters": int(args.kde_refine_max_iters),
        "kde_refine_adaptive_mass": float(args.kde_refine_adaptive_mass),
        "models": results,
        "best_model": ranked_by_objective[0]["model_id"] if ranked_by_objective else None,
        "best_model_by_median_km": ranked_by_median[0]["model_id"] if ranked_by_median else None,
        "ranked_by_objective": [row.get("model_id") for row in ranked_by_objective],
        "ranked_by_median_km": [row.get("model_id") for row in ranked_by_median],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0 if ranked_by_objective else 1


if __name__ == "__main__":
    raise SystemExit(main())
