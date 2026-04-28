"""
Run a retrieval fine-tuning loop against the fixed Paris benchmark split.

Each round:
1) evaluate the current retrieval source on the fixed eval subset,
2) mine a larger hard-negative triplet set from the resulting failures,
3) fine-tune the encoder directly on images,
4) rebuild the train index (and optional DBA companion),
5) evaluate the new model on the same fixed eval subset.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.core.geo import GeoRetrievalProvider
from src.core.logic.config import load_config
from src.tools import augment_geo_index_embeddings as dba_tools
from src.tools import benchmark_geo_backbones as bench
from src.tools import mine_hard_negative_triplets as miner
from src.tools import run_geo_eval
from src.tools import train_retrieval_encoder as encoder_tools
from src.tools.build_geo_index import build_index


HIGHER_IS_BETTER = {"within_1km_pct", "within_2km_pct", "within_5km_pct", "within_10km_pct", "within_50km_pct"}


def _slug(text: str) -> str:
    return bench._slug(text)


def _safe_float(value: object) -> Optional[float]:
    try:
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return num


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pad_list(values: Sequence[object], size: int, fill_value: object) -> List[object]:
    out = list(values[: max(0, int(size))])
    while len(out) < max(0, int(size)):
        out.append(fill_value)
    return out


def _positive_weight(value: object, default: float) -> float:
    num = _safe_float(value)
    if num is None or num <= 0.0:
        return float(default)
    return float(num)


def _sort_metric_key(metrics: dict, objective: str) -> tuple[float, float]:
    obj = str(objective).strip()
    primary = float(_safe_float(metrics.get(obj)) or 0.0)
    secondary = float(_safe_float(metrics.get("mean_km")) or float("inf"))
    if obj in HIGHER_IS_BETTER:
        return (primary, -secondary)
    return (-primary, -float(_safe_float(metrics.get("within_2km_pct")) or 0.0))


def _is_better(candidate: dict, incumbent: Optional[dict], objective: str) -> bool:
    if incumbent is None:
        return True
    obj = str(objective).strip()
    cand = _safe_float(candidate.get(obj))
    prev = _safe_float(incumbent.get(obj))
    if cand is None:
        return False
    if prev is None:
        return True
    if obj in HIGHER_IS_BETTER:
        if cand != prev:
            return cand > prev
        cand_mean = float(_safe_float(candidate.get("mean_km")) or float("inf"))
        prev_mean = float(_safe_float(incumbent.get("mean_km")) or float("inf"))
        return cand_mean < prev_mean
    if cand != prev:
        return cand < prev
    cand_w2 = float(_safe_float(candidate.get("within_2km_pct")) or 0.0)
    prev_w2 = float(_safe_float(incumbent.get("within_2km_pct")) or 0.0)
    return cand_w2 > prev_w2


def _build_provider_for_model(
    *,
    model_id: str,
    index_path: Path,
    dba_index_path: Optional[Path],
    top_k: int,
    min_score: float,
    min_keep_topk: int,
    dba_weight: float,
) -> GeoRetrievalProvider:
    index_paths: List[str] = []
    index_weights: List[float] = []
    if dba_index_path is not None and dba_index_path.exists():
        index_paths.append(str(dba_index_path))
        index_weights = [1.0, float(max(1e-6, dba_weight))]
    return GeoRetrievalProvider(
        index_path=str(index_path),
        index_paths=index_paths,
        index_weights=index_weights,
        model_id=str(model_id),
        top_k=int(top_k),
        min_score=float(min_score),
        min_keep_topk=int(min_keep_topk),
        source_fusion_mode="rrf" if index_paths else "weighted_score",
        query_tta_degrees=[0.0, 90.0, 180.0, 270.0],
        query_tta_modes=["rgb"],
        query_tta_scales=[1.0],
        query_tta_reduce="max",
        consensus_top_n=20,
        consensus_radius_km=3.0,
        consensus_score_power=1.0,
    )


def _sample_eval_rows(eval_metadata: Path, *, limit: int, seed: int) -> List[dict]:
    rows = bench._load_rows(eval_metadata)
    return bench._sample_rows(rows, limit=int(limit), seed=int(seed))


def _evaluate_with_diagnostics(
    *,
    rows: Sequence[dict],
    images_dir: Path,
    provider: GeoRetrievalProvider,
    diag_samples: int,
    config_label: str,
) -> dict:
    distances: List[float] = []
    missing = 0
    null_predictions = 0
    retrieval_scores: List[float] = []
    diagnostics: List[dict] = []
    for row in rows:
        rel = str(row["path"])
        image_path = Path(rel) if Path(rel).is_absolute() else run_geo_eval.resolve_image_path(images_dir, rel)
        if not image_path.exists():
            missing += 1
            continue
        pred, score, provider_error = run_geo_eval.predict_latlon_retrieval(str(image_path), provider)
        if score is not None:
            retrieval_scores.append(float(score))
        if pred is None:
            null_predictions += 1
            continue
        gt_lat = float(row["latitude"])
        gt_lon = float(row["longitude"])
        dist = run_geo_eval.haversine_km(gt_lat, gt_lon, pred[0], pred[1])
        distances.append(float(dist))
        if len(diagnostics) < max(0, int(diag_samples)):
            diagnostics.append(
                {
                    "image": str(image_path),
                    "gt_lat": gt_lat,
                    "gt_lon": gt_lon,
                    "pred_lat": float(pred[0]),
                    "pred_lon": float(pred[1]),
                    "dist_km": float(dist),
                    "retrieval_score": score,
                    "provider_error": provider_error,
                }
            )
    distances.sort()
    evaluated = len(distances)
    metrics = {
        "config": config_label,
        "total": len(rows),
        "evaluated": evaluated,
        "missing_files": int(missing),
        "null_predictions": int(null_predictions),
        "retrieval_score_mean": (sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else None,
        "retrieval_score_min": min(retrieval_scores) if retrieval_scores else None,
        "retrieval_score_max": max(retrieval_scores) if retrieval_scores else None,
        "mean_km": (sum(distances) / evaluated) if evaluated else None,
        "median_km": bench._percentile(distances, 50.0),
        "p90_km": bench._percentile(distances, 90.0),
        "within_1km_pct": bench._pct_within(distances, 1.0),
        "within_2km_pct": bench._pct_within(distances, 2.0),
        "within_5km_pct": bench._pct_within(distances, 5.0),
        "within_10km_pct": bench._pct_within(distances, 10.0),
        "within_50km_pct": bench._pct_within(distances, 50.0),
        "samples": diagnostics,
    }
    return metrics


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_aux_fused_config(
    *,
    bootstrap_config_path: Path,
    output_config_path: Path,
    tuned_model_ref: str,
    tuned_index_path: Path,
    tuned_dba_index_path: Optional[Path],
    aux_index_weight: float,
    aux_dba_weight: float,
) -> dict:
    payload = _load_json(bootstrap_config_path)
    geo = payload.setdefault("geolocator", {})
    primary_index = str(geo.get("retrieval_index_path") or "").strip()
    if not primary_index:
        raise ValueError("bootstrap_config_missing_primary_index")

    base_model_id = str(geo.get("retrieval_model_id") or "").strip() or "openai/clip-vit-large-patch14"
    base_projection_path = geo.get("retrieval_projection_path")

    existing_paths = [str(item).strip() for item in list(geo.get("retrieval_index_paths") or []) if str(item).strip()]
    total_existing_indices = 1 + len(existing_paths)
    existing_weights = [
        _positive_weight(item, 1.0)
        for item in _pad_list(list(geo.get("retrieval_index_weights") or []), total_existing_indices, 1.0)
    ]
    existing_model_ids = [
        str(item).strip() or base_model_id
        for item in _pad_list(list(geo.get("retrieval_index_model_ids") or []), len(existing_paths), base_model_id)
    ]
    existing_projection_paths = [
        (str(item).strip() if item is not None and str(item).strip() else None)
        for item in _pad_list(list(geo.get("retrieval_index_projection_paths") or []), len(existing_paths), base_projection_path)
    ]

    tuned_model_text = str(tuned_model_ref).strip()
    existing_paths.append(str(tuned_index_path))
    existing_weights.append(float(max(1e-6, aux_index_weight)))
    existing_model_ids.append(tuned_model_text)
    existing_projection_paths.append(None)

    if tuned_dba_index_path is not None and tuned_dba_index_path.exists():
        existing_paths.append(str(tuned_dba_index_path))
        existing_weights.append(float(max(1e-6, aux_dba_weight)))
        existing_model_ids.append(tuned_model_text)
        existing_projection_paths.append(None)

    geo["retrieval_index_paths"] = existing_paths
    geo["retrieval_index_weights"] = existing_weights
    geo["retrieval_index_model_ids"] = existing_model_ids
    geo["retrieval_index_projection_paths"] = existing_projection_paths
    geo["retrieval_source_fusion_mode"] = "rrf"

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _mine_round_triplets(
    *,
    eval_report_path: Path,
    eval_rows: Sequence[dict],
    train_metadata_path: Path,
    output_triplets_path: Path,
    output_summary_path: Path,
    args,
) -> dict:
    query_records = [
        miner.GeoRecord(
            path=str(row["path"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )
        for row in eval_rows
    ]
    reference_records = miner.load_metadata_csv(train_metadata_path)
    failures = miner.merge_failures(
        miner.load_eval_failures_many([eval_report_path]),
        max_failures_per_query=int(max(0, args.max_failures_per_query)),
    )
    triplets = miner.mine_triplets(
        query_records,
        failures,
        reference_records=reference_records,
        min_error_km=float(args.min_error_km),
        positive_radius_km=float(args.positive_radius_km),
        negative_pred_radius_km=float(args.negative_pred_radius_km),
        negative_min_gt_distance_km=float(args.negative_min_gt_distance_km),
        negative_max_gt_distance_km=float(args.negative_max_gt_distance_km),
        max_positives=int(args.max_positives),
        max_negatives=int(args.max_negatives),
        dedupe_by_scene=not bool(args.no_scene_dedupe),
        difficulty_mode=str(args.difficulty_mode),
        difficulty_reference_km=float(args.difficulty_reference_km),
        difficulty_max_weight=float(args.difficulty_max_weight),
    )
    written = miner._write_jsonl(output_triplets_path, triplets)
    summary = {
        "eval_report": str(eval_report_path),
        "train_metadata": str(train_metadata_path),
        "total_failures_considered": len(failures),
        "triplets_written": written,
        "max_failures_per_query": int(max(0, args.max_failures_per_query)),
        **miner._summarize_triplets(triplets),
    }
    _write_report(output_summary_path, summary)
    return summary


def _build_model_indices(
    *,
    model_id: str,
    train_images_dir: Path,
    train_rows: Sequence[dict],
    index_path: Path,
    dba_index_path: Optional[Path],
    dba_neighbors: int,
    dba_weight: float,
    dba_geo_radius_km: float,
) -> dict:
    count = build_index(
        images=[],
        meta=list(train_rows),
        output=index_path,
        model_id=str(model_id),
        root_dir=train_images_dir.parent,
        images_dir=train_images_dir,
    )
    if count <= 0:
        raise ValueError("fine_tuned_index_empty")
    dba_payload = {"index_count": int(count), "dba_index_path": None}
    if dba_index_path is not None:
        payload = dba_tools._load_index(index_path)
        embeddings = payload["embeddings"]
        latitudes = payload["latitudes"]
        longitudes = payload["longitudes"]
        augmented = dba_tools.augment_embeddings(
            embeddings,
            neighbors=int(dba_neighbors),
            self_weight=float(max(0.0, dba_weight)),
            min_similarity=0.0,
            temperature=0.07,
            latitudes=latitudes,
            longitudes=longitudes,
            max_geo_distance_km=float(max(0.0, dba_geo_radius_km)),
        )
        output_payload = dict(payload)
        output_payload["embeddings"] = augmented
        dba_index_path.parent.mkdir(parents=True, exist_ok=True)
        bench.np.savez(dba_index_path, **output_payload)
        dba_payload["dba_index_path"] = str(dba_index_path)
    return dba_payload


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run iterative retrieval encoder fine-tuning on Paris benchmark.")
    parser.add_argument("--train-images-dir", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--eval-images-dir", required=True)
    parser.add_argument("--eval-metadata", required=True)
    parser.add_argument("--base-model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument(
        "--bootstrap-config",
        default="src/config/paris_close_range_dual_rrf.json",
        help="Optional config used for round-0 mining/eval before any encoder fine-tuning.",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=0, help="0 uses all train rows; >0 samples a fixed subset for index rebuilds.")
    parser.add_argument("--eval-limit", type=int, default=180)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--rank-objective", default="within_2km_pct", choices=sorted(bench.RANK_OBJECTIVES))
    parser.add_argument("--retrieval-top-k", type=int, default=25)
    parser.add_argument("--retrieval-min-score", type=float, default=0.05)
    parser.add_argument("--retrieval-min-keep-topk", type=int, default=0)
    parser.add_argument("--use-dba", action="store_true")
    parser.add_argument("--dba-neighbors", type=int, default=5)
    parser.add_argument("--dba-self-weight", type=float, default=1.0)
    parser.add_argument("--dba-eval-weight", type=float, default=0.5)
    parser.add_argument("--dba-geo-radius-km", type=float, default=2.0)
    parser.add_argument("--min-error-km", type=float, default=2.0)
    parser.add_argument("--positive-radius-km", type=float, default=0.35)
    parser.add_argument("--negative-pred-radius-km", type=float, default=2.0)
    parser.add_argument("--negative-min-gt-distance-km", type=float, default=2.0)
    parser.add_argument("--negative-max-gt-distance-km", type=float, default=25.0)
    parser.add_argument("--max-positives", type=int, default=3)
    parser.add_argument("--max-negatives", type=int, default=12)
    parser.add_argument("--difficulty-mode", default="error_km_predmix", choices=["none", "error_km", "error_km_predmix"])
    parser.add_argument("--difficulty-reference-km", type=float, default=10.0)
    parser.add_argument("--difficulty-max-weight", type=float, default=3.0)
    parser.add_argument("--max-failures-per-query", type=int, default=1)
    parser.add_argument("--no-scene-dedupe", action="store_true")
    parser.add_argument("--train-scope", default="vision_encoder", choices=["visual_projection", "vision_encoder", "all"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--ce-weight", type=float, default=0.2)
    parser.add_argument("--sample-weight-mode", default="triplet_weight", choices=["auto", "uniform", "triplet_weight"])
    parser.add_argument("--sample-weight-power", type=float, default=1.0)
    parser.add_argument("--sample-weight-max", type=float, default=3.0)
    parser.add_argument("--aux-index-weight", type=float, default=0.35)
    parser.add_argument("--aux-dba-weight", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/retrieval_finetune_loop")
    args = parser.parse_args(argv)

    train_images_dir = Path(args.train_images_dir)
    train_metadata_path = Path(args.train_metadata)
    eval_images_dir = Path(args.eval_images_dir)
    eval_metadata_path = Path(args.eval_metadata)
    if not train_metadata_path.exists():
        raise FileNotFoundError(f"train_metadata_not_found:{train_metadata_path}")
    if not eval_metadata_path.exists():
        raise FileNotFoundError(f"eval_metadata_not_found:{eval_metadata_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows_all = bench._load_rows(train_metadata_path)
    train_rows = bench._sample_rows(train_rows_all, limit=int(args.train_limit), seed=int(args.eval_seed))
    eval_rows = _sample_eval_rows(eval_metadata_path, limit=int(args.eval_limit), seed=int(args.eval_seed))
    if not eval_rows:
        raise ValueError("eval_rows_empty")
    if not train_rows:
        raise ValueError("train_rows_empty")

    best_metrics: Optional[dict] = None
    best_model_ref: Optional[str] = None
    best_index_path: Optional[Path] = None
    best_dba_index_path: Optional[Path] = None
    best_serving_metrics: Optional[dict] = None
    best_serving_config: Optional[str] = None
    rounds_payload: List[dict] = []

    bootstrap_config = str(args.bootstrap_config).strip()
    bootstrap_provider: Optional[GeoRetrievalProvider] = None
    if bootstrap_config:
        bootstrap_provider = run_geo_eval.build_retrieval_provider(load_config(bootstrap_config))
        if bootstrap_provider is None:
            raise ValueError("bootstrap_config_missing_retrieval_index")
        bootstrap_report = _evaluate_with_diagnostics(
            rows=eval_rows,
            images_dir=eval_images_dir,
            provider=bootstrap_provider,
            diag_samples=len(eval_rows),
            config_label=bootstrap_config,
        )
        bootstrap_report_path = output_dir / "bootstrap_eval_180.json"
        _write_report(bootstrap_report_path, bootstrap_report)
        rounds_payload.append(
            {
                "stage": "bootstrap_serving",
                "eval_report_path": str(bootstrap_report_path),
                "metrics": bootstrap_report,
            }
        )
        best_serving_metrics = dict(bootstrap_report)
        best_serving_config = bootstrap_config

    baseline_index_path = output_dir / "baseline_model.npz"
    baseline_dba_index_path = output_dir / "baseline_model_dba.npz" if bool(args.use_dba) else None
    index_summary = _build_model_indices(
        model_id=str(args.base_model_id),
        train_images_dir=train_images_dir,
        train_rows=train_rows,
        index_path=baseline_index_path,
        dba_index_path=baseline_dba_index_path,
        dba_neighbors=int(args.dba_neighbors),
        dba_weight=float(args.dba_self_weight),
        dba_geo_radius_km=float(args.dba_geo_radius_km),
    )
    baseline_provider = _build_provider_for_model(
        model_id=str(args.base_model_id),
        index_path=baseline_index_path,
        dba_index_path=baseline_dba_index_path if bool(args.use_dba) else None,
        top_k=int(args.retrieval_top_k),
        min_score=float(args.retrieval_min_score),
        min_keep_topk=int(args.retrieval_min_keep_topk),
        dba_weight=float(args.dba_eval_weight),
    )
    baseline_report = _evaluate_with_diagnostics(
        rows=eval_rows,
        images_dir=eval_images_dir,
        provider=baseline_provider,
        diag_samples=len(eval_rows),
        config_label=str(args.base_model_id),
    )
    baseline_report_path = output_dir / "baseline_eval_180.json"
    _write_report(baseline_report_path, baseline_report)
    best_metrics = dict(baseline_report)
    best_model_ref = str(args.base_model_id)
    best_index_path = baseline_index_path
    best_dba_index_path = baseline_dba_index_path
    rounds_payload.append(
        {
            "stage": "baseline_model",
            "eval_report_path": str(baseline_report_path),
            "metrics": baseline_report,
            "index_path": str(baseline_index_path),
            "dba_index_path": (str(baseline_dba_index_path) if baseline_dba_index_path is not None else None),
            "index_summary": index_summary,
        }
    )

    for round_idx in range(1, int(args.rounds) + 1):
        round_dir = output_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        mining_report_path = round_dir / "mining_eval.json"

        if round_idx == 1 and bootstrap_provider is not None:
            mining_metrics = _evaluate_with_diagnostics(
                rows=eval_rows,
                images_dir=eval_images_dir,
                provider=bootstrap_provider,
                diag_samples=len(eval_rows),
                config_label=bootstrap_config,
            )
            _write_report(mining_report_path, mining_metrics)
            train_init_model_id = str(best_model_ref or args.base_model_id)
        else:
            if best_model_ref is None or best_index_path is None:
                train_init_model_id = str(args.base_model_id)
                initial_index_path = round_dir / "seed_index.npz"
                build_index(
                    images=[],
                    meta=train_rows,
                    output=initial_index_path,
                    model_id=train_init_model_id,
                    root_dir=train_images_dir.parent,
                    images_dir=train_images_dir,
                )
                best_index_path = initial_index_path
                best_dba_index_path = None
                best_model_ref = train_init_model_id
            provider = _build_provider_for_model(
                model_id=str(best_model_ref),
                index_path=best_index_path,
                dba_index_path=best_dba_index_path if bool(args.use_dba) else None,
                top_k=int(args.retrieval_top_k),
                min_score=float(args.retrieval_min_score),
                min_keep_topk=int(args.retrieval_min_keep_topk),
                dba_weight=float(args.dba_eval_weight),
            )
            mining_metrics = _evaluate_with_diagnostics(
                rows=eval_rows,
                images_dir=eval_images_dir,
                provider=provider,
                diag_samples=len(eval_rows),
                config_label=str(best_model_ref),
            )
            _write_report(mining_report_path, mining_metrics)
            train_init_model_id = str(best_model_ref)

        triplets_path = round_dir / "triplets.jsonl"
        triplets_summary_path = round_dir / "triplets.summary.json"
        triplet_summary = _mine_round_triplets(
            eval_report_path=mining_report_path,
            eval_rows=eval_rows,
            train_metadata_path=train_metadata_path,
            output_triplets_path=triplets_path,
            output_summary_path=triplets_summary_path,
            args=args,
        )

        model_dir = round_dir / "model"
        encoder_report_path = round_dir / "encoder.report.json"
        encoder_tools.main(
            [
                "--triplets",
                str(triplets_path),
                "--query-images-dir",
                str(eval_images_dir),
                "--reference-images-dir",
                str(train_images_dir),
                "--model-id",
                str(train_init_model_id),
                "--output-dir",
                str(model_dir),
                "--report-output",
                str(encoder_report_path),
                "--train-scope",
                str(args.train_scope),
                "--epochs",
                str(int(args.epochs)),
                "--batch-size",
                str(int(args.batch_size)),
                "--learning-rate",
                str(float(args.learning_rate)),
                "--weight-decay",
                str(float(args.weight_decay)),
                "--margin",
                str(float(args.margin)),
                "--temperature",
                str(float(args.temperature)),
                "--ce-weight",
                str(float(args.ce_weight)),
                "--sample-weight-mode",
                str(args.sample_weight_mode),
                "--sample-weight-power",
                str(float(args.sample_weight_power)),
                "--sample-weight-max",
                str(float(args.sample_weight_max)),
                "--device",
                str(args.device),
            ]
        )

        index_path = round_dir / f"{_slug(model_dir.name)}.npz"
        dba_index_path = round_dir / f"{_slug(model_dir.name)}_dba.npz" if bool(args.use_dba) else None
        index_summary = _build_model_indices(
            model_id=str(model_dir),
            train_images_dir=train_images_dir,
            train_rows=train_rows,
            index_path=index_path,
            dba_index_path=dba_index_path,
            dba_neighbors=int(args.dba_neighbors),
            dba_weight=float(args.dba_self_weight),
            dba_geo_radius_km=float(args.dba_geo_radius_km),
        )

        candidate_provider = _build_provider_for_model(
            model_id=str(model_dir),
            index_path=index_path,
            dba_index_path=dba_index_path if bool(args.use_dba) else None,
            top_k=int(args.retrieval_top_k),
            min_score=float(args.retrieval_min_score),
            min_keep_topk=int(args.retrieval_min_keep_topk),
            dba_weight=float(args.dba_eval_weight),
        )
        eval_metrics = _evaluate_with_diagnostics(
            rows=eval_rows,
            images_dir=eval_images_dir,
            provider=candidate_provider,
            diag_samples=len(eval_rows),
            config_label=str(model_dir),
        )
        eval_report_path = round_dir / "eval_180.json"
        _write_report(eval_report_path, eval_metrics)

        improved = _is_better(eval_metrics, best_metrics, str(args.rank_objective))
        if improved:
            best_metrics = dict(eval_metrics)
            best_model_ref = str(model_dir)
            best_index_path = index_path
            best_dba_index_path = dba_index_path

        aux_fused_config_path: Optional[Path] = None
        aux_fused_eval_report_path: Optional[Path] = None
        aux_fused_metrics: Optional[dict] = None
        aux_fused_improved = False
        if bootstrap_config:
            aux_fused_config_path = round_dir / "aux_fused_config.json"
            _build_aux_fused_config(
                bootstrap_config_path=Path(bootstrap_config),
                output_config_path=aux_fused_config_path,
                tuned_model_ref=str(model_dir),
                tuned_index_path=index_path,
                tuned_dba_index_path=dba_index_path,
                aux_index_weight=float(args.aux_index_weight),
                aux_dba_weight=float(args.aux_dba_weight),
            )
            aux_provider = run_geo_eval.build_retrieval_provider(load_config(str(aux_fused_config_path)))
            if aux_provider is None:
                raise ValueError("aux_fused_config_missing_retrieval_index")
            aux_fused_metrics = _evaluate_with_diagnostics(
                rows=eval_rows,
                images_dir=eval_images_dir,
                provider=aux_provider,
                diag_samples=len(eval_rows),
                config_label=str(aux_fused_config_path),
            )
            aux_fused_eval_report_path = round_dir / "aux_fused_eval_180.json"
            _write_report(aux_fused_eval_report_path, aux_fused_metrics)
            aux_fused_improved = _is_better(aux_fused_metrics, best_serving_metrics, str(args.rank_objective))
            if aux_fused_improved:
                best_serving_metrics = dict(aux_fused_metrics)
                best_serving_config = str(aux_fused_config_path)

        rounds_payload.append(
            {
                "stage": f"round_{round_idx:02d}",
                "train_init_model_id": train_init_model_id,
                "mining_eval_report_path": str(mining_report_path),
                "triplets_path": str(triplets_path),
                "triplets_summary_path": str(triplets_summary_path),
                "triplet_summary": triplet_summary,
                "encoder_report_path": str(encoder_report_path),
                "index_path": str(index_path),
                "dba_index_path": (str(dba_index_path) if dba_index_path is not None else None),
                "index_summary": index_summary,
                "eval_report_path": str(eval_report_path),
                "metrics": eval_metrics,
                "improved": bool(improved),
                "aux_fused_config_path": (str(aux_fused_config_path) if aux_fused_config_path is not None else None),
                "aux_fused_eval_report_path": (
                    str(aux_fused_eval_report_path) if aux_fused_eval_report_path is not None else None
                ),
                "aux_fused_metrics": aux_fused_metrics,
                "aux_fused_improved": bool(aux_fused_improved),
            }
        )

    summary = {
        "rank_objective": str(args.rank_objective),
        "train_limit": int(args.train_limit),
        "eval_limit": int(args.eval_limit),
        "eval_seed": int(args.eval_seed),
        "base_model_id": str(args.base_model_id),
        "bootstrap_config": (bootstrap_config or None),
        "best_model_ref": best_model_ref,
        "best_index_path": (str(best_index_path) if best_index_path is not None else None),
        "best_dba_index_path": (str(best_dba_index_path) if best_dba_index_path is not None else None),
        "best_metrics": best_metrics,
        "best_serving_config": best_serving_config,
        "best_serving_metrics": best_serving_metrics,
        "rounds": rounds_payload,
    }
    summary_path = output_dir / "loop_summary.json"
    _write_report(summary_path, summary)
    print(f"loop summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
