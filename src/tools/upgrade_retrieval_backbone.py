"""
Automate aerial-domain retrieval backbone upgrade:
1) benchmark candidate backbones,
2) select best by objective,
3) rebuild final index with best model,
4) patch target config.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import List

from src.tools import benchmark_geo_backbones as bench
from src.tools.build_geo_index import build_index


def _slug(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return clean or "model"


def _norm_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _load_rows(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
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

    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
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

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("json_metadata_must_be_list")
        rows = []
        for row in payload:
            if not isinstance(row, dict):
                continue
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

    raise ValueError("unsupported_metadata_format")


def _sample_rows(rows: List[dict], limit: int, seed: int) -> List[dict]:
    sampled = list(rows)
    random.Random(seed).shuffle(sampled)
    if limit > 0:
        sampled = sampled[:limit]
    return sampled


def _resolve_final_index_path(args, best_model: str) -> Path:
    if args.final_index_output:
        return Path(args.final_index_output)
    train_images_dir = Path(args.train_images_dir)
    dataset_tag = _slug(f"{train_images_dir.parent.name}_{train_images_dir.name}")
    model_tag = _slug(best_model)
    return Path("data/geo_index") / f"{dataset_tag}_{model_tag}.npz"


def _patch_config(
    *,
    config_path: Path,
    best_model: str,
    final_index_path: Path,
    query_expansion_top_n: int,
    query_expansion_beta: float,
    query_expansion_alpha: float,
    local_match_top_n: int,
    local_match_weight: float,
    local_match_ratio: float,
    local_match_max_features: int,
    graph_rerank_top_n: int,
    graph_rerank_sigma_km: float,
    graph_rerank_score_alpha: float,
    graph_rerank_support_beta: float,
    graph_rerank_center_radius_km: float,
    kde_refine_top_n: int,
    kde_refine_sigma_km: float,
    kde_refine_score_power: float,
    kde_refine_margin_threshold: float,
    kde_refine_switch_radius_km: float,
    kde_refine_max_iters: int,
    preserve_multi_index: bool,
) -> None:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    geolocator = raw.setdefault("geolocator", {})
    geolocator["retrieval_model_id"] = str(best_model)
    geolocator["retrieval_index_path"] = _norm_path(final_index_path)
    geolocator["retrieval_query_expansion_top_n"] = int(max(0, int(query_expansion_top_n)))
    geolocator["retrieval_query_expansion_beta"] = float(max(0.0, min(1.0, float(query_expansion_beta))))
    geolocator["retrieval_query_expansion_alpha"] = float(max(0.0, min(1.0, float(query_expansion_alpha))))
    geolocator["retrieval_local_match_top_n"] = int(max(0, int(local_match_top_n)))
    geolocator["retrieval_local_match_weight"] = float(max(0.0, min(1.0, float(local_match_weight))))
    geolocator["retrieval_local_match_ratio"] = float(max(0.5, min(0.95, float(local_match_ratio))))
    geolocator["retrieval_local_match_max_features"] = int(max(128, min(5000, int(local_match_max_features))))
    geolocator["retrieval_graph_rerank_top_n"] = int(max(0, int(graph_rerank_top_n)))
    geolocator["retrieval_graph_rerank_sigma_km"] = float(max(0.1, min(50.0, float(graph_rerank_sigma_km))))
    geolocator["retrieval_graph_rerank_score_alpha"] = float(max(0.0, min(3.0, float(graph_rerank_score_alpha))))
    geolocator["retrieval_graph_rerank_support_beta"] = float(max(0.0, min(5.0, float(graph_rerank_support_beta))))
    geolocator["retrieval_graph_rerank_center_radius_km"] = float(
        max(0.0, min(50.0, float(graph_rerank_center_radius_km)))
    )
    geolocator["retrieval_kde_refine_top_n"] = int(max(0, int(kde_refine_top_n)))
    geolocator["retrieval_kde_refine_sigma_km"] = float(max(0.1, min(50.0, float(kde_refine_sigma_km))))
    geolocator["retrieval_kde_refine_score_power"] = float(max(0.0, min(5.0, float(kde_refine_score_power))))
    geolocator["retrieval_kde_refine_margin_threshold"] = float(
        max(0.0, min(1.0, float(kde_refine_margin_threshold)))
    )
    geolocator["retrieval_kde_refine_switch_radius_km"] = float(
        max(0.0, min(50.0, float(kde_refine_switch_radius_km)))
    )
    geolocator["retrieval_kde_refine_max_iters"] = int(max(1, min(32, int(kde_refine_max_iters))))
    if not preserve_multi_index:
        geolocator["retrieval_index_paths"] = []
        geolocator["retrieval_index_weights"] = []
        geolocator["retrieval_index_model_ids"] = []
    config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark and apply best aerial retrieval backbone.")
    parser.add_argument("--train-images-dir", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--eval-images-dir", required=True)
    parser.add_argument("--eval-metadata", required=True)
    parser.add_argument("--config", default="src/config/paris.json")
    parser.add_argument(
        "--model-ids",
        default="",
        help="Comma-separated model ids. Overrides --model-preset when non-empty.",
    )
    parser.add_argument(
        "--model-preset",
        default="aerial_rtx5060_precise",
        choices=sorted(bench.MODEL_PRESETS.keys()),
    )
    parser.add_argument(
        "--rank-objective",
        default="within_2km_pct",
        choices=sorted(bench.RANK_OBJECTIVES),
        help="Objective used to choose best model.",
    )
    parser.add_argument("--benchmark-train-limit", type=int, default=600)
    parser.add_argument("--benchmark-eval-limit", type=int, default=200)
    parser.add_argument("--final-train-limit", type=int, default=0, help="0 means all available training rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-top-k", type=int, default=50)
    parser.add_argument("--retrieval-min-score", type=float, default=0.1)
    parser.add_argument("--retrieval-min-keep-topk", type=int, default=2)
    parser.add_argument("--query-tta-degrees", default="0,90,180,270")
    parser.add_argument("--query-tta-modes", default="rgb")
    parser.add_argument("--query-tta-auto-modality", action="store_true")
    parser.add_argument("--query-tta-reduce", default="max")
    parser.add_argument("--query-expansion-top-n", type=int, default=0)
    parser.add_argument("--query-expansion-beta", type=float, default=0.0)
    parser.add_argument("--query-expansion-alpha", type=float, default=0.5)
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
    parser.add_argument("--reuse-indices", action="store_true")
    parser.add_argument("--final-index-output", default="")
    parser.add_argument("--preserve-multi-index", action="store_true")
    parser.add_argument("--output-dir", default="runs/backbone_upgrade")
    parser.add_argument("--dry-run", action="store_true", help="Skip config patch step.")
    args = parser.parse_args(argv)

    train_meta_path = Path(args.train_metadata)
    eval_meta_path = Path(args.eval_metadata)
    config_path = Path(args.config)
    if not train_meta_path.exists():
        raise FileNotFoundError(f"train_metadata_not_found:{train_meta_path}")
    if not eval_meta_path.exists():
        raise FileNotFoundError(f"eval_metadata_not_found:{eval_meta_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config_not_found:{config_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_output = output_dir / "backbone_benchmark.json"

    benchmark_argv: list[str] = [
        "--train-images-dir",
        str(args.train_images_dir),
        "--train-metadata",
        str(args.train_metadata),
        "--eval-images-dir",
        str(args.eval_images_dir),
        "--eval-metadata",
        str(args.eval_metadata),
        "--train-limit",
        str(int(args.benchmark_train_limit)),
        "--eval-limit",
        str(int(args.benchmark_eval_limit)),
        "--seed",
        str(int(args.seed)),
        "--retrieval-top-k",
        str(int(args.retrieval_top_k)),
        "--retrieval-min-score",
        str(float(args.retrieval_min_score)),
        "--retrieval-min-keep-topk",
        str(int(args.retrieval_min_keep_topk)),
        "--query-tta-degrees",
        str(args.query_tta_degrees),
        "--query-tta-modes",
        str(args.query_tta_modes),
        *(
            ["--query-tta-auto-modality"]
            if bool(args.query_tta_auto_modality)
            else []
        ),
        "--query-tta-reduce",
        str(args.query_tta_reduce),
        "--query-expansion-top-n",
        str(int(args.query_expansion_top_n)),
        "--query-expansion-beta",
        str(float(args.query_expansion_beta)),
        "--query-expansion-alpha",
        str(float(args.query_expansion_alpha)),
        "--local-match-top-n",
        str(int(args.local_match_top_n)),
        "--local-match-weight",
        str(float(args.local_match_weight)),
        "--local-match-ratio",
        str(float(args.local_match_ratio)),
        "--local-match-max-features",
        str(int(args.local_match_max_features)),
        "--graph-rerank-top-n",
        str(int(args.graph_rerank_top_n)),
        "--graph-rerank-sigma-km",
        str(float(args.graph_rerank_sigma_km)),
        "--graph-rerank-score-alpha",
        str(float(args.graph_rerank_score_alpha)),
        "--graph-rerank-support-beta",
        str(float(args.graph_rerank_support_beta)),
        "--graph-rerank-center-radius-km",
        str(float(args.graph_rerank_center_radius_km)),
        "--kde-refine-top-n",
        str(int(args.kde_refine_top_n)),
        "--kde-refine-sigma-km",
        str(float(args.kde_refine_sigma_km)),
        "--kde-refine-score-power",
        str(float(args.kde_refine_score_power)),
        "--kde-refine-margin-threshold",
        str(float(args.kde_refine_margin_threshold)),
        "--kde-refine-switch-radius-km",
        str(float(args.kde_refine_switch_radius_km)),
        "--kde-refine-max-iters",
        str(int(args.kde_refine_max_iters)),
        "--rank-objective",
        str(args.rank_objective),
        "--output",
        str(benchmark_output),
    ]
    if str(args.model_ids).strip():
        benchmark_argv.extend(["--model-ids", str(args.model_ids)])
    else:
        benchmark_argv.extend(["--model-preset", str(args.model_preset)])
    if args.reuse_indices:
        benchmark_argv.append("--reuse-indices")

    bench_code = bench.main(benchmark_argv)
    if bench_code != 0:
        print(f"Backbone benchmark failed; inspect {benchmark_output}")
        return int(bench_code)

    bench_payload = json.loads(benchmark_output.read_text(encoding="utf-8"))
    best_model = str(bench_payload.get("best_model") or "").strip()
    if not best_model:
        print(f"No best model found in {benchmark_output}")
        return 1

    final_index_path = _resolve_final_index_path(args, best_model)
    final_index_path.parent.mkdir(parents=True, exist_ok=True)

    train_rows_all = _load_rows(train_meta_path)
    final_rows = _sample_rows(train_rows_all, int(args.final_train_limit), int(args.seed))
    if not final_rows:
        print("No train rows available for final index build.")
        return 1

    count = build_index(
        images=[],
        meta=final_rows,
        output=final_index_path,
        model_id=best_model,
        root_dir=Path(args.train_images_dir).parent,
        images_dir=Path(args.train_images_dir),
    )
    if count <= 0:
        print("Final index build wrote no embeddings.")
        return 1

    if not args.dry_run:
        _patch_config(
            config_path=config_path,
            best_model=best_model,
            final_index_path=final_index_path,
            query_expansion_top_n=int(args.query_expansion_top_n),
            query_expansion_beta=float(args.query_expansion_beta),
            query_expansion_alpha=float(args.query_expansion_alpha),
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
            preserve_multi_index=bool(args.preserve_multi_index),
        )

    report = {
        "benchmark_output": _norm_path(benchmark_output),
        "rank_objective": str(args.rank_objective),
        "best_model": best_model,
        "final_index_path": _norm_path(final_index_path),
        "final_index_rows": int(count),
        "config_path": _norm_path(config_path),
        "config_patched": not bool(args.dry_run),
        "query_expansion_top_n": int(args.query_expansion_top_n),
        "query_expansion_beta": float(args.query_expansion_beta),
        "query_expansion_alpha": float(args.query_expansion_alpha),
        "query_tta_modes": str(args.query_tta_modes),
        "query_tta_auto_modality": bool(args.query_tta_auto_modality),
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
        "preserve_multi_index": bool(args.preserve_multi_index),
    }
    report_path = output_dir / "backbone_upgrade_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Backbone upgrade complete. Report: {report_path}")
    print(f"Best model: {best_model}")
    print(f"Final index: {final_index_path}")
    if args.dry_run:
        print("Config patch skipped (dry-run).")
    else:
        print(f"Updated config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
