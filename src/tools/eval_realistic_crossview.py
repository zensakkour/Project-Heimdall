"""Evaluate street-to-aerial cross-view retrieval on the realistic Paris dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.core.geo import GeoRetrievalProvider
from src.tools.build_geo_index import _load_metadata, build_index
from src.tools.run_geo_eval import haversine_km, resolve_image_path


def _optional_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _load_pairs(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lat = _optional_float(row.get("lat") or row.get("latitude"))
            lon = _optional_float(row.get("lon") or row.get("longitude"))
            rel = str(row.get("street_path") or row.get("path") or "").strip()
            if lat is None or lon is None or not rel:
                continue
            item = dict(row)
            item["lat"] = float(lat)
            item["lon"] = float(lon)
            item["street_path"] = rel
            rows.append(item)
    return rows


def _percentile(sorted_values: Sequence[float], p: float) -> Optional[float]:
    if not sorted_values:
        return None
    idx = int(round((float(p) / 100.0) * (len(sorted_values) - 1)))
    idx = max(0, min(len(sorted_values) - 1, idx))
    return float(sorted_values[idx])


def _pct_within(values_km: Sequence[float], threshold_km: float) -> float:
    values = list(values_km)
    if not values:
        return 0.0
    count = sum(1 for value in values if float(value) <= float(threshold_km))
    return 100.0 * float(count) / float(len(values))


def _topn_recall(samples: Sequence[dict], *, top_n: int, threshold_km: float) -> float:
    items = list(samples)
    if not items:
        return 0.0
    hits = 0
    for sample in items:
        candidates = list(sample.get("top_candidates") or [])[: max(1, int(top_n))]
        if any((_optional_float(item.get("distance_km")) or float("inf")) <= float(threshold_km) for item in candidates):
            hits += 1
    return 100.0 * float(hits) / float(len(items))


def summarize_samples(samples: Sequence[dict]) -> dict:
    distances = sorted(
        float(item["distance_km"])
        for item in samples
        if _optional_float(item.get("distance_km")) is not None
    )
    return {
        "evaluated": len(distances),
        "mean_km": (sum(distances) / len(distances)) if distances else None,
        "median_km": _percentile(distances, 50.0),
        "p90_km": _percentile(distances, 90.0),
        "within_100m_pct": _pct_within(distances, 0.1),
        "within_250m_pct": _pct_within(distances, 0.25),
        "within_500m_pct": _pct_within(distances, 0.5),
        "within_1km_pct": _pct_within(distances, 1.0),
        "within_2km_pct": _pct_within(distances, 2.0),
        "within_5km_pct": _pct_within(distances, 5.0),
        "top1_recall_100m_pct": _topn_recall(samples, top_n=1, threshold_km=0.1),
        "top1_recall_250m_pct": _topn_recall(samples, top_n=1, threshold_km=0.25),
        "top1_recall_500m_pct": _topn_recall(samples, top_n=1, threshold_km=0.5),
        "top1_recall_1km_pct": _topn_recall(samples, top_n=1, threshold_km=1.0),
        "top5_recall_100m_pct": _topn_recall(samples, top_n=5, threshold_km=0.1),
        "top5_recall_250m_pct": _topn_recall(samples, top_n=5, threshold_km=0.25),
        "top5_recall_500m_pct": _topn_recall(samples, top_n=5, threshold_km=0.5),
        "top5_recall_1km_pct": _topn_recall(samples, top_n=5, threshold_km=1.0),
        "top10_recall_100m_pct": _topn_recall(samples, top_n=10, threshold_km=0.1),
        "top10_recall_250m_pct": _topn_recall(samples, top_n=10, threshold_km=0.25),
        "top10_recall_500m_pct": _topn_recall(samples, top_n=10, threshold_km=0.5),
        "top10_recall_1km_pct": _topn_recall(samples, top_n=10, threshold_km=1.0),
    }


def _build_or_load_aerial_index(
    *,
    aerial_metadata_path: Path,
    aerial_index_path: Path,
    embedding_model: str,
) -> int:
    if aerial_index_path.exists():
        return -1
    metadata = _load_metadata(aerial_metadata_path)
    aerial_images_dir = aerial_metadata_path.parent / "images"
    images = sorted([p for p in aerial_images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise ValueError("aerial_images_not_found")
    return build_index(
        images,
        metadata,
        aerial_index_path,
        embedding_model,
        aerial_metadata_path.parent.parent,
        aerial_images_dir,
        projection_path=None,
    )


def evaluate_realistic_crossview(
    *,
    test_pairs_path: Path,
    aerial_metadata_path: Path,
    street_images_dir: Path,
    aerial_index_path: Path,
    embedding_model: str,
    projection_path: Optional[str],
    top_k: int,
) -> dict:
    _build_or_load_aerial_index(
        aerial_metadata_path=aerial_metadata_path,
        aerial_index_path=aerial_index_path,
        embedding_model=embedding_model,
    )
    projection_text = None if projection_path is None else (str(projection_path).strip() or None)
    provider = GeoRetrievalProvider(
        index_path=str(aerial_index_path),
        model_id=str(embedding_model),
        projection_path=projection_text,
        top_k=int(max(1, top_k)),
        min_score=-1.0,
        min_keep_topk=int(max(1, top_k)),
        source_fusion_mode="weighted_score",
    )
    rows = _load_pairs(test_pairs_path)
    samples: List[dict] = []
    missing = 0
    null_predictions = 0
    for row in rows:
        query_path = resolve_image_path(street_images_dir, str(row["street_path"]))
        if not query_path.exists():
            missing += 1
            continue
        candidates = provider.candidates(str(query_path))
        gt_lat = float(row["lat"])
        gt_lon = float(row["lon"])
        if not candidates:
            null_predictions += 1
            samples.append(
                {
                    "query_path": str(query_path),
                    "gt_lat": gt_lat,
                    "gt_lon": gt_lon,
                    "pred_lat": None,
                    "pred_lon": None,
                    "distance_km": None,
                    "top_candidates": [],
                    "provider_error": provider.last_error,
                }
            )
            continue
        top_candidates = []
        for item in candidates[: max(1, int(top_k))]:
            cand_distance_km = haversine_km(gt_lat, gt_lon, float(item.latitude), float(item.longitude))
            top_candidates.append(
                {
                    "path": str(getattr(item, "supporting_image", "") or ""),
                    "lat": float(item.latitude),
                    "lon": float(item.longitude),
                    "score": float(getattr(item, "retrieval_score", 0.0) or 0.0),
                    "distance_km": float(cand_distance_km),
                }
            )
        top1 = top_candidates[0]
        samples.append(
            {
                "query_path": str(query_path),
                "gt_lat": gt_lat,
                "gt_lon": gt_lon,
                "pred_lat": top1["lat"],
                "pred_lon": top1["lon"],
                "distance_km": float(top1["distance_km"]),
                "top_candidates": top_candidates,
                "provider_error": provider.last_error,
            }
        )

    summary = summarize_samples(samples)
    summary.update(
        {
            "test_pairs": str(test_pairs_path),
            "aerial_metadata": str(aerial_metadata_path),
            "street_images_dir": str(street_images_dir),
            "aerial_index": str(aerial_index_path),
            "projection": projection_text,
            "embedding_model": str(embedding_model),
            "top_k": int(top_k),
            "total_queries": len(rows),
            "missing_queries": int(missing),
            "null_predictions": int(null_predictions),
            "samples": samples,
        }
    )
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate realistic Paris street-to-aerial retrieval.")
    parser.add_argument("--test-pairs", required=True)
    parser.add_argument("--aerial-metadata", required=True)
    parser.add_argument("--projection", default="")
    parser.add_argument("--embedding-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--aerial-index", default="")
    parser.add_argument("--street-images-dir", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    test_pairs_path = Path(args.test_pairs)
    aerial_metadata_path = Path(args.aerial_metadata)
    street_images_dir = (
        Path(args.street_images_dir)
        if str(args.street_images_dir).strip()
        else (test_pairs_path.parent.parent / "street_panoramax")
    )
    aerial_index_path = (
        Path(args.aerial_index)
        if str(args.aerial_index).strip()
        else (aerial_metadata_path.parent.parent / "indices" / "aerial_clip_index.npz")
    )
    report = evaluate_realistic_crossview(
        test_pairs_path=test_pairs_path,
        aerial_metadata_path=aerial_metadata_path,
        street_images_dir=street_images_dir,
        aerial_index_path=aerial_index_path,
        embedding_model=str(args.embedding_model),
        projection_path=(str(args.projection).strip() or None),
        top_k=int(max(1, args.top_k)),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
