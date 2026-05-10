"""
Mine street-to-aerial hard-negative triplets from the current retrieval stack.

This differs from generic geo-ring mining: negatives come from the model's own
retrieved candidates, so training pressure is focused on current failure modes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.core.logic.config import load_config
from src.core.logic.types import GeoCandidate
from src.tools.run_geo_eval import build_retrieval_provider, load_metadata_records, resolve_image_path


@dataclass(frozen=True)
class GeoRecord:
    path: str
    latitude: float
    longitude: float


def load_geo_records(path: Path) -> list[GeoRecord]:
    rows: list[GeoRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = row.get("path") or row.get("image") or row.get("aerial_path") or row.get("street_path")
            lat = row.get("latitude") or row.get("lat")
            lon = row.get("longitude") or row.get("lon") or row.get("lng")
            if not rel or lat is None or lon is None:
                continue
            try:
                rows.append(GeoRecord(str(rel), float(lat), float(lon)))
            except Exception:
                continue
    return rows


def mine_triplets_for_query(
    *,
    query_path: str,
    gt_latitude: float,
    gt_longitude: float,
    candidates: Sequence[GeoCandidate],
    reference_records: Sequence[GeoRecord],
    positive_radius_km: float,
    positive_fallback_top_k: int,
    negative_min_gt_distance_km: float,
    negative_max_gt_distance_km: float,
    max_positives: int,
    max_negatives: int,
    positive_source: str = "reference",
    positive_candidate_source_filter: Sequence[str] = (),
    negative_candidate_source_filter: Sequence[str] = (),
) -> Optional[dict]:
    positive_candidates = _filter_candidates_by_source(candidates, positive_candidate_source_filter)
    negative_candidates = _filter_candidates_by_source(candidates, negative_candidate_source_filter)
    if str(positive_source).strip().lower() == "closest_candidate":
        positives = _retrieval_positive_items(
            positive_candidates,
            gt_latitude,
            gt_longitude,
            max_radius_km=max(0.0, float(positive_radius_km)),
            fallback_top_k=max(0, int(positive_fallback_top_k)),
            limit=max(1, int(max_positives)),
        )
    else:
        positives = _nearest_reference_items(
            reference_records,
            gt_latitude,
            gt_longitude,
            max_radius_km=max(0.0, float(positive_radius_km)),
            fallback_top_k=max(0, int(positive_fallback_top_k)),
            limit=max(1, int(max_positives)),
        )
    negatives = _retrieval_negative_items(
        negative_candidates,
        gt_latitude,
        gt_longitude,
        min_distance_km=max(0.0, float(negative_min_gt_distance_km)),
        max_distance_km=max(float(negative_min_gt_distance_km), float(negative_max_gt_distance_km)),
        limit=max(1, int(max_negatives)),
    )
    if not positives or not negatives:
        return None

    closest_neg = min(float(item["distance_to_gt_km"]) for item in negatives)
    farthest_neg = max(float(item["distance_to_gt_km"]) for item in negatives)
    hardest_score = max(float(item.get("retrieval_score") or 0.0) for item in negatives)
    triplet_weight = _triplet_weight(closest_neg, farthest_neg, hardest_score)
    return {
        "query_path": Path(str(query_path)).as_posix(),
        "gt_latitude": float(gt_latitude),
        "gt_longitude": float(gt_longitude),
        "positives": positives,
        "hard_negatives": negatives,
        "triplet_weight": triplet_weight,
        "min_retrieved_negative_gt_km": closest_neg,
        "max_retrieved_negative_gt_km": farthest_neg,
        "hardest_negative_retrieval_score": hardest_score,
        "mined_from": "retrieval_candidates",
        "positive_source": str(positive_source).strip().lower() or "reference",
    }


def mine_retrieval_triplets(
    *,
    records: Sequence[dict],
    images_dir: Path,
    reference_records: Sequence[GeoRecord],
    provider,
    limit: int,
    seed: int,
    positive_radius_km: float,
    positive_fallback_top_k: int,
    negative_min_gt_distance_km: float,
    negative_max_gt_distance_km: float,
    max_positives: int,
    max_negatives: int,
    max_negative_reuse: int = 0,
    positive_source: str = "reference",
    positive_candidate_source_filter: Sequence[str] = (),
    negative_candidate_source_filter: Sequence[str] = (),
) -> tuple[list[dict], dict]:
    ordered = list(records)
    random.Random(int(seed)).shuffle(ordered)
    if limit > 0:
        ordered = ordered[: int(limit)]

    triplets: list[dict] = []
    missing_files = 0
    no_candidates = 0
    no_triplet = 0
    candidate_counts: list[int] = []
    negative_reuse_counts: dict[str, int] = {}
    for item in ordered:
        rel_path = str(item.get("path") or item.get("street_path") or item.get("image_path") or "")
        if not rel_path:
            no_triplet += 1
            continue
        image_path = Path(rel_path) if Path(rel_path).is_absolute() else resolve_image_path(images_dir, rel_path)
        if not image_path.exists():
            missing_files += 1
            continue
        candidates = provider.candidates(str(image_path))
        if not candidates:
            no_candidates += 1
            continue
        candidate_counts.append(len(candidates))
        gt_lat = float(item.get("latitude", item.get("lat")))
        gt_lon = float(item.get("longitude", item.get("lon", item.get("lng"))))
        triplet = mine_triplets_for_query(
            query_path=rel_path,
            gt_latitude=gt_lat,
            gt_longitude=gt_lon,
            candidates=candidates,
            reference_records=reference_records,
            positive_radius_km=positive_radius_km,
            positive_fallback_top_k=positive_fallback_top_k,
            negative_min_gt_distance_km=negative_min_gt_distance_km,
            negative_max_gt_distance_km=negative_max_gt_distance_km,
            max_positives=max_positives,
            max_negatives=max_negatives,
            positive_source=positive_source,
            positive_candidate_source_filter=positive_candidate_source_filter,
            negative_candidate_source_filter=negative_candidate_source_filter,
        )
        if triplet is None:
            no_triplet += 1
            continue
        if max_negative_reuse > 0:
            filtered_negatives = []
            for negative in triplet["hard_negatives"]:
                key = _reuse_key(str(negative.get("path") or ""))
                if negative_reuse_counts.get(key, 0) >= max_negative_reuse:
                    continue
                filtered_negatives.append(negative)
                if len(filtered_negatives) >= max_negatives:
                    break
            if not filtered_negatives:
                no_triplet += 1
                continue
            triplet["hard_negatives"] = filtered_negatives
            closest_neg = min(float(item["distance_to_gt_km"]) for item in filtered_negatives)
            farthest_neg = max(float(item["distance_to_gt_km"]) for item in filtered_negatives)
            hardest_score = max(float(item.get("retrieval_score") or 0.0) for item in filtered_negatives)
            triplet["min_retrieved_negative_gt_km"] = closest_neg
            triplet["max_retrieved_negative_gt_km"] = farthest_neg
            triplet["hardest_negative_retrieval_score"] = hardest_score
            triplet["triplet_weight"] = _triplet_weight(closest_neg, farthest_neg, hardest_score)
        for negative in triplet["hard_negatives"]:
            key = _reuse_key(str(negative.get("path") or ""))
            negative_reuse_counts[key] = negative_reuse_counts.get(key, 0) + 1
        triplets.append(triplet)

    positive_paths = {
        _reuse_key(str(item.get("path") or ""))
        for triplet in triplets
        for item in triplet.get("positives", [])
        if item.get("path")
    }
    negative_paths = {
        _reuse_key(str(item.get("path") or ""))
        for triplet in triplets
        for item in triplet.get("hard_negatives", [])
        if item.get("path")
    }
    top_negative_reuse = sorted(negative_reuse_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    summary = {
        "records_seen": len(ordered),
        "triplets_written": len(triplets),
        "missing_files": missing_files,
        "no_candidates": no_candidates,
        "no_triplet": no_triplet,
        "candidate_count_mean": float(sum(candidate_counts) / len(candidate_counts)) if candidate_counts else None,
        "positive_radius_km": float(positive_radius_km),
        "positive_fallback_top_k": int(positive_fallback_top_k),
        "negative_min_gt_distance_km": float(negative_min_gt_distance_km),
        "negative_max_gt_distance_km": float(negative_max_gt_distance_km),
        "max_positives": int(max_positives),
        "max_negatives": int(max_negatives),
        "max_negative_reuse": int(max_negative_reuse),
        "positive_source": str(positive_source).strip().lower() or "reference",
        "positive_candidate_source_filter": list(positive_candidate_source_filter),
        "negative_candidate_source_filter": list(negative_candidate_source_filter),
        "unique_positive_paths": len(positive_paths),
        "unique_negative_paths": len(negative_paths),
        "top_negative_reuse": [{"path": path, "count": count} for path, count in top_negative_reuse],
    }
    return triplets, summary


def _filter_candidates_by_source(
    candidates: Sequence[GeoCandidate],
    filters: Sequence[str],
) -> list[GeoCandidate]:
    needles = [str(item).strip().lower() for item in filters if str(item).strip()]
    if not needles:
        return list(candidates)
    out: list[GeoCandidate] = []
    for cand in candidates:
        haystack = " ".join(
            [
                str(cand.match_id or ""),
                str(cand.image_path or ""),
            ]
        ).lower()
        if any(needle in haystack for needle in needles):
            out.append(cand)
    return out


def _parse_filter_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _nearest_reference_items(
    reference_records: Sequence[GeoRecord],
    gt_lat: float,
    gt_lon: float,
    *,
    max_radius_km: float,
    fallback_top_k: int,
    limit: int,
) -> list[dict]:
    ranked = []
    for rec in reference_records:
        dist = haversine_km(gt_lat, gt_lon, rec.latitude, rec.longitude)
        ranked.append((dist, rec))
    ranked.sort(key=lambda item: item[0])
    selected = [(dist, rec) for dist, rec in ranked if dist <= max_radius_km]
    if not selected and fallback_top_k > 0:
        selected = ranked[: int(fallback_top_k)]
    out = []
    seen = set()
    for dist, rec in selected:
        key = Path(rec.path).name
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "path": Path(rec.path).as_posix(),
                "distance_to_gt_km": float(dist),
                "source": "gt_nearby_reference",
            }
        )
        if len(out) >= limit:
            break
    return out


def _retrieval_positive_items(
    candidates: Sequence[GeoCandidate],
    gt_lat: float,
    gt_lon: float,
    *,
    max_radius_km: float,
    fallback_top_k: int,
    limit: int,
) -> list[dict]:
    ranked = []
    for rank, cand in enumerate(candidates, start=1):
        path = str(cand.image_path or "").strip()
        if not path:
            continue
        dist = haversine_km(gt_lat, gt_lon, cand.latitude, cand.longitude)
        ranked.append((dist, rank, cand, path))
    ranked.sort(key=lambda item: item[0])
    selected = [item for item in ranked if item[0] <= max_radius_km]
    if not selected and fallback_top_k > 0:
        selected = ranked[: int(fallback_top_k)]
    out = []
    seen = set()
    for dist, rank, cand, path in selected:
        key = Path(path).name
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "path": _reference_style_path(path),
                "distance_to_gt_km": float(dist),
                "retrieval_rank": int(rank),
                "retrieval_score": float(cand.retrieval_score),
                "match_id": cand.match_id,
                "source": "retrieved_closest_candidate",
            }
        )
        if len(out) >= limit:
            break
    return out


def _retrieval_negative_items(
    candidates: Sequence[GeoCandidate],
    gt_lat: float,
    gt_lon: float,
    *,
    min_distance_km: float,
    max_distance_km: float,
    limit: int,
) -> list[dict]:
    out = []
    seen = set()
    for rank, cand in enumerate(candidates, start=1):
        path = str(cand.image_path or "").strip()
        if not path:
            continue
        dist = haversine_km(gt_lat, gt_lon, cand.latitude, cand.longitude)
        if dist < min_distance_km or dist > max_distance_km:
            continue
        key = Path(path).name
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "path": _reference_style_path(path),
                "distance_to_gt_km": float(dist),
                "retrieval_rank": int(rank),
                "retrieval_score": float(cand.retrieval_score),
                "match_id": cand.match_id,
                "source": "retrieved_wrong_candidate",
            }
        )
        if len(out) >= limit:
            break
    return out


def _reference_style_path(path: str) -> str:
    normalized = Path(str(path)).as_posix()
    lower = normalized.lower()
    marker = "/chips/"
    if marker in lower:
        return normalized[lower.index(marker) + 1 :]
    if lower.startswith("chips/"):
        return normalized
    return normalized


def _reuse_key(path: str) -> str:
    return Path(str(path)).as_posix().lower()


def _triplet_weight(closest_neg_km: float, farthest_neg_km: float, hardest_score: float) -> float:
    closeness = 1.0 / max(1.0, float(closest_neg_km))
    spread = min(1.0, max(0.0, (float(farthest_neg_km) - float(closest_neg_km)) / 25.0))
    score = max(0.0, min(1.0, float(hardest_score)))
    return float(min(4.0, max(1.0, 1.0 + 1.2 * closeness + 0.8 * score + 0.4 * spread)))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return float(radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0))))


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            count += 1
    return count


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mine hard-negative triplets from current retrieval candidates.")
    parser.add_argument("--config", default="src/config/paris.json")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--query-metadata", default="")
    parser.add_argument("--reference-metadata", required=True)
    parser.add_argument("--output", default="runs/retrieval_hard_triplets.jsonl")
    parser.add_argument("--summary-output", default="runs/retrieval_hard_triplets_summary.json")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-radius-km", type=float, default=0.35)
    parser.add_argument("--positive-fallback-top-k", type=int, default=3)
    parser.add_argument("--negative-min-gt-distance-km", type=float, default=1.0)
    parser.add_argument("--negative-max-gt-distance-km", type=float, default=25.0)
    parser.add_argument("--max-positives", type=int, default=4)
    parser.add_argument("--max-negatives", type=int, default=12)
    parser.add_argument(
        "--positive-source",
        choices=["reference", "closest_candidate"],
        default="reference",
        help="Use either nearby reference metadata positives or the closest returned retrieval candidate as the positive.",
    )
    parser.add_argument(
        "--max-negative-reuse",
        type=int,
        default=0,
        help="Optional cap on how often the same hard-negative reference chip can appear across the mined set.",
    )
    parser.add_argument(
        "--positive-candidate-source-filter",
        default="",
        help="Comma-separated substrings that positive retrieval candidates must match in match_id or image_path.",
    )
    parser.add_argument(
        "--negative-candidate-source-filter",
        default="",
        help="Comma-separated substrings that negative retrieval candidates must match in match_id or image_path.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    provider = build_retrieval_provider(cfg)
    if provider is None:
        raise ValueError("config_has_no_retrieval_provider")
    images_dir = Path(args.images_dir)
    metadata_path = Path(args.metadata)
    query_metadata = Path(args.query_metadata) if str(args.query_metadata).strip() else None
    records = load_metadata_records(metadata_path, query_metadata_path=query_metadata, images_dir=images_dir)
    reference_records = load_geo_records(Path(args.reference_metadata))
    if not reference_records:
        raise ValueError("reference_metadata_empty")

    triplets, summary = mine_retrieval_triplets(
        records=records,
        images_dir=images_dir,
        reference_records=reference_records,
        provider=provider,
        limit=int(args.limit),
        seed=int(args.seed),
        positive_radius_km=float(args.positive_radius_km),
        positive_fallback_top_k=int(args.positive_fallback_top_k),
        negative_min_gt_distance_km=float(args.negative_min_gt_distance_km),
        negative_max_gt_distance_km=float(args.negative_max_gt_distance_km),
        max_positives=int(args.max_positives),
        max_negatives=int(args.max_negatives),
        max_negative_reuse=int(args.max_negative_reuse),
        positive_source=str(args.positive_source),
        positive_candidate_source_filter=_parse_filter_list(args.positive_candidate_source_filter),
        negative_candidate_source_filter=_parse_filter_list(args.negative_candidate_source_filter),
    )
    out_path = Path(args.output)
    written = write_jsonl(out_path, triplets)
    summary.update(
        {
            "config": str(args.config),
            "metadata": str(metadata_path),
            "images_dir": str(images_dir),
            "reference_metadata": str(Path(args.reference_metadata)),
            "output": str(out_path),
            "triplets_written": written,
        }
    )
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
