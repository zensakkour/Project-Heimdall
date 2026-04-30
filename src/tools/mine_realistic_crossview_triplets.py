"""Mine street-to-aerial triplets from the realistic Paris dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class StreetPair:
    street_id: str
    street_path: str
    aerial_id: str
    lat: float
    lon: float
    heading_deg: Optional[float]


@dataclass(frozen=True)
class AerialRecord:
    aerial_id: str
    path: str
    lat: float
    lon: float


def _safe_float(value: object) -> Optional[float]:
    try:
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return num


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = (math.sin(dphi / 2.0) ** 2) + (math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    return radius * c


def _haversine_vector_km(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    radius = 6371.0
    phi1 = math.radians(float(lat))
    phi2 = np.radians(lats.astype(np.float64))
    dphi = phi2 - phi1
    dlambda = np.radians(lons.astype(np.float64) - float(lon))
    a = (np.sin(dphi / 2.0) ** 2) + (math.cos(phi1) * np.cos(phi2) * (np.sin(dlambda / 2.0) ** 2))
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    return np.asarray(radius * c, dtype=np.float64)


def load_pairs_csv(path: Path) -> List[StreetPair]:
    rows: List[StreetPair] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            street_path = str(row.get("street_path") or "").strip()
            aerial_id = str(row.get("aerial_id") or "").strip()
            lat = _safe_float(row.get("lat") or row.get("latitude"))
            lon = _safe_float(row.get("lon") or row.get("longitude"))
            if not street_path or not aerial_id or lat is None or lon is None:
                continue
            rows.append(
                StreetPair(
                    street_id=str(row.get("street_id") or Path(street_path).stem),
                    street_path=street_path,
                    aerial_id=aerial_id,
                    lat=float(lat),
                    lon=float(lon),
                    heading_deg=_safe_float(row.get("heading_deg")),
                )
            )
    return rows


def load_aerial_metadata(path: Path) -> List[AerialRecord]:
    rows: List[AerialRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            aerial_id = str(row.get("aerial_id") or row.get("image_id") or "").strip()
            rel = str(row.get("path") or "").strip()
            lat = _safe_float(row.get("lat") or row.get("latitude"))
            lon = _safe_float(row.get("lon") or row.get("longitude"))
            if not aerial_id or not rel or lat is None or lon is None:
                continue
            rows.append(AerialRecord(aerial_id=aerial_id, path=rel, lat=float(lat), lon=float(lon)))
    return rows


def _compute_triplet_weight(closest_negative_km: Optional[float], negative_count: int) -> float:
    if closest_negative_km is None:
        return 1.0
    proximity = max(0.0, 1.0 - (float(closest_negative_km) / 5.0))
    richness = min(1.0, float(max(0, negative_count)) / 20.0)
    return float(min(3.0, 1.0 + (1.5 * proximity) + (0.5 * richness)))


def mine_triplets(
    *,
    pairs: Sequence[StreetPair],
    aerial_records: Sequence[AerialRecord],
    positive_radius_m: float,
    negative_min_distance_m: float,
    negative_max_distance_m: float,
    max_positives: int,
    max_negatives: int,
    limit: int,
    seed: int,
) -> tuple[List[dict], dict]:
    if not pairs or not aerial_records:
        return [], {
            "total_queries": 0,
            "triplets_written": 0,
            "avg_positives": 0.0,
            "avg_negatives": 0.0,
            "skipped_no_positive": 0,
            "skipped_no_negative": 0,
        }

    rng = random.Random(int(seed))
    queries = list(pairs)
    rng.shuffle(queries)
    if limit > 0:
        queries = queries[: int(limit)]

    aerial_lats = np.asarray([row.lat for row in aerial_records], dtype=np.float64)
    aerial_lons = np.asarray([row.lon for row in aerial_records], dtype=np.float64)
    aerial_ids = [row.aerial_id for row in aerial_records]
    aerial_paths = [row.path for row in aerial_records]
    aerial_by_id = {row.aerial_id: row for row in aerial_records}

    positive_radius_km = float(positive_radius_m) / 1000.0
    negative_min_km = float(negative_min_distance_m) / 1000.0
    negative_max_km = float(negative_max_distance_m) / 1000.0

    skipped_no_positive = 0
    skipped_no_negative = 0
    triplets: List[dict] = []
    pos_counts: List[int] = []
    neg_counts: List[int] = []

    for query in queries:
        dists = _haversine_vector_km(query.lat, query.lon, aerial_lats, aerial_lons)
        used_ids = set()
        positives: List[dict] = []

        exact = aerial_by_id.get(query.aerial_id)
        if exact is not None:
            exact_dist = haversine_km(query.lat, query.lon, exact.lat, exact.lon)
            positives.append({"id": exact.aerial_id, "path": exact.path, "distance_to_gt_km": float(exact_dist)})
            used_ids.add(exact.aerial_id)

        pos_idx = np.where(dists <= positive_radius_km)[0]
        if pos_idx.size > 0:
            for idx in pos_idx[np.argsort(dists[pos_idx])]:
                aerial_id = aerial_ids[int(idx)]
                if aerial_id in used_ids:
                    continue
                positives.append(
                    {
                        "id": aerial_id,
                        "path": aerial_paths[int(idx)],
                        "distance_to_gt_km": float(dists[int(idx)]),
                    }
                )
                used_ids.add(aerial_id)
                if len(positives) >= max(1, int(max_positives)):
                    break
        if not positives:
            skipped_no_positive += 1
            continue

        neg_mask = (dists >= negative_min_km) & (dists <= negative_max_km)
        neg_idx = np.where(neg_mask)[0]
        negatives: List[dict] = []
        if neg_idx.size > 0:
            for idx in neg_idx[np.argsort(dists[neg_idx])]:
                aerial_id = aerial_ids[int(idx)]
                if aerial_id in used_ids:
                    continue
                negatives.append(
                    {
                        "id": aerial_id,
                        "path": aerial_paths[int(idx)],
                        "distance_to_gt_km": float(dists[int(idx)]),
                        "source": "gt_ring",
                    }
                )
                used_ids.add(aerial_id)
                if len(negatives) >= max(1, int(max_negatives)):
                    break
        if not negatives:
            skipped_no_negative += 1
            continue

        closest_negative_km = min(float(item["distance_to_gt_km"]) for item in negatives)
        triplets.append(
            {
                "query_id": query.street_id,
                "query_path": query.street_path,
                "query_lat": float(query.lat),
                "query_lon": float(query.lon),
                "query_heading_deg": query.heading_deg,
                "positive_ids": [str(item["id"]) for item in positives],
                "positive_paths": [str(item["path"]) for item in positives],
                "negative_ids": [str(item["id"]) for item in negatives],
                "negative_paths": [str(item["path"]) for item in negatives],
                "negative_distances_m": [round(float(item["distance_to_gt_km"]) * 1000.0, 3) for item in negatives],
                "triplet_weight": _compute_triplet_weight(closest_negative_km, len(negatives)),
                # Compatibility with existing training utilities.
                "positives": [{"id": item["id"], "path": item["path"], "distance_to_gt_km": item["distance_to_gt_km"]} for item in positives],
                "hard_negatives": [
                    {
                        "id": item["id"],
                        "path": item["path"],
                        "distance_to_gt_km": item["distance_to_gt_km"],
                        "source": item["source"],
                    }
                    for item in negatives
                ],
            }
        )
        pos_counts.append(len(positives))
        neg_counts.append(len(negatives))

    summary = {
        "total_queries": len(queries),
        "triplets_written": len(triplets),
        "avg_positives": float(sum(pos_counts) / len(pos_counts)) if pos_counts else 0.0,
        "avg_negatives": float(sum(neg_counts) / len(neg_counts)) if neg_counts else 0.0,
        "skipped_no_positive": int(skipped_no_positive),
        "skipped_no_negative": int(skipped_no_negative),
        "positive_radius_m": float(positive_radius_m),
        "negative_min_distance_m": float(negative_min_distance_m),
        "negative_max_distance_m": float(negative_max_distance_m),
        "max_positives": int(max_positives),
        "max_negatives": int(max_negatives),
        "seed": int(seed),
    }
    return triplets, summary


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            count += 1
    return count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mine realistic street-to-aerial cross-view triplets.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--street-metadata", default="")
    parser.add_argument("--aerial-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--positive-radius-m", type=float, default=80.0)
    parser.add_argument("--negative-min-distance-m", type=float, default=300.0)
    parser.add_argument("--negative-max-distance-m", type=float, default=5000.0)
    parser.add_argument("--max-positives", type=int, default=3)
    parser.add_argument("--max-negatives", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    pairs = load_pairs_csv(Path(args.pairs))
    aerial_records = load_aerial_metadata(Path(args.aerial_metadata))
    triplets, summary = mine_triplets(
        pairs=pairs,
        aerial_records=aerial_records,
        positive_radius_m=float(args.positive_radius_m),
        negative_min_distance_m=float(args.negative_min_distance_m),
        negative_max_distance_m=float(args.negative_max_distance_m),
        max_positives=int(args.max_positives),
        max_negatives=int(args.max_negatives),
        limit=int(args.limit),
        seed=int(args.seed),
    )
    written = _write_jsonl(Path(args.output), triplets)
    summary["triplets_written"] = int(written)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
