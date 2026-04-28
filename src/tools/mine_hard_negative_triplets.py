"""
Mine hard-negative retrieval triplets from geo metadata and evaluation failures.
"""
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
class GeoRecord:
    path: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class EvalFailure:
    query_path: str
    gt_latitude: float
    gt_longitude: float
    pred_latitude: Optional[float]
    pred_longitude: Optional[float]
    distance_km: Optional[float]


def scene_key(path: str) -> str:
    name = Path(str(path)).name
    for prefix in ("RGB-PanSharpen_", "MUL-PanSharpen_", "PAN_", "RGB_", "MUL_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = (math.sin(dphi / 2.0) ** 2) + (math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
    return radius * c


def _haversine_vector_km(
    lat: float,
    lon: float,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    radius = 6371.0
    phi1 = math.radians(float(lat))
    phi2 = np.radians(latitudes.astype(np.float64))
    dphi = phi2 - phi1
    dlambda = np.radians(longitudes.astype(np.float64) - float(lon))
    a = (np.sin(dphi / 2.0) ** 2) + (math.cos(phi1) * np.cos(phi2) * (np.sin(dlambda / 2.0) ** 2))
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    return np.asarray(radius * c, dtype=np.float64)


def load_metadata_csv(path: Path) -> List[GeoRecord]:
    out: List[GeoRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rel = row.get("path") or row.get("image")
            lat = row.get("latitude") or row.get("lat")
            lon = row.get("longitude") or row.get("lon")
            if not rel or lat is None or lon is None:
                continue
            out.append(
                GeoRecord(
                    path=str(rel),
                    latitude=float(lat),
                    longitude=float(lon),
                )
            )
    return out


def load_eval_failures(eval_report: Path) -> List[EvalFailure]:
    payload = json.loads(eval_report.read_text(encoding="utf-8"))
    raw_samples = payload.get("samples", []) if isinstance(payload, dict) else []
    out: List[EvalFailure] = []
    for item in raw_samples:
        if not isinstance(item, dict):
            continue
        q = item.get("image")
        gt_lat = item.get("gt_lat")
        gt_lon = item.get("gt_lon")
        if not q or gt_lat is None or gt_lon is None:
            continue
        pred_lat = item.get("pred_lat")
        pred_lon = item.get("pred_lon")
        dist_km = item.get("dist_km")
        out.append(
            EvalFailure(
                query_path=str(q),
                gt_latitude=float(gt_lat),
                gt_longitude=float(gt_lon),
                pred_latitude=float(pred_lat) if pred_lat is not None else None,
                pred_longitude=float(pred_lon) if pred_lon is not None else None,
                distance_km=float(dist_km) if dist_km is not None else None,
            )
        )
    return out


def _record_maps(records: Sequence[GeoRecord]) -> tuple[Dict[str, GeoRecord], Dict[str, GeoRecord]]:
    by_path: Dict[str, GeoRecord] = {}
    by_name: Dict[str, GeoRecord] = {}
    for rec in records:
        key = Path(rec.path).as_posix()
        by_path[key] = rec
        by_name.setdefault(Path(key).name, rec)
    return by_path, by_name


def _resolve_query_record(
    query_path: str,
    *,
    by_path: Dict[str, GeoRecord],
    by_name: Dict[str, GeoRecord],
) -> Optional[GeoRecord]:
    as_posix = Path(query_path).as_posix()
    direct = by_path.get(as_posix)
    if direct is not None:
        return direct
    return by_name.get(Path(as_posix).name)


def _safe_float(value: object) -> Optional[float]:
    try:
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return num


def _distance_summary(values: Sequence[Optional[float]]) -> Optional[float]:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(min(cleaned))


def _negative_source_counts(items: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        source = str(item.get("source") or "unknown").strip() or "unknown"
        counts[source] = int(counts.get(source, 0)) + 1
    return counts


def _compute_triplet_weight(
    *,
    fail: EvalFailure,
    negatives: Sequence[dict],
    difficulty_mode: str,
    difficulty_reference_km: float,
    difficulty_max_weight: float,
) -> float:
    mode = str(difficulty_mode).strip().lower()
    if mode == "none":
        return 1.0

    error_km = max(0.0, _safe_float(fail.distance_km) or 0.0)
    ref_km = max(0.1, float(difficulty_reference_km))
    max_weight = max(1.0, float(difficulty_max_weight))
    denom = math.log1p(ref_km)
    severity = (math.log1p(error_km) / denom) if denom > 0.0 else 0.0
    severity = max(0.0, severity)
    weight = 1.0 + severity

    if mode == "error_km_predmix":
        pred_count = sum(1 for item in negatives if str(item.get("source") or "") == "pred_neighborhood")
        total = max(1, len(negatives))
        weight = weight + (0.5 * (float(pred_count) / float(total)))

    return float(min(max_weight, max(1.0, weight)))


def _summarize_triplets(triplets: Sequence[dict]) -> dict:
    if not triplets:
        return {
            "avg_positives": 0.0,
            "avg_hard_negatives": 0.0,
            "triplet_weight_min": None,
            "triplet_weight_mean": None,
            "triplet_weight_max": None,
            "negative_source_totals": {},
        }

    pos_counts = [len(row.get("positives", [])) for row in triplets]
    neg_counts = [len(row.get("hard_negatives", [])) for row in triplets]
    weights = [
        float(value)
        for value in (_safe_float(row.get("triplet_weight")) for row in triplets)
        if value is not None and value > 0.0
    ]
    source_totals: Dict[str, int] = {}
    for row in triplets:
        for source, count in dict(row.get("hard_negative_source_counts") or {}).items():
            source_totals[str(source)] = int(source_totals.get(str(source), 0)) + int(count)

    return {
        "avg_positives": float(sum(pos_counts) / len(pos_counts)),
        "avg_hard_negatives": float(sum(neg_counts) / len(neg_counts)),
        "triplet_weight_min": float(min(weights)) if weights else None,
        "triplet_weight_mean": float(sum(weights) / len(weights)) if weights else None,
        "triplet_weight_max": float(max(weights)) if weights else None,
        "negative_source_totals": source_totals,
    }


def mine_triplets(
    query_records: Sequence[GeoRecord],
    failures: Sequence[EvalFailure],
    *,
    reference_records: Optional[Sequence[GeoRecord]] = None,
    min_error_km: float = 2.0,
    positive_radius_km: float = 0.35,
    negative_pred_radius_km: float = 2.0,
    negative_min_gt_distance_km: float = 2.0,
    negative_max_gt_distance_km: float = 25.0,
    max_positives: int = 3,
    max_negatives: int = 10,
    dedupe_by_scene: bool = True,
    difficulty_mode: str = "error_km_predmix",
    difficulty_reference_km: float = 10.0,
    difficulty_max_weight: float = 3.0,
) -> List[dict]:
    if not query_records:
        return []
    query_list = list(query_records)
    ref_list = list(reference_records) if reference_records else list(query_records)
    if not ref_list:
        return []
    by_path, by_name = _record_maps(query_list)

    lats = np.asarray([item.latitude for item in ref_list], dtype=np.float64)
    lons = np.asarray([item.longitude for item in ref_list], dtype=np.float64)
    paths = [item.path for item in ref_list]

    out: List[dict] = []
    for fail in failures:
        if fail.distance_km is not None and float(fail.distance_km) < float(min_error_km):
            continue
        query_rec = _resolve_query_record(fail.query_path, by_path=by_path, by_name=by_name)
        if query_rec is None:
            continue
        d_gt = _haversine_vector_km(query_rec.latitude, query_rec.longitude, lats, lons)

        query_key = Path(query_rec.path).as_posix()
        is_query = np.asarray([Path(path).as_posix() == query_key for path in paths], dtype=bool)

        pos_mask = (d_gt > 0.0) & (d_gt <= float(positive_radius_km)) & (~is_query)
        pos_idx = np.where(pos_mask)[0]
        if pos_idx.size <= 0:
            continue
        pos_sorted = pos_idx[np.argsort(d_gt[pos_idx])]
        positives: List[dict] = []
        used = {query_key}
        used_scene = {scene_key(query_rec.path)} if dedupe_by_scene else set()
        for idx in pos_sorted:
            path_i = paths[int(idx)]
            key_i = Path(path_i).as_posix()
            scene_i = scene_key(path_i)
            if key_i in used:
                continue
            if dedupe_by_scene and scene_i in used_scene:
                continue
            positives.append(
                {
                    "path": path_i,
                    "distance_to_gt_km": float(d_gt[int(idx)]),
                }
            )
            used.add(key_i)
            if dedupe_by_scene:
                used_scene.add(scene_i)
            if len(positives) >= max(1, int(max_positives)):
                break
        if not positives:
            continue

        used.update(Path(item["path"]).as_posix() for item in positives)

        negatives: List[dict] = []
        if fail.pred_latitude is not None and fail.pred_longitude is not None:
            d_pred = _haversine_vector_km(float(fail.pred_latitude), float(fail.pred_longitude), lats, lons)
            primary_mask = (
                (d_pred <= float(negative_pred_radius_km))
                & (d_gt >= float(negative_min_gt_distance_km))
                & (~is_query)
            )
            primary_idx = np.where(primary_mask)[0]
            if primary_idx.size > 0:
                primary_sorted = primary_idx[np.argsort(d_pred[primary_idx])]
                for idx in primary_sorted:
                    p = Path(paths[int(idx)]).as_posix()
                    if p in used:
                        continue
                    if dedupe_by_scene and scene_key(paths[int(idx)]) in used_scene:
                        continue
                    negatives.append(
                        {
                            "path": paths[int(idx)],
                            "distance_to_pred_km": float(d_pred[int(idx)]),
                            "distance_to_gt_km": float(d_gt[int(idx)]),
                            "source": "pred_neighborhood",
                        }
                    )
                    used.add(p)
                    if dedupe_by_scene:
                        used_scene.add(scene_key(paths[int(idx)]))
                    if len(negatives) >= max(1, int(max_negatives)):
                        break

        if len(negatives) < max(1, int(max_negatives)):
            fallback_mask = (
                (d_gt >= float(negative_min_gt_distance_km))
                & (d_gt <= float(negative_max_gt_distance_km))
                & (~is_query)
            )
            fallback_idx = np.where(fallback_mask)[0]
            if fallback_idx.size > 0:
                fallback_sorted = fallback_idx[np.argsort(d_gt[fallback_idx])]
                for idx in fallback_sorted:
                    p = Path(paths[int(idx)]).as_posix()
                    if p in used:
                        continue
                    if dedupe_by_scene and scene_key(paths[int(idx)]) in used_scene:
                        continue
                    negatives.append(
                        {
                            "path": paths[int(idx)],
                            "distance_to_pred_km": None,
                            "distance_to_gt_km": float(d_gt[int(idx)]),
                            "source": "gt_ring",
                        }
                    )
                    used.add(p)
                    if dedupe_by_scene:
                        used_scene.add(scene_key(paths[int(idx)]))
                    if len(negatives) >= max(1, int(max_negatives)):
                        break

        if not negatives:
            continue

        source_counts = _negative_source_counts(negatives)
        triplet_weight = _compute_triplet_weight(
            fail=fail,
            negatives=negatives,
            difficulty_mode=difficulty_mode,
            difficulty_reference_km=difficulty_reference_km,
            difficulty_max_weight=difficulty_max_weight,
        )

        out.append(
            {
                "query_path": query_rec.path,
                "gt_latitude": float(query_rec.latitude),
                "gt_longitude": float(query_rec.longitude),
                "pred_latitude": fail.pred_latitude,
                "pred_longitude": fail.pred_longitude,
                "eval_error_km": fail.distance_km,
                "positives": positives,
                "hard_negatives": negatives,
                "triplet_weight": triplet_weight,
                "difficulty_mode": str(difficulty_mode),
                "hard_negative_source_counts": source_counts,
                "closest_positive_gt_km": _distance_summary(
                    [_safe_float(item.get("distance_to_gt_km")) for item in positives]
                ),
                "closest_negative_gt_km": _distance_summary(
                    [_safe_float(item.get("distance_to_gt_km")) for item in negatives]
                ),
                "closest_negative_pred_km": _distance_summary(
                    [_safe_float(item.get("distance_to_pred_km")) for item in negatives]
                ),
            }
        )
    return out


def _synthetic_failures_from_metadata(
    records: Sequence[GeoRecord],
    *,
    sample_count: int,
    seed: int,
) -> List[EvalFailure]:
    if not records:
        return []
    rng = random.Random(int(seed))
    picks = list(records)
    rng.shuffle(picks)
    picks = picks[: max(1, int(sample_count))]
    out: List[EvalFailure] = []
    for rec in picks:
        out.append(
            EvalFailure(
                query_path=rec.path,
                gt_latitude=rec.latitude,
                gt_longitude=rec.longitude,
                pred_latitude=None,
                pred_longitude=None,
                distance_km=None,
            )
        )
    return out


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
            count += 1
    return count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mine hard-negative triplets from eval failures.")
    parser.add_argument("--metadata", required=True, help="Metadata CSV (path, latitude, longitude).")
    parser.add_argument(
        "--reference-metadata",
        default="",
        help="Optional reference metadata CSV used as the retrieval pool for positives/negatives.",
    )
    parser.add_argument(
        "--eval-report",
        default="",
        help="Optional run_geo_eval JSON report containing sample-level GT/pred diagnostics.",
    )
    parser.add_argument("--output", default="runs/hard_negative_triplets.jsonl", help="Output triplet JSONL path.")
    parser.add_argument("--summary-output", default="runs/hard_negative_triplets_summary.json")
    parser.add_argument("--min-error-km", type=float, default=2.0)
    parser.add_argument("--positive-radius-km", type=float, default=0.35)
    parser.add_argument("--negative-pred-radius-km", type=float, default=2.0)
    parser.add_argument("--negative-min-gt-distance-km", type=float, default=2.0)
    parser.add_argument("--negative-max-gt-distance-km", type=float, default=25.0)
    parser.add_argument("--max-positives", type=int, default=3)
    parser.add_argument("--max-negatives", type=int, default=10)
    parser.add_argument(
        "--difficulty-mode",
        default="error_km_predmix",
        choices=["none", "error_km", "error_km_predmix"],
        help="How to compute triplet_weight metadata for downstream training.",
    )
    parser.add_argument(
        "--difficulty-reference-km",
        type=float,
        default=10.0,
        help="Reference error scale used by weighted triplet mining.",
    )
    parser.add_argument(
        "--difficulty-max-weight",
        type=float,
        default=3.0,
        help="Upper bound for emitted triplet_weight values.",
    )
    parser.add_argument("--no-scene-dedupe", action="store_true")
    parser.add_argument("--sample-count", type=int, default=800, help="Used when --eval-report is not provided.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    metadata_path = Path(args.metadata)
    records = load_metadata_csv(metadata_path)
    if not records:
        raise ValueError("metadata_empty_or_invalid")
    reference_records: Optional[List[GeoRecord]] = None
    if str(args.reference_metadata).strip():
        reference_records = load_metadata_csv(Path(args.reference_metadata))
        if not reference_records:
            raise ValueError("reference_metadata_empty_or_invalid")

    eval_report = args.eval_report.strip()
    if eval_report:
        failures = load_eval_failures(Path(eval_report))
    else:
        failures = _synthetic_failures_from_metadata(records, sample_count=args.sample_count, seed=args.seed)

    triplets = mine_triplets(
        records,
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

    out_path = Path(args.output)
    written = _write_jsonl(out_path, triplets)
    triplet_summary = _summarize_triplets(triplets)
    summary = {
        "metadata": str(metadata_path),
        "reference_metadata": str(Path(args.reference_metadata)) if str(args.reference_metadata).strip() else None,
        "eval_report": eval_report or None,
        "total_records": len(records),
        "total_reference_records": len(reference_records) if reference_records is not None else len(records),
        "total_failures_considered": len(failures),
        "triplets_written": written,
        "output": str(out_path),
        "min_error_km": float(args.min_error_km),
        "positive_radius_km": float(args.positive_radius_km),
        "negative_pred_radius_km": float(args.negative_pred_radius_km),
        "negative_min_gt_distance_km": float(args.negative_min_gt_distance_km),
        "negative_max_gt_distance_km": float(args.negative_max_gt_distance_km),
        "max_positives": int(args.max_positives),
        "max_negatives": int(args.max_negatives),
        "dedupe_by_scene": not bool(args.no_scene_dedupe),
        "difficulty_mode": str(args.difficulty_mode),
        "difficulty_reference_km": float(args.difficulty_reference_km),
        "difficulty_max_weight": float(args.difficulty_max_weight),
        **triplet_summary,
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {written} triplets -> {out_path}")
    print(f"Wrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
