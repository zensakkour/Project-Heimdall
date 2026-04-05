"""
Evaluation metrics and calibration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class GroundTruth:
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Candidate:
    latitude: float
    longitude: float
    score: float


def load_ground_truth(path: Path) -> dict[str, GroundTruth]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        return _load_ground_truth_csv(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        return _load_ground_truth_json(path)
    raise ValueError("ground truth must be .csv, .json, or .jsonl")


def _load_ground_truth_csv(path: Path) -> dict[str, GroundTruth]:
    data: dict[str, GroundTruth] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image = row.get("image") or row.get("path")
            if not image:
                continue
            lat = row.get("latitude") or row.get("lat")
            lon = row.get("longitude") or row.get("lon")
            if lat is None or lon is None:
                continue
            data[str(image)] = GroundTruth(latitude=float(lat), longitude=float(lon))
    return data


def _load_ground_truth_json(path: Path) -> dict[str, GroundTruth]:
    raw = path.read_text(encoding="utf-8")
    data: dict[str, GroundTruth] = {}
    if path.suffix.lower() == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image = item.get("image") or item.get("path")
            if not image:
                continue
            lat = item.get("latitude") or item.get("lat")
            lon = item.get("longitude") or item.get("lon")
            if lat is None or lon is None:
                continue
            data[str(image)] = GroundTruth(latitude=float(lat), longitude=float(lon))
        return data

    payload = json.loads(raw)
    if isinstance(payload, dict):
        for image, value in payload.items():
            if isinstance(value, dict):
                lat = value.get("latitude") or value.get("lat")
                lon = value.get("longitude") or value.get("lon")
                if lat is None or lon is None:
                    continue
                data[str(image)] = GroundTruth(latitude=float(lat), longitude=float(lon))
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            image = item.get("image") or item.get("path")
            if not image:
                continue
            lat = item.get("latitude") or item.get("lat")
            lon = item.get("longitude") or item.get("lon")
            if lat is None or lon is None:
                continue
            data[str(image)] = GroundTruth(latitude=float(lat), longitude=float(lon))
    return data


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def extract_candidates(row: dict) -> list[Candidate]:
    fusion = extract_fusion(row)

    candidates: list[Candidate] = []
    if isinstance(fusion, dict):
        items = fusion.get("candidates")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                cand = item.get("candidate", {})
                if not isinstance(cand, dict):
                    continue
                lat = cand.get("latitude")
                lon = cand.get("longitude")
                score = item.get("posterior_weight", cand.get("retrieval_score"))
                if lat is None or lon is None or score is None:
                    continue
                candidates.append(Candidate(float(lat), float(lon), float(score)))
        return candidates

    # fallback to retrieval candidates
    items = row.get("candidates")
    if isinstance(row.get("result"), dict) and isinstance(row["result"].get("candidates"), list):
        items = row["result"].get("candidates")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            lat = item.get("latitude")
            lon = item.get("longitude")
            score = item.get("retrieval_score", item.get("confidence"))
            if lat is None or lon is None or score is None:
                continue
            candidates.append(Candidate(float(lat), float(lon), float(score)))
        return candidates

    # fallback to geo estimate
    geo = None
    if isinstance(row.get("result"), dict):
        geo = row["result"].get("geo")
    if isinstance(geo, dict):
        lat = geo.get("latitude")
        lon = geo.get("longitude")
        score = geo.get("confidence")
        if lat is not None and lon is not None and score is not None:
            candidates.append(Candidate(float(lat), float(lon), float(score)))
    return candidates


def extract_uncertainty_radius(row: dict) -> Optional[float]:
    fusion = extract_fusion(row)
    if isinstance(fusion, dict):
        radius = fusion.get("uncertainty_radius_m")
        if isinstance(radius, (int, float)):
            return float(radius)
    geo = None
    if isinstance(row.get("result"), dict):
        geo = row["result"].get("geo")
    if isinstance(geo, dict):
        radius = geo.get("uncertainty_radius_m") or geo.get("uncertainty_m")
        if isinstance(radius, (int, float)):
            return float(radius)
    return None


def extract_fusion(row: dict) -> Optional[dict]:
    fusion = row.get("fusion")
    if isinstance(row.get("result"), dict) and isinstance(row["result"].get("fusion"), dict):
        fusion = row["result"].get("fusion")
    return fusion if isinstance(fusion, dict) else None


def extract_confidence_tier(row: dict) -> Optional[str]:
    fusion = extract_fusion(row)
    if fusion is None:
        return None
    tier = fusion.get("confidence_tier")
    if isinstance(tier, str):
        return tier.lower().strip()
    return None


def extract_top1_cross_source_support(row: dict) -> Optional[float]:
    fusion = extract_fusion(row)
    if fusion is None:
        return None
    support = fusion.get("top1_cross_source_support")
    if isinstance(support, (int, float)):
        return max(0.0, min(1.0, float(support)))
    return None


def normalize_confidence(score: float) -> float:
    if math.isnan(score):
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def compute_ece(confidences: Iterable[float], correctness: Iterable[int], bins: int = 10) -> float:
    conf_list = list(confidences)
    corr_list = list(correctness)
    if not conf_list:
        return 0.0
    ece = 0.0
    n = len(conf_list)
    for b in range(bins):
        low = b / bins
        high = (b + 1) / bins
        idx = [i for i, c in enumerate(conf_list) if (c >= low and c < high) or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(corr_list[i] for i in idx) / len(idx)
        conf = sum(conf_list[i] for i in idx) / len(idx)
        ece += abs(acc - conf) * (len(idx) / n)
    return ece


def compute_brier(confidences: Iterable[float], correctness: Iterable[int]) -> float:
    conf_list = list(confidences)
    corr_list = list(correctness)
    if not conf_list:
        return 0.0
    return sum((c - float(y)) ** 2 for c, y in zip(conf_list, corr_list)) / len(conf_list)


def compute_nll(confidences: Iterable[float], correctness: Iterable[int], eps: float = 1e-6) -> float:
    conf_list = list(confidences)
    corr_list = list(correctness)
    if not conf_list:
        return 0.0
    total = 0.0
    for c, y in zip(conf_list, corr_list):
        p = max(eps, min(1.0 - eps, c))
        if int(y) == 1:
            total += -math.log(p)
        else:
            total += -math.log(1.0 - p)
    return total / len(conf_list)


def _distance_bucket_km(distance_km: float) -> str:
    if distance_km <= 1.0:
        return "0-1km"
    if distance_km <= 5.0:
        return "1-5km"
    if distance_km <= 25.0:
        return "5-25km"
    if distance_km <= 100.0:
        return "25-100km"
    return ">100km"


def load_results(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute evaluation metrics from JSONL.")
    parser.add_argument("jsonl", help="Path to JSONL results")
    parser.add_argument("--ground-truth", help="CSV/JSON/JSONL with image, latitude, longitude")
    parser.add_argument("--radius-m", type=float, default=1000.0, help="Hit radius in meters")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for accuracy")
    parser.add_argument("--ece-bins", type=int, default=10, help="ECE bin count")
    parser.add_argument("--output", default="runs/metrics.json")
    args = parser.parse_args()

    rows = load_results(Path(args.jsonl))
    ground_truth = None
    if args.ground_truth:
        ground_truth = load_ground_truth(Path(args.ground_truth))

    total = 0
    with_gt = 0
    top1_hits = 0
    topk_hits = 0
    confidences: list[float] = []
    correctness: list[int] = []
    uncertainty_values: list[float] = []
    top1_distances_km: list[float] = []
    top5_25km_hits = 0
    top1_cross_source_support_values: list[float] = []
    high_conf_total = 0
    high_conf_hits = 0
    medium_or_higher_total = 0
    medium_or_higher_hits = 0

    for row in rows:
        total += 1
        image = row.get("image") or row.get("path")
        if not image:
            continue
        candidates = extract_candidates(row)
        if not candidates:
            continue
        candidates.sort(key=lambda item: item.score, reverse=True)

        radius = extract_uncertainty_radius(row)
        if radius is not None:
            uncertainty_values.append(radius)
        top1_cross_source_support = extract_top1_cross_source_support(row)
        if top1_cross_source_support is not None:
            top1_cross_source_support_values.append(top1_cross_source_support)

        if ground_truth is None:
            continue

        gt = ground_truth.get(str(image))
        if gt is None:
            gt = ground_truth.get(Path(str(image)).name)
        if gt is None:
            continue

        with_gt += 1
        best = candidates[0]
        best_dist = haversine_m(gt.latitude, gt.longitude, best.latitude, best.longitude)
        top1_distances_km.append(best_dist / 1000.0)
        best_hit = 1 if best_dist <= args.radius_m else 0
        top1_hits += best_hit
        tier = extract_confidence_tier(row)
        if tier == "high":
            high_conf_total += 1
            high_conf_hits += best_hit
        if tier in {"high", "medium"}:
            medium_or_higher_total += 1
            medium_or_higher_hits += best_hit

        for cand in candidates[: args.top_k]:
            dist = haversine_m(gt.latitude, gt.longitude, cand.latitude, cand.longitude)
            if dist <= args.radius_m:
                topk_hits += 1
                break
        for cand in candidates[:5]:
            dist = haversine_m(gt.latitude, gt.longitude, cand.latitude, cand.longitude)
            if dist <= 25_000.0:
                top5_25km_hits += 1
                break

        confidences.append(normalize_confidence(best.score))
        correctness.append(best_hit)

    distance_buckets = {"0-1km": 0, "1-5km": 0, "5-25km": 0, "25-100km": 0, ">100km": 0}
    for dist_km in top1_distances_km:
        distance_buckets[_distance_bucket_km(dist_km)] += 1

    payload = {
        "total": total,
        "with_ground_truth": with_gt,
        "top1": (top1_hits / with_gt) if with_gt else None,
        "topk": (topk_hits / with_gt) if with_gt else None,
        "top1_mean_km": (sum(top1_distances_km) / len(top1_distances_km)) if top1_distances_km else None,
        "top1_median_km": (
            sorted(top1_distances_km)[len(top1_distances_km) // 2] if top1_distances_km else None
        ),
        "top5_within_25km_pct": (100.0 * top5_25km_hits / with_gt) if with_gt else None,
        "avg_top1_cross_source_support": (
            sum(top1_cross_source_support_values) / len(top1_cross_source_support_values)
        )
        if top1_cross_source_support_values
        else None,
        "high_confidence_coverage_pct": (100.0 * high_conf_total / with_gt) if with_gt else None,
        "high_confidence_top1": (high_conf_hits / high_conf_total) if high_conf_total else None,
        "medium_or_higher_coverage_pct": (100.0 * medium_or_higher_total / with_gt) if with_gt else None,
        "medium_or_higher_top1": (medium_or_higher_hits / medium_or_higher_total)
        if medium_or_higher_total
        else None,
        "failure_buckets_top1": distance_buckets,
        "ece": compute_ece(confidences, correctness, bins=args.ece_bins) if with_gt else None,
        "brier": compute_brier(confidences, correctness) if with_gt else None,
        "nll": compute_nll(confidences, correctness) if with_gt else None,
        "avg_uncertainty_radius_m": (sum(uncertainty_values) / len(uncertainty_values))
        if uncertainty_values
        else None,
        "radius_m": args.radius_m,
        "top_k": args.top_k,
        "notes": "Uses fusion candidates when available; falls back to retrieval candidates or geo estimate.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
