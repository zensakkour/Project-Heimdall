"""
Generate a hard-negative benchmark report from geo evaluation outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.tools.eval_metrics import extract_candidates, haversine_m, load_results


def _load_ground_truth_with_groups(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_gt_csv(path)
    if suffix in {".json", ".jsonl"}:
        return _load_gt_json(path)
    raise ValueError("ground truth must be .csv, .json, or .jsonl")


def _load_gt_csv(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image = row.get("image") or row.get("path")
            lat = row.get("latitude") or row.get("lat")
            lon = row.get("longitude") or row.get("lon")
            if not image or lat is None or lon is None:
                continue
            group = row.get("hard_negative_group") or row.get("group") or row.get("scene_group")
            out[str(image)] = {
                "latitude": float(lat),
                "longitude": float(lon),
                "group": str(group) if group else "ungrouped",
            }
    return out


def _normalize_gt_item(item: dict) -> Optional[dict]:
    image = item.get("image") or item.get("path")
    lat = item.get("latitude") or item.get("lat")
    lon = item.get("longitude") or item.get("lon")
    if not image or lat is None or lon is None:
        return None
    group = item.get("hard_negative_group") or item.get("group") or item.get("scene_group")
    return {
        "image": str(image),
        "latitude": float(lat),
        "longitude": float(lon),
        "group": str(group) if group else "ungrouped",
    }


def _load_gt_json(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                continue
            norm = _normalize_gt_item(item)
            if norm is None:
                continue
            out[norm["image"]] = {
                "latitude": norm["latitude"],
                "longitude": norm["longitude"],
                "group": norm["group"],
            }
        return out

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            norm = _normalize_gt_item(item)
            if norm is None:
                continue
            out[norm["image"]] = {
                "latitude": norm["latitude"],
                "longitude": norm["longitude"],
                "group": norm["group"],
            }
    elif isinstance(payload, dict):
        for image, value in payload.items():
            if not isinstance(value, dict):
                continue
            lat = value.get("latitude") or value.get("lat")
            lon = value.get("longitude") or value.get("lon")
            if lat is None or lon is None:
                continue
            group = value.get("hard_negative_group") or value.get("group") or value.get("scene_group")
            out[str(image)] = {
                "latitude": float(lat),
                "longitude": float(lon),
                "group": str(group) if group else "ungrouped",
            }
    return out


def _distance_bucket(km: float) -> str:
    if km <= 1.0:
        return "0-1km"
    if km <= 5.0:
        return "1-5km"
    if km <= 25.0:
        return "5-25km"
    if km <= 100.0:
        return "25-100km"
    return ">100km"


def _topk_hit(candidates: List[object], gt_lat: float, gt_lon: float, radius_km: float, top_k: int) -> bool:
    limit = max(1, int(top_k))
    for cand in sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]:
        if haversine_m(gt_lat, gt_lon, cand.latitude, cand.longitude) <= radius_km * 1000.0:
            return True
    return False


def build_hard_negative_report(
    rows: Iterable[dict],
    ground_truth: Dict[str, dict],
    *,
    top_k: int = 5,
    radius_km: float = 25.0,
    hardest_n: int = 25,
) -> dict:
    distances_km: List[float] = []
    topk_hits = 0
    evaluated = 0
    bucket_counts = {"0-1km": 0, "1-5km": 0, "5-25km": 0, "25-100km": 0, ">100km": 0}
    hardest = []
    per_group: Dict[str, dict] = {}

    for row in rows:
        image = row.get("image") or row.get("path")
        if not image:
            continue
        gt = ground_truth.get(str(image))
        if gt is None:
            gt = ground_truth.get(Path(str(image)).name)
        if gt is None:
            continue
        candidates = extract_candidates(row)
        if not candidates:
            continue

        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        top = ranked[0]
        dist_km = haversine_m(gt["latitude"], gt["longitude"], top.latitude, top.longitude) / 1000.0
        hit = _topk_hit(ranked, gt["latitude"], gt["longitude"], radius_km=radius_km, top_k=top_k)
        group = str(gt.get("group") or "ungrouped")

        evaluated += 1
        distances_km.append(dist_km)
        if hit:
            topk_hits += 1
        bucket_counts[_distance_bucket(dist_km)] += 1
        hardest.append(
            {
                "image": str(image),
                "group": group,
                "top1_distance_km": dist_km,
                "pred_latitude": top.latitude,
                "pred_longitude": top.longitude,
                "gt_latitude": gt["latitude"],
                "gt_longitude": gt["longitude"],
                "top1_score": top.score,
            }
        )

        group_stats = per_group.setdefault(
            group,
            {"count": 0, "top1_distances_km": [], "topk_hits": 0},
        )
        group_stats["count"] += 1
        group_stats["top1_distances_km"].append(dist_km)
        if hit:
            group_stats["topk_hits"] += 1

    hardest.sort(key=lambda item: item["top1_distance_km"], reverse=True)
    group_report = {}
    for group, item in per_group.items():
        vals = item["top1_distances_km"]
        group_report[group] = {
            "count": item["count"],
            "top1_mean_km": (sum(vals) / len(vals)) if vals else None,
            "top1_median_km": statistics.median(vals) if vals else None,
            "topk_within_radius_pct": 100.0 * item["topk_hits"] / item["count"] if item["count"] else 0.0,
        }

    return {
        "evaluated": evaluated,
        "top1_mean_km": (sum(distances_km) / len(distances_km)) if distances_km else None,
        "top1_median_km": statistics.median(distances_km) if distances_km else None,
        "top5_within_25km_pct": (100.0 * topk_hits / evaluated) if evaluated else 0.0,
        "distance_buckets": bucket_counts,
        "per_group": group_report,
        "hardest_samples": hardest[: max(1, int(hardest_n))],
        "top_k": int(top_k),
        "radius_km": float(radius_km),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create hard-negative geo benchmark report.")
    parser.add_argument("--results", required=True, help="JSONL results.")
    parser.add_argument("--ground-truth", required=True, help="CSV/JSON/JSONL with optional group labels.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--radius-km", type=float, default=25.0)
    parser.add_argument("--hardest-n", type=int, default=25)
    parser.add_argument("--output", default="runs/hard_negative_report.json")
    args = parser.parse_args(argv)

    rows = load_results(Path(args.results))
    gt = _load_ground_truth_with_groups(Path(args.ground_truth))
    report = build_hard_negative_report(
        rows,
        gt,
        top_k=args.top_k,
        radius_km=args.radius_km,
        hardest_n=args.hardest_n,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

