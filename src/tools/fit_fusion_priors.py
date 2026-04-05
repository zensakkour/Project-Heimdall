"""
Fit source priors for geo fusion from evaluation outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from src.tools.eval_metrics import haversine_m, load_ground_truth, load_results


def _source_from_match_id(match_id: str | None) -> str:
    key = (match_id or "").lower()
    if key.startswith("exif:"):
        return "exif"
    if key == "geoclip" or key.startswith("geoclip:"):
        return "geoclip"
    return "retrieval"


def _retrieval_source_from_match_id(match_id: str | None) -> Optional[str]:
    key = (match_id or "").strip().lower()
    if not key.startswith("retrieval:"):
        return None
    parts = key.split(":")
    # Legacy format is often "retrieval:<item>"; only treat >=3 parts as source-aware.
    if len(parts) < 3:
        return None
    source = parts[1].strip()
    if not source:
        return None
    return f"retrieval:{source}"


def _estimate_source_reliability(
    rows: Iterable[dict],
    ground_truth: Dict[str, object],
    radius_km: float,
) -> Dict[str, dict]:
    radius_m = max(0.1, float(radius_km)) * 1000.0
    stats = {
        "retrieval": {"count": 0, "hits": 0},
        "geoclip": {"count": 0, "hits": 0},
        "exif": {"count": 0, "hits": 0},
    }
    stats["retrieval_by_source"] = {}
    for row in rows:
        image = row.get("image") or row.get("path")
        if not image:
            continue
        gt = ground_truth.get(str(image))
        if gt is None:
            gt = ground_truth.get(Path(str(image)).name)
        if gt is None:
            continue

        fusion = row.get("fusion")
        if isinstance(row.get("result"), dict) and isinstance(row["result"].get("fusion"), dict):
            fusion = row["result"].get("fusion")
        if not isinstance(fusion, dict):
            continue
        items = fusion.get("candidates")
        if not isinstance(items, list):
            continue

        best_by_source: Dict[str, tuple[float, float, float]] = {}
        best_by_retrieval_source: Dict[str, tuple[float, float, float]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            cand = item.get("candidate")
            if not isinstance(cand, dict):
                continue
            lat = cand.get("latitude")
            lon = cand.get("longitude")
            if lat is None or lon is None:
                continue
            score = item.get("posterior_weight", cand.get("retrieval_score", 0.0))
            try:
                score_value = float(score)
                lat_value = float(lat)
                lon_value = float(lon)
            except Exception:
                continue
            source = _source_from_match_id(cand.get("match_id"))
            prev = best_by_source.get(source)
            if prev is None or score_value > prev[2]:
                best_by_source[source] = (lat_value, lon_value, score_value)
            retrieval_source = _retrieval_source_from_match_id(cand.get("match_id"))
            if retrieval_source:
                prev_retrieval = best_by_retrieval_source.get(retrieval_source)
                if prev_retrieval is None or score_value > prev_retrieval[2]:
                    best_by_retrieval_source[retrieval_source] = (lat_value, lon_value, score_value)

        for source, (lat, lon, _score) in best_by_source.items():
            stats[source]["count"] += 1
            dist_m = haversine_m(gt.latitude, gt.longitude, lat, lon)
            if dist_m <= radius_m:
                stats[source]["hits"] += 1
        per_source = stats["retrieval_by_source"]
        if isinstance(per_source, dict):
            for source, (lat, lon, _score) in best_by_retrieval_source.items():
                bucket = per_source.setdefault(source, {"count": 0, "hits": 0})
                bucket["count"] += 1
                dist_m = haversine_m(gt.latitude, gt.longitude, lat, lon)
                if dist_m <= radius_m:
                    bucket["hits"] += 1

    return stats


def _recommended_priors(stats: Dict[str, dict], smoothing: float) -> Dict[str, float]:
    smooth = max(0.0, float(smoothing))
    reliability: Dict[str, float] = {}
    for source in ("retrieval", "geoclip", "exif"):
        item = stats.get(source, {})
        if not isinstance(item, dict):
            item = {}
        count = float(item.get("count", 0))
        hits = float(item.get("hits", 0))
        reliability[source] = (hits + smooth) / (count + 2.0 * smooth) if count > 0.0 else 0.5
    active = [value for source, value in reliability.items() if stats[source]["count"] > 0]
    if not active:
        return {"source_prior_retrieval": 1.0, "source_prior_geoclip": 1.0, "source_prior_exif": 1.0}
    mean_rel = sum(active) / len(active)
    rec = {}
    rec["source_prior_retrieval"] = max(0.25, min(4.0, reliability["retrieval"] / mean_rel))
    rec["source_prior_geoclip"] = max(0.25, min(4.0, reliability["geoclip"] / mean_rel))
    rec["source_prior_exif"] = max(0.25, min(4.0, reliability["exif"] / mean_rel))
    return rec


def _recommended_retrieval_source_priors(
    source_stats: Dict[str, dict],
    *,
    global_retrieval_prior: float,
    smoothing: float,
    min_count: int,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    smooth = max(0.0, float(smoothing))
    threshold = max(1, int(min_count))
    reliability: Dict[str, float] = {}
    for source, item in source_stats.items():
        count = float(item.get("count", 0))
        hits = float(item.get("hits", 0))
        if count < threshold:
            continue
        reliability[source] = (hits + smooth) / (count + 2.0 * smooth) if count > 0.0 else 0.5
    if not reliability:
        return out

    mean_rel = sum(reliability.values()) / len(reliability)
    base = max(0.25, min(4.0, float(global_retrieval_prior)))
    for source, rel in reliability.items():
        scaled = base * (rel / max(1e-6, mean_rel))
        out[source] = max(0.25, min(4.0, scaled))
    return out


def _apply_priors_to_config(
    config_path: Path,
    priors: Dict[str, float],
    retrieval_source_priors: Dict[str, float],
) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    fusion = payload.setdefault("fusion", {})
    if not isinstance(fusion, dict):
        fusion = {}
        payload["fusion"] = fusion
    fusion["source_prior_retrieval"] = float(priors["source_prior_retrieval"])
    fusion["source_prior_geoclip"] = float(priors["source_prior_geoclip"])
    fusion["source_prior_exif"] = float(priors["source_prior_exif"])
    fusion["source_prior_retrieval_by_source"] = {
        key: float(value) for key, value in sorted(retrieval_source_priors.items())
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit source priors for geo fusion.")
    parser.add_argument("--results", required=True, help="JSONL results from pipeline runs.")
    parser.add_argument("--ground-truth", required=True, help="Ground truth CSV/JSON/JSONL.")
    parser.add_argument("--radius-km", type=float, default=25.0, help="Hit radius in km.")
    parser.add_argument("--smoothing", type=float, default=3.0, help="Laplace smoothing pseudo-count.")
    parser.add_argument(
        "--per-source-min-count",
        type=int,
        default=5,
        help="Minimum sample count required to emit a retrieval source-specific prior.",
    )
    parser.add_argument("--apply-config", action="store_true", help="Apply recommended priors to --config.")
    parser.add_argument("--config", default="", help="Config JSON to patch when --apply-config is set.")
    parser.add_argument("--output", default="runs/fusion_priors.json", help="Output JSON.")
    args = parser.parse_args(argv)

    rows = load_results(Path(args.results))
    ground_truth = load_ground_truth(Path(args.ground_truth))
    stats = _estimate_source_reliability(rows, ground_truth, radius_km=args.radius_km)
    priors = _recommended_priors(stats, smoothing=args.smoothing)
    retrieval_source_priors = _recommended_retrieval_source_priors(
        stats.get("retrieval_by_source", {}),
        global_retrieval_prior=priors["source_prior_retrieval"],
        smoothing=args.smoothing,
        min_count=args.per_source_min_count,
    )

    fusion_patch = {
        "source_prior_retrieval": priors["source_prior_retrieval"],
        "source_prior_geoclip": priors["source_prior_geoclip"],
        "source_prior_exif": priors["source_prior_exif"],
    }
    if retrieval_source_priors:
        fusion_patch["source_prior_retrieval_by_source"] = retrieval_source_priors

    applied_config = None
    if args.apply_config:
        config_path = Path(args.config).expanduser() if args.config else Path("src/config/defaults.json")
        if not config_path.exists():
            raise FileNotFoundError(f"config_not_found:{config_path}")
        _apply_priors_to_config(config_path, priors, retrieval_source_priors)
        applied_config = str(config_path)

    payload = {
        "results": str(Path(args.results)),
        "ground_truth": str(Path(args.ground_truth)),
        "radius_km": args.radius_km,
        "smoothing": args.smoothing,
        "per_source_min_count": args.per_source_min_count,
        "source_stats": stats,
        "recommended_priors": priors,
        "recommended_retrieval_source_priors": retrieval_source_priors,
        "applied_config": applied_config,
        "config_patch": {
            "fusion": fusion_patch
        },
        "notes": "Priors are reliability-normalized and clipped to [0.25, 4.0].",
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
