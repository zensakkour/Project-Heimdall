"""
Fit confidence calibration and confidence-tier thresholds from geo eval outputs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

from src.tools.eval_metrics import (
    compute_brier,
    compute_ece,
    compute_nll,
    extract_candidates,
    haversine_m,
    load_ground_truth,
    load_results,
)


def _extract_top1_confidence(row: dict) -> float | None:
    fusion = row.get("fusion")
    if isinstance(row.get("result"), dict) and isinstance(row["result"].get("fusion"), dict):
        fusion = row["result"].get("fusion")
    if isinstance(fusion, dict):
        top1 = fusion.get("top1_posterior")
        if isinstance(top1, (int, float)):
            return float(top1)

    candidates = extract_candidates(row)
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item.score)
    return float(best.score)


def _top1_distance_m(row: dict, gt_lat: float, gt_lon: float) -> float | None:
    candidates = extract_candidates(row)
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item.score)
    return haversine_m(gt_lat, gt_lon, best.latitude, best.longitude)


def calibrate_probability(prob: float, scale: float, bias: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(prob)))
    logit = math.log(p / (1.0 - p))
    z = float(scale) * logit + float(bias)
    out = 1.0 / (1.0 + math.exp(-z))
    return max(0.0, min(1.0, out))


def find_best_logit_calibration(
    confidences: Iterable[float],
    correctness: Iterable[int],
    *,
    scale_values: Iterable[float],
    bias_values: Iterable[float],
) -> dict:
    conf = list(confidences)
    corr = list(correctness)
    if not conf:
        return {
            "scale": 1.0,
            "bias": 0.0,
            "nll": None,
            "ece": None,
            "brier": None,
            "calibrated_confidences": [],
        }

    best = None
    for scale in scale_values:
        for bias in bias_values:
            calibrated = [calibrate_probability(c, scale, bias) for c in conf]
            nll = compute_nll(calibrated, corr)
            if best is None or nll < best["nll"]:
                best = {
                    "scale": float(scale),
                    "bias": float(bias),
                    "nll": float(nll),
                    "ece": float(compute_ece(calibrated, corr)),
                    "brier": float(compute_brier(calibrated, corr)),
                    "calibrated_confidences": calibrated,
                }
    return best


def _recommend_threshold(
    confidences: List[float],
    correctness: List[int],
    *,
    target_precision: float,
    min_support: int,
) -> float | None:
    if not confidences:
        return None
    pairs = sorted(zip(confidences, correctness), key=lambda item: item[0], reverse=True)
    best = None
    for threshold in sorted({round(c, 3) for c in confidences}, reverse=True):
        selected = [item for item in pairs if item[0] >= threshold]
        if len(selected) < min_support:
            continue
        precision = sum(int(y) for _, y in selected) / len(selected)
        if precision >= target_precision:
            best = threshold
            break
    return best


def _build_fusion_calibration_patch(
    scale: float,
    bias: float,
    high_threshold: float | None,
    medium_threshold: float | None,
) -> dict:
    high = float(high_threshold) if high_threshold is not None else 0.72
    medium = float(medium_threshold) if medium_threshold is not None else 0.46
    # Keep tiering monotonic even on noisy/small calibration sets.
    if medium > high:
        medium = high
    return {
        "confidence_calibration_logit_scale": float(scale),
        "confidence_calibration_logit_bias": float(bias),
        "confidence_high_threshold": high,
        "confidence_medium_threshold": medium,
    }


def _apply_calibration_to_config(config_path: Path, fusion_patch: dict) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    fusion = payload.setdefault("fusion", {})
    if not isinstance(fusion, dict):
        fusion = {}
        payload["fusion"] = fusion
    for key, value in fusion_patch.items():
        fusion[key] = value
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit confidence calibration for geo fusion top-1 posterior.")
    parser.add_argument("--results", required=True, help="JSONL results from pipeline runs.")
    parser.add_argument("--ground-truth", required=True, help="Ground truth CSV/JSON/JSONL.")
    parser.add_argument("--radius-km", type=float, default=25.0, help="Hit radius in km.")
    parser.add_argument("--ece-bins", type=int, default=10, help="ECE bins.")
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--scale-steps", type=int, default=31)
    parser.add_argument("--bias-min", type=float, default=-1.0)
    parser.add_argument("--bias-max", type=float, default=1.0)
    parser.add_argument("--bias-steps", type=int, default=41)
    parser.add_argument("--target-high-precision", type=float, default=0.80)
    parser.add_argument("--target-medium-precision", type=float, default=0.60)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--apply-config", action="store_true", help="Apply recommended calibration patch to --config.")
    parser.add_argument("--config", default="", help="Config JSON to patch when --apply-config is set.")
    parser.add_argument("--output", default="runs/confidence_calibration.json", help="Output JSON.")
    args = parser.parse_args(argv)

    rows = load_results(Path(args.results))
    ground_truth = load_ground_truth(Path(args.ground_truth))
    radius_m = max(0.1, float(args.radius_km)) * 1000.0

    confidences: List[float] = []
    correctness: List[int] = []
    for row in rows:
        image = row.get("image") or row.get("path")
        if not image:
            continue
        gt = ground_truth.get(str(image))
        if gt is None:
            gt = ground_truth.get(Path(str(image)).name)
        if gt is None:
            continue
        conf = _extract_top1_confidence(row)
        dist = _top1_distance_m(row, gt.latitude, gt.longitude)
        if conf is None or dist is None:
            continue
        confidences.append(max(0.0, min(1.0, conf)))
        correctness.append(1 if dist <= radius_m else 0)

    scale_values = [
        args.scale_min + (args.scale_max - args.scale_min) * i / max(1, args.scale_steps - 1)
        for i in range(max(1, args.scale_steps))
    ]
    bias_values = [
        args.bias_min + (args.bias_max - args.bias_min) * i / max(1, args.bias_steps - 1)
        for i in range(max(1, args.bias_steps))
    ]
    best = find_best_logit_calibration(
        confidences,
        correctness,
        scale_values=scale_values,
        bias_values=bias_values,
    )
    calibrated = best["calibrated_confidences"]
    high_thr = _recommend_threshold(
        calibrated,
        correctness,
        target_precision=max(0.0, min(1.0, args.target_high_precision)),
        min_support=max(1, int(args.min_support)),
    )
    medium_thr = _recommend_threshold(
        calibrated,
        correctness,
        target_precision=max(0.0, min(1.0, args.target_medium_precision)),
        min_support=max(1, int(args.min_support)),
    )
    fusion_patch = _build_fusion_calibration_patch(best["scale"], best["bias"], high_thr, medium_thr)

    applied_config = None
    if args.apply_config:
        config_path = Path(args.config).expanduser() if args.config else Path("src/config/defaults.json")
        if not config_path.exists():
            raise FileNotFoundError(f"config_not_found:{config_path}")
        _apply_calibration_to_config(config_path, fusion_patch)
        applied_config = str(config_path)

    payload = {
        "results": str(Path(args.results)),
        "ground_truth": str(Path(args.ground_truth)),
        "samples": len(confidences),
        "radius_km": args.radius_km,
        "raw_metrics": {
            "ece": compute_ece(confidences, correctness, bins=args.ece_bins) if confidences else None,
            "brier": compute_brier(confidences, correctness) if confidences else None,
            "nll": compute_nll(confidences, correctness) if confidences else None,
        },
        "best_logit_calibration": {
            "scale": best["scale"],
            "bias": best["bias"],
            "ece": best["ece"],
            "brier": best["brier"],
            "nll": best["nll"],
        },
        "recommended_thresholds": {
            "confidence_high_threshold": fusion_patch["confidence_high_threshold"],
            "confidence_medium_threshold": fusion_patch["confidence_medium_threshold"],
            "target_high_precision": args.target_high_precision,
            "target_medium_precision": args.target_medium_precision,
            "min_support": args.min_support,
        },
        "applied_config": applied_config,
        "config_patch": {"fusion": fusion_patch},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
