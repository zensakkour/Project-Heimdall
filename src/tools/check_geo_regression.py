"""
Gate geo-eval quality against a committed baseline report.

Usage:
  python -m src.tools.check_geo_regression \
    --baseline docs/eval/geo_eval_baseline.json \
    --candidate docs/eval/geo_eval_current.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HIGHER_IS_BETTER = {
    "within_1km_pct": 2.0,
    "within_5km_pct": 2.0,
    "within_10km_pct": 1.5,
    "within_50km_pct": 1.0,
}

LOWER_IS_BETTER = {
    "mean_km": {"relative": 0.10, "absolute": 1.0},
    "median_km": {"relative": 0.10, "absolute": 0.5},
    "p90_km": {"relative": 0.12, "absolute": 2.0},
    "p95_km": {"relative": 0.12, "absolute": 3.0},
}


def _load_report(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "report" in raw and isinstance(raw["report"], dict):
        return raw["report"]
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid report payload at {path}")
    return raw


def compare_reports(baseline: dict, candidate: dict) -> list[str]:
    regressions: list[str] = []

    base_eval = baseline.get("evaluated")
    cand_eval = candidate.get("evaluated")
    if isinstance(base_eval, (int, float)) and isinstance(cand_eval, (int, float)) and base_eval > 0:
        if cand_eval < base_eval * 0.95:
            regressions.append(
                f"evaluated dropped too much: baseline={base_eval}, candidate={cand_eval}, min={base_eval * 0.95:.2f}"
            )

    for metric, max_drop in HIGHER_IS_BETTER.items():
        b = baseline.get(metric)
        c = candidate.get(metric)
        if b is None or c is None:
            regressions.append(f"{metric}: missing in baseline or candidate")
            continue
        min_allowed = float(b) - max_drop
        if float(c) < min_allowed:
            regressions.append(
                f"{metric} regressed: baseline={b:.3f}, candidate={c:.3f}, min_allowed={min_allowed:.3f}"
            )

    for metric, tol in LOWER_IS_BETTER.items():
        b = baseline.get(metric)
        c = candidate.get(metric)
        if b is None or c is None:
            regressions.append(f"{metric}: missing in baseline or candidate")
            continue
        b = float(b)
        c = float(c)
        allowed_delta = max(float(tol["absolute"]), b * float(tol["relative"]))
        max_allowed = b + allowed_delta
        if c > max_allowed:
            regressions.append(
                f"{metric} regressed: baseline={b:.3f}, candidate={c:.3f}, max_allowed={max_allowed:.3f}"
            )

    return regressions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare geo eval report against baseline.")
    parser.add_argument("--baseline", required=True, help="Baseline JSON path.")
    parser.add_argument("--candidate", required=True, help="Candidate JSON path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    if not baseline_path.exists():
        print(f"baseline file missing: {baseline_path}")
        return 2
    if not candidate_path.exists():
        print(f"candidate file missing: {candidate_path}")
        return 2

    baseline = _load_report(baseline_path)
    candidate = _load_report(candidate_path)
    regressions = compare_reports(baseline, candidate)
    if regressions:
        print("Geo regression gate failed:")
        for item in regressions:
            print(f" - {item}")
        return 1
    print("Geo regression gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

