"""
Build an impact report between two geo evaluation reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable


HIGHER_IS_BETTER = {
    "top1",
    "topk",
    "within_1km_pct",
    "within_5km_pct",
    "within_10km_pct",
    "within_50km_pct",
    "top5_within_25km_pct",
    "avg_top1_cross_source_support",
    "high_confidence_top1",
    "medium_or_higher_top1",
}

LOWER_IS_BETTER = {
    "mean_km",
    "median_km",
    "p90_km",
    "p95_km",
    "ece",
    "brier",
    "nll",
    "top1_mean_km",
    "top1_median_km",
    "avg_uncertainty_radius_m",
}


def _load_report(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "report" in raw and isinstance(raw["report"], dict):
        return raw["report"]
    if not isinstance(raw, dict):
        raise ValueError(f"invalid report payload: {path}")
    return raw


def _to_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def compare_impact(baseline: dict, candidate: dict, metrics: Iterable[str] | None = None) -> dict:
    keys = list(metrics) if metrics is not None else sorted(set(HIGHER_IS_BETTER | LOWER_IS_BETTER))
    rows = []
    improved = 0
    regressed = 0
    unchanged = 0
    missing = 0
    for metric in keys:
        b = _to_float(baseline.get(metric))
        c = _to_float(candidate.get(metric))
        if b is None or c is None:
            rows.append(
                {
                    "metric": metric,
                    "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                    "baseline": b,
                    "candidate": c,
                    "delta": None,
                    "delta_pct": None,
                    "status": "missing",
                }
            )
            missing += 1
            continue
        delta = c - b
        delta_pct = (delta / abs(b) * 100.0) if abs(b) > 1e-12 else None
        if metric in HIGHER_IS_BETTER:
            if delta > 0:
                status = "improved"
                improved += 1
            elif delta < 0:
                status = "regressed"
                regressed += 1
            else:
                status = "unchanged"
                unchanged += 1
        else:
            if delta < 0:
                status = "improved"
                improved += 1
            elif delta > 0:
                status = "regressed"
                regressed += 1
            else:
                status = "unchanged"
                unchanged += 1

        rows.append(
            {
                "metric": metric,
                "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                "baseline": b,
                "candidate": c,
                "delta": delta,
                "delta_pct": delta_pct,
                "status": status,
            }
        )

    summary = {
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "missing": missing,
        "net_score": improved - regressed,
    }
    return {"summary": summary, "metrics": rows}


def to_markdown(impact: dict, *, title: str = "Geo Quality Impact Report") -> str:
    lines = [f"# {title}", ""]
    summary = impact.get("summary", {})
    lines.append(
        f"- Improved: {summary.get('improved', 0)} | Regressed: {summary.get('regressed', 0)} | "
        f"Unchanged: {summary.get('unchanged', 0)} | Missing: {summary.get('missing', 0)} | "
        f"Net: {summary.get('net_score', 0)}"
    )
    lines.append("")
    lines.append("| Metric | Direction | Baseline | Candidate | Delta | Delta % | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in impact.get("metrics", []):
        delta_pct = row["delta_pct"]
        lines.append(
            "| {metric} | {direction} | {baseline} | {candidate} | {delta} | {delta_pct} | {status} |".format(
                metric=row["metric"],
                direction=row["direction"],
                baseline=_fmt(row["baseline"]),
                candidate=_fmt(row["candidate"]),
                delta=_fmt(row["delta"]),
                delta_pct=_fmt(delta_pct, suffix="%"),
                status=row["status"],
            )
        )
    return "\n".join(lines) + "\n"


def _fmt(value, *, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}{suffix}"
    return f"{value}{suffix}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two geo reports and emit an impact summary.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-json", default="runs/geo_impact.json")
    parser.add_argument("--output-md", default="runs/geo_impact.md")
    args = parser.parse_args(argv)

    baseline = _load_report(Path(args.baseline))
    candidate = _load_report(Path(args.candidate))
    impact = compare_impact(baseline, candidate)
    payload = {
        "baseline": str(Path(args.baseline)),
        "candidate": str(Path(args.candidate)),
        "impact": impact,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(to_markdown(impact), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
