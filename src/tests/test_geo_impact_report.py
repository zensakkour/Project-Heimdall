from __future__ import annotations

from src.tools.geo_impact_report import compare_impact, to_markdown


def test_compare_impact_marks_improvement_and_regression() -> None:
    baseline = {
        "within_5km_pct": 40.0,
        "mean_km": 12.0,
        "ece": 0.2,
    }
    candidate = {
        "within_5km_pct": 45.0,
        "mean_km": 10.0,
        "ece": 0.25,
    }
    impact = compare_impact(baseline, candidate, metrics=["within_5km_pct", "mean_km", "ece"])
    statuses = {row["metric"]: row["status"] for row in impact["metrics"]}
    assert statuses["within_5km_pct"] == "improved"
    assert statuses["mean_km"] == "improved"
    assert statuses["ece"] == "regressed"
    assert impact["summary"]["net_score"] == 1


def test_to_markdown_contains_table() -> None:
    impact = {
        "summary": {"improved": 1, "regressed": 0, "unchanged": 0, "missing": 0, "net_score": 1},
        "metrics": [
            {
                "metric": "within_10km_pct",
                "direction": "higher",
                "baseline": 50.0,
                "candidate": 55.0,
                "delta": 5.0,
                "delta_pct": 10.0,
                "status": "improved",
            }
        ],
    }
    md = to_markdown(impact)
    assert "| Metric | Direction | Baseline | Candidate | Delta | Delta % | Status |" in md
    assert "within_10km_pct" in md

