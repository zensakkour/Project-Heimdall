from __future__ import annotations

from src.tools.check_geo_regression import compare_reports


def _baseline() -> dict:
    return {
        "evaluated": 100,
        "mean_km": 10.0,
        "median_km": 7.0,
        "p90_km": 20.0,
        "p95_km": 30.0,
        "within_1km_pct": 10.0,
        "within_5km_pct": 35.0,
        "within_10km_pct": 60.0,
        "within_50km_pct": 95.0,
    }


def test_compare_reports_passes_when_within_threshold() -> None:
    baseline = _baseline()
    candidate = dict(baseline)
    candidate["mean_km"] = 10.8
    candidate["within_5km_pct"] = 34.0
    regressions = compare_reports(baseline, candidate)
    assert regressions == []


def test_compare_reports_detects_regression() -> None:
    baseline = _baseline()
    candidate = dict(baseline)
    candidate["mean_km"] = 15.0
    candidate["within_1km_pct"] = 6.0
    regressions = compare_reports(baseline, candidate)
    assert regressions
    assert any("mean_km" in item for item in regressions)
    assert any("within_1km_pct" in item for item in regressions)

