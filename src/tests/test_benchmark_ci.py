from __future__ import annotations

from pathlib import Path
import tempfile

from src.tools.benchmark_ci import compare_runs, evaluate_policy, promote_baseline, render_pr_summary


def _run_payload(run_id: str, realistic_mean: float, realistic_w10: float) -> dict:
    return {
        "run_id": run_id,
        "generated_at": "2026-04-05T20:00:00Z",
        "profile": "core",
        "geo_scenarios": [
            {
                "scenario": "realistic_single",
                "mean_km": realistic_mean,
                "median_km": 10.0,
                "within_5km_pct": 20.0,
                "within_10km_pct": realistic_w10,
            },
            {
                "scenario": "candidate_multi",
                "mean_km": realistic_mean + 1.0,
                "median_km": 11.0,
                "within_5km_pct": 18.0,
                "within_10km_pct": realistic_w10 - 2.0,
            },
        ],
        "backbone_benchmark": {
            "best_model": "google/siglip-base-patch16-224",
            "models": [
                {
                    "model_id": "openai/clip-vit-large-patch14",
                    "mean_km": 27.0,
                    "median_km": 18.0,
                    "within_5km_pct": 20.0,
                    "within_10km_pct": 36.0,
                },
                {
                    "model_id": "google/siglip-base-patch16-224",
                    "mean_km": 26.0,
                    "median_km": 17.0,
                    "within_5km_pct": 15.0,
                    "within_10km_pct": 33.0,
                },
            ],
        },
    }


def test_compare_runs_builds_expected_deltas() -> None:
    baseline = _run_payload("baseline", realistic_mean=20.0, realistic_w10=40.0)
    candidate = _run_payload("candidate", realistic_mean=18.5, realistic_w10=44.0)
    compare = compare_runs(baseline, candidate)

    realistic = [row for row in compare["scenario_deltas"] if row["scenario"] == "realistic_single"][0]
    assert realistic["delta"]["mean_km"] == -1.5
    assert realistic["delta"]["within_10km_pct"] == 4.0


def test_evaluate_policy_detects_regression() -> None:
    baseline = _run_payload("baseline", realistic_mean=20.0, realistic_w10=40.0)
    candidate = _run_payload("candidate", realistic_mean=21.0, realistic_w10=35.0)
    rules = [
        {
            "metric": "scenario.realistic_single.mean_km",
            "direction": "lower",
            "max_regression": 0.75,
        },
        {
            "metric": "scenario.realistic_single.within_10km_pct",
            "direction": "higher",
            "max_regression": 2.0,
        },
    ]
    result = evaluate_policy(baseline, candidate, rules)
    assert result["passed"] is False
    assert result["failures"] == 2


def test_promote_baseline_updates_contract_and_summary() -> None:
    payload = _run_payload("20260405T220000_000001Z", realistic_mean=18.0, realistic_w10=45.0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        contract_path = root / "baseline.json"
        summary_path = root / "baseline_summary.json"
        promote_baseline(
            run_payload=payload,
            profile="core",
            baseline_contract_path=contract_path,
            baseline_summary_path=summary_path,
            commit_sha="abc123",
        )
        contract_text = contract_path.read_text(encoding="utf-8")
        summary_text = summary_path.read_text(encoding="utf-8")
        assert "20260405T220000_000001Z" in contract_text
        assert "abc123" in contract_text
        assert "scenario.realistic_single.mean_km" in summary_text


def test_render_pr_summary_contains_policy() -> None:
    candidate = _run_payload("candidate", realistic_mean=18.0, realistic_w10=44.0)
    summary = render_pr_summary(
        candidate=candidate,
        compare=None,
        policy_result={"passed": True},
        baseline_contract={"baseline_run_id": "baseline"},
    )
    assert "## Benchmark Summary" in summary
    assert "Policy: **PASS**" in summary
