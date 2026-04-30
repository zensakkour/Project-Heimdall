from __future__ import annotations

from src.tools.eval_realistic_crossview import summarize_samples


def test_summarize_samples_computes_distance_metrics_and_recall() -> None:
    samples = [
        {
            "distance_km": 0.2,
            "top_candidates": [
                {"distance_km": 0.2},
                {"distance_km": 0.6},
            ],
        },
        {
            "distance_km": 1.5,
            "top_candidates": [
                {"distance_km": 1.5},
                {"distance_km": 0.3},
                {"distance_km": 0.2},
            ],
        },
    ]

    summary = summarize_samples(samples)
    assert summary["evaluated"] == 2
    assert summary["within_500m_pct"] == 50.0
    assert summary["within_2km_pct"] == 100.0
    assert summary["top1_recall_500m_pct"] == 50.0
    assert summary["top5_recall_500m_pct"] == 100.0


def test_summarize_samples_handles_empty_input() -> None:
    summary = summarize_samples([])
    assert summary["evaluated"] == 0
    assert summary["mean_km"] is None
    assert summary["top10_recall_1km_pct"] == 0.0
