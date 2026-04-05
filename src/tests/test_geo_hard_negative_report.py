from __future__ import annotations

from src.tools.geo_hard_negative_report import build_hard_negative_report


def test_hard_negative_report_outputs_group_metrics() -> None:
    rows = [
        {
            "image": "a.jpg",
            "result": {
                "fusion": {
                    "candidates": [
                        {
                            "candidate": {"latitude": 48.8566, "longitude": 2.3522, "retrieval_score": 0.8},
                            "posterior_weight": 0.8,
                        }
                    ]
                }
            },
        },
        {
            "image": "b.jpg",
            "result": {
                "fusion": {
                    "candidates": [
                        {
                            "candidate": {"latitude": 0.0, "longitude": 0.0, "retrieval_score": 0.8},
                            "posterior_weight": 0.8,
                        }
                    ]
                }
            },
        },
    ]
    gt = {
        "a.jpg": {"latitude": 48.8566, "longitude": 2.3522, "group": "bridge"},
        "b.jpg": {"latitude": 35.6764, "longitude": 139.65, "group": "bridge"},
    }
    report = build_hard_negative_report(rows, gt, top_k=5, radius_km=25.0, hardest_n=5)
    assert report["evaluated"] == 2
    assert "bridge" in report["per_group"]
    assert report["distance_buckets"][">100km"] >= 1
    assert len(report["hardest_samples"]) >= 1

