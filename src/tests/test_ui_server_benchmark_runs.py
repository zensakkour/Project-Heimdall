from __future__ import annotations

import json
from pathlib import Path
import tempfile

from fastapi.testclient import TestClient

from src.tools import ui_server


def _make_run(
    run_id: str,
    generated_at: str,
    best_model: str,
    realistic_mean: float,
    realistic_median: float,
    realistic_w10: float,
) -> dict:
    return {
        "run_id": run_id,
        "generated_at": generated_at,
        "geo_scenarios": [
            {
                "scenario": "realistic_single",
                "mean_km": realistic_mean,
                "median_km": realistic_median,
                "within_5km_pct": 20.0,
                "within_10km_pct": realistic_w10,
            }
        ],
        "backbone_benchmark": {
            "best_model": best_model,
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


def test_benchmark_runs_list_and_load(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runs_dir = Path(tmp) / "benchmark_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        newer = {
            "run_id": "20260405T210101_000001Z",
            "generated_at": "2026-04-05T21:01:01Z",
            "geo_scenarios": [],
            "backbone_benchmark": {
                "best_model": "google/siglip-base-patch16-224",
                "models": [
                    {"model_id": "openai/clip-vit-large-patch14"},
                    {"model_id": "google/siglip-base-patch16-224"},
                ],
            },
        }
        older = {
            "run_id": "20260405T205959_000001Z",
            "generated_at": "2026-04-05T20:59:59Z",
            "geo_scenarios": [],
            "backbone_benchmark": {
                "best_model": "openai/clip-vit-large-patch14",
                "models": [{"model_id": "openai/clip-vit-large-patch14"}],
            },
        }
        (runs_dir / f"{newer['run_id']}.json").write_text(json.dumps(newer), encoding="utf-8")
        (runs_dir / f"{older['run_id']}.json").write_text(json.dumps(older), encoding="utf-8")

        monkeypatch.setattr(ui_server, "_benchmark_runs_dir", lambda: runs_dir)
        client = TestClient(ui_server.app)

        listed = client.get("/eval/benchmarks/runs")
        assert listed.status_code == 200
        rows = listed.json().get("runs", [])
        assert len(rows) == 2
        assert rows[0]["run_id"] == newer["run_id"]
        assert rows[0]["best_model"] == "google/siglip-base-patch16-224"
        assert rows[0]["model_count"] == 2

        loaded = client.get(f"/eval/benchmarks/runs/{newer['run_id']}")
        assert loaded.status_code == 200
        payload = loaded.json()
        assert payload["run_id"] == newer["run_id"]
        assert payload["backbone_benchmark"]["best_model"] == "google/siglip-base-patch16-224"


def test_benchmark_run_rejects_unsafe_id() -> None:
    client = TestClient(ui_server.app)
    res = client.get("/eval/benchmarks/runs/run..bad")
    assert res.status_code == 400
    assert res.json().get("error") == "invalid_run_id"


def test_compare_benchmark_runs_and_append_progress(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs_dir = root / "benchmark_runs"
        compares_dir = root / "benchmark_compares"
        runs_dir.mkdir(parents=True, exist_ok=True)
        compares_dir.mkdir(parents=True, exist_ok=True)
        progress_path = root / "PROGRESS.md"
        progress_path.write_text("# Progress\n", encoding="utf-8")

        baseline = _make_run(
            run_id="20260405T200000_000001Z",
            generated_at="2026-04-05T20:00:00Z",
            best_model="openai/clip-vit-large-patch14",
            realistic_mean=20.0,
            realistic_median=12.0,
            realistic_w10=40.0,
        )
        candidate = _make_run(
            run_id="20260405T210000_000001Z",
            generated_at="2026-04-05T21:00:00Z",
            best_model="google/siglip-base-patch16-224",
            realistic_mean=18.0,
            realistic_median=10.0,
            realistic_w10=46.0,
        )
        (runs_dir / f"{baseline['run_id']}.json").write_text(json.dumps(baseline), encoding="utf-8")
        (runs_dir / f"{candidate['run_id']}.json").write_text(json.dumps(candidate), encoding="utf-8")

        monkeypatch.setattr(ui_server, "_benchmark_runs_dir", lambda: runs_dir)
        monkeypatch.setattr(ui_server, "_benchmark_compares_dir", lambda: compares_dir)
        monkeypatch.setattr(ui_server, "_progress_log_path", lambda: progress_path)
        client = TestClient(ui_server.app)

        res = client.post(
            "/eval/benchmarks/compare",
            params={
                "baseline_run_id": baseline["run_id"],
                "candidate_run_id": candidate["run_id"],
                "append_progress": "1",
            },
        )
        assert res.status_code == 200
        payload = res.json()
        assert payload["baseline_run_id"] == baseline["run_id"]
        assert payload["candidate_run_id"] == candidate["run_id"]
        assert payload["progress_appended"] is True
        assert "progress_md_snippet" in payload

        realistic_rows = [
            row for row in payload.get("scenario_deltas", []) if row.get("scenario") == "realistic_single"
        ]
        assert realistic_rows
        realistic = realistic_rows[0]
        assert realistic["delta"]["mean_km"] == -2.0
        assert realistic["delta"]["within_10km_pct"] == 6.0

        compare_path = Path(payload["compare_path"])
        assert compare_path.exists()
        progress_text = progress_path.read_text(encoding="utf-8")
        assert baseline["run_id"] in progress_text
        assert candidate["run_id"] in progress_text
