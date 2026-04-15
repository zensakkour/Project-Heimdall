from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from src.core.logic.types import GeoCandidate
from src.tools import benchmark_geo_backbones as bench


def _write_meta(path: Path, rows: list[tuple[str, float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "latitude", "longitude"])
        for rel, lat, lon in rows:
            writer.writerow([rel, lat, lon])


def test_benchmark_geo_backbones_runs_with_mock_provider(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        train_images = root / "train_images"
        eval_images = root / "eval_images"
        train_images.mkdir(parents=True, exist_ok=True)
        eval_images.mkdir(parents=True, exist_ok=True)

        for name in ("a.jpg", "b.jpg", "c.jpg"):
            (train_images / name).write_bytes(b"x")
            (eval_images / name).write_bytes(b"x")

        train_meta = root / "train.csv"
        eval_meta = root / "eval.csv"
        _write_meta(
            train_meta,
            [
                ("a.jpg", 48.85, 2.35),
                ("b.jpg", 48.86, 2.36),
                ("c.jpg", 48.87, 2.37),
            ],
        )
        _write_meta(
            eval_meta,
            [
                ("a.jpg", 48.85, 2.35),
                ("b.jpg", 48.86, 2.36),
            ],
        )
        output = root / "out" / "bench.json"

        def _fake_build_index(**kwargs):
            Path(kwargs["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(kwargs["output"]).write_bytes(b"index")
            return 3

        class _FakeProvider:
            def __init__(self, *args, **kwargs):
                self.last_error = None

            def candidates(self, _image_path: str):
                return [GeoCandidate(latitude=48.85, longitude=2.35, retrieval_score=0.9, match_id="x")]

        monkeypatch.setattr(bench, "build_index", _fake_build_index)
        monkeypatch.setattr(bench, "GeoRetrievalProvider", _FakeProvider)

        code = bench.main(
            [
                "--train-images-dir",
                str(train_images),
                "--train-metadata",
                str(train_meta),
                "--eval-images-dir",
                str(eval_images),
                "--eval-metadata",
                str(eval_meta),
                "--model-ids",
                "m1,m2",
                "--train-limit",
                "3",
                "--eval-limit",
                "2",
                "--output",
                str(output),
            ]
        )
        assert code == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["best_model"] in {"m1", "m2"}
        assert len(payload["models"]) == 2
        assert all(item["status"] == "ok" for item in payload["models"])


def test_benchmark_geo_backbones_supports_model_preset(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        train_images = root / "train_images"
        eval_images = root / "eval_images"
        train_images.mkdir(parents=True, exist_ok=True)
        eval_images.mkdir(parents=True, exist_ok=True)

        for name in ("a.jpg", "b.jpg", "c.jpg"):
            (train_images / name).write_bytes(b"x")
            (eval_images / name).write_bytes(b"x")

        train_meta = root / "train.csv"
        eval_meta = root / "eval.csv"
        _write_meta(train_meta, [("a.jpg", 48.85, 2.35), ("b.jpg", 48.86, 2.36), ("c.jpg", 48.87, 2.37)])
        _write_meta(eval_meta, [("a.jpg", 48.85, 2.35), ("b.jpg", 48.86, 2.36)])
        output = root / "out" / "bench.json"

        def _fake_build_index(**kwargs):
            Path(kwargs["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(kwargs["output"]).write_bytes(b"index")
            return 3

        class _FakeProvider:
            def __init__(self, *args, **kwargs):
                self.last_error = None

            def candidates(self, _image_path: str):
                return [GeoCandidate(latitude=48.85, longitude=2.35, retrieval_score=0.9, match_id="x")]

        monkeypatch.setattr(bench, "build_index", _fake_build_index)
        monkeypatch.setattr(bench, "GeoRetrievalProvider", _FakeProvider)

        code = bench.main(
            [
                "--train-images-dir",
                str(train_images),
                "--train-metadata",
                str(train_meta),
                "--eval-images-dir",
                str(eval_images),
                "--eval-metadata",
                str(eval_meta),
                "--model-preset",
                "legacy_clip_siglip",
                "--train-limit",
                "3",
                "--eval-limit",
                "2",
                "--output",
                str(output),
            ]
        )
        assert code == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["model_preset"] == "legacy_clip_siglip"
        assert payload["model_ids"] == ["openai/clip-vit-large-patch14", "google/siglip-base-patch16-224"]


def test_benchmark_geo_backbones_ranks_by_objective(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        train_images = root / "train_images"
        eval_images = root / "eval_images"
        train_images.mkdir(parents=True, exist_ok=True)
        eval_images.mkdir(parents=True, exist_ok=True)

        for name in ("a.jpg", "b.jpg"):
            (train_images / name).write_bytes(b"x")
            (eval_images / name).write_bytes(b"x")

        train_meta = root / "train.csv"
        eval_meta = root / "eval.csv"
        _write_meta(train_meta, [("a.jpg", 48.85, 2.35), ("b.jpg", 48.86, 2.36)])
        _write_meta(eval_meta, [("a.jpg", 48.85, 2.35), ("b.jpg", 48.86, 2.36)])
        output = root / "out" / "bench.json"

        def _fake_build_index(**kwargs):
            Path(kwargs["output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(kwargs["output"]).write_bytes(b"index")
            return 2

        class _FakeProvider:
            def __init__(self, *args, **kwargs):
                self.last_error = None
                self.model_id = kwargs.get("model_id")

            def candidates(self, _image_path: str):
                return [GeoCandidate(latitude=48.85, longitude=2.35, retrieval_score=0.9, match_id="x")]

        def _fake_eval(provider, rows, images_dir):
            if getattr(provider, "model_id", "") == "m2":
                return {
                    "evaluated": 2,
                    "missing_files": 0,
                    "null_predictions": 0,
                    "provider_errors": {},
                    "retrieval_score_mean": 0.9,
                    "retrieval_score_min": 0.9,
                    "retrieval_score_max": 0.9,
                    "mean_km": 3.5,
                    "median_km": 3.0,
                    "p90_km": 3.0,
                    "within_1km_pct": 30.0,
                    "within_2km_pct": 80.0,
                    "within_5km_pct": 100.0,
                    "within_10km_pct": 100.0,
                    "within_50km_pct": 100.0,
                }
            return {
                "evaluated": 2,
                "missing_files": 0,
                "null_predictions": 0,
                "provider_errors": {},
                "retrieval_score_mean": 0.9,
                "retrieval_score_min": 0.9,
                "retrieval_score_max": 0.9,
                "mean_km": 2.0,
                "median_km": 1.5,
                "p90_km": 1.5,
                "within_1km_pct": 40.0,
                "within_2km_pct": 50.0,
                "within_5km_pct": 100.0,
                "within_10km_pct": 100.0,
                "within_50km_pct": 100.0,
            }

        monkeypatch.setattr(bench, "build_index", _fake_build_index)
        monkeypatch.setattr(bench, "GeoRetrievalProvider", _FakeProvider)
        monkeypatch.setattr(bench, "_evaluate_top1", _fake_eval)

        code = bench.main(
            [
                "--train-images-dir",
                str(train_images),
                "--train-metadata",
                str(train_meta),
                "--eval-images-dir",
                str(eval_images),
                "--eval-metadata",
                str(eval_meta),
                "--model-ids",
                "m1,m2",
                "--rank-objective",
                "within_2km_pct",
                "--output",
                str(output),
            ]
        )
        assert code == 0
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["best_model"] == "m2"
        assert payload["best_model_by_median_km"] == "m1"
