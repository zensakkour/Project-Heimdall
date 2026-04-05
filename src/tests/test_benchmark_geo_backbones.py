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
