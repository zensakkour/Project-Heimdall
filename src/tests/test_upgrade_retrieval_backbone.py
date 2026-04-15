from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from src.tools import upgrade_retrieval_backbone as upgrade


def _write_meta(path: Path, rows: list[tuple[str, float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "latitude", "longitude"])
        for rel, lat, lon in rows:
            writer.writerow([rel, lat, lon])


def test_upgrade_retrieval_backbone_patches_config(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        train_images = root / "train_images"
        eval_images = root / "eval_images"
        train_images.mkdir(parents=True, exist_ok=True)
        eval_images.mkdir(parents=True, exist_ok=True)
        (train_images / "a.jpg").write_bytes(b"x")
        (eval_images / "a.jpg").write_bytes(b"x")

        train_meta = root / "train.csv"
        eval_meta = root / "eval.csv"
        _write_meta(train_meta, [("a.jpg", 48.85, 2.35)])
        _write_meta(eval_meta, [("a.jpg", 48.85, 2.35)])

        cfg_path = root / "cfg.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "geolocator": {
                        "retrieval_model_id": "old-model",
                        "retrieval_index_path": "old-index.npz",
                        "retrieval_index_paths": ["other.npz"],
                        "retrieval_index_weights": [1.0],
                        "retrieval_index_model_ids": ["old-model"],
                    }
                }
            ),
            encoding="utf-8",
        )

        out_dir = root / "out"

        def _fake_bench_main(argv):
            output_path = Path(argv[argv.index("--output") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "best_model": "best/model",
                        "models": [
                            {
                                "model_id": "best/model",
                                "status": "ok",
                                "index_path": str(output_path.parent / "indices" / "best_model.npz"),
                                "within_2km_pct": 99.0,
                                "median_km": 1.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return 0

        def _fake_build_index(**kwargs):
            output = Path(kwargs["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"index")
            return 12

        monkeypatch.setattr(upgrade.bench, "main", _fake_bench_main)
        monkeypatch.setattr(upgrade, "build_index", _fake_build_index)

        code = upgrade.main(
            [
                "--train-images-dir",
                str(train_images),
                "--train-metadata",
                str(train_meta),
                "--eval-images-dir",
                str(eval_images),
                "--eval-metadata",
                str(eval_meta),
                "--config",
                str(cfg_path),
                "--output-dir",
                str(out_dir),
                "--model-ids",
                "best/model,other/model",
                "--rank-objective",
                "within_2km_pct",
            ]
        )
        assert code == 0

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        geo = cfg["geolocator"]
        assert geo["retrieval_model_id"] == "best/model"
        assert geo["retrieval_index_path"].endswith("train_images_best_model.npz")
        assert geo["retrieval_index_paths"] == []
        assert geo["retrieval_index_weights"] == []
        assert geo["retrieval_index_model_ids"] == []

        report_path = out_dir / "backbone_upgrade_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["best_model"] == "best/model"
        assert report["config_patched"] is True
