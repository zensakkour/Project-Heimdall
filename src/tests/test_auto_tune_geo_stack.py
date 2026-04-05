from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.tools import auto_tune_geo_stack


def _write_basic_files(root: Path) -> tuple[Path, Path, Path]:
    config = root / "cfg.json"
    config.write_text(json.dumps({"geolocator": {}, "fusion": {}}), encoding="utf-8")
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata = root / "metadata.csv"
    metadata.write_text("path,latitude,longitude\nx.jpg,48.8566,2.3522\n", encoding="utf-8")
    return config, images_dir, metadata


def test_auto_tune_uses_existing_results_and_runs_all_steps(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config, images_dir, metadata = _write_basic_files(root)
        results = root / "results.jsonl"
        results.write_text("", encoding="utf-8")
        output_dir = root / "out"

        calls = []

        def _ok(argv):
            calls.append(argv)
            return 0

        monkeypatch.setattr(auto_tune_geo_stack, "_run_tune_retrieval", _ok)
        monkeypatch.setattr(auto_tune_geo_stack, "_run_fit_priors", _ok)
        monkeypatch.setattr(auto_tune_geo_stack, "_run_fit_calibration", _ok)

        code = auto_tune_geo_stack.main(
            [
                "--config",
                str(config),
                "--images-dir",
                str(images_dir),
                "--metadata",
                str(metadata),
                "--results",
                str(results),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert code == 0
        assert len(calls) == 3
        summary = json.loads((output_dir / "auto_tune_summary.json").read_text(encoding="utf-8"))
        statuses = {item["name"]: item["status"] for item in summary["steps"]}
        assert statuses["tune_retrieval_geo"] == "ok"
        assert statuses["fit_fusion_priors"] == "ok"
        assert statuses["fit_confidence_calibration"] == "ok"
        markdown = (output_dir / "auto_tune_summary.md").read_text(encoding="utf-8")
        assert "| tune_retrieval_geo | ok |" in markdown


def test_auto_tune_skips_fusion_steps_when_results_missing(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config, images_dir, metadata = _write_basic_files(root)
        output_dir = root / "out"

        monkeypatch.setattr(auto_tune_geo_stack, "_run_tune_retrieval", lambda _argv: 0)
        monkeypatch.setattr(auto_tune_geo_stack, "_run_fit_priors", lambda _argv: 0)
        monkeypatch.setattr(auto_tune_geo_stack, "_run_fit_calibration", lambda _argv: 0)

        code = auto_tune_geo_stack.main(
            [
                "--config",
                str(config),
                "--images-dir",
                str(images_dir),
                "--metadata",
                str(metadata),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert code == 0
        summary = json.loads((output_dir / "auto_tune_summary.json").read_text(encoding="utf-8"))
        statuses = {item["name"]: item["status"] for item in summary["steps"]}
        assert statuses["fit_fusion_priors"] == "skipped"
        assert statuses["fit_confidence_calibration"] == "skipped"


def test_auto_tune_can_generate_results_then_fit(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config, images_dir, metadata = _write_basic_files(root)
        output_dir = root / "out"

        calls = {"priors": 0, "calibration": 0}

        monkeypatch.setattr(auto_tune_geo_stack, "_run_tune_retrieval", lambda _argv: 0)

        def _generate(**kwargs):
            Path(kwargs["output_path"]).write_text("", encoding="utf-8")
            return 3

        monkeypatch.setattr(auto_tune_geo_stack, "_generate_results_jsonl", _generate)

        def _priors(_argv):
            calls["priors"] += 1
            return 0

        def _cal(_argv):
            calls["calibration"] += 1
            return 0

        monkeypatch.setattr(auto_tune_geo_stack, "_run_fit_priors", _priors)
        monkeypatch.setattr(auto_tune_geo_stack, "_run_fit_calibration", _cal)

        code = auto_tune_geo_stack.main(
            [
                "--config",
                str(config),
                "--images-dir",
                str(images_dir),
                "--metadata",
                str(metadata),
                "--generate-results-if-missing",
                "--output-dir",
                str(output_dir),
            ]
        )
        assert code == 0
        assert calls["priors"] == 1
        assert calls["calibration"] == 1
        summary = json.loads((output_dir / "auto_tune_summary.json").read_text(encoding="utf-8"))
        statuses = {item["name"]: item["status"] for item in summary["steps"]}
        assert statuses["generate_results"] == "ok"


def test_auto_tune_restores_config_when_later_step_fails(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        config, images_dir, metadata = _write_basic_files(root)
        original = config.read_text(encoding="utf-8")
        results = root / "results.jsonl"
        results.write_text("", encoding="utf-8")
        output_dir = root / "out"

        def _tune(argv):
            cfg_idx = argv.index("--config") + 1
            cfg_path = Path(argv[cfg_idx])
            cfg_path.write_text(json.dumps({"geolocator": {"retrieval_top_k": 999}, "fusion": {}}), encoding="utf-8")
            return 0

        monkeypatch.setattr(auto_tune_geo_stack, "_run_tune_retrieval", _tune)
        monkeypatch.setattr(auto_tune_geo_stack, "_run_fit_priors", lambda _argv: 1)
        monkeypatch.setattr(auto_tune_geo_stack, "_run_fit_calibration", lambda _argv: 0)

        code = auto_tune_geo_stack.main(
            [
                "--config",
                str(config),
                "--images-dir",
                str(images_dir),
                "--metadata",
                str(metadata),
                "--results",
                str(results),
                "--output-dir",
                str(output_dir),
            ]
        )
        assert code == 1
        assert config.read_text(encoding="utf-8") == original
        summary = json.loads((output_dir / "auto_tune_summary.json").read_text(encoding="utf-8"))
        statuses = {item["name"]: item["status"] for item in summary["steps"]}
        assert statuses["fit_fusion_priors"] == "failed"
        assert statuses["restore_config"] == "ok"
        assert summary["config_restored"] is True
