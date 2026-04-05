"""
Run retrieval tuning + fusion prior fitting + confidence calibration in one command.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class StepResult:
    name: str
    status: str
    details: str = ""
    output: Optional[str] = None


def _step_result_markdown(step: StepResult) -> str:
    details = step.details.replace("\n", " ").strip() if step.details else ""
    output = step.output or ""
    return f"| {step.name} | {step.status} | {details} | {output} |"


def _run_tune_retrieval(argv: List[str]) -> int:
    from src.tools.tune_retrieval_geo import main as tune_main

    return int(tune_main(argv))


def _run_fit_priors(argv: List[str]) -> int:
    from src.tools.fit_fusion_priors import main as fit_main

    return int(fit_main(argv))


def _run_fit_calibration(argv: List[str]) -> int:
    from src.tools.fit_confidence_calibration import main as fit_main

    return int(fit_main(argv))


def _load_metadata_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image = row.get("path") or row.get("image")
            lat = row.get("latitude") or row.get("lat")
            lon = row.get("longitude") or row.get("lon")
            if not image or lat is None or lon is None:
                continue
            try:
                records.append({"path": str(image), "latitude": float(lat), "longitude": float(lon)})
            except Exception:
                continue
    return records


def _generate_results_jsonl(
    *,
    config_path: Path,
    images_dir: Path,
    metadata_path: Path,
    output_path: Path,
    limit: int,
    seed: int,
) -> int:
    from src.core.logic.config import load_config
    from src.core.logic.serialize import assessment_to_dict
    from src.tools.run_geo_eval import build_pipeline, resolve_image_path

    cfg = load_config(str(config_path))
    pipeline = build_pipeline(cfg)
    records = _load_metadata_records(metadata_path)
    random.Random(seed).shuffle(records)
    if limit > 0:
        records = records[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            rel = str(row["path"])
            image_path = Path(rel) if Path(rel).is_absolute() else resolve_image_path(images_dir, rel)
            if not image_path.exists():
                continue
            result = pipeline.run(str(image_path))
            payload = {"image": str(image_path), "result": assessment_to_dict(result)}
            handle.write(json.dumps(payload) + "\n")
            written += 1
    return written


def _run_step(name: str, runner, argv: List[str], output: Optional[Path] = None) -> StepResult:
    try:
        code = int(runner(argv))
    except Exception as exc:  # pragma: no cover - defensive
        return StepResult(name=name, status="failed", details=str(exc), output=str(output) if output else None)
    if code != 0:
        return StepResult(
            name=name,
            status="failed",
            details=f"exit_code={code}",
            output=str(output) if output else None,
        )
    return StepResult(name=name, status="ok", output=str(output) if output else None)


def _write_markdown_summary(
    *,
    path: Path,
    config_path: Path,
    images_dir: Path,
    metadata_path: Path,
    results_path: Optional[Path],
    config_restored: bool,
    steps: List[StepResult],
) -> None:
    lines = [
        "# Auto Tune Geo Stack Summary",
        "",
        f"- Config: `{config_path}`",
        f"- Images: `{images_dir}`",
        f"- Metadata: `{metadata_path}`",
        f"- Results: `{results_path}`" if results_path else "- Results: (none)",
        f"- Config restored on failure: `{config_restored}`",
        "",
        "| Step | Status | Details | Output |",
        "|---|---|---|---|",
    ]
    lines.extend(_step_result_markdown(step) for step in steps)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="One-command geo stack auto-tuning.")
    parser.add_argument("--config", default="src/config/defaults.json")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--results", default="", help="Optional existing JSONL results for fusion prior/calibration fit.")
    parser.add_argument("--generate-results-if-missing", action="store_true")
    parser.add_argument("--results-limit", type=int, default=200)
    parser.add_argument("--limit", type=int, default=300, help="Sample limit for retrieval tuning.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval-source-balance-beta", default="0.0,0.35,0.7")
    parser.add_argument("--output-dir", default="runs/auto_tune_geo")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"config_not_found:{config_path}")
    original_config_text = config_path.read_text(encoding="utf-8")
    images_dir = Path(args.images_dir)
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_not_found:{metadata_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tune_output = out_dir / "tune_retrieval_geo.json"
    priors_output = out_dir / "fusion_priors.json"
    calibration_output = out_dir / "confidence_calibration.json"
    generated_results = out_dir / "results_for_fusion_fit.jsonl"
    summary_path = out_dir / "auto_tune_summary.json"
    markdown_summary_path = out_dir / "auto_tune_summary.md"

    steps: List[StepResult] = []
    config_restored = False

    tune_args = [
        "--config",
        str(config_path),
        "--images-dir",
        str(images_dir),
        "--metadata",
        str(metadata_path),
        "--output",
        str(tune_output),
        "--limit",
        str(int(args.limit)),
        "--seed",
        str(int(args.seed)),
        "--retrieval-source-balance-beta",
        str(args.retrieval_source_balance_beta),
        "--apply-best-config",
    ]
    steps.append(_run_step("tune_retrieval_geo", _run_tune_retrieval, tune_args, output=tune_output))

    results_path: Optional[Path] = None
    if args.results.strip():
        results_path = Path(args.results.strip())
    elif args.generate_results_if_missing:
        try:
            count = _generate_results_jsonl(
                config_path=config_path,
                images_dir=images_dir,
                metadata_path=metadata_path,
                output_path=generated_results,
                limit=max(1, int(args.results_limit)),
                seed=int(args.seed),
            )
            if count > 0:
                steps.append(
                    StepResult(
                        name="generate_results",
                        status="ok",
                        details=f"written={count}",
                        output=str(generated_results),
                    )
                )
                results_path = generated_results
            else:
                steps.append(
                    StepResult(
                        name="generate_results",
                        status="skipped",
                        details="written=0",
                        output=str(generated_results),
                    )
                )
        except Exception as exc:
            steps.append(StepResult(name="generate_results", status="failed", details=str(exc)))

    if results_path is not None and results_path.exists():
        priors_args = [
            "--results",
            str(results_path),
            "--ground-truth",
            str(metadata_path),
            "--apply-config",
            "--config",
            str(config_path),
            "--output",
            str(priors_output),
        ]
        steps.append(_run_step("fit_fusion_priors", _run_fit_priors, priors_args, output=priors_output))

        calibration_args = [
            "--results",
            str(results_path),
            "--ground-truth",
            str(metadata_path),
            "--apply-config",
            "--config",
            str(config_path),
            "--output",
            str(calibration_output),
        ]
        steps.append(
            _run_step(
                "fit_confidence_calibration",
                _run_fit_calibration,
                calibration_args,
                output=calibration_output,
            )
        )
    else:
        steps.append(
            StepResult(
                name="fit_fusion_priors",
                status="skipped",
                details="results_missing",
            )
        )
        steps.append(
            StepResult(
                name="fit_confidence_calibration",
                status="skipped",
                details="results_missing",
            )
        )

    summary = {
        "config": str(config_path),
        "images_dir": str(images_dir),
        "metadata": str(metadata_path),
        "results": str(results_path) if results_path else None,
        "config_restored": config_restored,
        "steps": [asdict(step) for step in steps],
    }
    failed = [step for step in steps if step.status == "failed"]
    if failed:
        try:
            config_path.write_text(original_config_text, encoding="utf-8")
            config_restored = True
            steps.append(
                StepResult(
                    name="restore_config",
                    status="ok",
                    details="restored_original_on_failure",
                    output=str(config_path),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            steps.append(StepResult(name="restore_config", status="failed", details=str(exc), output=str(config_path)))

    summary["config_restored"] = config_restored
    summary["steps"] = [asdict(step) for step in steps]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_summary(
        path=markdown_summary_path,
        config_path=config_path,
        images_dir=images_dir,
        metadata_path=metadata_path,
        results_path=results_path,
        config_restored=config_restored,
        steps=steps,
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {markdown_summary_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
