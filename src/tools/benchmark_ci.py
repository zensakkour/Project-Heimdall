"""
Canonical benchmark runner and regression gate.

Usage:
  python -m src.tools.benchmark_ci --profile core
  python -m src.tools.benchmark_ci --profile core --promote <run_id>
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = APP_ROOT / "benchmarks" / "manifest.json"
DEFAULT_POLICY_PATH = APP_ROOT / "benchmarks" / "policy.json"
DEFAULT_BASELINE_PATH = APP_ROOT / "docs" / "eval" / "baseline.json"
DEFAULT_HISTORY_PATH = APP_ROOT / "docs" / "eval" / "history.jsonl"
DEFAULT_LATEST_REPORT_PATH = APP_ROOT / "docs" / "eval" / "latest_report.md"
DEFAULT_LATEST_PR_SUMMARY_PATH = APP_ROOT / "docs" / "eval" / "latest_pr_summary.md"
DEFAULT_BASELINE_SUMMARY_PATH = APP_ROOT / "docs" / "eval" / "baseline_summary.json"
DEFAULT_UI_RUNS_DIR = APP_ROOT / "src" / "dashboard" / "data" / "benchmark_runs"
DEFAULT_RUN_HISTORY_ROOT = APP_ROOT / "runs" / "benchmark_history"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return APP_ROOT / path


def _to_repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(APP_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt_metric(value: object, digits: int = 3) -> str:
    num = _safe_float(value)
    if num is None:
        return "-"
    return f"{num:.{digits}f}"


def _load_profile_section(path: Path, profile: str) -> dict:
    payload = _load_json(path)
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"invalid profiles map: {path}")
    section = profiles.get(profile)
    if not isinstance(section, dict):
        raise ValueError(f"profile_not_found:{profile}:{path}")
    return section


def _extract_eval_summary(path: Path) -> dict:
    payload = _load_json(path)
    return {
        "name": path.stem,
        "path": str(path),
        "mean_km": payload.get("mean_km"),
        "median_km": payload.get("median_km"),
        "within_5km_pct": payload.get("within_5km_pct"),
        "within_10km_pct": payload.get("within_10km_pct"),
        "evaluated": payload.get("evaluated"),
    }


def _flatten_metrics(summary: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in summary.get("geo_scenarios", []):
        name = str(row.get("scenario") or row.get("name") or "").strip()
        if not name:
            continue
        for metric in ("mean_km", "median_km", "within_5km_pct", "within_10km_pct", "evaluated"):
            val = _safe_float(row.get(metric))
            if val is not None:
                out[f"scenario.{name}.{metric}"] = val
    for row in summary.get("backbone_benchmark", {}).get("models", []):
        model_id = str(row.get("model_id") or "").strip()
        if not model_id:
            continue
        for metric in ("mean_km", "median_km", "within_5km_pct", "within_10km_pct"):
            val = _safe_float(row.get(metric))
            if val is not None:
                out[f"backbone.{model_id}.{metric}"] = val
    return out


def compare_runs(baseline: dict, candidate: dict) -> dict:
    baseline_scenarios = {
        str(item.get("scenario") or item.get("name")): item
        for item in baseline.get("geo_scenarios", [])
        if item.get("scenario") or item.get("name")
    }
    candidate_scenarios = {
        str(item.get("scenario") or item.get("name")): item
        for item in candidate.get("geo_scenarios", [])
        if item.get("scenario") or item.get("name")
    }

    scenario_rows = []
    for name in sorted(set(baseline_scenarios.keys()) | set(candidate_scenarios.keys())):
        b = baseline_scenarios.get(name, {})
        c = candidate_scenarios.get(name, {})
        b_mean = _safe_float(b.get("mean_km"))
        c_mean = _safe_float(c.get("mean_km"))
        b_med = _safe_float(b.get("median_km"))
        c_med = _safe_float(c.get("median_km"))
        b_w10 = _safe_float(b.get("within_10km_pct"))
        c_w10 = _safe_float(c.get("within_10km_pct"))
        scenario_rows.append(
            {
                "scenario": name,
                "baseline": {
                    "mean_km": b_mean,
                    "median_km": b_med,
                    "within_10km_pct": b_w10,
                },
                "candidate": {
                    "mean_km": c_mean,
                    "median_km": c_med,
                    "within_10km_pct": c_w10,
                },
                "delta": {
                    "mean_km": (c_mean - b_mean) if b_mean is not None and c_mean is not None else None,
                    "median_km": (c_med - b_med) if b_med is not None and c_med is not None else None,
                    "within_10km_pct": (c_w10 - b_w10) if b_w10 is not None and c_w10 is not None else None,
                },
            }
        )

    baseline_models = {
        str(item.get("model_id")): item
        for item in baseline.get("backbone_benchmark", {}).get("models", [])
        if item.get("model_id")
    }
    candidate_models = {
        str(item.get("model_id")): item
        for item in candidate.get("backbone_benchmark", {}).get("models", [])
        if item.get("model_id")
    }

    model_rows = []
    for model_id in sorted(set(baseline_models.keys()) | set(candidate_models.keys())):
        b = baseline_models.get(model_id, {})
        c = candidate_models.get(model_id, {})
        b_mean = _safe_float(b.get("mean_km"))
        c_mean = _safe_float(c.get("mean_km"))
        b_med = _safe_float(b.get("median_km"))
        c_med = _safe_float(c.get("median_km"))
        b_w10 = _safe_float(b.get("within_10km_pct"))
        c_w10 = _safe_float(c.get("within_10km_pct"))
        model_rows.append(
            {
                "model_id": model_id,
                "baseline": {
                    "mean_km": b_mean,
                    "median_km": b_med,
                    "within_10km_pct": b_w10,
                },
                "candidate": {
                    "mean_km": c_mean,
                    "median_km": c_med,
                    "within_10km_pct": c_w10,
                },
                "delta": {
                    "mean_km": (c_mean - b_mean) if b_mean is not None and c_mean is not None else None,
                    "median_km": (c_med - b_med) if b_med is not None and c_med is not None else None,
                    "within_10km_pct": (c_w10 - b_w10) if b_w10 is not None and c_w10 is not None else None,
                },
            }
        )

    return {
        "generated_at": _utc_now_iso(),
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "baseline_best_model": baseline.get("backbone_benchmark", {}).get("best_model"),
        "candidate_best_model": candidate.get("backbone_benchmark", {}).get("best_model"),
        "scenario_deltas": scenario_rows,
        "model_deltas": model_rows,
    }


def evaluate_policy(baseline: dict, candidate: dict, rules: list[dict]) -> dict:
    baseline_metrics = _flatten_metrics(baseline)
    candidate_metrics = _flatten_metrics(candidate)
    checks = []
    failures = 0
    for rule in rules:
        metric = str(rule.get("metric") or "").strip()
        direction = str(rule.get("direction") or "").strip().lower()
        tol_abs = float(rule.get("max_regression", 0.0) or 0.0)
        tol_pct = float(rule.get("max_regression_pct", 0.0) or 0.0)
        baseline_value = baseline_metrics.get(metric)
        candidate_value = candidate_metrics.get(metric)
        note = ""
        passed = False
        threshold: Optional[float] = None
        allowed_delta: Optional[float] = None
        if not metric or direction not in {"lower", "higher"}:
            note = "invalid_rule"
        elif baseline_value is None or candidate_value is None:
            note = "metric_missing"
        else:
            allowed_delta = max(tol_abs, abs(float(baseline_value)) * (tol_pct / 100.0))
            if direction == "lower":
                threshold = float(baseline_value) + allowed_delta
                passed = float(candidate_value) <= threshold
            else:
                threshold = float(baseline_value) - allowed_delta
                passed = float(candidate_value) >= threshold
            if not passed:
                note = "regression"
        if not passed:
            failures += 1
        checks.append(
            {
                "metric": metric,
                "direction": direction,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "threshold": threshold,
                "allowed_delta": allowed_delta,
                "delta": (
                    float(candidate_value) - float(baseline_value)
                    if baseline_value is not None and candidate_value is not None
                    else None
                ),
                "passed": passed,
                "note": note,
            }
        )

    return {
        "status": "pass" if failures == 0 else "fail",
        "passed": failures == 0,
        "failures": failures,
        "checks": checks,
    }


def _render_compare_table(compare: dict, key: str, label_key: str) -> list[str]:
    rows = compare.get(key, []) if compare else []
    if not rows:
        return ["No data."]
    lines = [
        "| Name | Baseline Mean | Candidate Mean | Delta Mean | Baseline <=10km | Candidate <=10km | Delta <=10km |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {b_mean} | {c_mean} | {d_mean} | {b_w10} | {c_w10} | {d_w10} |".format(
                name=row.get(label_key),
                b_mean=_fmt_metric(row.get("baseline", {}).get("mean_km")),
                c_mean=_fmt_metric(row.get("candidate", {}).get("mean_km")),
                d_mean=_fmt_metric(row.get("delta", {}).get("mean_km")),
                b_w10=_fmt_metric(row.get("baseline", {}).get("within_10km_pct"), digits=2),
                c_w10=_fmt_metric(row.get("candidate", {}).get("within_10km_pct"), digits=2),
                d_w10=_fmt_metric(row.get("delta", {}).get("within_10km_pct"), digits=2),
            )
        )
    return lines


def _render_policy_table(policy_result: Optional[dict]) -> list[str]:
    if not policy_result:
        return ["Policy not evaluated (no baseline)."]
    checks = policy_result.get("checks", [])
    if not checks:
        return ["No policy rules configured."]
    lines = [
        "| Metric | Direction | Baseline | Candidate | Threshold | Delta | Result |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in checks:
        result = "PASS" if row.get("passed") else "FAIL"
        lines.append(
            "| {metric} | {direction} | {baseline} | {candidate} | {threshold} | {delta} | {result} |".format(
                metric=row.get("metric"),
                direction=row.get("direction"),
                baseline=_fmt_metric(row.get("baseline")),
                candidate=_fmt_metric(row.get("candidate")),
                threshold=_fmt_metric(row.get("threshold")),
                delta=_fmt_metric(row.get("delta")),
                result=result,
            )
        )
    return lines


def render_latest_report(
    *,
    candidate: dict,
    baseline_contract: dict,
    compare: Optional[dict],
    policy_result: Optional[dict],
    manifest_path: Path,
    policy_path: Path,
    baseline_path: Path,
) -> str:
    run_id = candidate.get("run_id", "-")
    generated_at = candidate.get("generated_at", "-")
    baseline_run_id = baseline_contract.get("baseline_run_id") or "none"
    baseline_commit = baseline_contract.get("baseline_commit_sha") or "none"
    policy_status = (
        "SKIPPED (no baseline)"
        if policy_result is None
        else ("PASS" if policy_result.get("passed") else "FAIL")
    )
    lines = [
        "# Benchmark Report",
        "",
        f"- Generated at (UTC): {generated_at}",
        f"- Profile: `{candidate.get('profile', '-')}`",
        f"- Candidate run: `{run_id}`",
        f"- Baseline run: `{baseline_run_id}`",
        f"- Baseline commit: `{baseline_commit}`",
        f"- Policy result: **{policy_status}**",
        "",
        "## Scenario Comparison",
    ]
    if compare:
        lines.extend(_render_compare_table(compare, "scenario_deltas", "scenario"))
    else:
        lines.append("No baseline comparison available yet.")

    lines.extend(["", "## Backbone Comparison"])
    if compare:
        lines.extend(_render_compare_table(compare, "model_deltas", "model_id"))
    else:
        lines.append("No baseline comparison available yet.")

    lines.extend(["", "## Policy Checks"])
    lines.extend(_render_policy_table(policy_result))

    lines.extend(
        [
            "",
            "## References",
            f"- Manifest: `{_to_repo_rel(manifest_path)}`",
            f"- Policy: `{_to_repo_rel(policy_path)}`",
            f"- Baseline contract: `{_to_repo_rel(baseline_path)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def render_pr_summary(
    *,
    candidate: dict,
    compare: Optional[dict],
    policy_result: Optional[dict],
    baseline_contract: dict,
) -> str:
    policy_label = (
        "SKIPPED (no baseline)"
        if policy_result is None
        else ("PASS" if policy_result.get("passed") else "FAIL")
    )
    run_id = candidate.get("run_id", "-")
    baseline_run_id = baseline_contract.get("baseline_run_id") or "none"
    lines = [
        "## Benchmark Summary",
        "",
        f"- Profile: `{candidate.get('profile', '-')}`",
        f"- Candidate run: `{run_id}`",
        f"- Baseline run: `{baseline_run_id}`",
        f"- Policy: **{policy_label}**",
        "",
        "### Scenario Deltas",
    ]
    if compare:
        lines.extend(_render_compare_table(compare, "scenario_deltas", "scenario"))
    else:
        lines.append("No baseline comparison available.")
    return "\n".join(lines) + "\n"


def _git_commit_sha() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=APP_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def _default_baseline_contract(profile: str) -> dict:
    return {
        "version": 1,
        "profile": profile,
        "baseline_run_id": None,
        "baseline_commit_sha": None,
        "promoted_at": None,
        "baseline_summary_path": "docs/eval/baseline_summary.json",
    }


def _load_baseline_contract(path: Path, profile: str) -> tuple[dict, Path, Optional[dict]]:
    contract = _default_baseline_contract(profile)
    if path.exists():
        loaded = _load_json(path)
        if isinstance(loaded, dict):
            contract.update(loaded)
    baseline_summary_path = _as_path(contract.get("baseline_summary_path", str(DEFAULT_BASELINE_SUMMARY_PATH)))
    baseline_payload: Optional[dict] = None
    if baseline_summary_path.exists():
        summary_payload = _load_json(baseline_summary_path)
        if isinstance(summary_payload, dict) and isinstance(summary_payload.get("summary"), dict):
            baseline_payload = summary_payload["summary"]
    return contract, baseline_summary_path, baseline_payload


def promote_baseline(
    *,
    run_payload: dict,
    profile: str,
    baseline_contract_path: Path,
    baseline_summary_path: Path,
    commit_sha: str,
) -> dict:
    promoted_at = _utc_now_iso()
    baseline_summary = {
        "version": 1,
        "profile": profile,
        "status": "ready",
        "run_id": run_payload.get("run_id"),
        "commit_sha": commit_sha,
        "promoted_at": promoted_at,
        "metrics": _flatten_metrics(run_payload),
        "summary": run_payload,
    }
    _write_json(baseline_summary_path, baseline_summary)
    contract = _default_baseline_contract(profile)
    if baseline_contract_path.exists():
        loaded = _load_json(baseline_contract_path)
        if isinstance(loaded, dict):
            contract.update(loaded)
    contract.update(
        {
            "profile": profile,
            "baseline_run_id": run_payload.get("run_id"),
            "baseline_commit_sha": commit_sha,
            "promoted_at": promoted_at,
            "baseline_summary_path": _to_repo_rel(baseline_summary_path),
        }
    )
    _write_json(baseline_contract_path, contract)
    return contract


def _run_benchmark_suite(profile: str, profile_cfg: dict, run_id: str) -> tuple[dict, Path, Path]:
    from src.tools.benchmark_geo_backbones import main as run_backbone_bench
    from src.tools.run_geo_eval import main as run_geo_eval

    history_dir = DEFAULT_RUN_HISTORY_ROOT / run_id
    history_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_UI_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_summary_path = DEFAULT_UI_RUNS_DIR / f"{run_id}.json"

    geo_results = []
    for scenario in profile_cfg.get("geo_scenarios", []):
        name = str(scenario.get("name") or "").strip()
        if not name:
            continue
        scenario_output = history_dir / f"geo_{name}.json"
        args = [
            "--config",
            str(_as_path(scenario.get("config"))),
            "--images-dir",
            str(_as_path(scenario.get("images_dir"))),
            "--metadata",
            str(_as_path(scenario.get("metadata"))),
            "--limit",
            str(int(scenario.get("limit", 0) or 0)),
            "--output",
            str(scenario_output),
        ]
        if bool(scenario.get("retrieval_only", False)):
            args.append("--retrieval-only")
        run_geo_eval(args)
        row = _extract_eval_summary(scenario_output)
        row["scenario"] = name
        geo_results.append(row)

    backbone_cfg = profile_cfg.get("backbone_benchmark", {})
    backbone_output = history_dir / "backbone_benchmark.json"
    model_ids = backbone_cfg.get("model_ids", [])
    if isinstance(model_ids, str):
        model_ids = [item.strip() for item in model_ids.split(",") if item.strip()]
    tta_degrees = backbone_cfg.get("query_tta_degrees", [0, 90, 180, 270])
    if isinstance(tta_degrees, (int, float)):
        tta_degrees = [tta_degrees]
    bench_args = [
        "--train-images-dir",
        str(_as_path(backbone_cfg.get("train_images_dir"))),
        "--train-metadata",
        str(_as_path(backbone_cfg.get("train_metadata"))),
        "--eval-images-dir",
        str(_as_path(backbone_cfg.get("eval_images_dir"))),
        "--eval-metadata",
        str(_as_path(backbone_cfg.get("eval_metadata"))),
        "--model-ids",
        ",".join(str(item) for item in model_ids if str(item).strip()),
        "--train-limit",
        str(int(backbone_cfg.get("train_limit", 0) or 0)),
        "--eval-limit",
        str(int(backbone_cfg.get("eval_limit", 0) or 0)),
        "--seed",
        str(int(backbone_cfg.get("seed", 42) or 42)),
        "--retrieval-top-k",
        str(int(backbone_cfg.get("retrieval_top_k", 50) or 50)),
        "--retrieval-min-score",
        str(float(backbone_cfg.get("retrieval_min_score", 0.1) or 0.1)),
        "--retrieval-min-keep-topk",
        str(int(backbone_cfg.get("retrieval_min_keep_topk", 2) or 2)),
        "--query-tta-degrees",
        ",".join(str(item) for item in tta_degrees),
        "--query-tta-reduce",
        str(backbone_cfg.get("query_tta_reduce", "max")),
        "--output",
        str(backbone_output),
    ]
    if bool(backbone_cfg.get("reuse_indices", True)):
        bench_args.append("--reuse-indices")
    code = int(run_backbone_bench(bench_args))
    if code != 0:
        raise RuntimeError(f"backbone_benchmark_failed:{code}")

    backbone_payload = _load_json(backbone_output)
    summary = {
        "run_id": run_id,
        "generated_at": _utc_now_iso(),
        "profile": profile,
        "manifest_version": profile_cfg.get("version"),
        "geo_scenarios": geo_results,
        "backbone_benchmark": {
            "best_model": backbone_payload.get("best_model"),
            "ranked_by_median_km": backbone_payload.get("ranked_by_median_km", []),
            "models": backbone_payload.get("models", []),
            "path": str(backbone_output),
        },
    }
    _write_json(history_dir / "summary.json", summary)
    _write_json(run_summary_path, summary)
    return summary, run_summary_path, history_dir


def _promote_from_run_id(
    *,
    run_id: str,
    profile: str,
    baseline_path: Path,
) -> int:
    run_path = DEFAULT_UI_RUNS_DIR / f"{run_id}.json"
    if not run_path.exists():
        print(f"run_not_found: {run_path}")
        return 2
    run_payload = _load_json(run_path)
    contract, baseline_summary_path, _ = _load_baseline_contract(baseline_path, profile)
    summary_path_raw = contract.get("baseline_summary_path")
    if summary_path_raw:
        baseline_summary_path = _as_path(summary_path_raw)
    promote_baseline(
        run_payload=run_payload,
        profile=profile,
        baseline_contract_path=baseline_path,
        baseline_summary_path=baseline_summary_path,
        commit_sha=_git_commit_sha(),
    )
    print(f"Promoted run {run_id} to baseline.")
    print(f"Updated {baseline_path}")
    print(f"Updated {baseline_summary_path}")
    return 0


def _pick_scenario(summary: dict, name: str) -> dict:
    for row in summary.get("geo_scenarios", []):
        if str(row.get("scenario")) == name:
            return row
    return {}


def _build_history_row(
    *,
    summary: dict,
    baseline_contract: dict,
    policy_result: Optional[dict],
) -> dict:
    realistic = _pick_scenario(summary, "realistic_single")
    candidate_multi = _pick_scenario(summary, "candidate_multi")
    return {
        "run_id": summary.get("run_id"),
        "generated_at": summary.get("generated_at"),
        "profile": summary.get("profile"),
        "commit_sha": _git_commit_sha(),
        "baseline_run_id": baseline_contract.get("baseline_run_id"),
        "policy_status": "skipped" if policy_result is None else policy_result.get("status"),
        "policy_passed": None if policy_result is None else bool(policy_result.get("passed")),
        "best_model": summary.get("backbone_benchmark", {}).get("best_model"),
        "realistic_single_mean_km": _safe_float(realistic.get("mean_km")),
        "realistic_single_within_10km_pct": _safe_float(realistic.get("within_10km_pct")),
        "candidate_multi_mean_km": _safe_float(candidate_multi.get("mean_km")),
        "candidate_multi_within_10km_pct": _safe_float(candidate_multi.get("within_10km_pct")),
    }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical benchmark suite and regression gates.")
    parser.add_argument("--profile", default="core", help="Manifest/policy profile name.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument("--history", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--latest-report", default=str(DEFAULT_LATEST_REPORT_PATH))
    parser.add_argument("--latest-pr-summary", default=str(DEFAULT_LATEST_PR_SUMMARY_PATH))
    parser.add_argument("--run-id", default="", help="Optional run id override.")
    parser.add_argument("--promote", default="", help="Promote an existing run id to baseline.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    profile = str(args.profile).strip()
    manifest_path = _as_path(args.manifest)
    policy_path = _as_path(args.policy)
    baseline_path = _as_path(args.baseline)
    history_path = _as_path(args.history)
    latest_report_path = _as_path(args.latest_report)
    latest_pr_summary_path = _as_path(args.latest_pr_summary)

    if args.promote:
        return _promote_from_run_id(run_id=str(args.promote).strip(), profile=profile, baseline_path=baseline_path)

    try:
        profile_cfg = _load_profile_section(manifest_path, profile)
        policy_cfg = _load_profile_section(policy_path, profile)
    except Exception as exc:
        print(f"config_load_error:{exc}")
        return 2

    run_id = str(args.run_id).strip() or _new_run_id()
    try:
        summary, run_summary_path, history_dir = _run_benchmark_suite(profile, profile_cfg, run_id)
    except Exception as exc:
        print(f"benchmark_run_failed:{exc}")
        return 2

    contract, _, baseline_payload = _load_baseline_contract(baseline_path, profile)
    compare_payload: Optional[dict] = None
    policy_result: Optional[dict] = None
    if baseline_payload is not None:
        compare_payload = compare_runs(baseline_payload, summary)
        compare_payload["generated_at"] = _utc_now_iso()
        compare_payload["baseline_commit_sha"] = contract.get("baseline_commit_sha")
        _write_json(history_dir / "compare_vs_baseline.json", compare_payload)
        policy_rules = policy_cfg.get("rules", [])
        if not isinstance(policy_rules, list):
            policy_rules = []
        policy_result = evaluate_policy(baseline_payload, summary, policy_rules)
        _write_json(history_dir / "policy_result.json", policy_result)

    report_md = render_latest_report(
        candidate=summary,
        baseline_contract=contract,
        compare=compare_payload,
        policy_result=policy_result,
        manifest_path=manifest_path,
        policy_path=policy_path,
        baseline_path=baseline_path,
    )
    _write_text(latest_report_path, report_md)
    pr_summary_md = render_pr_summary(
        candidate=summary,
        compare=compare_payload,
        policy_result=policy_result,
        baseline_contract=contract,
    )
    _write_text(latest_pr_summary_path, pr_summary_md)

    history_row = _build_history_row(summary=summary, baseline_contract=contract, policy_result=policy_result)
    _append_jsonl(history_path, history_row)

    print(f"Run complete: {summary.get('run_id')}")
    print(f"Run summary: {_to_repo_rel(run_summary_path)}")
    print(f"Report: {_to_repo_rel(latest_report_path)}")
    print(f"PR summary: {_to_repo_rel(latest_pr_summary_path)}")
    print(f"History ledger: {_to_repo_rel(history_path)}")
    if policy_result is None:
        print("Policy: skipped (no baseline promoted yet)")
        return 0
    if policy_result.get("passed"):
        print("Policy: PASS")
        return 0
    print("Policy: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
