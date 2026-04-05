"""
FastAPI server for live analysis UI.
"""
from __future__ import annotations

import base64
import importlib
from importlib import metadata
import json
import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.core.logic.config import HeimdallConfig, has_retrieval_index, load_config
from src.core.logic.serialize import assessment_to_dict
from src.core.logic.types import (
    Assessment,
    Detection,
    Evidence,
    FusionCandidate,
    FusionResult,
    GeoCandidate,
    GeoEstimate,
    UncertaintyEllipse,
    Verification,
)
from src.core.logic.visualize import draw_detections


APP_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = APP_ROOT / "src" / "dashboard" / "analysis"
DASHBOARD_DIR = APP_ROOT / "src" / "dashboard"

app = FastAPI()

_EVAL_STATE = {"status": "idle", "last_result": None}
_GEO_EVAL_STATE = {"status": "idle", "last_result": None, "progress": None, "progress_path": None}
_BENCHMARK_STATE = {"status": "idle", "stage": None, "last_result": None, "progress": None, "run_id": None}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _benchmark_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _benchmark_runs_dir() -> Path:
    path = APP_ROOT / "src" / "dashboard" / "data" / "benchmark_runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _benchmark_history_root() -> Path:
    path = APP_ROOT / "runs" / "benchmark_history"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _benchmark_compares_dir() -> Path:
    path = APP_ROOT / "src" / "dashboard" / "data" / "benchmark_compares"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _progress_log_path() -> Path:
    return APP_ROOT / "PROGRESS.md"


def _is_safe_run_id(run_id: str) -> bool:
    text = (run_id or "").strip()
    if not text:
        return False
    if "/" in text or "\\" in text or ".." in text:
        return False
    return Path(text).name == text


def _list_benchmark_runs(limit: int = 100) -> list[dict]:
    safe_limit = max(1, min(int(limit), 500))
    runs_dir = _benchmark_runs_dir()
    files = sorted(runs_dir.glob("*.json"), key=lambda p: p.name, reverse=True)
    rows: list[dict] = []
    for path in files[:safe_limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        models = payload.get("backbone_benchmark", {}).get("models", [])
        rows.append(
            {
                "run_id": payload.get("run_id") or path.stem,
                "generated_at": payload.get("generated_at"),
                "best_model": payload.get("backbone_benchmark", {}).get("best_model"),
                "model_count": len(models) if isinstance(models, list) else 0,
                "path": str(path),
            }
        )
    return rows


def _load_benchmark_run_payload(run_id: str) -> Optional[dict]:
    if not _is_safe_run_id(run_id):
        return None
    path = _benchmark_runs_dir() / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _to_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _diff_metric(baseline: object, candidate: object) -> Optional[float]:
    b = _to_float(baseline)
    c = _to_float(candidate)
    if b is None or c is None:
        return None
    return c - b


def _fmt_num(value: object, digits: int = 3) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _build_progress_benchmark_snippet(compare: dict) -> str:
    baseline_run_id = compare.get("baseline_run_id", "-")
    candidate_run_id = compare.get("candidate_run_id", "-")
    baseline_ts = compare.get("baseline_generated_at", "-")
    candidate_ts = compare.get("candidate_generated_at", "-")
    baseline_best = compare.get("baseline_best_model", "-")
    candidate_best = compare.get("candidate_best_model", "-")

    lines = [
        f"## {_utc_now_iso()[:10]}",
        f"- Benchmark comparison: candidate `{candidate_run_id}` ({candidate_ts}) vs baseline `{baseline_run_id}` ({baseline_ts}).",
    ]
    scenario_deltas = compare.get("scenario_deltas", [])
    for row in scenario_deltas:
        name = row.get("scenario") or row.get("key") or "scenario"
        lines.append(
            "- "
            + f"{name}: "
            + f"mean_km {_fmt_num(row.get('baseline', {}).get('mean_km'))} -> {_fmt_num(row.get('candidate', {}).get('mean_km'))} "
            + f"(delta {_fmt_num(row.get('delta', {}).get('mean_km'))}), "
            + f"median_km {_fmt_num(row.get('baseline', {}).get('median_km'))} -> {_fmt_num(row.get('candidate', {}).get('median_km'))} "
            + f"(delta {_fmt_num(row.get('delta', {}).get('median_km'))}), "
            + f"<=10km {_fmt_num(row.get('baseline', {}).get('within_10km_pct'), 2)} -> {_fmt_num(row.get('candidate', {}).get('within_10km_pct'), 2)} "
            + f"(delta {_fmt_num(row.get('delta', {}).get('within_10km_pct'), 2)})."
        )
    lines.append(f"- Backbone best model: `{baseline_best}` -> `{candidate_best}`.")
    lines.append(
        "- Artifacts: "
        + f"`src/dashboard/data/benchmark_runs/{baseline_run_id}.json`, "
        + f"`src/dashboard/data/benchmark_runs/{candidate_run_id}.json`."
    )
    return "\n".join(lines)


def _compare_benchmark_payloads(baseline: dict, candidate: dict) -> dict:
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
    scenario_keys = sorted(set(baseline_scenarios.keys()) | set(candidate_scenarios.keys()))
    scenario_deltas = []
    for key in scenario_keys:
        b_row = baseline_scenarios.get(key, {})
        c_row = candidate_scenarios.get(key, {})
        scenario_deltas.append(
            {
                "scenario": key,
                "baseline": {
                    "mean_km": _to_float(b_row.get("mean_km")),
                    "median_km": _to_float(b_row.get("median_km")),
                    "within_5km_pct": _to_float(b_row.get("within_5km_pct")),
                    "within_10km_pct": _to_float(b_row.get("within_10km_pct")),
                },
                "candidate": {
                    "mean_km": _to_float(c_row.get("mean_km")),
                    "median_km": _to_float(c_row.get("median_km")),
                    "within_5km_pct": _to_float(c_row.get("within_5km_pct")),
                    "within_10km_pct": _to_float(c_row.get("within_10km_pct")),
                },
                "delta": {
                    "mean_km": _diff_metric(b_row.get("mean_km"), c_row.get("mean_km")),
                    "median_km": _diff_metric(b_row.get("median_km"), c_row.get("median_km")),
                    "within_5km_pct": _diff_metric(b_row.get("within_5km_pct"), c_row.get("within_5km_pct")),
                    "within_10km_pct": _diff_metric(b_row.get("within_10km_pct"), c_row.get("within_10km_pct")),
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
    model_keys = sorted(set(baseline_models.keys()) | set(candidate_models.keys()))
    model_deltas = []
    for key in model_keys:
        b_row = baseline_models.get(key, {})
        c_row = candidate_models.get(key, {})
        model_deltas.append(
            {
                "model_id": key,
                "baseline": {
                    "mean_km": _to_float(b_row.get("mean_km")),
                    "median_km": _to_float(b_row.get("median_km")),
                    "within_5km_pct": _to_float(b_row.get("within_5km_pct")),
                    "within_10km_pct": _to_float(b_row.get("within_10km_pct")),
                },
                "candidate": {
                    "mean_km": _to_float(c_row.get("mean_km")),
                    "median_km": _to_float(c_row.get("median_km")),
                    "within_5km_pct": _to_float(c_row.get("within_5km_pct")),
                    "within_10km_pct": _to_float(c_row.get("within_10km_pct")),
                },
                "delta": {
                    "mean_km": _diff_metric(b_row.get("mean_km"), c_row.get("mean_km")),
                    "median_km": _diff_metric(b_row.get("median_km"), c_row.get("median_km")),
                    "within_5km_pct": _diff_metric(b_row.get("within_5km_pct"), c_row.get("within_5km_pct")),
                    "within_10km_pct": _diff_metric(b_row.get("within_10km_pct"), c_row.get("within_10km_pct")),
                },
            }
        )

    compare = {
        "compare_id": _benchmark_run_id(),
        "generated_at": _utc_now_iso(),
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "baseline_generated_at": baseline.get("generated_at"),
        "candidate_generated_at": candidate.get("generated_at"),
        "baseline_best_model": baseline.get("backbone_benchmark", {}).get("best_model"),
        "candidate_best_model": candidate.get("backbone_benchmark", {}).get("best_model"),
        "scenario_deltas": scenario_deltas,
        "model_deltas": model_deltas,
    }
    compare["progress_md_snippet"] = _build_progress_benchmark_snippet(compare)
    return compare


def _append_progress_snippet(snippet: str) -> None:
    text = (snippet or "").strip()
    if not text:
        return
    path = _progress_log_path()
    prefix = "\n\n" if path.exists() and path.read_text(encoding="utf-8").strip() else ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + text + "\n")


def build_pipeline(cfg: Optional[HeimdallConfig]) -> "HeimdallPipeline":
    # Lazy imports keep app startup resilient when optional heavy deps are unavailable.
    from src.core.detection.factory import create_detector
    from src.core.geo import GeoCLIPProvider, GeoLocator, GeoRetrievalProvider, MultiCandidateProvider
    from src.core.logic.pipeline import HeimdallPipeline

    if cfg is None:
        return HeimdallPipeline()
    detector = create_detector(cfg.detector)
    geolocator = GeoLocator(
        cfg.geolocator.model_path,
        use_sidecar=cfg.geolocator.use_sidecar,
        use_exif=cfg.geolocator.use_exif,
    )
    retrieval_provider = GeoRetrievalProvider(
        index_path=cfg.geolocator.retrieval_index_path,
        index_paths=cfg.geolocator.retrieval_index_paths,
        index_weights=cfg.geolocator.retrieval_index_weights,
        index_model_ids=cfg.geolocator.retrieval_index_model_ids,
        model_id=cfg.geolocator.retrieval_model_id or "openai/clip-vit-large-patch14",
        top_k=cfg.geolocator.retrieval_top_k,
        per_index_top_k=cfg.geolocator.retrieval_per_index_top_k,
        index_score_norm=cfg.geolocator.retrieval_index_score_norm,
        source_balance_beta=cfg.geolocator.retrieval_source_balance_beta,
        min_score=cfg.geolocator.retrieval_min_score,
        min_keep_topk=cfg.geolocator.retrieval_min_keep_topk,
        diversity_radius_km=cfg.geolocator.retrieval_diversity_radius_km,
        diversity_lambda=cfg.geolocator.retrieval_diversity_lambda,
        diversity_min_keep=cfg.geolocator.retrieval_diversity_min_keep,
        locality_radius_km=cfg.geolocator.retrieval_locality_radius_km,
        locality_weight=cfg.geolocator.retrieval_locality_weight,
        query_tta_degrees=cfg.geolocator.retrieval_query_tta_degrees,
        query_tta_reduce=cfg.geolocator.retrieval_query_tta_reduce,
    )
    geoclip_provider = GeoCLIPProvider(
        model_path=cfg.geolocator.model_path,
        model_id=cfg.geolocator.model_id,
        model_cache_dir=cfg.geolocator.model_cache_dir,
        encoder_name=cfg.geolocator.encoder_name,
        top_n=cfg.geolocator.top_n,
        use_sidecar=cfg.geolocator.use_sidecar,
        use_exif=cfg.geolocator.use_exif,
        score_scale=cfg.geolocator.geospot_score_scale,
    )
    if has_retrieval_index(cfg.geolocator):
        candidate_provider = MultiCandidateProvider(
            [retrieval_provider, geoclip_provider],
            dedupe_radius_m=cfg.geolocator.candidate_dedupe_radius_m,
            source_balance_beta=cfg.geolocator.candidate_source_balance_beta,
            max_candidates=cfg.geolocator.candidate_max_results,
        )
    else:
        candidate_provider = geoclip_provider
    return HeimdallPipeline(
        detector=detector,
        geolocator=geolocator,
        candidate_provider=candidate_provider,
        fusion_config=cfg.fusion,
        score_config=cfg.score,
        verification_config=cfg.verification,
    )


def _to_bool_flag(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _make_demo_detections(width: int, height: int) -> list[Detection]:
    w = max(320, width)
    h = max(240, height)
    return [
        Detection(
            label="vehicle",
            confidence=0.86,
            obb=(
                (w * 0.22, h * 0.30),
                (w * 0.40, h * 0.27),
                (w * 0.43, h * 0.42),
                (w * 0.24, h * 0.45),
            ),
            heading_deg=31.5,
            shadow_azimuth_deg=126.2,
            shadow_length_ratio=1.14,
        ),
        Detection(
            label="building",
            confidence=0.73,
            obb=(
                (w * 0.58, h * 0.38),
                (w * 0.80, h * 0.35),
                (w * 0.84, h * 0.56),
                (w * 0.61, h * 0.59),
            ),
            heading_deg=12.0,
            shadow_azimuth_deg=118.0,
            shadow_length_ratio=0.58,
        ),
    ]


def _build_demo_assessment(width: int, height: int, reason: Optional[str]) -> Assessment:
    detections = _make_demo_detections(width, height)
    candidates = [
        GeoCandidate(latitude=48.85661, longitude=2.35222, retrieval_score=0.82, match_id="paris-center"),
        GeoCandidate(latitude=48.86601, longitude=2.33334, retrieval_score=0.64, match_id="paris-louvre"),
        GeoCandidate(latitude=48.84590, longitude=2.37470, retrieval_score=0.52, match_id="paris-east"),
    ]
    posterior = [0.56, 0.28, 0.16]
    fusion_candidates = []
    for cand, weight in zip(candidates, posterior):
        fusion_candidates.append(
            FusionCandidate(
                candidate=cand,
                posterior_weight=weight,
                evidence=Evidence(
                    retrieval_score=cand.retrieval_score,
                    shadow_residual_deg=7.5,
                    terrain_residual=18.0,
                    likelihoods={"retrieval": cand.retrieval_score, "shadow": 0.79, "terrain": 0.66},
                    posterior_weight=weight,
                    explanation="Safe demo fallback candidate",
                ),
            )
        )
    mean_lat = sum(item.candidate.latitude * item.posterior_weight for item in fusion_candidates)
    mean_lon = sum(item.candidate.longitude * item.posterior_weight for item in fusion_candidates)
    fusion = FusionResult(
        candidates=fusion_candidates,
        mean_latitude=mean_lat,
        mean_longitude=mean_lon,
        covariance_m=((260.0, 0.0), (0.0, 180.0)),
        ellipse=UncertaintyEllipse(major_axis_m=520.0, minor_axis_m=300.0, orientation_deg=19.0),
        uncertainty_radius_m=420.0,
        normalized_entropy=0.76,
        effective_candidate_count=2.2,
        top1_posterior=0.56,
        top2_margin=0.28,
        confidence_tier="medium",
        ambiguous=False,
        credible_set_size=2,
    )
    return Assessment(
        detections=detections,
        geo=GeoEstimate(
            latitude=mean_lat,
            longitude=mean_lon,
            confidence=0.74,
            landmarks=["paris-demo-fallback"],
            uncertainty_m=420.0,
        ),
        verification=Verification(
            shadow_ok=True,
            topo_ok=True,
            notes=(
                "Safe demo mode fallback output."
                if not reason
                else f"Safe demo mode fallback output ({reason[:180]})."
            ),
        ),
        score=0.72,
        candidates=candidates,
        fusion=fusion,
    )


def _make_demo_image_payload(image_path: Path, reason: Optional[str]) -> dict:
    with Image.open(image_path) as img:
        width, height = img.size
    result = _build_demo_assessment(width, height, reason)
    annotated = draw_detections(str(image_path), result.detections)
    return {
        "generated_at": _utc_now_iso(),
        "result": assessment_to_dict(result),
        "image_data": _image_to_data_url(annotated),
        "geo_debug": {
            "candidate_count": len(result.candidates),
            "fusion": bool(result.fusion),
            "error": None,
            "safe_demo": True,
            "fallback_reason": reason,
        },
        "safe_demo": True,
    }


def _load_config_from_env(profile: Optional[str] = None) -> Optional[HeimdallConfig]:
    config_dir = APP_ROOT / "src" / "config"
    default_path = config_dir / "defaults.json"
    if profile:
        key = profile.strip().lower()
        profile_map = {
            "paris": "paris.json",
            "paris-focused": "paris.json",
            "paris-test": "paris_test.json",
            "paris_test": "paris_test.json",
            "legacy": "open_geo.json",
            "open_geo": "open_geo.json",
            "open-geo": "open_geo.json",
        }
        config_name = profile_map.get(key)
        if config_name:
            config_path = config_dir / config_name
            if config_path.exists():
                return load_config(str(config_path))
    if default_path.exists():
        return load_config(str(default_path))
    return None


def _config_path_for_profile(profile: Optional[str]) -> Path:
    config_dir = APP_ROOT / "src" / "config"
    default_path = config_dir / "defaults.json"
    if profile:
        key = profile.strip().lower()
        profile_map = {
            "paris": "paris.json",
            "paris-focused": "paris.json",
            "paris-test": "paris_test.json",
            "paris_test": "paris_test.json",
            "legacy": "open_geo.json",
            "open_geo": "open_geo.json",
            "open-geo": "open_geo.json",
        }
        config_name = profile_map.get(key)
        if config_name:
            config_path = config_dir / config_name
            if config_path.exists():
                return config_path
    return default_path


def _resolve_local_path(raw: Optional[str]) -> Optional[Path]:
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return APP_ROOT / path


def _check_python_deps() -> dict:
    checks = {}
    modules = [
        ("torch", True),
        ("ultralytics", True),
        ("fastapi", True),
        ("cv2", True),
        ("PIL", True),
    ]
    for name, required in modules:
        try:
            spec = importlib.util.find_spec(name)
            if spec is None:
                checks[name] = {"ok": False, "required": required, "error": "module not found"}
                continue
            version = None
            try:
                package_name = "Pillow" if name == "PIL" else name
                version = metadata.version(package_name)
            except Exception:
                version = None
            checks[name] = {
                "ok": True,
                "required": required,
                "version": version,
            }
        except Exception as exc:
            checks[name] = {"ok": False, "required": required, "error": str(exc)}
    return checks


def _check_config_paths() -> dict:
    config_dir = APP_ROOT / "src" / "config"
    files = {
        "defaults": config_dir / "defaults.json",
        "paris": config_dir / "paris.json",
        "paris_test": config_dir / "paris_test.json",
        "open_geo": config_dir / "open_geo.json",
    }
    out = {}
    for key, path in files.items():
        out[key] = {"path": str(path), "exists": path.exists()}
    return out


def _check_model_paths() -> dict:
    defaults_path = APP_ROOT / "src" / "config" / "defaults.json"
    out = {}
    if not defaults_path.exists():
        return {"defaults_config": {"path": str(defaults_path), "exists": False}}
    try:
        cfg = load_config(str(defaults_path))
    except Exception as exc:
        return {
            "defaults_config": {"path": str(defaults_path), "exists": True},
            "error": str(exc),
        }
    checks = {
        "detector_weights": _resolve_local_path(cfg.detector.weights_path),
        "geolocator_model_path": _resolve_local_path(cfg.geolocator.model_path),
        "geolocator_model_cache_dir": _resolve_local_path(cfg.geolocator.model_cache_dir),
        "retrieval_index_path": _resolve_local_path(cfg.geolocator.retrieval_index_path),
    }
    for key, path in checks.items():
        if path is None:
            out[key] = {"path": None, "exists": False, "configured": False}
        else:
            out[key] = {
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "configured": True,
            }
    if cfg.geolocator.retrieval_index_paths:
        multi = []
        for raw in cfg.geolocator.retrieval_index_paths:
            path = _resolve_local_path(raw)
            if path is None:
                continue
            multi.append(
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "is_dir": path.is_dir(),
                }
            )
        out["retrieval_index_paths"] = {
            "configured": True,
            "count": len(multi),
            "paths": multi,
        }
    return out


def _can_write(directory: Path) -> tuple[bool, Optional[str]]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=True,
            dir=str(directory),
            prefix="heimdall-health-",
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp:
            tmp.write("ok")
        return True, None
    except Exception as exc:
        return False, str(exc)


def _check_write_permissions() -> dict:
    targets = {
        "runs": APP_ROOT / "runs",
        "dashboard_data": APP_ROOT / "src" / "dashboard" / "data",
        "data": APP_ROOT / "data",
    }
    out = {}
    for key, path in targets.items():
        ok, error = _can_write(path)
        out[key] = {"path": str(path), "ok": ok, "error": error}
    return out


def _health_snapshot() -> dict:
    deps = _check_python_deps()
    config_paths = _check_config_paths()
    model_paths = _check_model_paths()
    write_permissions = _check_write_permissions()

    required_failures = []
    for name, check in deps.items():
        if check.get("required") and not check.get("ok"):
            required_failures.append(f"python_dep:{name}")
    for name, check in config_paths.items():
        if not check.get("exists"):
            required_failures.append(f"config:{name}")
    for name, check in write_permissions.items():
        if not check.get("ok"):
            required_failures.append(f"write:{name}")
    for name, check in model_paths.items():
        if not isinstance(check, dict) or not check.get("configured"):
            continue
        if name == "retrieval_index_paths":
            paths = check.get("paths")
            if isinstance(paths, list):
                any_exists = any(isinstance(item, dict) and item.get("exists") for item in paths)
                if not any_exists:
                    required_failures.append(f"model_path:{name}")
            continue
        if not check.get("exists"):
            required_failures.append(f"model_path:{name}")

    status = "ok" if not required_failures else "degraded"
    return {
        "status": status,
        "timestamp": _utc_now_iso(),
        "required_failures": required_failures,
        "deps": deps,
        "config_paths": config_paths,
        "model_paths": model_paths,
        "write_permissions": write_permissions,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/analysis")
def analysis_index() -> RedirectResponse:
    return RedirectResponse(url="/analysis/")


@app.get("/health")
def health() -> JSONResponse:
    snapshot = _health_snapshot()
    payload = {
        "status": snapshot["status"],
        "timestamp": snapshot["timestamp"],
        "required_failures": snapshot["required_failures"],
        "summary": {
            "deps_checked": len(snapshot["deps"]),
            "config_paths_checked": len(snapshot["config_paths"]),
            "model_paths_checked": len(snapshot["model_paths"]),
            "write_targets_checked": len(snapshot["write_permissions"]),
        },
    }
    return JSONResponse(payload, status_code=200 if snapshot["status"] == "ok" else 503)


@app.get("/health/deps")
def health_deps() -> JSONResponse:
    snapshot = _health_snapshot()
    return JSONResponse(snapshot, status_code=200 if snapshot["status"] == "ok" else 503)


@app.post("/analyze/image")
async def analyze_image(
    image: UploadFile = File(...),
    det_json: Optional[UploadFile] = File(None),
    geo_json: Optional[UploadFile] = File(None),
    profile: Optional[str] = None,
    safe_demo: Optional[str] = None,
) -> JSONResponse:
    force_safe_demo = _to_bool_flag(safe_demo)
    pipeline = None
    fallback_reason: Optional[str] = "forced" if force_safe_demo else None
    if not force_safe_demo:
        try:
            cfg = _load_config_from_env(profile)
            pipeline = build_pipeline(cfg)
        except Exception as exc:
            fallback_reason = f"pipeline init failed: {exc}"

    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / Path(image.filename).name
        image_path.write_bytes(await image.read())

        if det_json is not None:
            det_path = Path(str(image_path) + ".detections.json")
            det_path.write_bytes(await det_json.read())
        if geo_json is not None:
            geo_path = Path(str(image_path) + ".geo.json")
            geo_path.write_bytes(await geo_json.read())

        if fallback_reason is not None or pipeline is None:
            return JSONResponse(_make_demo_image_payload(image_path, fallback_reason))

        try:
            result = pipeline.run(str(image_path))
            annotated = draw_detections(str(image_path), result.detections)
            geo_debug = {
                "candidate_count": len(result.candidates),
                "fusion": bool(result.fusion),
                "error": getattr(pipeline.candidate_provider, "last_error", None),
                "safe_demo": False,
            }
            payload = {
                "generated_at": _utc_now_iso(),
                "result": assessment_to_dict(result),
                "image_data": _image_to_data_url(annotated),
                "geo_debug": geo_debug,
                "safe_demo": False,
            }
            return JSONResponse(payload)
        except Exception as exc:
            return JSONResponse(_make_demo_image_payload(image_path, f"pipeline run failed: {exc}"))


@app.post("/analyze/video")
async def analyze_video(
    video: UploadFile = File(...),
    interval_s: float = Form(2.0),
    max_frames: int = Form(12),
    profile: Optional[str] = None,
    safe_demo: Optional[str] = None,
) -> JSONResponse:
    force_safe_demo = _to_bool_flag(safe_demo)
    pipeline = None
    fallback_reason: Optional[str] = "forced" if force_safe_demo else None
    if not force_safe_demo:
        try:
            cfg = _load_config_from_env(profile)
            pipeline = build_pipeline(cfg)
        except Exception as exc:
            fallback_reason = f"pipeline init failed: {exc}"

    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / Path(video.filename).name
        video_path.write_bytes(await video.read())

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return JSONResponse({"error": "unable to read video"}, status_code=400)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval_frames = max(1, int(fps * interval_s))

        frames = []
        frame_index = 0
        extracted = 0
        while extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % interval_frames == 0:
                image_path = Path(tmp) / f"frame_{frame_index}.png"
                cv2.imwrite(str(image_path), frame)
                if fallback_reason is None and pipeline is not None:
                    try:
                        result = pipeline.run(str(image_path))
                    except Exception as exc:
                        fallback_reason = f"pipeline run failed: {exc}"
                        with Image.open(image_path) as img:
                            w, h = img.size
                        result = _build_demo_assessment(w, h, fallback_reason)
                else:
                    with Image.open(image_path) as img:
                        w, h = img.size
                    result = _build_demo_assessment(w, h, fallback_reason)
                annotated = draw_detections(str(image_path), result.detections)
                frames.append(
                    {
                        "timestamp_s": frame_index / fps,
                        "result": assessment_to_dict(result),
                        "image_data": _image_to_data_url(annotated),
                    }
                )
                extracted += 1
            frame_index += 1

        return JSONResponse({"frames": frames, "safe_demo": fallback_reason is not None})


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/png;base64," + encoded


@app.post("/eval/dota/start")
def start_eval() -> JSONResponse:
    if _EVAL_STATE["status"] == "running":
        return JSONResponse({"status": "running"})

    def _run() -> None:
        _EVAL_STATE["status"] = "running"
        try:
            from src.tools.run_dota_eval import main as run_dota_eval

            cfg = load_config("src/config/defaults.json")
            run_dota_eval(
                [
                    "--weights",
                    (cfg.detector.weights_path or "yolo11x-obb.pt"),
                    "--data",
                    "data/dota/dota.yaml",
                    "--imgsz",
                    str(cfg.detector.imgsz),
                    "--output",
                    "src/dashboard/data/dota_eval.json",
                ]
            )
            report = Path("src/dashboard/data/dota_eval.json")
            if report.exists():
                _EVAL_STATE["last_result"] = report.read_text(encoding="utf-8")
            _EVAL_STATE["status"] = "done"
        except Exception as exc:
            _EVAL_STATE["status"] = "error"
            _EVAL_STATE["last_result"] = str(exc)

    import threading

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "running"})


@app.get("/eval/dota/status")
def eval_status() -> JSONResponse:
    return JSONResponse(_EVAL_STATE)


@app.post("/eval/geo/start")
def start_geo_eval(
    images_dir: str,
    metadata: str,
    limit: int = 0,
    profile: Optional[str] = None,
    retrieval_only: Optional[str] = None,
) -> JSONResponse:
    if _GEO_EVAL_STATE["status"] == "running":
        return JSONResponse({"status": "running"})

    def _run() -> None:
        _GEO_EVAL_STATE["status"] = "running"
        try:
            output_path = Path("src/dashboard/data/geo_eval.json")
            progress_path = Path("src/dashboard/data/geo_eval.progress.json")
            _GEO_EVAL_STATE["progress_path"] = str(progress_path)
            from src.tools.run_geo_eval import main as run_geo_eval

            args = [
                "--images-dir",
                images_dir,
                "--metadata",
                metadata,
                "--output",
                str(output_path),
                "--progress",
                str(progress_path),
            ]
            if limit and limit > 0:
                args.extend(["--limit", str(limit)])
            args.extend(["--config", str(_config_path_for_profile(profile))])
            retrieval_flag = "1"
            if retrieval_only is not None and str(retrieval_only).lower() in {"0", "false", "no"}:
                retrieval_flag = "0"
            if retrieval_flag == "1":
                args.append("--retrieval-only")
            run_geo_eval(args)
            if output_path.exists():
                _GEO_EVAL_STATE["last_result"] = output_path.read_text(encoding="utf-8")
            _GEO_EVAL_STATE["status"] = "done"
        except Exception as exc:
            _GEO_EVAL_STATE["status"] = "error"
            _GEO_EVAL_STATE["last_result"] = str(exc)

    import threading

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "running"})


@app.get("/eval/geo/status")
def geo_eval_status() -> JSONResponse:
    progress_path = _GEO_EVAL_STATE.get("progress_path")
    if progress_path:
        path = Path(progress_path)
        if path.exists():
            try:
                _GEO_EVAL_STATE["progress"] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return JSONResponse(_GEO_EVAL_STATE)


def _extract_eval_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": path.stem,
        "path": str(path),
        "mean_km": payload.get("mean_km"),
        "median_km": payload.get("median_km"),
        "within_5km_pct": payload.get("within_5km_pct"),
        "within_10km_pct": payload.get("within_10km_pct"),
        "evaluated": payload.get("evaluated"),
    }


@app.post("/eval/benchmarks/start")
def start_benchmarks(
    images_dir: str = "data/spacenet_paris_test/chips",
    metadata: str = "data/spacenet_paris_test/metadata.csv",
    limit: int = 120,
    train_images_dir: str = "data/spacenet_paris/chips",
    train_metadata: str = "data/spacenet_paris/metadata.csv",
    eval_images_dir: str = "data/spacenet_paris_test/chips",
    eval_metadata: str = "data/spacenet_paris_test/metadata.csv",
    train_limit: int = 120,
    eval_limit: int = 60,
    model_ids: str = "openai/clip-vit-large-patch14,google/siglip-base-patch16-224",
    reuse_indices: Optional[str] = "1",
) -> JSONResponse:
    if _BENCHMARK_STATE["status"] == "running":
        return JSONResponse({"status": "running"})

    def _set_progress(current: int, total: int, stage: str, message: str) -> None:
        safe_total = max(1, int(total))
        safe_current = max(0, min(int(current), safe_total))
        pct = int(round(100.0 * (float(safe_current) / float(safe_total))))
        _BENCHMARK_STATE["stage"] = stage
        _BENCHMARK_STATE["progress"] = {
            "current": safe_current,
            "total": safe_total,
            "percent": pct,
            "message": message,
        }

    def _run() -> None:
        run_id = _benchmark_run_id()
        _BENCHMARK_STATE["status"] = "running"
        _BENCHMARK_STATE["run_id"] = run_id
        _BENCHMARK_STATE["last_result"] = None
        total_steps = 4
        _set_progress(0, total_steps, "init", "Preparing benchmark jobs")
        try:
            from src.tools.run_geo_eval import main as run_geo_eval
            from src.tools.benchmark_geo_backbones import main as run_backbone_bench

            out_dir = _benchmark_history_root() / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            ui_geo_leaky = out_dir / "bench_ui_current_leaky.json"
            ui_geo_realistic = out_dir / "bench_ui_realistic_single.json"
            ui_geo_multi = out_dir / "bench_ui_candidate_multi.json"
            ui_backbone = out_dir / "bench_ui_backbone.json"
            summary_out = APP_ROOT / "src" / "dashboard" / "data" / "benchmark_compare.json"
            summary_out.parent.mkdir(parents=True, exist_ok=True)
            run_summary_out = _benchmark_runs_dir() / f"{run_id}.json"

            geo_jobs = [
                ("leaky_reference", APP_ROOT / "runs" / "bench_cfg" / "cfg_current_leaky.json", ui_geo_leaky),
                ("realistic_single", APP_ROOT / "runs" / "bench_cfg" / "cfg_realistic_single.json", ui_geo_realistic),
                ("candidate_multi", APP_ROOT / "runs" / "bench_cfg" / "cfg_candidate_multi.json", ui_geo_multi),
            ]

            geo_results = []
            for name, cfg_path, out_path in geo_jobs:
                step_idx = len(geo_results)
                _set_progress(
                    step_idx,
                    total_steps,
                    f"geo_eval:{name}",
                    f"Running scenario: {name}",
                )
                run_geo_eval(
                    [
                        "--config",
                        str(cfg_path),
                        "--images-dir",
                        images_dir,
                        "--metadata",
                        metadata,
                        "--retrieval-only",
                        "--limit",
                        str(max(1, int(limit))),
                        "--output",
                        str(out_path),
                    ]
                )
                item = _extract_eval_summary(out_path)
                item["scenario"] = name
                geo_results.append(item)

            _set_progress(
                3,
                total_steps,
                "backbone_benchmark",
                "Running backbone model comparison",
            )
            bench_args = [
                "--train-images-dir",
                train_images_dir,
                "--train-metadata",
                train_metadata,
                "--eval-images-dir",
                eval_images_dir,
                "--eval-metadata",
                eval_metadata,
                "--model-ids",
                model_ids,
                "--train-limit",
                str(max(1, int(train_limit))),
                "--eval-limit",
                str(max(1, int(eval_limit))),
                "--output",
                str(ui_backbone),
            ]
            if _to_bool_flag(reuse_indices):
                bench_args.append("--reuse-indices")
            run_backbone_bench(bench_args)
            backbone_payload = json.loads(ui_backbone.read_text(encoding="utf-8"))
            backbone_rows = backbone_payload.get("models", [])

            summary = {
                "run_id": run_id,
                "generated_at": _utc_now_iso(),
                "inputs": {
                    "images_dir": images_dir,
                    "metadata": metadata,
                    "limit": max(1, int(limit)),
                    "train_images_dir": train_images_dir,
                    "train_metadata": train_metadata,
                    "eval_images_dir": eval_images_dir,
                    "eval_metadata": eval_metadata,
                    "train_limit": max(1, int(train_limit)),
                    "eval_limit": max(1, int(eval_limit)),
                    "model_ids": [item.strip() for item in str(model_ids).split(",") if item.strip()],
                    "reuse_indices": _to_bool_flag(reuse_indices),
                },
                "geo_scenarios": geo_results,
                "backbone_benchmark": {
                    "best_model": backbone_payload.get("best_model"),
                    "ranked_by_median_km": backbone_payload.get("ranked_by_median_km", []),
                    "models": backbone_rows,
                    "path": str(ui_backbone),
                },
            }
            summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            run_summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            _BENCHMARK_STATE["last_result"] = json.dumps(summary, indent=2)
            _BENCHMARK_STATE["status"] = "done"
            _set_progress(4, total_steps, "done", "Benchmark run complete")
        except Exception as exc:
            _BENCHMARK_STATE["status"] = "error"
            _BENCHMARK_STATE["stage"] = "error"
            if _BENCHMARK_STATE.get("progress"):
                _BENCHMARK_STATE["progress"]["message"] = "Benchmark run failed"
            _BENCHMARK_STATE["last_result"] = str(exc)

    import threading

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "running"})


@app.get("/eval/benchmarks/status")
def benchmark_status() -> JSONResponse:
    return JSONResponse(_BENCHMARK_STATE)


@app.get("/eval/benchmarks/runs")
def list_benchmark_runs(limit: int = 100) -> JSONResponse:
    return JSONResponse({"runs": _list_benchmark_runs(limit=limit)})


@app.get("/eval/benchmarks/runs/{run_id}")
def get_benchmark_run(run_id: str) -> JSONResponse:
    if not _is_safe_run_id(run_id):
        return JSONResponse({"error": "invalid_run_id"}, status_code=400)
    path = _benchmark_runs_dir() / f"{run_id}.json"
    if not path.exists():
        return JSONResponse({"error": "run_not_found"}, status_code=404)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return JSONResponse({"error": "run_read_error"}, status_code=500)
    return JSONResponse(payload)


@app.post("/eval/benchmarks/compare")
def compare_benchmark_runs(
    baseline_run_id: str,
    candidate_run_id: str,
    append_progress: Optional[str] = "0",
) -> JSONResponse:
    if not _is_safe_run_id(baseline_run_id) or not _is_safe_run_id(candidate_run_id):
        return JSONResponse({"error": "invalid_run_id"}, status_code=400)
    baseline = _load_benchmark_run_payload(baseline_run_id)
    candidate = _load_benchmark_run_payload(candidate_run_id)
    if baseline is None:
        return JSONResponse({"error": f"baseline_not_found:{baseline_run_id}"}, status_code=404)
    if candidate is None:
        return JSONResponse({"error": f"candidate_not_found:{candidate_run_id}"}, status_code=404)

    compare = _compare_benchmark_payloads(baseline=baseline, candidate=candidate)
    out_path = _benchmark_compares_dir() / f"{compare['compare_id']}.json"
    out_path.write_text(json.dumps(compare, indent=2), encoding="utf-8")
    compare["compare_path"] = str(out_path)

    appended = False
    if _to_bool_flag(append_progress):
        _append_progress_snippet(str(compare.get("progress_md_snippet", "")))
        appended = True
    compare["progress_appended"] = appended
    return JSONResponse(compare)


@app.post("/fs/pick_dir")
def pick_dir() -> JSONResponse:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory()
        root.destroy()
        return JSONResponse({"path": path})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/fs/pick_file")
def pick_file() -> JSONResponse:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select metadata CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        return JSONResponse({"path": path})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# Static mounts come last so API routes are not shadowed.
app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR), name="dashboard")
app.mount("/analysis", StaticFiles(directory=LIVE_DIR, html=True), name="analysis")
app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="root")
