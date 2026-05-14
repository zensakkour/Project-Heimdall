"""
FastAPI server for live analysis UI.
"""
from __future__ import annotations

import base64
import asyncio
import hashlib
import importlib
from dataclasses import replace as dataclass_replace
from importlib import metadata
import json
import io
import logging
import math
import multiprocessing as mp
import os
import random
from queue import Empty as QueueEmpty
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
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


# Operator Session State
_OPERATOR_SESSION: dict = {
    "session_id": None,
    "status": "ready",
    "source": None,
    "fused_estimate": None,
    "candidates": [],
    "clues": [],
    "detections": [],
    "warnings": [],
    "timeline": [],
    "operator_notes": "",
    "operator_pins": [],
    "updated_at": None,
}

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _save_operator_session(custom_name: str = None):
    global _OPERATOR_SESSION
    if not _OPERATOR_SESSION.get("session_id"):
        return
    _OPERATOR_SESSION["updated_at"] = _utc_now_iso()
    if custom_name:
        _OPERATOR_SESSION["custom_name"] = custom_name
    elif "custom_name" in _OPERATOR_SESSION:
        custom_name = _OPERATOR_SESSION["custom_name"]

    from datetime import datetime, timezone
    dt_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H-%M")
    display_name = f"{dt_str} - {custom_name}" if custom_name else f"auto_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    _OPERATOR_SESSION["display_name"] = display_name

    session_id = _OPERATOR_SESSION["session_id"]
    sessions_dir = APP_ROOT / "operator_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Each session lives in its own folder named by session_id
    session_dir = sessions_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Pull out and save the source image as a separate file
    session_data = dict(_OPERATOR_SESSION)
    source = (session_data.get("source") or {}).copy()
    image_data_url = source.pop("image_data_url", None)

    if image_data_url:
        try:
            header, b64_data = image_data_url.split(",", 1)
            content_type = header.split(":")[1].split(";")[0]  # e.g. image/jpeg
            ext = content_type.split("/")[-1].replace("jpeg", "jpg")
            image_bytes = base64.b64decode(b64_data)
            img_filename = f"source.{ext}"
            (session_dir / img_filename).write_bytes(image_bytes)
            source["image_file"] = img_filename          # pointer, not the data
        except Exception:
            source["image_data_url"] = image_data_url    # fallback: keep embedded

    if session_data.get("source") is not None:
        session_data["source"] = source

    with open(session_dir / "session.json", "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)

    # Remove legacy flat files for this session_id if any
    for old in sessions_dir.glob(f"session_*_{session_id}.json"):
        old.unlink(missing_ok=True)

def _reset_operator_session():
    global _OPERATOR_SESSION
    _OPERATOR_SESSION = {
        "session_id": uuid.uuid4().hex,
        "status": "ready",
        "source": None,
        "fused_estimate": None,
        "candidates": [],
        "clues": [],
        "detections": [],
        "warnings": [],
        "timeline": [],
        "operator_notes": "",
        "operator_pins": [],
        "updated_at": _utc_now_iso(),
    }

_reset_operator_session()

def _add_timeline_event(message: str, level: str = "info"):
    _OPERATOR_SESSION["timeline"].append({
        "timestamp": _utc_now_iso(),
        "message": message,
        "level": level
    })


app = FastAPI()

_EVAL_STATE = {"status": "idle", "last_result": None}
_GEO_EVAL_STATE = {
    "status": "idle",
    "last_result": None,
    "progress": None,
    "progress_path": None,
    "profile_requested": None,
    "profile_effective": None,
    "profile_warning": None,
    "config_path": None,
}
_GEO_RANDOM_STATE = {
    "status": "idle",
    "last_result": None,
    "progress": None,
    "progress_path": None,
    "seed": None,
    "profile_requested": None,
    "profile_effective": None,
    "profile_warning": None,
    "config_path": None,
}
_BENCHMARK_STATE = {"status": "idle", "stage": None, "last_result": None, "progress": None, "run_id": None}


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value

_WORKER_ENABLED_DEFAULT = os.getenv("HEIMDALL_USE_INFERENCE_WORKER", "1")
_WORKER_IMAGE_TIMEOUT_S = float(os.getenv("HEIMDALL_INFERENCE_TIMEOUT_S", "900"))
_WORKER_VIDEO_TIMEOUT_S = float(os.getenv("HEIMDALL_VIDEO_TIMEOUT_S", "900"))
_MAX_IMAGE_BYTES = int(os.getenv("HEIMDALL_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
_MAX_VIDEO_BYTES = int(os.getenv("HEIMDALL_MAX_VIDEO_BYTES", str(256 * 1024 * 1024)))
_MAX_METADATA_BYTES = int(os.getenv("HEIMDALL_MAX_METADATA_BYTES", str(4 * 1024 * 1024)))
_ANALYSIS_CONCURRENCY = int(os.getenv("HEIMDALL_ANALYSIS_CONCURRENCY", "2"))
_ANALYSIS_QUEUE_TIMEOUT_S = float(os.getenv("HEIMDALL_ANALYSIS_QUEUE_TIMEOUT_S", "5"))
_LOGGER = logging.getLogger("heimdall.ui_server")
if not _LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
_ANALYSIS_SEMAPHORE = asyncio.Semaphore(max(1, _ANALYSIS_CONCURRENCY))
_ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
_ALLOWED_VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-msvideo",
    "application/octet-stream",
}


def build_pipeline(cfg: Optional[HeimdallConfig]) -> "HeimdallPipeline":
    # Lazy imports keep app startup resilient when optional heavy deps are unavailable.
    from src.core.detection.factory import create_detector
    from src.core.geo import GeoCLIPProvider, GeoLocator, GeoRetrievalProvider, MultiCandidateProvider
    from src.core.logic.pipeline import HeimdallPipeline

    if cfg is None:
        return HeimdallPipeline()
    detector_tuple = create_detector(cfg.detector)
    detector = detector_tuple[0] if detector_tuple else None
    backend = detector_tuple[1] if detector_tuple else "none"
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
        index_projection_paths=cfg.geolocator.retrieval_index_projection_paths,
        model_id=cfg.geolocator.retrieval_model_id or "openai/clip-vit-large-patch14",
        projection_path=cfg.geolocator.retrieval_projection_path,
        top_k=cfg.geolocator.retrieval_top_k,
        per_index_top_k=cfg.geolocator.retrieval_per_index_top_k,
        index_score_norm=cfg.geolocator.retrieval_index_score_norm,
        source_fusion_mode=cfg.geolocator.retrieval_source_fusion_mode,
        source_balance_beta=cfg.geolocator.retrieval_source_balance_beta,
        min_score=cfg.geolocator.retrieval_min_score,
        min_keep_topk=cfg.geolocator.retrieval_min_keep_topk,
        diversity_radius_km=cfg.geolocator.retrieval_diversity_radius_km,
        diversity_lambda=cfg.geolocator.retrieval_diversity_lambda,
        diversity_min_keep=cfg.geolocator.retrieval_diversity_min_keep,
        locality_radius_km=cfg.geolocator.retrieval_locality_radius_km,
        locality_weight=cfg.geolocator.retrieval_locality_weight,
        consensus_top_n=cfg.geolocator.retrieval_consensus_top_n,
        consensus_radius_km=cfg.geolocator.retrieval_consensus_radius_km,
        consensus_score_power=cfg.geolocator.retrieval_consensus_score_power,
        query_tta_degrees=cfg.geolocator.retrieval_query_tta_degrees,
        query_tta_modes=cfg.geolocator.retrieval_query_tta_modes,
        query_tta_scales=cfg.geolocator.retrieval_query_tta_scales,
        query_tta_auto_modality=cfg.geolocator.retrieval_query_tta_auto_modality,
        query_tta_reduce=cfg.geolocator.retrieval_query_tta_reduce,
        query_expansion_top_n=cfg.geolocator.retrieval_query_expansion_top_n,
        query_expansion_beta=cfg.geolocator.retrieval_query_expansion_beta,
        query_expansion_alpha=cfg.geolocator.retrieval_query_expansion_alpha,
        tta_agreement_top_n=cfg.geolocator.retrieval_tta_agreement_top_n,
        tta_agreement_weight=cfg.geolocator.retrieval_tta_agreement_weight,
        local_match_top_n=cfg.geolocator.retrieval_local_match_top_n,
        local_match_weight=cfg.geolocator.retrieval_local_match_weight,
        local_match_ratio=cfg.geolocator.retrieval_local_match_ratio,
        local_match_max_features=cfg.geolocator.retrieval_local_match_max_features,
        structure_rerank_top_n=cfg.geolocator.retrieval_structure_rerank_top_n,
        structure_rerank_weight=cfg.geolocator.retrieval_structure_rerank_weight,
        graph_rerank_top_n=cfg.geolocator.retrieval_graph_rerank_top_n,
        graph_rerank_sigma_km=cfg.geolocator.retrieval_graph_rerank_sigma_km,
        graph_rerank_score_alpha=cfg.geolocator.retrieval_graph_rerank_score_alpha,
        graph_rerank_support_beta=cfg.geolocator.retrieval_graph_rerank_support_beta,
        graph_rerank_center_radius_km=cfg.geolocator.retrieval_graph_rerank_center_radius_km,
        kde_refine_top_n=cfg.geolocator.retrieval_kde_refine_top_n,
        kde_refine_sigma_km=cfg.geolocator.retrieval_kde_refine_sigma_km,
        kde_refine_score_power=cfg.geolocator.retrieval_kde_refine_score_power,
        kde_refine_margin_threshold=cfg.geolocator.retrieval_kde_refine_margin_threshold,
        kde_refine_switch_radius_km=cfg.geolocator.retrieval_kde_refine_switch_radius_km,
        kde_refine_max_iters=cfg.geolocator.retrieval_kde_refine_max_iters,
        kde_refine_adaptive_mass=cfg.geolocator.retrieval_kde_refine_adaptive_mass,
        geo_prior_mode=cfg.geolocator.retrieval_geo_prior_mode,
        geo_prior_bbox=cfg.geolocator.retrieval_geo_prior_bbox,
        geo_prior_sigma_km=cfg.geolocator.retrieval_geo_prior_sigma_km,
        geo_prior_min_keep=cfg.geolocator.retrieval_geo_prior_min_keep,
    )
    if has_retrieval_index(cfg.geolocator):
        if cfg.geolocator.use_geoclip_with_retrieval:
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
            candidate_provider = MultiCandidateProvider(
                [retrieval_provider, geoclip_provider],
                dedupe_radius_m=cfg.geolocator.candidate_dedupe_radius_m,
                source_balance_beta=cfg.geolocator.candidate_source_balance_beta,
                max_candidates=cfg.geolocator.candidate_max_results,
            )
        else:
            candidate_provider = retrieval_provider
    else:
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
        candidate_provider = geoclip_provider
    return HeimdallPipeline(
        detector=detector,
        geolocator=geolocator,
        candidate_provider=candidate_provider,
        fusion_config=cfg.fusion,
        score_config=cfg.score,
        verification_config=cfg.verification,
        detector_backend=backend,
    )


def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "ts": _utc_now_iso()}
    payload.update(fields)
    try:
        _LOGGER.info(json.dumps(payload, sort_keys=True, default=str))
    except Exception:
        _LOGGER.info("%s %s", event, fields)


def _use_inference_worker() -> bool:
    return str(os.getenv("HEIMDALL_USE_INFERENCE_WORKER", _WORKER_ENABLED_DEFAULT)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _to_bool_flag(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_upload_name(filename: Optional[str], fallback: str) -> str:
    name = Path(filename or fallback).name
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_", ".", " "}).strip()
    return cleaned or fallback


def _normalized_content_type(upload: UploadFile) -> str:
    raw = (upload.content_type or "").split(";", 1)[0].strip().lower()
    return raw


def _validate_upload_content_type(upload: UploadFile, allowed: set[str], label: str) -> Optional[str]:
    ctype = _normalized_content_type(upload)
    if not ctype:
        return None
    if ctype in allowed:
        return None
    return f"unsupported {label} content-type: {ctype}"


async def _write_upload_limited(upload: UploadFile, target: Path, max_bytes: int) -> int:
    total = 0
    with target.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"upload too large ({total} bytes > {max_bytes})")
            handle.write(chunk)
    return total


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _runtime_manifest(profile: Optional[str]) -> dict:
    cfg_path = _config_path_for_profile(profile)

    def _version(pkg: str) -> Optional[str]:
        try:
            return metadata.version(pkg)
        except Exception:
            return None

    return {
        "profile": (profile or "default"),
        "config_path": str(cfg_path),
        "config_sha256": _file_sha256(cfg_path),
        "env": {
            "inference_worker_enabled": _use_inference_worker(),
            "timeouts_s": {
                "image": _WORKER_IMAGE_TIMEOUT_S,
                "video": _WORKER_VIDEO_TIMEOUT_S,
            },
            "limits": {
                "max_image_bytes": _MAX_IMAGE_BYTES,
                "max_video_bytes": _MAX_VIDEO_BYTES,
                "max_metadata_bytes": _MAX_METADATA_BYTES,
                "analysis_concurrency": max(1, _ANALYSIS_CONCURRENCY),
                "analysis_queue_timeout_s": _ANALYSIS_QUEUE_TIMEOUT_S,
            },
        },
        "packages": {
            "fastapi": _version("fastapi"),
            "uvicorn": _version("uvicorn"),
            "numpy": _version("numpy"),
            "pillow": _version("Pillow"),
        },
    }


@app.middleware("http")
async def request_telemetry_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        _log_event(
            "request.error",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            error=str(exc),
        )
        raise
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    _log_event(
        "request.complete",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


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
        backend="demo",
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
        "fallback_reason": reason,
    }


def _analysis_error_response(
    error: str,
    *,
    request_id: str,
    timings_ms: dict[str, float],
    worker_mode: str,
    manifest: Optional[dict] = None,
    status_code: int = 503,
) -> JSONResponse:
    payload = {
        "error": error,
        "safe_demo": False,
    }
    _attach_runtime_meta(
        payload,
        request_id=request_id,
        timings_ms=timings_ms,
        worker_mode=worker_mode,
        manifest=manifest,
    )
    return JSONResponse(payload, status_code=status_code)


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
        }
        config_name = profile_map.get(key)
        if config_name:
            config_path = config_dir / config_name
            if config_path.exists():
                return load_config(str(config_path))
    if default_path.exists():
        return load_config(str(default_path))
    return None


def _with_forced_sidecar(cfg: HeimdallConfig) -> HeimdallConfig:
    return dataclass_replace(
        cfg,
        detector=dataclass_replace(
            cfg.detector,
            backend="sidecar",
            use_sidecar=True,
            weights_path=None,
        ),
        geolocator=dataclass_replace(
            cfg.geolocator,
            use_sidecar=True,
        ),
    )


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
        }
        config_name = profile_map.get(key)
        if config_name:
            config_path = config_dir / config_name
            if config_path.exists():
                return config_path
    return default_path


def _infer_profile_from_paths(images_dir: str, metadata: str) -> Optional[str]:
    joined = f"{images_dir} {metadata}".replace("\\", "/").lower()
    if "spacenet_paris_test" in joined:
        return "paris_test"
    if "spacenet_paris" in joined:
        return "paris"
    return None


def _is_legacy_profile(profile: Optional[str]) -> bool:
    key = str(profile or "").strip().lower()
    return key == "legacy"


def _resolve_eval_profile(
    requested_profile: Optional[str],
    *,
    images_dir: str,
    metadata: str,
) -> tuple[Optional[str], Optional[str]]:
    inferred = _infer_profile_from_paths(images_dir=images_dir, metadata=metadata)
    requested = str(requested_profile or "").strip().lower() or None
    if inferred in {"paris", "paris_test"} and _is_legacy_profile(requested):
        warning = (
            f"profile auto-corrected to '{inferred}' because dataset paths are Paris SpaceNet "
            "(requested retired legacy profile)."
        )
        return inferred, warning
    if requested is None and inferred is not None:
        return inferred, None
    return requested_profile, None


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
        ("rfdetr", False),
        ("ultralytics", True),
        ("fastapi", True),
        ("cv2", False),
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


def _make_demo_video_payload(reason: Optional[str], interval_s: float, max_frames: int) -> dict:
    frame_count = max(1, min(max_frames, 4))
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for idx in range(frame_count):
            image_path = Path(tmp) / f"safe_demo_frame_{idx}.png"
            Image.new("RGB", (1280, 720), color=(10, 14, 16)).save(image_path, format="PNG")
            result = _build_demo_assessment(1280, 720, reason)
            annotated = draw_detections(str(image_path), result.detections)
            frames.append(
                {
                    "timestamp_s": round(idx * max(interval_s, 0.25), 3),
                    "result": assessment_to_dict(result),
                    "image_data": _image_to_data_url(annotated),
                }
            )
    return {
        "generated_at": _utc_now_iso(),
        "frames": frames,
        "safe_demo": True,
        "fallback_reason": reason,
    }


def _run_image_pipeline_local(
    image_path: Path,
    profile: Optional[str],
    force_safe_demo: bool,
    force_sidecar: bool = False,
    allow_demo_fallback: bool = True,
) -> dict:
    fallback_reason: Optional[str] = "forced" if force_safe_demo else None
    pipeline = None
    if fallback_reason is None:
        try:
            cfg = _load_config_from_env(profile)
            if cfg is not None and force_sidecar:
                cfg = _with_forced_sidecar(cfg)
            pipeline = build_pipeline(cfg)
        except Exception as exc:
            fallback_reason = f"pipeline init failed: {exc}"
    if fallback_reason is None and pipeline is not None:
        try:
            result = pipeline.run(str(image_path))
            annotated = draw_detections(str(image_path), result.detections)
            return {
                "generated_at": _utc_now_iso(),
                "result": assessment_to_dict(result),
                "image_data": _image_to_data_url(annotated),
                "geo_debug": {
                    "candidate_count": len(result.candidates),
                    "fusion": bool(result.fusion),
                    "error": getattr(pipeline.candidate_provider, "last_error", None),
                    "safe_demo": False,
                },
                "safe_demo": False,
            }
        except Exception as exc:
            fallback_reason = f"pipeline run failed: {exc}"
    if allow_demo_fallback:
        return _make_demo_image_payload(image_path, fallback_reason)
    raise RuntimeError(fallback_reason or "image pipeline failed")


def _run_video_pipeline_local(
    video_path: Path,
    interval_s: float,
    max_frames: int,
    profile: Optional[str],
    force_safe_demo: bool,
) -> dict:
    fallback_reason: Optional[str] = "forced" if force_safe_demo else None
    pipeline = None
    if fallback_reason is None:
        try:
            cfg = _load_config_from_env(profile)
            pipeline = build_pipeline(cfg)
        except Exception as exc:
            fallback_reason = f"pipeline init failed: {exc}"

    try:
        import cv2
    except Exception as exc:
        return _make_demo_video_payload(f"opencv unavailable: {exc}", interval_s, max_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("unable to read video")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval_frames = max(1, int(fps * interval_s))
        with tempfile.TemporaryDirectory() as tmp:
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
            return {
                "generated_at": _utc_now_iso(),
                "frames": frames,
                "safe_demo": fallback_reason is not None,
                "fallback_reason": fallback_reason,
            }
    finally:
        cap.release()


def _inference_worker(task: dict, result_queue: Any) -> None:
    started = time.perf_counter()
    action = task.get("action")
    try:
        if action == "image":
            payload = _run_image_pipeline_local(
                image_path=Path(task["image_path"]),
                profile=task.get("profile"),
                force_safe_demo=bool(task.get("force_safe_demo", False)),
                allow_demo_fallback=bool(task.get("force_safe_demo", False)),
            )
        elif action == "video":
            payload = _run_video_pipeline_local(
                video_path=Path(task["video_path"]),
                interval_s=float(task.get("interval_s", 2.0)),
                max_frames=int(task.get("max_frames", 12)),
                profile=task.get("profile"),
                force_safe_demo=bool(task.get("force_safe_demo", False)),
            )
        else:
            raise RuntimeError(f"unknown inference action: {action}")
        result_queue.put(
            {
                "ok": True,
                "payload": payload,
                "worker_time_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "ok": False,
                "error": str(exc),
                "worker_time_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )


def _run_inference_worker(task: dict, timeout_s: float) -> tuple[Optional[dict], Optional[str], float]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_inference_worker, args=(task, result_queue), daemon=True)
    started = time.perf_counter()
    process.start()
    try:
        result = result_queue.get(timeout=timeout_s)
    except QueueEmpty:
        if process.is_alive():
            process.terminate()
            process.join(2)
        return None, f"inference timeout after {timeout_s:.1f}s", round((time.perf_counter() - started) * 1000, 2)

    process.join(2)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)

    if not result.get("ok"):
        return None, str(result.get("error") or "worker failure"), duration_ms
    return result.get("payload"), None, duration_ms


def _attach_runtime_meta(
    payload: dict,
    request_id: str,
    timings_ms: dict,
    worker_mode: str,
    manifest: Optional[dict] = None,
) -> dict:
    payload["request_id"] = request_id
    payload["runtime"] = {
        "worker_mode": worker_mode,
        "timings_ms": timings_ms,
    }
    if manifest is not None:
        payload["runtime"]["manifest"] = manifest
    return payload


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


@app.get("/health/runtime")
def health_runtime() -> JSONResponse:
    payload = {
        "status": "ok",
        "timestamp": _utc_now_iso(),
        "inference_worker_enabled": _use_inference_worker(),
        "timeouts_s": {
            "image": _WORKER_IMAGE_TIMEOUT_S,
            "video": _WORKER_VIDEO_TIMEOUT_S,
        },
        "limits": {
            "max_image_bytes": _MAX_IMAGE_BYTES,
            "max_video_bytes": _MAX_VIDEO_BYTES,
            "max_metadata_bytes": _MAX_METADATA_BYTES,
            "analysis_concurrency": max(1, _ANALYSIS_CONCURRENCY),
            "analysis_queue_timeout_s": _ANALYSIS_QUEUE_TIMEOUT_S,
        },
    }
    return JSONResponse(payload)


@app.post("/analyze/image")
async def analyze_image(
    request: Request,
    image: UploadFile = File(...),
    det_json: Optional[UploadFile] = File(None),
    geo_json: Optional[UploadFile] = File(None),
    profile: Optional[str] = None,
    safe_demo: Optional[str] = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    timings_ms: dict[str, float] = {}
    started = time.perf_counter()
    force_safe_demo = _to_bool_flag(safe_demo)
    worker_mode = "process" if _use_inference_worker() else "inline"
    manifest = _runtime_manifest(profile)
    acquired = False
    try:
        try:
            await asyncio.wait_for(_ANALYSIS_SEMAPHORE.acquire(), timeout=_ANALYSIS_QUEUE_TIMEOUT_S)
            acquired = True
        except asyncio.TimeoutError:
            _log_event("analyze.image.rejected", request_id=request_id, reason="queue_timeout")
            return JSONResponse(
                {"error": "analysis queue is busy, retry shortly", "request_id": request_id},
                status_code=429,
            )

        image_type_error = _validate_upload_content_type(image, _ALLOWED_IMAGE_CONTENT_TYPES, "image")
        if image_type_error:
            _log_event("analyze.image.rejected", request_id=request_id, reason=image_type_error)
            return JSONResponse({"error": image_type_error, "request_id": request_id}, status_code=415)

        with tempfile.TemporaryDirectory() as tmp:
            io_started = time.perf_counter()
            image_path = Path(tmp) / _safe_upload_name(image.filename, "upload-image.bin")
            try:
                image_size = await _write_upload_limited(image, image_path, _MAX_IMAGE_BYTES)
            except ValueError as exc:
                _log_event("analyze.image.rejected", request_id=request_id, reason=str(exc))
                return JSONResponse({"error": str(exc), "request_id": request_id}, status_code=413)
            timings_ms["image_bytes"] = float(image_size)

            if det_json is not None:
                det_path = Path(str(image_path) + ".detections.json")
                try:
                    det_size = await _write_upload_limited(det_json, det_path, _MAX_METADATA_BYTES)
                    timings_ms["det_json_bytes"] = float(det_size)
                except ValueError as exc:
                    return JSONResponse({"error": str(exc), "request_id": request_id}, status_code=413)
            if geo_json is not None:
                geo_path = Path(str(image_path) + ".geo.json")
                try:
                    geo_size = await _write_upload_limited(geo_json, geo_path, _MAX_METADATA_BYTES)
                    timings_ms["geo_json_bytes"] = float(geo_size)
                except ValueError as exc:
                    return JSONResponse({"error": str(exc), "request_id": request_id}, status_code=413)
            timings_ms["io_write"] = round((time.perf_counter() - io_started) * 1000, 2)
            has_uploaded_sidecars = det_json is not None or geo_json is not None
            if has_uploaded_sidecars and worker_mode == "process":
                worker_mode = "inline"
                _log_event(
                    "analyze.image.sidecar_inline",
                    request_id=request_id,
                    reason="uploaded sidecars require same-process config/sidecar semantics",
                )

            if worker_mode == "process":
                payload, worker_error, worker_roundtrip_ms = _run_inference_worker(
                    {
                        "action": "image",
                        "image_path": str(image_path),
                        "profile": profile,
                        "force_safe_demo": force_safe_demo,
                    },
                    timeout_s=_WORKER_IMAGE_TIMEOUT_S,
                )
                timings_ms["worker_roundtrip"] = worker_roundtrip_ms
                if payload is not None and payload.get("safe_demo") and not force_safe_demo:
                    worker_error = (
                        payload.get("geo_debug", {}).get("fallback_reason")
                        or payload.get("fallback_reason")
                        or "worker returned demo fallback"
                    )
                    payload = None
                if payload is None and force_safe_demo:
                    fallback_reason = f"worker failure: {worker_error}"
                    payload = _make_demo_image_payload(image_path, fallback_reason)
                    _log_event(
                        "inference.worker_fallback",
                        request_id=request_id,
                        path="/analyze/image",
                        reason=fallback_reason,
                    )
            else:
                infer_started = time.perf_counter()
                try:
                    payload = _run_image_pipeline_local(
                        image_path=image_path,
                        profile=profile,
                        force_safe_demo=force_safe_demo,
                        force_sidecar=has_uploaded_sidecars,
                        allow_demo_fallback=force_safe_demo,
                    )
                except Exception as exc:
                    timings_ms["inline_inference"] = round((time.perf_counter() - infer_started) * 1000, 2)
                    timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
                    error = f"analysis failed: {exc}"
                    _log_event(
                        "analyze.image.failed",
                        request_id=request_id,
                        worker_mode=worker_mode,
                        reason=error,
                        total_ms=timings_ms["total"],
                    )
                    return _analysis_error_response(
                        error,
                        request_id=request_id,
                        timings_ms=timings_ms,
                        worker_mode=worker_mode,
                        manifest=manifest,
                    )
                timings_ms["inline_inference"] = round((time.perf_counter() - infer_started) * 1000, 2)
            if worker_mode == "process" and payload is None and not force_safe_demo:
                infer_started = time.perf_counter()
                worker_mode = "process-inline-fallback"
                try:
                    payload = _run_image_pipeline_local(
                        image_path=image_path,
                        profile=profile,
                        force_safe_demo=False,
                        force_sidecar=has_uploaded_sidecars,
                        allow_demo_fallback=False,
                    )
                except Exception as exc:
                    timings_ms["inline_inference"] = round((time.perf_counter() - infer_started) * 1000, 2)
                    timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
                    error = f"analysis failed: worker failure: {worker_error}; inline retry failed: {exc}"
                    _log_event(
                        "analyze.image.failed",
                        request_id=request_id,
                        worker_mode=worker_mode,
                        reason=error,
                        total_ms=timings_ms["total"],
                    )
                    return _analysis_error_response(
                        error,
                        request_id=request_id,
                        timings_ms=timings_ms,
                        worker_mode=worker_mode,
                        manifest=manifest,
                    )
                timings_ms["inline_inference"] = round((time.perf_counter() - infer_started) * 1000, 2)
            elif worker_mode == "process" and payload is None:
                payload = _make_demo_image_payload(image_path, f"worker failure: {worker_error}")
            if payload.get("safe_demo") and not force_safe_demo:
                timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
                reason = (
                    payload.get("geo_debug", {}).get("fallback_reason")
                    or payload.get("fallback_reason")
                    or "unexpected demo fallback"
                )
                error = f"analysis failed: {reason}"
                _log_event(
                    "analyze.image.failed",
                    request_id=request_id,
                    worker_mode=worker_mode,
                    reason=error,
                    total_ms=timings_ms["total"],
                )
                return _analysis_error_response(
                    error,
                    request_id=request_id,
                    timings_ms=timings_ms,
                    worker_mode=worker_mode,
                    manifest=manifest,
                )

        timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
        _attach_runtime_meta(
            payload,
            request_id=request_id,
            timings_ms=timings_ms,
            worker_mode=worker_mode,
            manifest=manifest,
        )
        _log_event(
            "analyze.image",
            request_id=request_id,
            worker_mode=worker_mode,
            safe_demo=bool(payload.get("safe_demo")),
            total_ms=timings_ms["total"],
        )
        return JSONResponse(payload)
    finally:
        if acquired:
            _ANALYSIS_SEMAPHORE.release()


@app.post("/analyze/video")
async def analyze_video(
    request: Request,
    video: UploadFile = File(...),
    interval_s: float = Form(2.0),
    max_frames: int = Form(12),
    profile: Optional[str] = None,
    safe_demo: Optional[str] = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    timings_ms: dict[str, float] = {}
    started = time.perf_counter()
    force_safe_demo = _to_bool_flag(safe_demo)
    worker_mode = "process" if _use_inference_worker() else "inline"
    manifest = _runtime_manifest(profile)
    acquired = False
    try:
        try:
            await asyncio.wait_for(_ANALYSIS_SEMAPHORE.acquire(), timeout=_ANALYSIS_QUEUE_TIMEOUT_S)
            acquired = True
        except asyncio.TimeoutError:
            _log_event("analyze.video.rejected", request_id=request_id, reason="queue_timeout")
            return JSONResponse(
                {"error": "analysis queue is busy, retry shortly", "request_id": request_id},
                status_code=429,
            )

        if interval_s <= 0 or interval_s > 60:
            return JSONResponse(
                {"error": "interval_s must be in (0, 60]", "request_id": request_id},
                status_code=400,
            )
        if max_frames < 1 or max_frames > 400:
            return JSONResponse(
                {"error": "max_frames must be in [1, 400]", "request_id": request_id},
                status_code=400,
            )

        video_type_error = _validate_upload_content_type(video, _ALLOWED_VIDEO_CONTENT_TYPES, "video")
        if video_type_error:
            _log_event("analyze.video.rejected", request_id=request_id, reason=video_type_error)
            return JSONResponse({"error": video_type_error, "request_id": request_id}, status_code=415)

        with tempfile.TemporaryDirectory() as tmp:
            io_started = time.perf_counter()
            video_path = Path(tmp) / _safe_upload_name(video.filename, "upload-video.bin")
            try:
                video_size = await _write_upload_limited(video, video_path, _MAX_VIDEO_BYTES)
            except ValueError as exc:
                _log_event("analyze.video.rejected", request_id=request_id, reason=str(exc))
                return JSONResponse({"error": str(exc), "request_id": request_id}, status_code=413)
            timings_ms["video_bytes"] = float(video_size)
            timings_ms["io_write"] = round((time.perf_counter() - io_started) * 1000, 2)

            if worker_mode == "process":
                payload, worker_error, worker_roundtrip_ms = _run_inference_worker(
                    {
                        "action": "video",
                        "video_path": str(video_path),
                        "interval_s": interval_s,
                        "max_frames": max_frames,
                        "profile": profile,
                        "force_safe_demo": force_safe_demo,
                    },
                    timeout_s=_WORKER_VIDEO_TIMEOUT_S,
                )
                timings_ms["worker_roundtrip"] = worker_roundtrip_ms
                if payload is None:
                    if worker_error and "unable to read video" in worker_error:
                        return JSONResponse({"error": "unable to read video", "request_id": request_id}, status_code=400)
                    fallback_reason = f"worker failure: {worker_error}"
                    payload = _make_demo_video_payload(fallback_reason, interval_s, max_frames)
                    _log_event(
                        "inference.worker_fallback",
                        request_id=request_id,
                        path="/analyze/video",
                        reason=fallback_reason,
                    )
            else:
                infer_started = time.perf_counter()
                try:
                    payload = _run_video_pipeline_local(
                        video_path=video_path,
                        interval_s=interval_s,
                        max_frames=max_frames,
                        profile=profile,
                        force_safe_demo=force_safe_demo,
                    )
                except ValueError as exc:
                    if "unable to read video" in str(exc):
                        return JSONResponse({"error": "unable to read video", "request_id": request_id}, status_code=400)
                    raise
                timings_ms["inline_inference"] = round((time.perf_counter() - infer_started) * 1000, 2)

        timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
        _attach_runtime_meta(
            payload,
            request_id=request_id,
            timings_ms=timings_ms,
            worker_mode=worker_mode,
            manifest=manifest,
        )
        _log_event(
            "analyze.video",
            request_id=request_id,
            worker_mode=worker_mode,
            safe_demo=bool(payload.get("safe_demo")),
            frame_count=len(payload.get("frames", [])),
            total_ms=timings_ms["total"],
        )
        return JSONResponse(payload)
    finally:
        if acquired:
            _ANALYSIS_SEMAPHORE.release()


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
    if _GEO_EVAL_STATE["status"] == "running" or _GEO_RANDOM_STATE["status"] == "running":
        return JSONResponse({"status": "running"})

    def _run() -> None:
        _GEO_EVAL_STATE["status"] = "running"
        try:
            output_path = Path("src/dashboard/data/geo_eval.json")
            progress_path = Path("src/dashboard/data/geo_eval.progress.json")
            _GEO_EVAL_STATE["progress_path"] = str(progress_path)
            effective_profile, warning = _resolve_eval_profile(
                profile,
                images_dir=images_dir,
                metadata=metadata,
            )
            config_path = _config_path_for_profile(effective_profile)
            _GEO_EVAL_STATE["profile_requested"] = profile
            _GEO_EVAL_STATE["profile_effective"] = effective_profile
            _GEO_EVAL_STATE["profile_warning"] = warning
            _GEO_EVAL_STATE["config_path"] = str(config_path)
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
            args.extend(["--config", str(config_path)])
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


def _build_random_geo_eval_summary(
    payload: dict,
    *,
    sample_size: int,
    seed: int,
    profile_requested: Optional[str],
    profile_effective: Optional[str],
    profile_warning: Optional[str],
    config_path: str,
    retrieval_only: bool,
) -> dict:
    diagnostics = payload.get("samples")
    samples = diagnostics if isinstance(diagnostics, list) else []
    dists = []
    for row in samples:
        if not isinstance(row, dict):
            continue
        dist = _to_float(row.get("dist_km"))
        if dist is None:
            continue
        dists.append(dist)

    within_2km = None
    if dists:
        within_2km = 100.0 * sum(1 for value in dists if value <= 2.0) / len(dists)

    return {
        "generated_at": _utc_now_iso(),
        "mode": "random_samples",
        "profile": (profile_effective or "").strip().lower() or "default",
        "profile_requested": (profile_requested or "").strip().lower() or "default",
        "profile_effective": (profile_effective or "").strip().lower() or "default",
        "profile_warning": profile_warning,
        "config_path": config_path,
        "retrieval_only": bool(retrieval_only),
        "requested_samples": int(sample_size),
        "seed": int(seed),
        "total": payload.get("total"),
        "evaluated": payload.get("evaluated"),
        "missing_files": payload.get("missing_files"),
        "null_predictions": payload.get("null_predictions"),
        "mean_km": payload.get("mean_km"),
        "median_km": payload.get("median_km"),
        "p90_km": payload.get("p90_km"),
        "within_1km_pct": payload.get("within_1km_pct"),
        "within_2km_pct": within_2km,
        "within_5km_pct": payload.get("within_5km_pct"),
        "within_10km_pct": payload.get("within_10km_pct"),
        "within_50km_pct": payload.get("within_50km_pct"),
        "samples": samples,
    }


@app.post("/eval/geo/random/start")
def start_geo_random_eval(
    images_dir: str,
    metadata: str,
    sample_size: int = 12,
    profile: Optional[str] = None,
    retrieval_only: Optional[str] = None,
) -> JSONResponse:
    if _GEO_EVAL_STATE["status"] == "running" or _GEO_RANDOM_STATE["status"] == "running":
        return JSONResponse({"status": "running"})

    def _run() -> None:
        _GEO_RANDOM_STATE["status"] = "running"
        _GEO_RANDOM_STATE["last_result"] = None
        try:
            output_path = Path("src/dashboard/data/geo_eval_random.json")
            progress_path = Path("src/dashboard/data/geo_eval_random.progress.json")
            _GEO_RANDOM_STATE["progress_path"] = str(progress_path)
            effective_profile, warning = _resolve_eval_profile(
                profile,
                images_dir=images_dir,
                metadata=metadata,
            )
            config_path = _config_path_for_profile(effective_profile)
            _GEO_RANDOM_STATE["profile_requested"] = profile
            _GEO_RANDOM_STATE["profile_effective"] = effective_profile
            _GEO_RANDOM_STATE["profile_warning"] = warning
            _GEO_RANDOM_STATE["config_path"] = str(config_path)
            from src.tools.run_geo_eval import main as run_geo_eval

            random_seed = random.randint(1, 2_147_483_647)
            _GEO_RANDOM_STATE["seed"] = int(random_seed)
            safe_sample_size = max(1, min(int(sample_size), 1000))
            args = [
                "--images-dir",
                images_dir,
                "--metadata",
                metadata,
                "--output",
                str(output_path),
                "--progress",
                str(progress_path),
                "--limit",
                str(safe_sample_size),
                "--diag-samples",
                str(safe_sample_size),
                "--seed",
                str(random_seed),
                "--config",
                str(config_path),
            ]
            retrieval_enabled = True
            if retrieval_only is not None and str(retrieval_only).lower() in {"0", "false", "no"}:
                retrieval_enabled = False
            if retrieval_enabled:
                args.append("--retrieval-only")
            run_geo_eval(args)
            if output_path.exists():
                report = json.loads(output_path.read_text(encoding="utf-8"))
                summary = _build_random_geo_eval_summary(
                    report,
                    sample_size=safe_sample_size,
                    seed=random_seed,
                    profile_requested=profile,
                    profile_effective=effective_profile,
                    profile_warning=warning,
                    config_path=str(config_path),
                    retrieval_only=retrieval_enabled,
                )
                _GEO_RANDOM_STATE["last_result"] = json.dumps(summary, indent=2, ensure_ascii=False)
            _GEO_RANDOM_STATE["status"] = "done"
        except Exception as exc:
            _GEO_RANDOM_STATE["status"] = "error"
            _GEO_RANDOM_STATE["last_result"] = str(exc)

    import threading

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "running"})


@app.get("/eval/geo/random/status")
def geo_random_eval_status() -> JSONResponse:
    progress_path = _GEO_RANDOM_STATE.get("progress_path")
    if progress_path:
        path = Path(progress_path)
        if path.exists():
            try:
                _GEO_RANDOM_STATE["progress"] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
    return JSONResponse(_GEO_RANDOM_STATE)


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
            scoped_cfg_dir = APP_ROOT / "runs" / "bench_cfg_scoped"
            legacy_cfg_dir = APP_ROOT / "runs" / "bench_cfg"

            def _resolve_bench_cfg(scoped_name: str, legacy_name: str) -> Path:
                scoped_path = scoped_cfg_dir / scoped_name
                if scoped_path.exists():
                    return scoped_path
                return legacy_cfg_dir / legacy_name

            geo_jobs = [
                (
                    "leaky_reference",
                    _resolve_bench_cfg("cfg_paris_current_leaky.json", "cfg_current_leaky.json"),
                    ui_geo_leaky,
                ),
                (
                    "realistic_single",
                    _resolve_bench_cfg("cfg_paris_realistic_single.json", "cfg_realistic_single.json"),
                    ui_geo_realistic,
                ),
                (
                    "candidate_multi",
                    _resolve_bench_cfg("cfg_mixed_candidate_multi.json", "cfg_candidate_multi.json"),
                    ui_geo_multi,
                ),
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
            summary = _json_safe(summary)
            serialized = json.dumps(summary, indent=2, allow_nan=False)
            run_summary_out.write_text(serialized, encoding="utf-8")
            summary_out.write_text(serialized, encoding="utf-8")
            _BENCHMARK_STATE["last_result"] = serialized
            _BENCHMARK_STATE["status"] = "done"
            _set_progress(4, total_steps, "done", "Benchmark run complete")
        except Exception as exc:
            _BENCHMARK_STATE["status"] = "error"
            _BENCHMARK_STATE["stage"] = "error"
            if _BENCHMARK_STATE.get("progress"):
                _BENCHMARK_STATE["progress"]["message"] = "Benchmark run failed"
            _BENCHMARK_STATE["last_result"] = json.dumps(
                {
                    "status": "error",
                    "run_id": run_id,
                    "error": str(exc),
                    "stage": _BENCHMARK_STATE.get("stage"),
                },
                indent=2,
            )

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


@app.get("/api/operator/session")
def operator_get_session() -> JSONResponse:
    return JSONResponse(_OPERATOR_SESSION)

def _session_summary(data: dict) -> dict:
    return {
        "session_id": data.get("session_id"),
        "custom_name": data.get("custom_name", ""),
        "display_name": data.get("display_name") or data.get("custom_name") or data.get("session_id", ""),
        "updated_at": data.get("updated_at"),
        "status": data.get("status"),
        "source_filename": (data.get("source") or {}).get("filename"),
    }

@app.get("/api/operator/sessions")
def operator_list_sessions() -> JSONResponse:
    sessions_dir = APP_ROOT / "operator_sessions"
    sessions = []
    if sessions_dir.exists():
        # New: folder-per-session
        for session_dir in sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            json_path = session_dir / "session.json"
            if not json_path.exists():
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append(_session_summary(data))
            except Exception:
                continue
        # Legacy: flat session_*.json files
        for file_path in sessions_dir.glob("session_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append(_session_summary(data))
            except Exception:
                continue
    sessions.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return JSONResponse({"sessions": sessions})

@app.get("/api/operator/sessions/{session_id}")
def operator_get_session_by_id(session_id: str) -> JSONResponse:
    global _OPERATOR_SESSION
    sessions_dir = APP_ROOT / "operator_sessions"

    # Per-session folder
    json_path = sessions_dir / session_id / "session.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Signal the frontend that an image is available via the image endpoint
            source = data.get("source") or {}
            if source.get("image_file") and not source.get("image_data_url"):
                data["source"]["has_session_image"] = True
            _OPERATOR_SESSION = data
            return JSONResponse(_OPERATOR_SESSION)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # Legacy: flat session_*.json files
    for file_path in sessions_dir.glob(f"session_*{session_id}.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _OPERATOR_SESSION = data
            return JSONResponse(_OPERATOR_SESSION)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    return JSONResponse({"error": "Session not found"}, status_code=404)


@app.get("/api/operator/sessions/{session_id}/image")
def operator_get_session_image(session_id: str):
    sessions_dir = APP_ROOT / "operator_sessions"
    session_dir = sessions_dir / session_id
    for ext in ["jpg", "jpeg", "png", "webp", "gif"]:
        img_path = session_dir / f"source.{ext}"
        if img_path.exists():
            ct = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
            return Response(content=img_path.read_bytes(), media_type=ct)
    return JSONResponse({"error": "Image not found"}, status_code=404)

@app.post("/api/operator/save")
async def operator_save_session(request: Request) -> JSONResponse:
    global _OPERATOR_SESSION
    try:
        data = await request.json()
        custom_name = data.get("name", "")
        save_as_new = data.get("save_as_new", False)

        if save_as_new:
            _OPERATOR_SESSION["session_id"] = uuid.uuid4().hex
            # Reset timeline and notes ID to differentiate if needed, though copying is fine
            if not custom_name and "custom_name" in _OPERATOR_SESSION:
                del _OPERATOR_SESSION["custom_name"]

        _save_operator_session(custom_name)
        return JSONResponse({"status": "session_saved", "session_id": _OPERATOR_SESSION["session_id"]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/api/operator/reset")
def operator_reset_session() -> JSONResponse:
    _reset_operator_session()
    return JSONResponse({"status": "reset_ok"})

@app.post("/api/operator/pin")
async def operator_add_pin(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        lat = data.get("lat")
        lon = data.get("lon")
        label = data.get("label", "Manual Pin")
        if lat is not None and lon is not None:
            _OPERATOR_SESSION["operator_pins"].append({
                "lat": float(lat),
                "lon": float(lon),
                "label": label,
                "added_at": _utc_now_iso()
            })
            _add_timeline_event(f"Operator added pin at {lat:.4f}, {lon:.4f}", "info")
            return JSONResponse({"status": "pin_added"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"error": "invalid payload"}, status_code=400)

@app.post("/api/operator/note")
async def operator_add_note(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        note = data.get("note", "")

        if "notes" not in _OPERATOR_SESSION:
            _OPERATOR_SESSION["notes"] = []

        note_entry = {
            "note_id": uuid.uuid4().hex,
            "text": note,
            "timestamp": _utc_now_iso(),
            "target_type": data.get("target_type")
        }

        target_type = data.get("target_type")

        # Check if note exists for this target, update if so
        updated = False
        for existing_note in _OPERATOR_SESSION["notes"]:
            if target_type == "note_id" and existing_note.get("note_id") == data.get("note_id"):
                existing_note["text"] = note
                existing_note["timestamp"] = _utc_now_iso()
                updated = True
                break
            elif target_type == "candidate" and existing_note.get("target_type") == "candidate":
                if existing_note.get("rank") == data.get("rank"):
                    existing_note["text"] = note
                    existing_note["timestamp"] = _utc_now_iso()
                    updated = True
                    break
            elif target_type == "manual_pin" and existing_note.get("target_type") == "manual_pin":
                if abs(existing_note.get("lat", 0) - data.get("lat", 0)) < 0.0001 and abs(existing_note.get("lon", 0) - data.get("lon", 0)) < 0.0001:
                    existing_note["text"] = note
                    existing_note["timestamp"] = _utc_now_iso()
                    updated = True
                    break

        if not updated:
            if target_type == "candidate":
                note_entry["rank"] = data.get("rank")
                note_entry["source"] = data.get("source")
            elif target_type == "manual_pin":
                note_entry["lat"] = data.get("lat")
                note_entry["lon"] = data.get("lon")
            _OPERATOR_SESSION["notes"].append(note_entry)

        _OPERATOR_SESSION["operator_notes"] = note # Keep for backwards compat

        _add_timeline_event("Operator added a note", "info")
        return JSONResponse({"status": "note_updated"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/api/operator/confirm")
async def operator_confirm_candidate(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        rank = data.get("rank")
        action = data.get("action") # "confirm" or "reject"
        _add_timeline_event(f"Operator {action}ed candidate rank {rank}", "info")
        return JSONResponse({"status": "action_recorded"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/api/operator/export.json")
def operator_export_session() -> JSONResponse:
    return JSONResponse(_OPERATOR_SESSION)

from src.core.geo.street_view import LocalStreetViewProvider
_STREET_VIEW_PROVIDER = None

def _get_sv_provider() -> "LocalStreetViewProvider":
    global _STREET_VIEW_PROVIDER
    if _STREET_VIEW_PROVIDER is None:
        data_dir = str(APP_ROOT / "data" / "paris_realistic_v1" / "street_combined")
        _STREET_VIEW_PROVIDER = LocalStreetViewProvider(data_dir=data_dir)
    return _STREET_VIEW_PROVIDER

@app.get("/api/operator/street_view")
def operator_street_view(lat: float, lon: float) -> JSONResponse:
    nearest = _get_sv_provider().find_nearest(lat, lon)
    if nearest:
        return JSONResponse(nearest)
    return JSONResponse({"error": "No street view imagery found near this location"}, status_code=404)

@app.get("/api/operator/street_view/image/{image_id}")
def operator_sv_image(image_id: str):
    provider = _get_sv_provider()
    point = provider._by_id.get(image_id)
    if not point:
        return JSONResponse({"error": "Image not found"}, status_code=404)
    img_path = provider.data_dir / point["path"]
    if not img_path.exists():
        return JSONResponse({"error": "Image file not found on disk"}, status_code=404)
    return FileResponse(str(img_path), media_type="image/jpeg")

@app.get("/api/operator/street_view/neighbors")
def operator_sv_neighbors(
    image_id: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    prefer_heading: Optional[float] = None,
) -> JSONResponse:
    provider = _get_sv_provider()

    # Resolve the current image
    current = None
    if image_id:
        point = provider._by_id.get(image_id)
        if point:
            current = provider._point_to_response(point, 0.0)
    if current is None and lat is not None and lon is not None:
        if prefer_heading is not None:
            current = provider.find_nearest_by_heading(lat, lon, prefer_heading)
        else:
            current = provider.find_nearest(lat, lon)

    if not current:
        return JSONResponse({"error": "No imagery found near this location"}, status_code=404)

    neighbors = provider.get_sequence_neighbors(current["image_id"])
    return JSONResponse({
        "current": current,
        "prev": neighbors["prev"],
        "next": neighbors["next"],
        "sequence_position": neighbors["position"],
        "sequence_total": neighbors["total"],
    })

@app.post("/api/operator/analyze")
async def operator_analyze(
    request: Request,
    image: UploadFile = File(...),
    profile: Optional[str] = Form(None),
    dev_mode: Optional[str] = Form(None),
) -> JSONResponse:
    _reset_operator_session()
    _OPERATOR_SESSION["status"] = "running"

    _add_timeline_event("Source uploaded", "info")

    is_dev = _to_bool_flag(dev_mode)
    if is_dev:
        _add_timeline_event("Running in DEV MODE (mock data)", "warning")
        import asyncio
        await asyncio.sleep(1) # simulate work

        # mock data
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        img_content_type = image.content_type or "image/jpeg"
        _OPERATOR_SESSION["source"] = {
            "filename": image.filename,
            "width": 800,
            "height": 600,
            "has_exif_gps": False,
            "quality": {"blur": 0.05, "brightness": 0.5},
            "image_data_url": f"data:{img_content_type};base64,{img_b64}"
        }
        _OPERATOR_SESSION["fused_estimate"] = {
            "lat": 48.8566,
            "lon": 2.3522,
            "display_lat": 48.8566,
            "display_lon": 2.3522,
            "radius_km": 1.5,
            "confidence": 0.85,
            "tier": "high",
            "sources": ["mock"]
        }
        _OPERATOR_SESSION["candidates"] = [
            {
                "rank": 1,
                "lat": 48.8566,
                "lon": 2.3522,
                "display_lat": 48.8566,
                "display_lon": 2.3522,
                "score": 0.95,
                "posterior": 0.95,
                "source": "mock",
                "source_support": ["mock"],
                "label": "Mock Candidate 1",
                "distance_to_fused_km": 0.0,
                "evidence": ["Eiffel Tower structure"]
            }
        ]
        _OPERATOR_SESSION["clues"] = [
            {"name": "mock clue", "score": 1.0, "description": "Dev mode clue", "reliability": "strong"}
        ]
        _OPERATOR_SESSION["detections"] = [
            {
                "label": "mock clue",
                "confidence": 1.0,
                "obb": [[240, 180], [560, 180], [560, 380], [240, 380]],
                "heading_deg": None,
                "shadow_azimuth_deg": None,
                "shadow_length_ratio": None,
            }
        ]
        _OPERATOR_SESSION["status"] = "completed"
        _add_timeline_event("Analysis complete (DEV MODE)", "success")
        return JSONResponse(_OPERATOR_SESSION)

    try:
        # Real processing
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / _safe_upload_name(image.filename, "upload-image.bin")
            await _write_upload_limited(image, image_path, _MAX_IMAGE_BYTES)

            img_bytes = image_path.read_bytes()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            img_content_type = image.content_type or "image/jpeg"
            _OPERATOR_SESSION["source"] = {
                "filename": image.filename,
                "width": 0,
                "height": 0,
                "has_exif_gps": False,
                "quality": {},
                "image_data_url": f"data:{img_content_type};base64,{img_b64}"
            }

            _add_timeline_event("Preprocessing complete", "info")
            _add_timeline_event("Model stages running...", "info")

            cfg = _load_config_from_env(profile)
            pipeline = build_pipeline(cfg)
            if pipeline is None:
                raise RuntimeError("Failed to build pipeline. Dependency missing?")

            _add_timeline_event("Detector complete", "info")

            # Extract clues
            detections = pipeline.detect(str(image_path))

            clues = []
            for d in detections:
                 clues.append({
                     "name": d.label,
                     "score": d.confidence,
                     "description": f"Detected {d.label}",
                     "reliability": "medium" if d.confidence > 0.5 else "weak"
                 })
            _OPERATOR_SESSION["clues"] = clues
            _OPERATOR_SESSION["detections"] = [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "obb": d.obb,
                    "heading_deg": d.heading_deg,
                    "shadow_azimuth_deg": d.shadow_azimuth_deg,
                    "shadow_length_ratio": d.shadow_length_ratio,
                }
                for d in detections
            ]

            # Geo loc
            _add_timeline_event("Geo candidates generated", "info")
            try:
                result = pipeline.run(str(image_path))
            except Exception as e:
                _add_timeline_event(f"Pipeline error: {e}", "error")
                _OPERATOR_SESSION["warnings"].append(f"Pipeline error: {e}")
                _OPERATOR_SESSION["status"] = "error"
                return JSONResponse(_OPERATOR_SESSION)

            # Detect no candidate generation error
            cand_err = getattr(pipeline.candidate_provider, "last_error", None)
            if cand_err:
                _OPERATOR_SESSION["warnings"].append(f"Candidate provider error: {cand_err}")
                _add_timeline_event(f"Candidate generation failed: {cand_err}", "error")
                _OPERATOR_SESSION["status"] = "error"
                return JSONResponse(_OPERATOR_SESSION)

            if not result.candidates:
                _OPERATOR_SESSION["warnings"].append("No geo candidates found.")
                _add_timeline_event("No geo candidates generated.", "warning")
                _OPERATOR_SESSION["status"] = "error"
                return JSONResponse(_OPERATOR_SESSION)

            _add_timeline_event("Fusion complete", "info")

            # Populate response structure
            allow_precise = os.environ.get("OPERATOR_ALLOW_PRECISE_COORDS", "false").lower() == "true"

            fused = result.fusion
            if fused:
                 _OPERATOR_SESSION["fused_estimate"] = {
                    "lat": fused.mean_latitude,
                    "lon": fused.mean_longitude,
                    "display_lat": fused.mean_latitude if allow_precise else round(fused.mean_latitude, 2),
                    "display_lon": fused.mean_longitude if allow_precise else round(fused.mean_longitude, 2),
                    "radius_km": fused.uncertainty_radius_m / 1000.0 if fused.uncertainty_radius_m else 0.0,
                    "confidence": fused.top1_posterior,
                    "tier": fused.confidence_tier,
                    "sources": ["fusion"]
                 }

                 for i, cand in enumerate(fused.candidates):
                     lat = cand.candidate.latitude
                     lon = cand.candidate.longitude
                     _OPERATOR_SESSION["candidates"].append({
                         "rank": i + 1,
                         "lat": lat,
                         "lon": lon,
                         "display_lat": lat if allow_precise else round(lat, 2),
                         "display_lon": lon if allow_precise else round(lon, 2),
                         "score": cand.evidence.retrieval_score if cand.evidence else cand.candidate.retrieval_score,
                         "posterior": cand.posterior_weight,
                         "source": cand.candidate.match_id or "unknown",
                         "source_support": [],
                         "label": f"Candidate {i + 1}",
                         "distance_to_fused_km": 0.0,
                         "evidence": [cand.evidence.explanation] if cand.evidence else []
                     })

            _OPERATOR_SESSION["status"] = "completed"
            _add_timeline_event("Analysis complete", "success")
            return JSONResponse(_OPERATOR_SESSION)

    except Exception as e:
        _OPERATOR_SESSION["status"] = "error"
        _OPERATOR_SESSION["warnings"].append(str(e))
        _add_timeline_event(f"Analysis failed: {str(e)}", "error")
        return JSONResponse({"error": str(e), "session": _OPERATOR_SESSION}, status_code=500)

# Static mounts come last so API routes are not shadowed.
app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR), name="dashboard")
app.mount("/analysis", StaticFiles(directory=LIVE_DIR, html=True), name="analysis")
app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="root")
