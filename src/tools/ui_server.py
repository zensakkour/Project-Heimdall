"""
FastAPI server for live analysis UI.
"""
from __future__ import annotations

import base64
import json
import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.core.logic.config import HeimdallConfig, load_config
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
        model_id=cfg.geolocator.retrieval_model_id or "openai/clip-vit-large-patch14",
        top_k=cfg.geolocator.retrieval_top_k,
        min_score=cfg.geolocator.retrieval_min_score,
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
    if cfg.geolocator.retrieval_index_path:
        candidate_provider = MultiCandidateProvider([retrieval_provider, geoclip_provider])
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
        "generated_at": datetime.utcnow().isoformat() + "Z",
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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/analysis")
def analysis_index() -> RedirectResponse:
    return RedirectResponse(url="/analysis/")


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
                "generated_at": datetime.utcnow().isoformat() + "Z",
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
