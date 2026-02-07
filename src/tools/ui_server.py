"""
FastAPI server for live analysis UI.
"""
from __future__ import annotations

import base64
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

from src.core.detection.factory import create_detector
from src.core.geo import GeoCLIPProvider, GeoLocator, GeoRetrievalProvider, MultiCandidateProvider
from src.core.logic.config import HeimdallConfig, load_config
from src.core.logic.pipeline import HeimdallPipeline
from src.core.logic.serialize import assessment_to_dict
from src.core.logic.visualize import draw_detections
from src.tools.run_dota_eval import main as run_dota_eval


APP_ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = APP_ROOT / "src" / "dashboard" / "analysis"
DASHBOARD_DIR = APP_ROOT / "src" / "dashboard"

app = FastAPI()

_EVAL_STATE = {"status": "idle", "last_result": None}


def build_pipeline(cfg: Optional[HeimdallConfig]) -> HeimdallPipeline:
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


def _load_config_from_env(profile: Optional[str] = None) -> Optional[HeimdallConfig]:
    config_dir = APP_ROOT / "src" / "config"
    default_path = config_dir / "defaults.json"
    if profile:
        key = profile.strip().lower()
        profile_map = {
            "paris": "paris.json",
            "paris-focused": "paris.json",
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
) -> JSONResponse:
    cfg = _load_config_from_env(profile)
    pipeline = build_pipeline(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / Path(image.filename).name
        image_path.write_bytes(await image.read())

        if det_json is not None:
            det_path = Path(str(image_path) + ".detections.json")
            det_path.write_bytes(await det_json.read())
        if geo_json is not None:
            geo_path = Path(str(image_path) + ".geo.json")
            geo_path.write_bytes(await geo_json.read())

        result = pipeline.run(str(image_path))
        annotated = draw_detections(str(image_path), result.detections)
        geo_debug = {
            "candidate_count": len(result.candidates),
            "fusion": bool(result.fusion),
            "error": getattr(pipeline.candidate_provider, "last_error", None),
        }
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "result": assessment_to_dict(result),
            "image_data": _image_to_data_url(annotated),
            "geo_debug": geo_debug,
        }
        return JSONResponse(payload)


@app.post("/analyze/video")
async def analyze_video(
    video: UploadFile = File(...),
    interval_s: float = Form(2.0),
    max_frames: int = Form(12),
    profile: Optional[str] = None,
) -> JSONResponse:
    cfg = _load_config_from_env(profile)
    pipeline = build_pipeline(cfg)

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
                result = pipeline.run(str(image_path))
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

        return JSONResponse({"frames": frames})


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


# Static mounts come last so API routes are not shadowed.
app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR), name="dashboard")
app.mount("/analysis", StaticFiles(directory=LIVE_DIR, html=True), name="analysis")
app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="root")


