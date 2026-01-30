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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from core.detection.factory import create_detector
from core.geo import GeoLocator
from core.logic.config import HeimdallConfig, load_config
from core.logic.pipeline import HeimdallPipeline
from core.logic.serialize import assessment_to_dict
from core.logic.visualize import draw_detections
from tools.run_dota_eval import main as run_dota_eval


APP_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_ROOT / "dashboard" / "live"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
    return HeimdallPipeline(
        detector=detector,
        geolocator=geolocator,
        score_config=cfg.score,
        verification_config=cfg.verification,
    )


def _load_config_from_env() -> Optional[HeimdallConfig]:
    default_path = APP_ROOT / "config" / "defaults.json"
    if default_path.exists():
        return load_config(str(default_path))
    return None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/analyze/image")
async def analyze_image(
    image: UploadFile = File(...),
    det_json: Optional[UploadFile] = File(None),
    geo_json: Optional[UploadFile] = File(None),
) -> JSONResponse:
    cfg = _load_config_from_env()
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
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "result": assessment_to_dict(result),
            "image_data": _image_to_data_url(annotated),
        }
        return JSONResponse(payload)


@app.post("/analyze/video")
async def analyze_video(
    video: UploadFile = File(...),
    interval_s: float = Form(2.0),
    max_frames: int = Form(12),
) -> JSONResponse:
    cfg = _load_config_from_env()
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
            cfg = load_config("config/defaults.json")
            run_dota_eval(
                [
                    "--weights",
                    (cfg.detector.weights_path or "yolo11x-obb.pt"),
                    "--data",
                    "data/dota/dota.yaml",
                    "--imgsz",
                    str(cfg.detector.imgsz),
                    "--output",
                    "dashboard/data/dota_eval.json",
                ]
            )
            report = Path("dashboard/data/dota_eval.json")
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
