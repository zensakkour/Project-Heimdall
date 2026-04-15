from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import types

from fastapi.testclient import TestClient
from PIL import Image

from src.tools import ui_server


def _png_bytes(width: int = 640, height: int = 360) -> bytes:
    img = Image.new("RGB", (width, height), color=(24, 36, 42))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_runtime_endpoint_shape() -> None:
    client = TestClient(ui_server.app)
    res = client.get("/health/runtime")
    assert res.status_code == 200
    payload = res.json()
    assert payload["status"] == "ok"
    assert "timestamp" in payload
    assert "inference_worker_enabled" in payload
    assert payload["timeouts_s"]["image"] > 0
    assert payload["timeouts_s"]["video"] > 0
    assert payload["limits"]["max_image_bytes"] > 0
    assert payload["limits"]["analysis_concurrency"] >= 1


def test_analyze_image_sets_request_id_and_runtime_meta(monkeypatch) -> None:
    monkeypatch.setenv("HEIMDALL_USE_INFERENCE_WORKER", "0")
    client = TestClient(ui_server.app)
    files = {"image": ("demo.png", _png_bytes(), "image/png")}
    res = client.post("/analyze/image?safe_demo=1", files=files, headers={"x-request-id": "req-test-123"})
    assert res.status_code == 200
    payload = res.json()
    assert res.headers["X-Request-ID"] == "req-test-123"
    assert payload["request_id"] == "req-test-123"
    assert payload["runtime"]["worker_mode"] == "inline"
    assert payload["runtime"]["timings_ms"]["total"] >= 0
    assert payload["runtime"]["manifest"]["config_path"].endswith(".json")
    assert payload["runtime"]["manifest"]["env"]["limits"]["max_image_bytes"] > 0


def test_analyze_image_worker_failure_falls_back_to_safe_demo(monkeypatch) -> None:
    monkeypatch.setenv("HEIMDALL_USE_INFERENCE_WORKER", "1")

    def _boom_worker(_task: dict, timeout_s: float):
        _ = timeout_s
        return None, "synthetic worker crash", 3.2

    monkeypatch.setattr(ui_server, "_run_inference_worker", _boom_worker)
    client = TestClient(ui_server.app)
    files = {"image": ("demo.png", _png_bytes(), "image/png")}
    res = client.post("/analyze/image", files=files)
    assert res.status_code == 200
    payload = res.json()
    assert payload["safe_demo"] is True
    assert payload["runtime"]["worker_mode"] == "process"
    assert "worker failure" in (payload["geo_debug"]["fallback_reason"] or "")


def test_analyze_image_rejects_unsupported_content_type(monkeypatch) -> None:
    monkeypatch.setenv("HEIMDALL_USE_INFERENCE_WORKER", "0")
    client = TestClient(ui_server.app)
    files = {"image": ("demo.txt", b"not-an-image", "text/plain")}
    res = client.post("/analyze/image", files=files)
    assert res.status_code == 415
    payload = res.json()
    assert "unsupported image content-type" in payload["error"]


def test_analyze_image_rejects_oversized_upload(monkeypatch) -> None:
    monkeypatch.setenv("HEIMDALL_USE_INFERENCE_WORKER", "0")
    monkeypatch.setattr(ui_server, "_MAX_IMAGE_BYTES", 32)
    client = TestClient(ui_server.app)
    files = {"image": ("demo.png", _png_bytes(), "image/png")}
    res = client.post("/analyze/image", files=files)
    assert res.status_code == 413
    payload = res.json()
    assert "upload too large" in payload["error"]


def test_start_geo_random_eval_reports_distance_summary(monkeypatch) -> None:
    class _ImmediateThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}
            self.daemon = daemon

        def start(self):
            if self._target is not None:
                self._target(*self._args, **self._kwargs)

    def _fake_geo_eval_main(argv: list[str]) -> None:
        out_path = Path(argv[argv.index("--output") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "total": 5,
                    "evaluated": 5,
                    "missing_files": 0,
                    "null_predictions": 0,
                    "mean_km": 3.2,
                    "median_km": 2.1,
                    "p90_km": 6.7,
                    "within_1km_pct": 20.0,
                    "within_5km_pct": 80.0,
                    "within_10km_pct": 100.0,
                    "within_50km_pct": 100.0,
                    "samples": [
                        {"image": "a.jpg", "dist_km": 0.8},
                        {"image": "b.jpg", "dist_km": 1.6},
                        {"image": "c.jpg", "dist_km": 2.4},
                        {"image": "d.jpg", "dist_km": 4.9},
                        {"image": "e.jpg", "dist_km": 6.7},
                    ],
                }
            ),
            encoding="utf-8",
        )

    with tempfile.TemporaryDirectory() as tmp:
        app_root = Path(tmp)
        monkeypatch.setattr(ui_server, "APP_ROOT", app_root)
        monkeypatch.setattr(ui_server.random, "randint", lambda a, b: 123456)
        fake_geo_mod = types.ModuleType("src.tools.run_geo_eval")
        fake_geo_mod.main = _fake_geo_eval_main
        monkeypatch.setitem(sys.modules, "src.tools.run_geo_eval", fake_geo_mod)

        import threading

        monkeypatch.setattr(threading, "Thread", _ImmediateThread)
        client = TestClient(ui_server.app)
        start_res = client.post(
            "/eval/geo/random/start",
            params={
                "images_dir": "data/spacenet_paris_test/chips",
                "metadata": "data/spacenet_paris_test/metadata.csv",
                "sample_size": "5",
                "profile": "paris",
                "retrieval_only": "1",
            },
        )
        assert start_res.status_code == 200
        assert start_res.json().get("status") == "running"

        status_res = client.get("/eval/geo/random/status")
        assert status_res.status_code == 200
        payload = status_res.json()
        assert payload.get("status") == "done"
        assert payload.get("seed") == 123456

        summary = json.loads(payload.get("last_result") or "{}")
        assert summary.get("requested_samples") == 5
        assert summary.get("within_1km_pct") == 20.0
        assert summary.get("within_2km_pct") == 40.0
        assert summary.get("within_5km_pct") == 80.0
        assert len(summary.get("samples", [])) == 5
