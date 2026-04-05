from __future__ import annotations

from io import BytesIO

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
