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


def test_analyze_image_force_safe_demo() -> None:
    client = TestClient(ui_server.app)
    files = {"image": ("demo.png", _png_bytes(), "image/png")}
    res = client.post("/analyze/image?safe_demo=1", files=files)
    assert res.status_code == 200
    payload = res.json()
    assert payload["safe_demo"] is True
    assert payload["result"]["score"] > 0
    assert payload["result"]["fusion"] is not None
    assert payload["geo_debug"]["safe_demo"] is True


def test_analyze_image_falls_back_when_pipeline_init_fails(monkeypatch) -> None:
    monkeypatch.setenv("HEIMDALL_USE_INFERENCE_WORKER", "0")

    def _boom(_cfg):
        raise RuntimeError("dependency unavailable")

    monkeypatch.setattr(ui_server, "build_pipeline", _boom)
    client = TestClient(ui_server.app)
    files = {"image": ("demo.png", _png_bytes(), "image/png")}
    res = client.post("/analyze/image", files=files)
    assert res.status_code == 200
    payload = res.json()
    assert payload["safe_demo"] is True
    assert payload["result"]["geo"] is not None
    assert "pipeline init failed" in (payload["geo_debug"]["fallback_reason"] or "")
