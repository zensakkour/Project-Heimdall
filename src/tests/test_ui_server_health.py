from __future__ import annotations

from fastapi.testclient import TestClient

from src.tools.ui_server import app


def test_health_endpoint_shape() -> None:
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code in {200, 503}
    payload = res.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "timestamp" in payload
    assert "required_failures" in payload
    assert "summary" in payload
    assert payload["summary"]["deps_checked"] >= 1


def test_health_deps_endpoint_shape() -> None:
    client = TestClient(app)
    res = client.get("/health/deps")
    assert res.status_code in {200, 503}
    payload = res.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "deps" in payload
    assert "torch" in payload["deps"]
    assert "config_paths" in payload
    assert "model_paths" in payload
    assert "write_permissions" in payload

