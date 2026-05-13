import pytest
from fastapi.testclient import TestClient
from src.tools.ui_server import app, APP_ROOT
import shutil

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_operator_sessions():
    sessions_dir = APP_ROOT / "operator_sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)
    yield
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)

def test_operator_session_lifecycle():
    # Reset session creates one implicitly, but let's just create a note to trigger save
    resp = client.post("/api/operator/reset")
    assert resp.status_code == 200

    resp = client.post("/api/operator/note", json={"note": "test note"})
    assert resp.status_code == 200

    # List sessions
    resp = client.get("/api/operator/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert len(data["sessions"]) >= 1

    session_id = data["sessions"][0]["session_id"]

    # Get session
    resp = client.get(f"/api/operator/sessions/{session_id}")
    assert resp.status_code == 200
    session_data = resp.json()
    assert session_data["operator_notes"] == "test note"
