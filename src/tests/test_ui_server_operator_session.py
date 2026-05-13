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

    # Manually save session
    resp = client.post("/api/operator/save", json={"name": "my session"})
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

def test_no_automatic_session_creation():
    client.post("/api/operator/reset")
    client.post("/api/operator/note", json={"note": "some note"})

    # Should not be any saved sessions yet
    resp = client.get("/api/operator/sessions")
    data = resp.json()
    assert len(data.get("sessions", [])) == 0

def test_manual_save_custom_name():
    client.post("/api/operator/reset")
    client.post("/api/operator/save", json={"name": "Mission Alpha"})

    resp = client.get("/api/operator/sessions")
    data = resp.json()
    assert len(data.get("sessions", [])) == 1
    assert data["sessions"][0]["custom_name"] == "Mission Alpha"

def test_manual_save_empty_name():
    client.post("/api/operator/reset")
    client.post("/api/operator/save", json={"name": ""})

    resp = client.get("/api/operator/sessions")
    data = resp.json()
    assert len(data.get("sessions", [])) == 1
    assert data["sessions"][0]["custom_name"] == ""
    assert data["sessions"][0]["display_name"].startswith("auto_")

def test_note_attached_to_candidate():
    client.post("/api/operator/reset")
    resp = client.post("/api/operator/note", json={
        "note": "A note on candidate",
        "target_type": "candidate",
        "rank": 1,
        "source": "some_source"
    })
    assert resp.status_code == 200

    session = client.get("/api/operator/session").json()
    assert len(session["notes"]) == 1
    assert session["notes"][0]["target_type"] == "candidate"
    assert session["notes"][0]["rank"] == 1

def test_note_attached_to_manual_pin():
    client.post("/api/operator/reset")
    resp = client.post("/api/operator/note", json={
        "note": "A note on manual pin",
        "target_type": "manual_pin",
        "lat": 12.34,
        "lon": 56.78
    })
    assert resp.status_code == 200

    session = client.get("/api/operator/session").json()
    assert len(session["notes"]) == 1
    assert session["notes"][0]["target_type"] == "manual_pin"
    assert session["notes"][0]["lat"] == 12.34

def test_street_walk_provider_lookup(monkeypatch):
    from src.tools import ui_server
    class MockStreetViewProvider:
        def find_nearest(self, lat, lon):
            return {"url": "http://example.com/image.jpg"}

    monkeypatch.setattr(ui_server, "_STREET_VIEW_PROVIDER", MockStreetViewProvider())

    resp = client.get("/api/operator/street_view?lat=48.8&lon=2.3")
    assert resp.status_code == 200
    assert resp.json()["url"] == "http://example.com/image.jpg"

def test_street_walk_provider_no_data(monkeypatch):
    from src.tools import ui_server
    class MockStreetViewProvider:
        def find_nearest(self, lat, lon):
            return None

    monkeypatch.setattr(ui_server, "_STREET_VIEW_PROVIDER", MockStreetViewProvider())

    resp = client.get("/api/operator/street_view?lat=48.8&lon=2.3")
    assert resp.status_code == 404
    assert "error" in resp.json()
