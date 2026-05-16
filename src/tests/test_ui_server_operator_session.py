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

def test_operator_session_image_serving():
    import base64
    # 1. Reset and simulate an analysis with an image
    client.post("/api/operator/reset")
    
    # Mock some image data
    pixel_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    data_url = f"data:image/png;base64,{pixel_b64}"
    
    # We need to manually inject it into the global state for the save test
    from src.tools import ui_server
    ui_server._OPERATOR_SESSION["source"] = {
        "filename": "test.png",
        "image_data_url": data_url
    }
    
    # 2. Save session
    resp = client.post("/api/operator/save", json={"name": "image session"})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    
    # 3. Verify session.json has image_file and has_session_image is set when loading
    resp = client.get(f"/api/operator/sessions/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"]["image_file"] == "source.png"
    assert data["source"]["has_session_image"] is True
    
    # 4. Verify image serving
    resp = client.get(f"/api/operator/sessions/{session_id}/image")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 0
    
    # Verify base64 content matches
    expected_bytes = base64.b64decode(pixel_b64)
    assert resp.content == expected_bytes

def test_session_image_serving_falls_back_to_embedded_source_data():
    import base64
    from src.tools import ui_server

    client.post("/api/operator/reset")
    pixel_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    data_url = f"data:image/png;base64,{pixel_b64}"
    ui_server._OPERATOR_SESSION["source_image_data"] = pixel_b64
    ui_server._OPERATOR_SESSION["source"] = {
        "filename": "fallback.png",
        "image_data_url": data_url,
    }

    save_resp = client.post("/api/operator/save", json={"name": "fallback image"})
    assert save_resp.status_code == 200
    session_id = save_resp.json()["session_id"]
    session_dir = ui_server._find_session_dir(session_id)
    assert session_dir is not None
    (session_dir / "source.png").unlink()

    resp = client.get(f"/api/operator/sessions/{session_id}/image")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == base64.b64decode(pixel_b64)

def test_saved_session_pin_note_merge_and_delete_persists():
    client.post("/api/operator/reset")
    save_resp = client.post("/api/operator/save", json={"name": "pin note session"})
    assert save_resp.status_code == 200
    session_id = save_resp.json()["session_id"]

    pin_resp = client.post("/api/operator/pin", json={"lat": 12.34, "lon": 56.78, "label": "Custom Pin"})
    assert pin_resp.status_code == 200
    pin = pin_resp.json()["pin"]

    note_resp = client.post("/api/operator/note", json={
        "note": "Pinned observation",
        "target_type": "manual_pin",
        "pin_id": pin["pin_id"],
        "lat": 12.34,
        "lon": 56.78,
    })
    assert note_resp.status_code == 200
    assert note_resp.json()["operator_pins"][0]["label"] == "Pinned observation"
    note_id = note_resp.json()["notes"][0]["note_id"]

    reload_resp = client.get(f"/api/operator/sessions/{session_id}")
    assert reload_resp.status_code == 200
    reloaded = reload_resp.json()
    assert reloaded["operator_pins"][0]["note_id"] == note_id
    assert reloaded["operator_pins"][0]["label"] == "Pinned observation"
    assert reloaded["notes"][0]["text"] == "Pinned observation"

    delete_resp = client.delete(f"/api/operator/pins/{pin['pin_id']}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["operator_pins"] == []
    assert delete_resp.json()["notes"] == []

    reload_after_delete = client.get(f"/api/operator/sessions/{session_id}")
    assert reload_after_delete.status_code == 200
    assert reload_after_delete.json()["operator_pins"] == []
    assert reload_after_delete.json()["notes"] == []

def test_removing_case_session_deletes_case_mirror(tmp_path, monkeypatch):
    from src.tools import ui_server

    cases_dir = tmp_path / "cases"
    monkeypatch.setattr(ui_server, "CASES_DIR", cases_dir)
    monkeypatch.setattr(ui_server, "ACTIVE_CASE_FILE", cases_dir / "_active_case.txt")

    client.post("/api/operator/reset")
    save_resp = client.post("/api/operator/save", json={"name": "case session"})
    assert save_resp.status_code == 200
    session_id = save_resp.json()["session_id"]

    case_resp = client.post("/api/cases", json={"name": "Delete Session Case"})
    assert case_resp.status_code == 201
    case_id = case_resp.json()["case_id"]

    add_resp = client.post(f"/api/cases/{case_id}/sessions", json={"session_id": session_id})
    assert add_resp.status_code == 200
    case_dir = ui_server._find_id_dir(cases_dir, case_id)
    mirrored_sessions = list((case_dir / "sessions").glob(f"*_{session_id}"))
    assert len(mirrored_sessions) == 1

    delete_resp = client.delete(f"/api/cases/{case_id}/sessions/{session_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["sessions"] == []
    assert list((case_dir / "sessions").glob(f"*_{session_id}")) == []

    reload_resp = client.get("/api/operator/sessions?include_case=1")
    assert reload_resp.status_code == 200
    assert all(s["session_id"] != session_id or s.get("case_id") != case_id for s in reload_resp.json()["sessions"])
