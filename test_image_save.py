import asyncio
from fastapi.testclient import TestClient
from src.tools.ui_server import app

client = TestClient(app)

def test_image_save():
    # Analyze with image
    with open("src/dashboard/assets/logo-mark.png", "rb") as f:
        response = client.post("/api/operator/analyze", files={"image": f}, data={"dev_mode": "true"})
    assert response.status_code == 200

    # Save session
    response = client.post("/api/operator/save", json={"name": "test_session", "save_as_new": True})
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    # Check if folder exists
    import os
    folders = [d for d in os.listdir("operator_sessions") if session_id in d]
    assert len(folders) == 1
    folder_path = os.path.join("operator_sessions", folders[0])

    # Check if image.jpg exists
    assert os.path.exists(os.path.join(folder_path, "image.jpg"))

    # Check if we can load it
    response = client.get(f"/api/operator/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["image_url"] == f"/api/operator/sessions/{session_id}/image"

    # Check image endpoint
    response = client.get(f"/api/operator/sessions/{session_id}/image")
    assert response.status_code == 200

    print("Test passed!")

if __name__ == "__main__":
    test_image_save()
