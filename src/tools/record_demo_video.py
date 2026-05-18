"""
Record clean desktop demo media of the analysis app with globe interactions.

Outputs:
- docs/images/analysis-demo.webm
- docs/images/analysis-desktop.png
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
import argparse

from playwright.sync_api import sync_playwright


def _find_open_port(host: str, start: int, end: int) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No open port in range {start}-{end}")


def _wait_for_server(url: str, timeout_s: int = 30) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as res:
                if res.status == 200:
                    return
        except Exception:
            time.sleep(0.4)
    raise RuntimeError(f"Server did not become ready: {url}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record Heimdall demo video + screenshot.")
    parser.add_argument(
        "--with-analyze",
        action="store_true",
        help="Run Analyze Image during recording (requires healthy model dependencies).",
    )
    parser.add_argument(
        "--sample-image",
        type=str,
        default="data/analysis_tests/paris_street/images/mapillary__1021055432583866.jpg",
        help="Sample image to upload when --with-analyze is enabled.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="paris_test",
        help="Analysis profile to select before running Analyze Image.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[2]
    py = root / ".venv" / "Scripts" / "python.exe"
    out_dir = root / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "analysis-demo.webm"
    screenshot_path = out_dir / "analysis-desktop.png"
    temp_dir = out_dir / "_video_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    sample_image = Path(args.sample_image)
    if not sample_image.is_absolute():
        sample_image = root / sample_image
    host = "127.0.0.1"
    port = _find_open_port(host, 8050, 8150)
    base = f"http://{host}:{port}"

    server = subprocess.Popen(
        [
            str(py),
            "-m",
            "uvicorn",
            "src.tools.ui_server:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_for_server(base + "/analysis/", timeout_s=90)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                record_video_dir=str(temp_dir),
                record_video_size={"width": 1920, "height": 1080},
            )
            page = context.new_page()
            page.goto(base + "/analysis/", wait_until="networkidle", timeout=60000)
            page.wait_for_selector("#live-map", timeout=30000)
            page.wait_for_timeout(1200)

            if args.with_analyze and sample_image.exists():
                page.select_option("#profile-select", value=str(args.profile))
                page.set_input_files("#image-file", str(sample_image))
                page.wait_for_timeout(300)
                with page.expect_response(
                    lambda response: response.request.method == "POST" and "/api/operator/analyze" in response.url,
                    timeout=180000,
                ) as analyze_response:
                    page.click("#geolocate-image")
                response = analyze_response.value
                payload = response.json()
                if not response.ok:
                    raise RuntimeError(f"Analyze Image failed with HTTP {response.status}: {payload}")
                if payload.get("safe_demo"):
                    raise RuntimeError(f"Analyze Image returned safe_demo payload: {payload}")
                if payload.get("error"):
                    raise RuntimeError(f"Analyze Image returned error payload: {payload}")
                page.wait_for_function(
                    """
                    () => {
                      const raw = document.querySelector("#raw-json")?.textContent || "";
                      const status = document.querySelector("#diag-model-status")?.textContent || "";
                      const progress = document.querySelector("#progress");
                      return raw.trim() !== "{}" && status.trim() && status.trim() !== "-" && progress?.style.display === "none";
                    }
                    """,
                    timeout=180000,
                )
                page.wait_for_timeout(1200)

            map_box = page.locator("#live-map").bounding_box()
            if not map_box:
                raise RuntimeError("Map element bounds unavailable.")
            cx = map_box["x"] + map_box["width"] * 0.5
            cy = map_box["y"] + map_box["height"] * 0.5

            # Rotate/pan with drag.
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.mouse.move(cx + 260, cy + 40, steps=40)
            page.mouse.up()
            page.wait_for_timeout(700)

            page.mouse.move(cx, cy)
            page.mouse.down()
            page.mouse.move(cx - 220, cy - 30, steps=35)
            page.mouse.up()
            page.wait_for_timeout(700)

            # Zoom in and out with wheel over the globe.
            page.mouse.move(cx, cy)
            for _ in range(6):
                page.mouse.wheel(0, -500)
                page.wait_for_timeout(180)
            page.wait_for_timeout(500)
            for _ in range(7):
                page.mouse.wheel(0, 500)
                page.wait_for_timeout(180)

            # Use map controls to demonstrate UI controls.
            page.wait_for_timeout(400)
            page.click("#map-zoom-in")
            page.wait_for_timeout(450)
            page.click("#map-zoom-out")
            page.wait_for_timeout(450)
            page.click("#map-reset-globe")
            page.wait_for_timeout(1200)

            # Capture the updated desktop screenshot from the same clean state.
            page.screenshot(path=str(screenshot_path), full_page=True)

            video = page.video
            page.close()
            context.close()
            browser.close()

            if video is None:
                raise RuntimeError("Demo video was not recorded.")
            recorded_path = Path(video.path())

        if destination.exists():
            destination.unlink()
        shutil.move(str(recorded_path), str(destination))
        print(f"Demo video created: {destination}")
        print(f"Demo screenshot created: {screenshot_path}")
        if args.with_analyze:
            print(f"Analyze sample used: {sample_image}")
            print(f"Analyze profile used: {args.profile}")
    finally:
        try:
            server.terminate()
            server.wait(timeout=5)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
