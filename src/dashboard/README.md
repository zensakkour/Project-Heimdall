# Heimdall Dashboard

Operator-facing web interface for Project Heimdall.

## What This UI Provides

- `Analysis` tab: upload and analyze image inputs, inspect detections, review geo ranking, and view fused map output.
- `Scoring` tab: run geo evaluation against a metadata CSV and monitor progress/results.

## Run Locally

From repository root:

```powershell
.\.venv\Scripts\python -m src.tools.generate_ui_data --jsonl runs/results.jsonl
.\.venv\Scripts\python -m src.tools.dev_app
```

Open:

- Open the URL printed in your terminal (example: `http://127.0.0.1:8000/analysis/`).

## Frontend Layout

- `index.html`: root redirect shell
- `analysis/index.html`: primary analysis interface
- `analysis/live.js`: UI logic and API calls
- `analysis/live.css`: analysis-specific styles
- `theme.css` and `styles.css`: shared styling system

## Backend Endpoints Used By UI

- `POST /analyze/image`
- `POST /analyze/video`
- `POST /eval/geo/start`
- `GET /eval/geo/status`
- `POST /eval/dota/start`
- `GET /eval/dota/status`

Safe demo mode:

- If runtime dependencies fail, analysis endpoints return mock outputs so the UI remains usable.
- Force fallback explicitly: `POST /analyze/image?safe_demo=1`

## Screenshot Assets

README screenshots are stored in `docs/images/` at repo root.

## Demo Video Asset

Place the desktop demo video at:

- `docs/images/analysis-demo.webm`
- `docs/images/analysis-desktop.png`

Generate/update video automatically:

```powershell
.\.venv\Scripts\python -m pip install playwright
.\.venv\Scripts\python -m playwright install chromium
.\.venv\Scripts\python -m src.tools.record_demo_video
```

Optional (include `Analyze Image` run if deps are healthy):

```powershell
.\.venv\Scripts\python -m src.tools.record_demo_video --with-analyze
```

Recommended demo interactions to include:

1. Upload image and run `Analyze Image`.
2. Rotate/pan the 3D globe by dragging.
3. Zoom in/out with mouse scroll on globe.
4. Use map buttons: `Zoom In`, `Zoom Out`, `Reset`.
