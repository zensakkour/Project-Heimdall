# Heimdall Dashboard

Operator-facing web interface for Project Heimdall.

## What This UI Provides

- `Analysis` tab: upload and analyze image inputs, inspect detections, review geo ranking, and view fused map output.
- `Scoring` tab: run geo evaluation against a metadata CSV and monitor progress/results.

## Run Locally

From repository root:

```powershell
.\.venv\Scripts\python -m src.tools.generate_ui_data --jsonl runs/results.jsonl
.\.venv\Scripts\python -m uvicorn src.tools.ui_server:app --reload --port 8000
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/analysis/`

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

## Screenshot Assets

README screenshots are stored in `docs/images/` at repo root.