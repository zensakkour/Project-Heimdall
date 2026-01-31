# Project Heimdall
Watchman of the Gods - Sovereign Intelligence

## Scope (2026-01-30)
This repo implements the perception, geolocation, fusion, explainability, and evaluation layers for Project Heimdall.
Data ingestion and dataset preparation are explicitly out of scope for this phase.

## Global Goal
Build a clean, testable, and reproducible system that:
1) detects objects in images,
2) proposes multiple geolocation candidates,
3) verifies and fuses them with probabilistic models to produce a posterior distribution and uncertainty,
4) exposes explainable results through a lightweight UI and reports.

---

## Modules (Engineering Spec)

### Module A - Detection Interface
- Unified detector interface: `detect(image) -> List[Detection]`.
- First backend: YOLO11-OBB (oriented bounding boxes + heading).
- Optional per-detection embedding for tracking.
- Tools: detector-only benchmark script.

Deliverables:
- `core/detection/base.py`
- `core/detection/yolo_obb.py`
- `schemas/detection.py`
- `tools/benchmark_detector.py`

### Module B - Geolocation Candidate Generation
- Candidate provider interface.
- Initial backend: embedding retrieval (GeoCLIP/GeoFT style).
- Always return ranked Top-N candidates (lat/lon, retrieval score, optional match id).

Deliverables:
- `core/geo/candidate_provider.py`
- `core/geo/geoclip_provider.py`
- `schemas/geo_candidate.py`
- `tools/run_geo_candidates.py`

### Module C - Probabilistic Verification & Fusion
- Likelihood models for shadow/terrain residuals.
- Temperature scaling for retrieval scores.
- Fuse signals in log-space into a posterior over candidates.
- Compute uncertainty ellipse and radius.

Deliverables:
- `core/logic/likelihoods.py`
- `core/logic/fusion.py`
- `schemas/fusion.py`
- `tools/calibrate_fusion.py`

### Module D - Explainability Layer
- Evidence object per candidate (residuals, likelihoods, weights).
- Compact explanation text per candidate.
- Include explainability fields in JSON outputs/UI.

Deliverables:
- `schemas/evidence.py`
- Fusion output extends evidence.
- `tools/generate_ui_data.py`

### Module E - Temporal / Multi-View Consistency (Tracking)
- Track abstraction across frames/posts.
- Association via embedding similarity + spatial + temporal proximity.
- Bayesian updates using previous posterior as prior.
- Optional and configurable.

Deliverables:
- `core/logic/tracking.py`
- `core/logic/filtering.py`
- `schemas/track.py`
- `tools/run_sequence_eval.py`

### Module F - UI & Reporting
- Map: candidates, posterior-weighted estimate, uncertainty ellipse.
- Table: residuals, likelihoods, weights.
- Object view: detections.
- Track timeline.
- Audit export: JSON per image with intermediate signals.

Deliverables:
- `dashboard/` extensions for table + uncertainty overlays + track timeline
- `tools/run_tests_report.py`
- `tools/export_audit.py`

### Module G - Evaluation & Reproducibility
- One-command evaluation (JSONL + HTML report).
- Metrics: top-1/top-5 accuracy, ECE, average uncertainty radius.
- Ablations: retrieval only vs fusion vs fusion+temporal.
- Persist config snapshot + git commit hash.

Deliverables:
- `tools/run_all.py`
- `tools/eval_metrics.py`
- `docs/REPRODUCIBILITY.md`

---

## Non-goals (Explicit)
- No data ingestion connectors, scraping logic, or dataset construction in this phase.
- No LLM or text processing components.
- No operational or real-world deployment claims.
- No automated decision or targeting functionality.

---

## Project Structure (Current)
```
Project-Heimdall/
|-- core/
|   |-- detection/   (detector interfaces and backends)
|   |-- geo/         (geo candidate providers)
|   `-- logic/       (fusion, verification, tracking)
|-- schemas/         (pydantic schemas)
|-- tools/           (CLI utilities)
|-- dashboard/       (lightweight UI)
|-- docs/            (reproducibility)
`-- data/            (local samples / weights)
```

---

## Getting Started (Local)

### Prerequisites
- Windows 10/11 (WSL2 optional)
- Python 3.10+
- Git

### Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the legacy pipeline (single image)
```powershell
python cli.py data/samples/real_port_miami.jpg --json
```

### Generate geo candidates
```powershell
python tools/run_geo_candidates.py data/samples/real_port_miami.jpg
```

### Geolocation model (GeoCLIP / GeoSpot Base)
The default config uses GeoSpot Base (`sdan/geospot-base`) with a SigLIP2 vision backbone.
On first run, the model will download if `huggingface_hub` is installed and network access is available.

Manual download:
```powershell
python tools/download_geospot_base.py
```

### Run fusion (stub pipeline)
```powershell
python tools/run_all.py data/samples/real_port_miami.jpg
python tools/eval_metrics.py runs/results.jsonl
```

### UI (Local)
Dashboard-only (static):
```powershell
python tools/generate_ui_data.py --jsonl runs/results.jsonl
cd dashboard
python -m http.server 8000
```

Full UI with analysis uploads (FastAPI):
```powershell
python -m uvicorn tools.ui_server:app --reload --port 8000
```

Open:
- `http://127.0.0.1:8000/` (Dashboard)
- `http://127.0.0.1:8000/analysis/` (Analysis)

---

## Status
Active development aligned to the 2026-01-30 engineering specification. See `PROGRESS.md` for the latest work and next steps.
