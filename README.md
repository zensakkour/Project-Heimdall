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
- `src/core/detection/base.py`
- `src/core/detection/yolo_obb.py`
- `src/schemas/detection.py`
- `src/tools/benchmark_detector.py`

### Module B - Geolocation Candidate Generation
- Candidate provider interface.
- Initial backend: embedding retrieval (GeoCLIP/GeoFT style).
- Always return ranked Top-N candidates (lat/lon, retrieval score, optional match id).

Deliverables:
- `src/core/geo/candidate_provider.py`
- `src/core/geo/geoclip_provider.py`
- `src/schemas/geo_candidate.py`
- `src/tools/run_geo_candidates.py`

### Module C - Probabilistic Verification & Fusion
- Likelihood models for shadow/terrain residuals.
- Temperature scaling for retrieval scores.
- Fuse signals in log-space into a posterior over candidates.
- Compute uncertainty ellipse and radius.

Deliverables:
- `src/core/logic/likelihoods.py`
- `src/core/logic/fusion.py`
- `src/schemas/fusion.py`
- `src/tools/calibrate_fusion.py`

### Module D - Explainability Layer
- Evidence object per candidate (residuals, likelihoods, weights).
- Compact explanation text per candidate.
- Include explainability fields in JSON outputs/UI.

Deliverables:
- `src/schemas/evidence.py`
- Fusion output extends evidence.
- `src/tools/generate_ui_data.py`

### Module E - Temporal / Multi-View Consistency (Tracking)
- Track abstraction across frames/posts.
- Association via embedding similarity + spatial + temporal proximity.
- Bayesian updates using previous posterior as prior.
- Optional and configurable.

Deliverables:
- `src/core/logic/tracking.py`
- `src/core/logic/filtering.py`
- `src/schemas/track.py`
- `src/tools/run_sequence_eval.py`

### Module F - UI & Reporting
- Map: candidates, posterior-weighted estimate, uncertainty ellipse.
- Table: residuals, likelihoods, weights.
- Object view: detections.
- Track timeline.
- Audit export: JSON per image with intermediate signals.

Deliverables:
- `src/dashboard/` extensions for table + uncertainty overlays + track timeline
- `src/tools/run_tests_report.py`
- `src/tools/export_audit.py`

### Module G - Evaluation & Reproducibility
- One-command evaluation (JSONL + HTML report).
- Metrics: top-1/top-5 accuracy, ECE, average uncertainty radius.
- Ablations: retrieval only vs fusion vs fusion+temporal.
- Persist config snapshot + git commit hash.

Deliverables:
- `src/tools/run_all.py`
- `src/tools/eval_metrics.py`
- `src/docs/REPRODUCIBILITY.md`

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
|-- src/
|   |-- core/
|   |-- tools/
|   |-- schemas/
|   |-- tests/
|   |-- dashboard/
|   |-- config/
|   |-- docs/
|   |-- scripts/
|   `-- ingestion/
`-- data/            (local samples / weights)
```

---

## Getting Started (Local)

### Prerequisites
- Windows 10/11 (WSL2 optional)
- Python 3.10+
- Git

### How to Launch
Dashboard-only (static):
```powershell
python -m src.tools.generate_ui_data --jsonl runs/results.jsonl
cd src/dashboard
python -m http.server 8000
```

Full UI with analysis uploads (FastAPI):
```powershell
python -m uvicorn src.tools.ui_server:app --reload --port 8000
```

Open:
- `http://127.0.0.1:8000/` (Redirects to Analysis)
- `http://127.0.0.1:8000/analysis/` (Analysis)

#### Map legend & confidence
- **Green dots** = geo candidates (size reflects confidence; brighter = higher confidence).
- **White dot** = fused mean location.
- **Dashed ring** = uncertainty radius around the fused mean (approx. distance where the true location is likely to fall).
- **Geo confidence** is shown in the summary (top candidate fusion weight or retrieval score).
- **Detection confidence** is shown next to each detection with a percent bar.
- **Hover/click** a candidate to see rank, lat/lon, and confidence details.

### Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### CUDA / GPU (PyTorch)
If you want GPU acceleration (recommended for RTX 50xx), install PyTorch separately using the official selector:
1. Open the PyTorch “Start Locally” page.
2. Choose **Preview (Nightly)** and **CUDA 12.8** for Windows.
3. Run the exact command it generates in your activated `.venv`.

RTX 50xx (`sm_120`) requires CUDA 12.8+ builds, and PyTorch maintainers recommend using nightly builds with CUDA 12.8 until the stable matrix fully aligns. citeturn0search2turn0search5

Official installer selector: citeturn0search0

### Local Data (Not Pushed To Git)
Large datasets are intentionally **not** stored in git. You must download them locally.

#### SpaceNet Paris (Aerial Geo Retrieval)
Prereqs:
- `awscli` (already in `requirements.txt`)
- `rasterio` (already in `requirements.txt`)

Download tiles (example: AOI_3_Paris PS-RGB):
```powershell
python -m awscli s3 cp --no-sign-request s3://spacenet-dataset/AOIs/AOI_3_Paris/PS-RGB/ data/spacenet_paris/PS-RGB/ --recursive
```

Chip GeoTIFFs into JPGs + metadata:
```powershell
python -m src.tools.ingest_spacenet_psrgb --input-dir data/spacenet_paris/PS-RGB --output-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --chip-size 512 --stride 512 --max-chips-per-tiff 80
```

Build the geo retrieval index:
```powershell
python -m src.tools.build_geo_index --images-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --output data/geo_index/spacenet_paris_clip.npz
```

#### Open Geo Demo (Small)
```powershell
python -m src.tools.download_open_geo --limit 100 --output data/open_geo
python -m src.tools.build_geo_index --images-dir data/open_geo/images --metadata data/open_geo/metadata.csv --output data/geo_index/open_geo_clip.npz
```

#### DOTA v1.0 (Object Detection)
```powershell
python -m src.tools.download_dota_v1 --output data/dota
python -m src.tools.prepare_dota_v1 --zip data/dota --out data/dota_v1 --yaml data/dota_v1/dota.yaml
```

### Geo Profiles (UI Tabs)
The analysis UI lets you switch between geo indices:
- Paris Focus: `src/config/paris.json` (SpaceNet Paris index)
- Legacy Open Geo: `src/config/open_geo.json` (Wikimedia demo index)

### Dataset (University-1652)
Download and export a working image subset:
```powershell
python -m src.tools.download_university1652
python -m src.tools.prepare_university1652 --split train --limit 200 --source-dir data/University-1652
```

Use in Live (upload from exported folder):
- `data/university-1652/images/train/`

Use in Non-Live (batch run + dashboard):
```powershell
python -m src.batch_run data/university-1652/images/train --output runs/results.jsonl
python -m src.tools.generate_ui_data --jsonl runs/results.jsonl
```

Important:
- The HuggingFace mirror is metadata-only; it does not contain the image files.
- You must download the full University-1652 dataset from the official source and set `--source-dir`.

### Retrieval Index (Reference Matching)
If you want the dataset to improve geo results, build an embedding index and point the config to it:
```powershell
python -m src.tools.build_geo_index --images-dir data/university-1652/images/train --metadata data/university-1652/metadata.csv
```

Then set in `src/config/defaults.json`:
- `geolocator.retrieval_index_path`
- `geolocator.retrieval_model_id`
- `geolocator.retrieval_top_k`
- `geolocator.retrieval_min_score`

### Open Geo Demo Dataset (No Request Needed)
Download a small open geotagged set from Wikimedia Commons:
```powershell
python -m src.tools.download_open_geo --limit 100 --output data/open_geo
python -m src.tools.build_geo_index --images-dir data/open_geo/images --metadata data/open_geo/metadata.csv --output data/geo_index/open_geo_clip.npz
```

Then set in `src/config/defaults.json`:
```json
"retrieval_index_path": "data/geo_index/open_geo_clip.npz"
```

### Run the legacy pipeline (single image)
```powershell
python -m src.cli data/samples/real_port_miami.jpg --json
```

### Generate geo candidates
```powershell
python -m src.tools.run_geo_candidates data/samples/real_port_miami.jpg
```

### Geolocation model (GeoCLIP / GeoSpot Base)
The default config uses GeoSpot Base (`sdan/geospot-base`) with a SigLIP2 vision backbone.
On first run, the model will download if `huggingface_hub` is installed and network access is available.

Manual download:
```powershell
python -m src.tools.download_geospot_base
```

### Run fusion (stub pipeline)
```powershell
python -m src.tools.run_all data/samples/real_port_miami.jpg
python -m src.tools.eval_metrics runs/results.jsonl
```

### UI (Local)
Dashboard-only (static):
```powershell
python -m src.tools.generate_ui_data --jsonl runs/results.jsonl
cd src/dashboard
python -m http.server 8000
```

Full UI with analysis uploads (FastAPI):
```powershell
python -m uvicorn src.tools.ui_server:app --reload --port 8000
```

Open:
- `http://127.0.0.1:8000/` (Dashboard)
- `http://127.0.0.1:8000/analysis/` (Analysis)

---

## Geolocation Tech\nSee `src/docs/GEO_TECH.md` for the current geo + detection stack and versions.\n\n## Status
Active development aligned to the 2026-01-30 engineering specification. See `PROGRESS.md` for the latest work and next steps.
