# Project Heimdall
Watchman of the Gods - Sovereign Intelligence

## 1. Core Architecture: The Heimdall Vision

Project Heimdall leverages high-performance C++ and Python to bridge the gap between "noisy" social media feeds (Telegram, etc.) and high-accuracy situational awareness.

### The Technical Stack

- Detection: YOLOv11-OBB (Oriented Bounding Boxes). Unlike standard boxes, OBB detects the rotation of a target, providing a heading that is useful for map matching.
- Geolocation: GeoFT / GeoCLIP. A pixel-to-coordinate engine that treats geolocation as a retrieval problem, matching ground-level features against a global satellite database.
- Intelligence: Telethon / Pyrogram for high-speed, asynchronous monitoring of specific reporting channels.
- Deep Tech Integration: PennyLane. A hybrid quantum-classical circuit used to enhance classification confidence in low-resolution or obfuscated imagery.

---

## 2. Phase-by-Phase Setup

### Phase 1: Foundation (Weeks 1-4)

- Environment: Windows 10/11, Python 3.10+. CUDA is optional for local training.
- Dataset Setup:
  - DOTA v1.0: Primary training set for oriented targets (tanks, ships, aircraft).
  - MSTAR: Specialized dataset for Synthetic Aperture Radar (SAR) recognition benchmarks.
- Labeling: Use Roboflow for OBB annotations, converting traditional DOTA formats to the `class_index x1 y1 x2 y2 x3 y3 x4 y4` YOLO format.

### Phase 2: The Heimdall Geolocation Core (Weeks 5-8)

- The Engine: Fork GeoFT and replace its backbone with a C3k2-optimized feature extractor to improve small-object detection in wide landscapes.
- Shadow Verification (Chronolocation): Implement a Python module using SunCalc. It matches shadow angles of detected vehicles to the sun's position at the estimated GPS coordinates and time to provide verified confirmation.
- Topographic Matching: Cross-reference background mountain silhouettes with NASA SRTM data to ensure the horizon matches the predicted coordinates.

### Phase 3: Live Mapping and Scale (Weeks 9-12)

- Ingestion: Set up a Telethon bot to scrape images and videos from priority channels.
- Heimdall Score: Develop a proprietary confirmation score that weights visual matches, shadow analysis, and topographic verification.
- Output: A live Mapbox dashboard showing verified threats with a timestamp and high-confidence location pins.

---

## 3. VC and Defense Strategy

To secure funding (e.g., from defense-focused firms like Anduril or Helsing), Project Heimdall emphasizes sovereign intelligence.

- GPS-Denied Capability: Visual geolocation is the primary fallback when GPS is jammed or spoofed.
- Explainability: Heimdall outputs specific visual landmarks used for the location, providing a rationale for human analysts.
- Computational Efficiency: By optimizing the C++ core for TensorRT, Heimdall is designed to run on-device for autonomous drones or tactical ruggedized tablets.

---

### Project Structure

```
Heimdall/
|-- core/
|   |-- detection/   (YOLOv11-OBB models)
|   |-- geo/         (GeoFT/GeoCLIP alignment logic)
|   `-- logic/       (Chronolocation and terrain verification)
|-- ingestion/
|   `-- telegram/    (Telethon scraper and parsers)
|-- data/
|   |-- dota/        (Aerial oriented images)
|   `-- weights/     (Trained yolo11-obb.pt)
`-- dashboard/       (Mapbox live visualization)
```

---

## 4. Getting Started (Local)

### Prerequisites (WSL2)
- Windows 10/11 with WSL2 enabled
- Ubuntu 22.04 (from Microsoft Store)
- Python 3.10+
- C++ toolchain (GCC/clang)
- Git

### Setup (WSL Ubuntu)

```powershell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run (WSL)

```bash
python3 cli.py /path/to/image.jpg
```

Or:

```bash
./scripts/run_pipeline.sh /path/to/image.jpg
```

JSON output:

```bash
python3 cli.py /path/to/image.jpg --json
```

### Sidecar Formats (Optional)

To run the pipeline without real models, drop sidecar JSON next to the image.

Detection sidecar (either `{image}.detections.json` or `{image_basename}.detections.json`):

```json
{
  "detections": [
    {
      "label": "vehicle",
      "confidence": 0.92,
      "obb": [[0,0],[10,0],[10,5],[0,5]],
      "heading_deg": 90.0,
      "shadow_azimuth_deg": 210.0,
      "shadow_length_ratio": 2.2
    }
  ]
}
```

Geolocation sidecar (any of `{image}.geo.json`, `{image_basename}.geo.json`,
`{image}.geoloc.json`, `{image_basename}.geoloc.json`):

```json
{
  "latitude": 35.0,
  "longitude": -120.0,
  "confidence": 0.7,
  "uncertainty_m": 80.0,
  "uncertainty_radius_m": 80.0,
  "landmarks": ["sidecar"]
}
```

### Config (Optional)

Edit `config/defaults.json` and run:

```bash
python3 cli.py /path/to/image.jpg --config config/defaults.json
```

### Batch Run (WSL)

```bash
python3 batch_run.py /path/to/images --output outputs.jsonl
```

Or:

```bash
./scripts/run_batch.sh /path/to/images --output outputs.jsonl
```

### Validate Output (WSL)

```bash
pytest -q
```

Schema validation for batch JSONL:

```bash
python3 tools/validate_jsonl.py outputs.jsonl
```

### UI (Local)

Generate test + score summaries:

```powershell
python tools/run_tests_report.py
python tools/generate_ui_data.py --jsonl outputs.jsonl
```

Serve the UI:

```powershell
cd dashboard
python -m http.server 8000
```

### Live Analysis UI (Local)

Start the live analysis server:

```powershell
python -m uvicorn tools.ui_server:app --reload --port 8001
```

Open: `http://localhost:8001`

Notes:
- If no YOLO weights are provided, the live UI will still detect blobs via a simple classical detector.
- Sidecar detections still work when no weights are provided.

### Best Working Model (Recommended)

Fastest path to strong detections is Ultralytics YOLO11 OBB (pretrained).

Set in `config/defaults.json`:

```json
{
  "detector": {
    "weights_path": "yolo11x-obb.pt"
  }
}
```

Then run the live UI and it will load the model automatically (Ultralytics will download the weights if missing).

### Sample Media (Local)

Generate sample image/video + sidecar JSONs:

```powershell
python tools/generate_sample_media.py
```

Sample files created:
- `data/samples/sample.jpg`
- `data/samples/sample.jpg.detections.json`
- `data/samples/sample.geo.json`
- `data/samples/sample.mp4`

Real photo sample (downloaded from a public-domain source):
- `data/samples/real_photo.jpg`
- `data/samples/real_photo.jpg.detections.json`
- `data/samples/real_photo.geo.json`

Another real aerial sample with ships (works with YOLO11-OBB):
- `data/samples/real_port_miami.jpg`

More real samples (for testing):
- `data/samples/real_glasgow_airport.jpg`
- `data/samples/real_larnaca_airport.jpg`
- `data/samples/real_melbourne_airport.webm`

### Benchmark (Detector Only)

```powershell
python tools/benchmark_detector.py --images data/samples/real_port_miami.jpg
```

### Dataset: DOTA v1.0 (OBB)

Download:

```powershell
python tools/download_dota_v1.py
```

This downloads `data/dota/DOTAv1.zip`.

Prepare (extract + dataset YAML):

```powershell
python tools/prepare_dota_v1.py
```

Run evaluation (CLI):

```powershell
python tools/run_dota_eval.py
```

Or run evaluation from the Live UI using the “Run Eval” button.

### Notes (WSL)
- Use Windows tools for Git GUI if you want, but run installs and scripts inside WSL.
- CUDA tooling is optional unless you plan to train locally; NVIDIA WSL2 GPU support is required for CUDA.

---

## 5. Status

Current work is paused. Resume by checking `PROGRESS.md` for the latest entries and next steps.

### Current Status
- Project skeleton and README cleaned up.
- Core directories created under `core/`, `ingestion/`, `data/`, and `dashboard/`.
- Dependencies listed in `requirements.txt` (pin versions when ready).
