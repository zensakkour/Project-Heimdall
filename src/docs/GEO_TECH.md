# Geolocation + Object Localization Tech (Current)

## Overview
This doc describes the current geolocation and object localization stack, how it is wired, and the exact software/model versions detected in this environment.

## Dataset (Current)
- University-1652 (HF mirror): `layumi/university-1652`
- Local path: `data/university-1652`
- Export tool: `python -m src.tools.prepare_university1652 --split train --limit 200 --source-dir data/University-1652`
- Note: the HuggingFace mirror is metadata-only (no image files). You need the official dataset download.

## Retrieval Index (Reference Matching)
Retrieval uses a CLIP image encoder to build an embedding index of reference images with known GPS.
At runtime, new images are embedded and nearest neighbors are returned as geo candidates.

Build an index (requires images + GPS metadata or sidecar files):
```powershell
python -m src.tools.build_geo_index --images-dir data/university-1652/images/train --metadata data/university-1652/metadata.csv
```

Or use sidecar files next to images (`.geo.json` / `.geoloc.json`) instead of a metadata CSV/JSON:
```powershell
python -m src.tools.build_geo_index --images-dir data/samples/with_sidecars
```

Configure retrieval in `src/config/defaults.json`:
- `geolocator.retrieval_index_path`
- `geolocator.retrieval_model_id` (default `openai/clip-vit-large-patch14`)
- `geolocator.retrieval_top_k`
- `geolocator.retrieval_min_score`

### Open Geo Demo (Wikimedia Commons)
You can bootstrap retrieval with open geotagged images from Wikimedia Commons:
```powershell
python -m src.tools.download_open_geo --limit 100 --output data/open_geo
python -m src.tools.build_geo_index --images-dir data/open_geo/images --metadata data/open_geo/metadata.csv --output data/geo_index/open_geo_clip.npz
```

## Object Localization (Detection)
- Model runtime: Ultralytics YOLO OBB (oriented bounding boxes)
- Weights (default): `yolo11x-obb.pt`
- Adapter: `src/core/detection/ultralytics_obb.py`
- Factory: `src/core/detection/factory.py`
- Config: `src/config/defaults.json` -> `detector.*`

Key settings in config:
- `detector.weights_path`
- `detector.min_confidence`
- `detector.nms_iou`
- `detector.max_detections`
- `detector.imgsz`

## Geolocation (Geo Candidates + Fusion)
### Candidate generation
- Primary model: GeoSpot Base (Hugging Face)
- Model ID: `sdan/geospot-base`
- Cache: `data/models/geospot-base`
- Provider: `src/core/geo/geoclip_provider.py`
- Config: `src/config/defaults.json` -> `geolocator.*`

Implementation details:
- GeoSpot Base uses a SigLIP2 vision backbone (patched in code to pass required inputs).
- The GeoCLIP location encoder + GPS gallery are used to produce candidate GPS points.
- Candidates are returned as `GeoCandidate` items with a retrieval score and match id.

### Fusion and uncertainty
- Fusion: `src/core/logic/fusion.py`
- Likelihoods: `src/core/logic/likelihoods.py`
- Uncertainty ellipse and radius are computed from candidate covariance.
- Config: `src/config/defaults.json` -> `fusion.*`

Key settings in config:
- `geolocator.top_n` (candidate pool size)
- `fusion.top_k` (candidates kept for fusion)
- `fusion.retrieval_temperature` (score scaling)
- `fusion.use_shadow` / `fusion.use_terrain`
- `fusion.shadow_sigma_deg` / `fusion.terrain_sigma`

## Scoring and Geo Tier
- Score: `src/core/logic/score.py`
- Geo tier computed in serialization: `src/core/logic/serialize.py`
- If Geo is missing, UI falls back to fusion/candidate scores for tier display.

## Versions (Detected in this environment)
- torch: 2.10.0+cpu
- ultralytics: 8.4.9
- transformers: 5.0.0
- fastapi: 0.128.0
- uvicorn: 0.40.0
- opencv-python: 4.13.0
- pillow: 12.1.0
- huggingface_hub: 1.3.5
- safetensors: 0.7.0
- geoclip: (no version string reported by package)

Notes:
- Dependencies are currently unpinned in `requirements.txt`.
- The GeoSpot Base model weights live under `data/models/geospot-base`.

## Entry Points
- CLI: `python -m src.cli <image>`
- Full UI: `python -m uvicorn src.tools.ui_server:app --reload --port 8000`
- Analysis page: `http://127.0.0.1:8000/analysis/`
