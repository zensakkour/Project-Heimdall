# Geolocation + Object Localization Tech (Current)

## Status Snapshot (As of April 5, 2026)

This is the current technical state of the geo stack in the development branch:

- Candidate generation is multi-source and resilient:
  - Retrieval index provider (`GeoRetrievalProvider`)
  - GeoSpot/GeoCLIP provider (`GeoCLIPProvider`)
  - EXIF and sidecar fallbacks
- Candidate quality controls are in place:
  - Invalid coordinate/score filtering
  - Near-duplicate clustering/merging across providers
  - Capped merged candidate list before fusion
- Detection quality controls are configurable:
  - Minimum OBB area filter (`detector.min_area_px`)
  - Class-aware vs class-agnostic NMS (`detector.class_agnostic_nms`)
  - Optional test-time augmentation (`detector.use_tta`)
- Fusion is robust and configurable:
  - Retrieval score normalization modes (`none`, `zscore_sigmoid`, `minmax`, `rank_exp`)
  - Source priors (`source_prior_retrieval`, `source_prior_geoclip`, `source_prior_exif`)
  - Spatial consensus likelihood (`use_spatial_consensus`, `spatial_sigma_km`, `spatial_consensus_weight`)
  - Dateline-safe longitude averaging and covariance
  - Posterior uncertainty radius + ellipse
  - Confidence diagnostics (`normalized_entropy`, `effective_candidate_count`, `top1_posterior`, `top2_margin`, `confidence_tier`, `ambiguous`)
  - Credible-set fusion stats (`credible_mass`, `min_credible_candidates`)
  - Optional top-cluster stats mode to avoid multimodal midpoint bias (`use_top_cluster_for_stats`, `credible_cluster_radius_km`, `min_credible_cluster_weight`)
- Evaluation tooling is operational:
  - Geo regression gate baseline checks
  - Fusion sweep/tuning script
  - Calibration/error metrics: `ece`, `brier`, `nll`

Current focus is benchmark-driven tuning (retrieval coverage + fusion calibration), not new UI scope.

## Overview
This doc describes the current geolocation and object localization stack, how it is wired, and the exact software/model versions detected in this environment.

## Dataset (Current, Local)
This section reflects what exists on disk right now under `data/`.

### Geo (local)
- Open geo demo set (Wikimedia Commons): `data/open_geo/`
  - Images: `data/open_geo/images/`
  - Metadata: `data/open_geo/metadata.csv`
- Retrieval index built from open geo demo set:
  - `data/geo_index/open_geo_clip.npz`
- SpaceNet (AOI_3_Paris) PS-RGB tiles (proof-of-concept):
  - Raw tiles: `data/spacenet_paris/PS-RGB/`
  - Chips: `data/spacenet_paris/chips/`
  - Metadata: `data/spacenet_paris/metadata.csv`
  - Retrieval index: `data/geo_index/spacenet_paris_clip.npz`
- Paris test set (not in training/index):
  - `data/test_paris/`
- Samples (used in UI/demo):
  - `data/samples/sample_port.jpg`

### Detection (local)
- DOTA v1.0 (prepared):
  - Dataset root: `data/dota_v1/`
  - YAML: `data/dota_v1/dota.yaml`

### Models / weights (local)
- GeoSpot Base weights cache:
  - `data/models/geospot-base/`
- Detection weights:
  - `yolo11x-obb.pt` (repo root)

## Optional Datasets (Not in repo by default)
These are the datasets the codebase supports or is designed to work with, but they are not included in the repository.

### University-1652 (Geo)
- Official: `layumi/University1652-Baseline` (dataset available on request)
- HF mirror (metadata only): `layumi/university-1652`
- Export tool:
  - `python -m src.tools.prepare_university1652 --split train --limit 200 --source-dir data/University-1652`
- Note: you must request and download the official dataset for images.

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

### UI Geo Profiles
The analysis UI can switch between geo profiles (tabs in the Inputs section):
- Paris Focus: `src/config/paris.json` (SpaceNet Paris index)
- Legacy Open Geo: `src/config/open_geo.json` (Wikimedia demo index)

### Open Geo Demo (Wikimedia Commons)
You can bootstrap retrieval with open geotagged images from Wikimedia Commons:
```powershell
python -m src.tools.download_open_geo --limit 100 --output data/open_geo
python -m src.tools.build_geo_index --images-dir data/open_geo/images --metadata data/open_geo/metadata.csv --output data/geo_index/open_geo_clip.npz
```

## Data You Can Add Now (Suggested)
The following datasets are commonly used to improve geo-localization and overhead object detection. See their official pages for license and download instructions.

### Geo / Landmark Retrieval
- Google Landmarks Dataset v2 (GLDv2): large-scale landmark retrieval dataset.
- University-1652: drone/satellite/ground multi-view geo-localization (requires request).

### Aerial / Overhead Object Detection
- DOTA: oriented bounding boxes for aerial imagery (high-res, many rotations).
- xView: large-scale overhead imagery with bounding boxes (60 classes).

If you want to add DOTA v1.0 now (supported by tools):
```powershell
python -m src.tools.download_dota_v1 --output data/dota
python -m src.tools.prepare_dota_v1 --zip data/dota --out data/dota_v1 --yaml data/dota_v1/dota.yaml
```

## Improvement Plan (Practical, Next Steps)
This is a minimal, high-impact path to improving geo accuracy and reducing landmark confusions.

1) Expand geo retrieval data:
- Add a larger landmark set (GLDv2) and rebuild the retrieval index.
- Add "hard negatives" for common confusions (bridges, skylines, harbors).

2) Tune geo candidate + fusion:
- Increase retrieval top_k and fusion top_k to improve candidate diversity.
- Adjust retrieval temperature to reduce overconfident wrong matches.

3) Evaluate with targeted error sets:
- Build a small labeled set of confusing look-alikes and track top-1/top-5.

### Data You Can Add Now (Quick Wins)
- GLDv2 (landmarks): best for disambiguating landmark-level confusions.
- More Wikimedia geotagged images: expand coverage cheaply.
- DOTA v1.0: train/finetune OBB detection to improve object signal quality for geo fusion.

## SpaceNet Aerial Geo (Proof-of-Concept)
SpaceNet imagery is hosted in a public S3 bucket. We use the AOI_3_Paris PS-RGB tiles for a small PoC.

Prereqs:
- `awscli` (for S3 listing/downloads)
- `rasterio` (for GeoTIFF chipping; included in `requirements.txt`)

List available AOIs:
```powershell
python -m awscli s3 ls --no-sign-request s3://spacenet-dataset/AOIs/
```

Download a few PS-RGB tiles (example: 6 tiles from AOI_3_Paris):
```powershell
python -m awscli s3 cp --no-sign-request s3://spacenet-dataset/AOIs/AOI_3_Paris/PS-RGB/16FEB29111912-S2AS_R01C01-056155973010_01_P001.TIF data/spacenet_paris/PS-RGB/
python -m awscli s3 cp --no-sign-request s3://spacenet-dataset/AOIs/AOI_3_Paris/PS-RGB/16FEB29111912-S2AS_R01C02-056155973010_01_P001.TIF data/spacenet_paris/PS-RGB/
python -m awscli s3 cp --no-sign-request s3://spacenet-dataset/AOIs/AOI_3_Paris/PS-RGB/16FEB29111912-S2AS_R01C03-056155973010_01_P001.TIF data/spacenet_paris/PS-RGB/
python -m awscli s3 cp --no-sign-request s3://spacenet-dataset/AOIs/AOI_3_Paris/PS-RGB/16FEB29111912-S2AS_R01C04-056155973010_01_P001.TIF data/spacenet_paris/PS-RGB/
python -m awscli s3 cp --no-sign-request s3://spacenet-dataset/AOIs/AOI_3_Paris/PS-RGB/16FEB29111912-S2AS_R01C05-056155973010_01_P001.TIF data/spacenet_paris/PS-RGB/
python -m awscli s3 cp --no-sign-request s3://spacenet-dataset/AOIs/AOI_3_Paris/PS-RGB/16FEB29111912-S2AS_R01C06-056155973010_01_P001.TIF data/spacenet_paris/PS-RGB/
```

Chip GeoTIFFs into JPGs + build metadata:
```powershell
python -m src.tools.ingest_spacenet_psrgb --input-dir data/spacenet_paris/PS-RGB --output-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --chip-size 512 --stride 512 --max-chips-per-tiff 80
```

Build the geo retrieval index:
```powershell
python -m src.tools.build_geo_index --images-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --output data/geo_index/spacenet_paris_clip.npz
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
- PyTorch is installed separately to match CUDA/GPU builds (see CUDA notes below).
- The GeoSpot Base model weights live under `data/models/geospot-base`.

## CUDA / GPU Install Notes
For RTX 50xx (Blackwell) GPUs, you need a PyTorch build based on CUDA 12.8+.
The PyTorch team recommends using the official “Start Locally” selector and choosing **Preview (Nightly)** + **CUDA 12.8** on Windows. citeturn0search0turn0search2turn0search5

Install steps:
1. Activate `.venv`.
2. Use the PyTorch selector to generate the command and run it.
3. Verify with:
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## Entry Points
- CLI: `python -m src.cli <image>`
- Full UI: `python -m uvicorn src.tools.ui_server:app --reload --port 8000`
- Analysis page: `http://127.0.0.1:8000/analysis/`


