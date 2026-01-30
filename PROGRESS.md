# Progress Log (Append-Only)

Do not delete or edit past entries. Append new work at the end.

## 2026-01-29
- Initialized repo structure: core/, ingestion/, data/, dashboard/ with .gitkeep placeholders.
- Added .gitignore for Python, C++ artifacts, and data/weights folders.
- Added requirements.txt with initial Python dependencies (un-pinned).
- Cleaned and reformatted README, then updated setup for WSL2 (Ubuntu 22.04).
- Updated Phase 1 environment line to be WSL-friendly.
- Created this PROGRESS.md log.

- Added core pipeline skeleton with domain types, scoring stub, and pipeline wiring in core/logic.

- Added detection adapter stub (YOLOv11-OBB) and geolocation adapter stub (GeoFT/GeoCLIP).

- Added verification stubs (shadow/topo) and wired adapters into the core pipeline with dependency injection.

- Added cli.py to run the pipeline on a local image with optional detector/geolocator paths.

- Added WSL helper script scripts/run_pipeline.sh and documented CLI usage in README.

- Added batch_run.py (folder runner with JSONL output) and scripts/run_batch.sh; documented batch usage in README.

- Added JSON schema for batch output in schemas/batch_result.schema.json.
- Added a minimal pytest for JSONL roundtrip in tests/test_batch_output.py.
- Added pytest to requirements and .pytest_cache to .gitignore; documented test run in README.

- Added config/defaults.json and core/logic/config.py for JSON-based pipeline configuration.
- Updated cli.py and batch_run.py to accept --config.
- Documented config usage in README.

- Added serialization helper core/logic/serialize.py.
- Updated cli.py to support --json output; batch_run.py now uses shared serializer.
- Documented JSON output in README.

- Paused work per user request; added resume note to README.

- Added tools/validate_jsonl.py for schema validation of batch JSONL output.
- Added jsonschema to requirements and documented validation command in README.

## 2026-01-30
- Added EXIF metadata helper for GPS/time extraction in core/logic/image_meta.py.
- GeoLocator now returns a GeoEstimate when EXIF GPS is present (confidence + landmark tag).
- YOLO OBB adapter can load detections from optional sidecar .detections.json files for local testing.
- Added pillow dependency for EXIF access.

## 2026-01-30
- Expanded detector config with min_confidence, NMS IoU, and max_detections; added post-processing utilities (AABB NMS + OBB validation).
- Added approximate solar position helper and improved verification logic with EXIF time + geo heuristics and notes.
- GeoLocator now supports .geo/.geoloc sidecar files and configurable EXIF/sidecar usage.
- Scoring now weights detections, geo, and verification with configurable weights.
- Updated CLI/batch pipeline wiring for new config options; extended defaults.json accordingly.
- Added tests for scoring and post-processing.
- Added core/logic/astro.py and core/logic/postprocess.py.


## 2026-01-30
- Added OBB normalization and stricter validity checks in post-processing; detections now have deterministic point order.
- Added geo range validation for EXIF and sidecar geolocation.
- Extended post-processing tests for OBB normalization.
- Ran pytest (4 passed).


## 2026-01-30
- Added detection metadata fields (heading_deg, shadow_azimuth_deg) and geo uncertainty_m across types/serialization/schema.
- Implemented OBB heading normalization and shadow-vs-heading heuristic in verification using sun azimuth + observed shadow directions.
- Scoring now downweights geo confidence when uncertainty_m is high.
- Added sidecar format docs in README and new pipeline sidecar smoke test.
- Ran pytest (5 passed).


## 2026-01-30
- Added shadow length ratio support and heuristic check tied to sun elevation; verification can disable shadow heuristics via config/CLI.
- Added detection metadata field shadow_length_ratio and geo uncertainty_radius_m in serialization/schema/README sidecar docs.
- Added CLI and batch flags --no-shadow and verification config defaults.
- Added tests for shadow length heuristic and updated sidecar test; ran pytest (6 passed).


## 2026-01-30
- Added shadow extraction stub to infer shadow azimuth/length from image dark pixels and wired into pipeline enrichment.
- Added geo confidence_tier to serialized output and schema; UI now surfaces scores and test status.
- Added lightweight dashboard UI (HTML/CSS/JS) plus tools to generate test reports and summary JSON.
- Added tests for shadow extraction; ran pytest (7 passed).


## 2026-01-30
- Added live analysis UI (dashboard/live) with manual file upload for images/videos and periodic snapshot analysis.
- Implemented FastAPI server tools/ui_server.py to run pipeline on uploads and return annotated frames.
- Added confidence_tier serialization + schema and test; added UI error handling for empty video frames.
- Updated requirements with FastAPI/uvicorn/opencv/python-multipart and README with live UI instructions.
- Ran pytest (8 passed).


## 2026-01-30
- Added tools/generate_sample_media.py to create sample image/video and sidecar JSONs for the live UI.
- Generated sample media in data/samples and documented usage in README.


## 2026-01-30
- Downloaded a real aerial photo sample into data/samples and added demo sidecar detections/geo so UI shows non-zero results.
- Documented real sample files in README.


## 2026-01-30
- Added classic (non-YOLO) detector and sidecar detector; UI now uses sidecar/classic when no weights.
- Updated cli/batch to use sidecar detector without weights; added detector config flags use_sidecar/use_classic.
- Added tests for classic detector and updated sidecar pipeline test; ran pytest (9 passed).


## 2026-01-30
- Added Ultralytics OBB detector adapter and detector factory; pipeline now supports pretrained YOLO11-OBB without rewriting code.
- Updated cli/batch/UI server to use detector factory; added ultralytics to requirements.
- Documented recommended YOLO11-OBB weights in README.
- Ran pytest (9 passed).


## 2026-01-30
- Verified YOLO11-OBB detects objects on a real aerial port image; added data/samples/real_port_miami.jpg.
- Added detector imgsz config and benchmark script tools/benchmark_detector.py.
- Ran pytest (9 passed).


## 2026-01-30
- Added real-world test media: Glasgow airport image, Larnaca airport image, and Melbourne airport drone video for YOLO11-OBB evaluation.
- Updated README with new sample files.


## 2026-01-30
- Updated live UI to a darker theme with globe/eye glyph and added interactive detection list + detail panel.
- Added canvas-based overlay rendering with selectable detections for visual inspection.


## 2026-01-30
- Added tools/download_dota_v1.py to fetch the DOTA v1.0 OBB dataset zip and documented usage in README.


## 2026-01-30
- Added DOTA v1.0 prepare script (unzip + dataset YAML) and evaluation runner with UI trigger.
- Live UI now shows eval status and results; includes a Run Eval button.


## 2026-01-30
- Fixed DOTA v1.0 YAML to include class names; regenerated data/dota/dota.yaml.


## 2026-01-30
- Enhanced DOTA eval reporting with overall mAP/mAP50 and per-class metrics; added UI tables for eval results.

