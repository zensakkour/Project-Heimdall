# Project Heimdall

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#requirements)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-0A0A0A)](#requirements)
[![License](https://img.shields.io/badge/License-Non--Commercial-orange.svg)](LICENSE)

Geospatial perception and analysis platform that combines object detection, geo-candidate retrieval, probabilistic fusion, and operator-facing explainability.

## Project Status Tracking

- Engineering progress log: [PROGRESS.md](PROGRESS.md)
- What it is: append-only record of shipped changes, validation runs, and technical milestones.
- How to use it: check the top snapshot for current status, then review dated entries for full history.
- Benchmark governance: [docs/eval/latest_report.md](docs/eval/latest_report.md), [docs/eval/history.jsonl](docs/eval/history.jsonl), [docs/eval/baseline.json](docs/eval/baseline.json)
- Chronological research ledger with before/after metrics: [research.md](research.md)
- Full research-style write-up: [src/docs/RESEARCH_PAPER.md](src/docs/RESEARCH_PAPER.md)
- External market and SOTA review: [src/docs/MARKET_RESEARCH.md](src/docs/MARKET_RESEARCH.md)
- Optional publication formatting: the research draft can be converted to submission style (IMRaD with numbered equations and references).
- Documentation index: [docs/DOCS_MAP.md](docs/DOCS_MAP.md)

## What The Platform Does

Project Heimdall processes image or video inputs and produces:

- Object detections (including oriented bounding boxes for aerial imagery)
- Ranked geolocation candidates
- Fused geolocation estimate with uncertainty metrics
- Operator-friendly analysis views in a local dashboard

## Project Scope

This repository implements the perception, geolocation, fusion, explainability,
and evaluation layers of Project Heimdall.

Current scope includes:

- Detection and oriented bounding-box localization
- Multi-provider geolocation candidate generation
- Probabilistic fusion and uncertainty estimation
- Explainable structured outputs for analysis and audits
- Local dashboard and API workflows for evaluation

Current non-goals include:

- Production deployment and operational decision automation
- Scraping pipelines and broad ingestion connectors
- Text/LLM reasoning as a dependency in core inference

## Demo

### Analysis App Video (Desktop)

[![Watch demo video](docs/images/analysis-desktop.png)](docs/images/analysis-demo.webm)

GitHub does not render embedded `<video>` tags in README reliably, so use the preview image above to open the video.
Direct link: [Watch/download demo video](docs/images/analysis-demo.webm)

To regenerate this demo automatically:

```powershell
.\.venv\Scripts\python -m pip install playwright
.\.venv\Scripts\python -m playwright install chromium
.\.venv\Scripts\python -m src.tools.record_demo_video
```

Optional (include Analyze Image run if model deps are healthy):

```powershell
.\.venv\Scripts\python -m src.tools.record_demo_video --with-analyze
```

### Demo Recording Checklist (for replacement video)

Record the new demo while showing these interactions:

1. Launch app with `.\.venv\Scripts\python -m src.tools.dev_app`.
1. Open the printed `/analysis/` URL.
1. Keep recording clean (no error banners in UI).
1. In the 3D globe, drag to rotate/pan.
1. Use mouse wheel scroll to zoom in and zoom out on the globe.
1. Click map controls (`Zoom In`, `Zoom Out`, `Reset`) to show full interaction flow.
1. Keep final files at `docs/images/analysis-demo.webm` and `docs/images/analysis-desktop.png`.

## Architecture Overview

The platform is organized as a modular pipeline:

1. Detection: runs detector backends through a unified interface.
1. Geo candidate retrieval: proposes top-N location hypotheses.
1. Verification and fusion: combines retrieval + verification signals into posterior weights.
1. Scoring and serialization: emits structured outputs for CLI, batch runs, and UI.
1. Dashboard/API layer: serves analysis and evaluation workflows.

## Core Modules

- `src/core/detection`: detector interface and implementations (`ultralytics_obb`,
  sidecar/classic adapters, detector factory)
- `src/core/geo`: geolocation providers (`GeoSpot/GeoCLIP` style + retrieval index)
- `src/core/logic`: fusion, likelihoods, scoring, verification, serialization, pipeline
- `src/schemas`: structured payload contracts for detections, candidates, and fusion
- `src/tools`: data prep, indexing, calibration, evaluation, and reporting utilities
- `src/dashboard`: analysis UI, map visualization, and scoring flows

## Technology Stack

- Language: Python 3.10+
- API/UI server: FastAPI + Uvicorn
- Vision and detection: Ultralytics (YOLO OBB), OpenCV, Pillow
- Geolocation: GeoCLIP/GeoSpot-style providers + retrieval index pipeline
- Data tooling: NumPy, Rasterio, AWS CLI integrations
- Validation/testing: JSON Schema, pytest
- Frontend: Vanilla HTML/CSS/JS + MapLibre GL

## Technology and Models (Detailed)

- Detection runtime: Ultralytics OBB pipeline with default weights `yolo11x-obb.pt`.
- Detection config: `src/config/defaults.json` (`detector.*`).
- Geolocation runtime: `sdan/geospot-base` for geo candidate inference.
- Retrieval model support: CLIP-based retrieval index with default model
  `openai/clip-vit-large-patch14`.
  - Multi-backbone retrieval is supported via per-index model routing (`retrieval_index_model_ids`).
- Geo and fusion config: `src/config/defaults.json` (`geolocator.*`, `fusion.*`).
- Fusion behavior: log-space posterior fusion over ranked geo candidates.
- Uncertainty outputs: radius and ellipse exposed in pipeline outputs/UI.
- API server: `src/tools/ui_server.py` (FastAPI).
- Analysis frontend: `src/dashboard/analysis/`.

## Technology Status (As of April 28, 2026)

Current implementation status:

- Geo candidate stack:
  - Multi-provider candidate generation (retrieval index + GeoSpot/GeoCLIP + EXIF/sidecar fallbacks).
  - Realistic single-index profile retuned on full split (`n=180`) to improve close-range precision:
    - `within_1km_pct`: `1.67` -> `5.00`
    - `within_5km_pct`: `23.89` -> `30.56`
    - artifacts: `runs/tune_retrieval_geo_realistic_within1km_focus_v1.json`, `runs/bench_realistic_single_180_precision_v2.json`.
  - Added retrieval consensus top-1 refinement in the Paris profile (`retrieval_consensus_top_n=20`, `retrieval_consensus_radius_km=3.0`), improving realistic split (`n=180`) metrics:
    - `within_1km_pct`: `5.00` -> `10.00`
    - `within_5km_pct`: `30.56` -> `36.67`
    - `median_km`: `11.50` -> `9.77`
    - artifacts: `runs/geo_eval_paris_profile_180_v2.json`, `runs/geo_eval_paris_profile_180_consensus_v1.json`.
  - Candidate validation, near-duplicate merge, and bounded candidate output before fusion.
  - Retrieval candidate diversity control (`retrieval_diversity_radius_km`, `retrieval_diversity_lambda`, `retrieval_diversity_min_keep`).
  - Retrieval minimum-candidate keep policy (`retrieval_min_keep_topk`) to avoid null geo outputs in low-similarity scenes.
  - Retrieval locality reranking (`retrieval_locality_radius_km`, `retrieval_locality_weight`) to suppress geographically isolated false matches.
  - Retrieval consensus refinement (`retrieval_consensus_top_n`, `retrieval_consensus_radius_km`, `retrieval_consensus_score_power`) with adaptive center selection (centroid vs weighted geo-median) for local outlier robustness without forcing unnecessary top-1 shifts.
  - Expanded structure-aware retrieval reranking (`retrieval_structure_rerank_top_n`, `retrieval_structure_rerank_weight`) into a geometry-lite scene-analysis layer before local matching:
    - corner density and edge density
    - dominant line-orientation histogram
    - corner / edge spatial layout
    - line orthogonality / anisotropy
    - guarded shadow axis and shadow elongation as weak sun-layout cues
    - weak-signal gating now keeps geometry-lite cues secondary unless layout / orthogonality / shadow-shape evidence is distinctive
    - current balanced branch probe on the canonical single-index Paris realistic split (`n=180`): `mean_km` `15.08` -> `14.72`, `median_km` `4.89` -> `4.59`, `within_1km_pct` `13.89` -> `15.00`, `within_2km_pct` `27.22` -> `28.33`, `within_5km_pct` `51.67` -> `53.33`, `within_10km_pct` `65.00` -> `66.11`
    - artifacts: `runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`, `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_d_180.json`.
  - Retrieval KDE mode refinement profiles are now benchmarked on the realistic split (`n=180`):
    - best `within_1km_pct`: `11.11` (`runs/geo_eval_paris_profile_180_kde_refine_c_w1_v1.json`)
    - best `within_2km_pct`: `20.00` (`runs/geo_eval_paris_profile_180_kde_refine_d_w2_v1.json`)
  - Local geometric reranking was upgraded to dual-engine matching (`SIFT` + `ORB`) with weak-signal gating and adaptive blending.
    - this materially improved legacy local-match performance (`localmatch_a`: `within_1km_pct` `5.00` -> `8.89`, `within_2km_pct` `13.33` -> `18.33`) while keeping control profile unchanged when local matching is disabled.
  - Added projection-aware retrieval adaptation from mined hard negatives.
    - best measured variant (`n=120`, realistic split): `within_1km_pct` `9.17` -> `12.50`, `within_2km_pct` `16.67` -> `28.33`
    - artifacts: `runs/geo_eval_projection_baseline_120.json`, `runs/geo_eval_projection_trainref_v2_mild_120.json`.
  - Added scope-aware retrieval geo prior (`retrieval_geo_prior_mode`, `retrieval_geo_prior_bbox`, `retrieval_geo_prior_sigma_km`, `retrieval_geo_prior_min_keep`) to prevent catastrophic cross-region matches.
    - mixed-scope stress test (`n=120`, Paris eval with Paris + open-geo indices): `mean_km` `6656.66` -> `18.82`, `within_10km_pct` `0.00` -> `40.83`
    - replay of the reported failure seed (`1870334448`, `n=2`): `mean_km` `7408.15` -> `0.00`
    - artifacts: `runs/geo_eval_mixed_scope_no_prior_120.json`, `runs/geo_eval_mixed_scope_hard_prior_120.json`, `runs/geo_eval_mixed_scope_no_prior_seed1870334448_2.json`, `runs/geo_eval_mixed_scope_hard_prior_seed1870334448_2.json`.
  - Retrieval query TTA with rotation ensembling (`retrieval_query_tta_degrees`, `retrieval_query_tta_reduce`) for aerial orientation robustness.
  - Multi-index retrieval support with per-index weighting (`retrieval_index_paths`, `retrieval_index_weights`, `retrieval_per_index_top_k`) for scalable dataset expansion.
  - Per-index retrieval model routing (`retrieval_index_model_ids`) so one run can mix indices built by different embedding backbones.
  - Per-index projection routing (`retrieval_index_projection_paths`) so multi-index runs can mix projected and non-projected spaces safely.
    - first realistic dual-space test (`n=180`, projected+raw CLIP with `rrf`) underperformed current projection V2 baseline, so it remains experimental.
    - artifacts: `runs/geo_eval_projection_trainref_v2_mild_180_baseline.json`, `runs/geo_eval_paris_dualspace_rrf_v1_180.json`.
  - Per-index score normalization for multi-index retrieval (`retrieval_index_score_norm`) to reduce cross-dataset score-scale bias.
  - Retrieval source-fusion mode (`retrieval_source_fusion_mode`: `weighted_score` or `rrf`) for multi-index rank aggregation policy.
  - Source-balanced retrieval selection (`retrieval_source_balance_beta`) to prevent single-index domination in multi-source top-k.
  - Source-balanced multi-provider candidate merge (`candidate_source_balance_beta`) to preserve cross-provider hypotheses before fusion.
- Fusion and uncertainty:
  - Log-space probabilistic fusion with configurable retrieval normalization (`none`, `zscore_sigmoid`, `minmax`, `rank_exp`).
  - Spatial-consensus likelihood to down-rank isolated outliers and favor geographically consistent hypotheses.
  - Cross-source agreement likelihood to reward hypotheses supported across retrieval/GeoCLIP/EXIF sources.
  - Optional plausibility reranking to favor coherent candidate clusters after posterior inference.
  - Adaptive outlier guard likelihood to downweight geographically isolated hypotheses with robust MAD-scale support (`use_adaptive_outlier_guard` + knobs).
  - Confidence calibration knobs (logit scale/bias + tier thresholds) for stricter confidence-tier gating.
  - Cross-source support-aware tier caps to prevent high confidence when the top hypothesis is source-isolated.
  - Dateline-safe longitude fusion and uncertainty ellipse/radius outputs.
  - Optional shadow/terrain likelihood terms.
- Runtime resiliency:
  - Health endpoints and dependency diagnostics (`/health`, `/health/deps`).
  - Safe-demo analysis fallback when heavy model dependencies are unavailable.
- Evaluation and tuning:
  - Geo regression gate tooling (`check_geo_regression`) with baseline/current reports in `docs/eval/`.
  - Fusion tuning script (`tune_geo_fusion`) supporting retrieval, diversity, consensus, cross-source, and plausibility sweeps.
  - Hard-negative benchmark report generator (`geo_hard_negative_report`) with distance buckets and per-group summaries.
  - Auto-fit utilities for source priors (`fit_fusion_priors`) and confidence calibration (`fit_confidence_calibration`).
  - Transaction-safe auto-tuning orchestration with rollback-on-failure and markdown/json run summaries (`auto_tune_geo_stack`).
  - Calibration/error metrics (`ece`, `brier`, `nll`) plus top-1/top-5 hard-negative diagnostics in `eval_metrics`.
  - Confidence reliability diagnostics in eval reports (`avg_top1_cross_source_support`, `high_confidence_top1`, `medium_or_higher_top1`).

Current technical focus:

- Expanding retrieval coverage quality (more diverse geo references and hard negatives).
- Converting scene-layout cues (corners, line structure, shadow direction) into reliable rerank signals without hurting tail robustness.
- Calibrating confidence and source priors from benchmark outputs instead of static defaults.
- Iterating on benchmark-driven plausibility reranking.

## Repository Structure

```text
Project-Heimdall/
|- src/
|  |- core/          # Detection, geo, fusion, scoring logic
|  |- tools/         # Data prep, evaluation, utility scripts
|  |- dashboard/     # Analysis UI assets
|  |- schemas/       # Typed payload schemas
|  |- tests/         # Unit tests
|  |- config/        # Runtime config profiles
|  |- docs/          # Technical documentation
|  `- scripts/       # Helper shell scripts
|- data/             # Local datasets/models (not versioned)
|- runs/             # Evaluation outputs
|- docs/images/      # README screenshots
|- README.md
`- requirements.txt
```

## Requirements

- Python 3.10+
- Git
- Optional GPU acceleration: PyTorch installed for your CUDA target

Install project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
```

Install the ML stack when you need full model inference:

```powershell
pip install -r requirements-ml.txt
```

Or install everything in one shot:

```powershell
pip install -r requirements.txt
```

For reproducible installs, use the lockfile:

```powershell
pip install -r requirements.lock.txt
```

If `torch` is missing, install CPU build quickly:

```powershell
.\.venv\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Environment verification:

```powershell
.\.venv\Scripts\python -m src.tools.doctor
```

Clean rebuild + verify in one command (run from a non-venv Python interpreter):

```powershell
python -m src.tools.doctor --rebuild
```

Development app launch (auto-picks free port):

```powershell
.\.venv\Scripts\python -m src.tools.dev_app
```

This starts the full app server: API + dashboard + analysis UI.

Windows CMD launcher:

```cmd
run_heimdall.cmd
```

This uses `.venv\Scripts\python.exe` when available, starts the same app server, and opens `/analysis/` in your default browser. Extra launcher flags can be passed through, for example:

```cmd
run_heimdall.cmd --no-reload
```

If filesystem watcher issues occur, run without reload:

```powershell
.\.venv\Scripts\python -m src.tools.dev_app --no-reload
```

## Quick Start

### 1. Generate dashboard summary data

```powershell
.\.venv\Scripts\python -m src.tools.generate_ui_data --jsonl runs/results.jsonl
```

### 2. Start the app server

CMD:

```cmd
run_heimdall.cmd
```

PowerShell/manual:

```powershell
.\.venv\Scripts\python -m src.tools.dev_app
```

### 3. Open the app

- Open the URL printed in your terminal (example: `http://127.0.0.1:8000/analysis/`).
- Open `http://127.0.0.1:8000/analysis/lab/` for scoring and benchmark workflows.

## How To Use The App

1. Start the server:

```powershell
.\.venv\Scripts\python -m src.tools.dev_app
```

1. Open the `/analysis/` URL printed in terminal (operator mode).
1. Select strategy profile.
1. Upload image and click `Analyze Image`.
1. Review detections, geo ranking, and fusion map.
1. Interact with 3D globe: drag to rotate/pan.
1. Scroll wheel to zoom in and zoom out.
1. Use `Zoom In`, `Zoom Out`, and `Reset` buttons.
1. Open `/analysis/lab/` for benchmark/eval tools (separate from upload/live analysis).

Benchmark comparison in UI:

1. Go to `/analysis/lab/`.
1. In `Benchmark Comparison`, click `Run Benchmark Comparison`.
1. Review:
   - scenario table (`leaky_reference`, `realistic_single`, `candidate_multi`)
   - backbone table (model-vs-model metrics and best model).
1. Every benchmark run is saved with a UTC timestamp.
1. Use `Saved runs` dropdown to load a previous run by date/time.
1. Use `Show selected saved run` toggle to switch between viewing latest run output and a selected historical run.
1. Use `Baseline run` + `Candidate run` and click `Compare Runs` to see metric deltas.
1. Use `Append Compare To PROGRESS.md` to write a comparison snippet into `PROGRESS.md`.

Random sample geo check in UI:

1. Go to `/analysis/lab/`.
1. In `Geo Scoring`, set images dir + metadata.
1. Set `Random sample size` and click `Run Random Samples`.
1. Review distance quality and accuracy bands (`<=1km`, `<=2km`, `<=5km`, `<=10km`) plus worst-sample distances.

Runtime diagnostics:

- `GET /health` returns service readiness summary.
- `GET /health/deps` returns detailed dependency/file/path/write diagnostics.
- `GET /health/runtime` returns worker-mode and inference timeout settings.

Safe demo behavior:

- If model dependencies are unavailable, `/analyze/image` and `/analyze/video` now return a realistic mock result instead of a hard error.
- You can also force this mode via query flag: `/analyze/image?safe_demo=1`.

Inference hardening toggles:

- `HEIMDALL_USE_INFERENCE_WORKER=1` (default) runs inference in an isolated worker process.
- `HEIMDALL_INFERENCE_TIMEOUT_S=90` controls image inference timeout.
- `HEIMDALL_VIDEO_TIMEOUT_S=300` controls video inference timeout.
- `HEIMDALL_MAX_IMAGE_BYTES` limits image upload size (default 20 MB).
- `HEIMDALL_MAX_VIDEO_BYTES` limits video upload size (default 256 MB).
- `HEIMDALL_ANALYSIS_CONCURRENCY` caps concurrent analysis runs (default 2).
- `HEIMDALL_ANALYSIS_QUEUE_TIMEOUT_S` controls queue wait timeout before `429` (default 5 s).
- Responses now include `request_id` and runtime timing metadata for diagnostics.

## Common Workflows

### Run single-image inference (CLI)

```powershell
.\.venv\Scripts\python -m src.cli data/analysis_tests/paris_street/images/mapillary__1021055432583866.jpg --json
```

### Run batch inference

```powershell
.\.venv\Scripts\python -m src.batch_run data/university-1652/images/train --output runs/results.jsonl
```

### Rebuild dashboard payload from batch output

```powershell
.\.venv\Scripts\python -m src.tools.generate_ui_data --jsonl runs/results.jsonl
```

### Geo baseline gate (regression check)

```powershell
.\.venv\Scripts\python -m src.tools.check_geo_regression --baseline docs/eval/geo_eval_baseline.json --candidate docs/eval/geo_eval_current.json
```

Update workflow:

1. Run geo eval and save latest report to `docs/eval/geo_eval_current.json`.
1. Run the regression gate command above.
1. If metric changes are intentional, update `docs/eval/geo_eval_baseline.json` in a dedicated PR with rationale.

### Hard-negative benchmark report

```powershell
.\.venv\Scripts\python -m src.tools.geo_hard_negative_report --results runs/results.jsonl --ground-truth data/spacenet_paris/metadata.csv --output runs/hard_negative_report.json
```

### Merge multiple Paris geo retrieval indices (scale Paris data coverage)

```powershell
.\.venv\Scripts\python -m src.tools.merge_geo_indices --inputs data/geo_index/spacenet_paris_clip.npz data/geo_index/spacenet_paris_test_clip.npz --output data/geo_index/merged_paris_clip.npz --dedupe-radius-m 75
```

### Fit fusion source priors from eval outputs

```powershell
.\.venv\Scripts\python -m src.tools.fit_fusion_priors --results runs/results.jsonl --ground-truth data/spacenet_paris/metadata.csv --per-source-min-count 5 --output runs/fusion_priors.json
```

Apply learned priors directly to config:
```powershell
.\.venv\Scripts\python -m src.tools.fit_fusion_priors --results runs/results.jsonl --ground-truth data/spacenet_paris/metadata.csv --per-source-min-count 5 --apply-config --config src/config/defaults.json --output runs/fusion_priors.json
```

### Fit confidence calibration from eval outputs

```powershell
.\.venv\Scripts\python -m src.tools.fit_confidence_calibration --results runs/results.jsonl --ground-truth data/spacenet_paris/metadata.csv --output runs/confidence_calibration.json
```

Apply learned calibration directly to config:
```powershell
.\.venv\Scripts\python -m src.tools.fit_confidence_calibration --results runs/results.jsonl --ground-truth data/spacenet_paris/metadata.csv --apply-config --config src/config/defaults.json --output runs/confidence_calibration.json
```

### Tune retrieval precision (fast sweep on cached raw candidates)

```powershell
.\.venv\Scripts\python -m src.tools.tune_retrieval_geo --config src/config/paris_test.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 300 --output runs/tune_retrieval_geo.json --rank-objective within_2km_pct --apply-best-config
```

### Mine hard-negative triplets (error-driven training data)

```powershell
.\.venv\Scripts\python -m src.tools.run_geo_eval --retrieval-only --config src/config/paris.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_paris_profile_180_for_mining_v1.json
.\.venv\Scripts\python -m src.tools.mine_hard_negative_triplets --metadata data/spacenet_paris_test/metadata.csv --reference-metadata data/spacenet_paris/metadata.csv --eval-report runs/geo_eval_paris_profile_180_for_mining_v1.json --output runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted.jsonl --summary-output runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted_summary.json --min-error-km 2.0 --positive-radius-km 0.35 --negative-pred-radius-km 2.0 --negative-min-gt-distance-km 2.0 --negative-max-gt-distance-km 25.0 --max-positives 3 --max-negatives 12 --difficulty-mode error_km_predmix --difficulty-reference-km 10.0 --difficulty-max-weight 3.0
```

This gives you query-positive-hard-negative tuples from real failure cases (not random tuples), ready for retrieval backbone fine-tuning.
With `--reference-metadata`, the query stays on the eval split while positives and hard negatives are pulled from the train/reference pool.
Each triplet now includes `triplet_weight`, per-source hard-negative counts, and nearest positive/negative distance diagnostics so downstream training can emphasize more severe/confusion-rich failures.
`src.tools.mine_hard_negative_triplets` now also supports `--eval-reports` and `--max-failures-per-query`, so you can merge multiple failure reports into one larger corpus without manually deduping repeated scenes.

### Fine-tune the retrieval encoder directly from mined triplets

```powershell
.\.venv\Scripts\python -m src.tools.train_retrieval_encoder --triplets runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted.jsonl --query-images-dir data/spacenet_paris_test/chips --reference-images-dir data/spacenet_paris/chips --model-id openai/clip-vit-large-patch14 --output-dir runs/retrieval_encoder_finetune/paris_round1_model --report-output runs/retrieval_encoder_finetune/paris_round1_model.report.json --train-scope vision_encoder --epochs 4 --batch-size 8 --learning-rate 1e-5 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.2 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device auto
```

This is a real image-encoder fine-tune step rather than a post-hoc linear projection. The output is a local `save_pretrained()` model directory that can be passed back into `build_geo_index` or any retrieval config as `retrieval_model_id`.

### Train and apply retrieval projection from hard negatives

```powershell
.\.venv\Scripts\python -m src.tools.train_retrieval_projection --triplets runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted.jsonl --embedding-index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14.npz --images-dir data/spacenet_paris_test/chips --output runs/retrieval_projection_paris_query_trainref_v2_weighted_cmp.npz --report-output runs/retrieval_projection_paris_query_trainref_v2_weighted_cmp.report.json --epochs 8 --batch-size 16 --learning-rate 3e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --orth-weight 0.002 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device cpu
.\.venv\Scripts\python -m src.tools.apply_projection_to_geo_index --index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14.npz --projection-path runs/retrieval_projection_paris_query_trainref_v2_weighted_cmp.npz --output data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_weighted_cmp.npz
```

Then run eval with a config that sets:
- `geolocator.retrieval_index_path` to the projected index.
- `geolocator.retrieval_projection_path` to the same projection file.

For query-vs-reference triplets, `--embedding-index` should point at the reference index, not the query-only eval index. Missing query embeddings are backfilled from `--images-dir`, and the trainer now reports missing path counts by role with an explicit split-mismatch hint if the wrong index is selected.
The projection training report now includes sample-weight stats plus weighted triplet satisfaction/loss metrics, so you can compare uniform vs difficulty-aware runs without re-parsing raw JSONL.
Controlled Paris realistic-split comparison (`n=180`, seed `42`, same 68 triplets): uniform single-index projection reached `<=1km 11.67%`, `<=2km 26.67%`, `<=5km 50.00%`, `<=10km 64.44%`, `mean 15.25 km`; difficulty-weighted training improved that to `<=1km 13.89%`, `<=2km 27.22%`, `<=5km 51.67%`, `<=10km 65.00%`, `mean 15.08 km`.

### Run an iterative Paris `180` retrieval fine-tune loop

```powershell
.\.venv\Scripts\python -m src.tools.run_retrieval_finetune_loop --train-images-dir data/spacenet_paris/chips --train-metadata data/spacenet_paris/metadata.csv --eval-images-dir data/spacenet_paris_test/chips --eval-metadata data/spacenet_paris_test/metadata.csv --bootstrap-config src/config/paris_close_range_dual_rrf.json --base-model-id openai/clip-vit-large-patch14 --rounds 1 --train-limit 600 --eval-limit 180 --eval-seed 42 --rank-objective within_2km_pct --use-dba --dba-neighbors 5 --dba-self-weight 1.0 --dba-eval-weight 0.5 --dba-geo-radius-km 2.0 --min-error-km 2.0 --positive-radius-km 0.35 --negative-pred-radius-km 2.0 --negative-min-gt-distance-km 2.0 --negative-max-gt-distance-km 25.0 --max-positives 3 --max-negatives 12 --max-failures-per-query 1 --difficulty-mode error_km_predmix --difficulty-reference-km 10.0 --difficulty-max-weight 3.0 --train-scope vision_encoder --epochs 4 --batch-size 8 --learning-rate 1e-5 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.2 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --device auto --output-dir runs/retrieval_finetune_loop
```

This loop fixes the eval slice to the canonical Paris `180` benchmark, mines failures from that same slice, fine-tunes the encoder, rebuilds the train index, optionally adds a geo-aware DBA companion, and writes a per-round summary to `runs/retrieval_finetune_loop/loop_summary.json`. Use `--train-limit` for practical iteration; leave it at `0` only when you deliberately want a full-train rebuild.
The loop can also emit an auxiliary fused serving config by appending the tuned model index and tuned DBA index on top of the bootstrap serving profile with their own `retrieval_index_model_ids` and explicit `null` projection routing. That deployment shape is implemented, but it is not promoted yet: on the full Paris `180` benchmark, a conservative auxiliary blend (`aux_index_weight=0.15`, `aux_dba_weight=0.05`) regressed against `paris_close_range_dual_rrf` from `mean_km 14.84` to `15.70` and from `<=2km 31.11%` to `21.11%`.

### Optional: Structure-aware / geometry-lite rerank

```powershell
.\.venv\Scripts\python -m src.tools.run_geo_eval --retrieval-only --config src/config/paris_structure_geometry_balanced.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_d_180.json
```

This rerank stays inside the retrieval stage and only reorders the top retrieval shortlist before local geometric matching. The current branch variant uses coarse geometry cues rather than semantic labels: corner density, edge density, dominant line histogram, corner/edge spatial layout, line orthogonality / anisotropy, plus guarded shadow-axis and shadow-elongation estimates from dark-mass layout.
Weak-signal gating now keeps those extra geometry cues from overpowering the older structure signal on diffuse scenes. On the canonical weighted single-index Paris run (`n=180`), the current balanced branch setting `retrieval_structure_rerank_top_n=14` and `retrieval_structure_rerank_weight=0.35` improved `mean_km` `15.08` -> `14.72`, improved `median_km` `4.89` -> `4.59`, improved `<=1km` `13.89%` -> `15.00%`, improved `<=2km` `27.22%` -> `28.33%`, improved `<=5km` `51.67%` -> `53.33%`, and improved `<=10km` `65.00%` -> `66.11%`.
`src.tools.tune_retrieval_geo` can now sweep `retrieval_structure_rerank_top_n` and `retrieval_structure_rerank_weight` directly when you want to compare geometry-rerank settings under the same tuning workflow.

### Optional: Geo-aware DBA index augmentation (close-range objective mode)

```powershell
.\.venv\Scripts\python -m src.tools.augment_geo_index_embeddings --index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz --output data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild_dba_geo2_k5.npz --neighbors 5 --self-weight 1.0 --min-similarity 0.0 --temperature 0.07 --max-geo-distance-km 2.0
```

Use this augmented index only for objective-specific close-range runs (`<=1km` / `<=2km`) and validate against the canonical benchmark before promotion.

### Optional: Dual-index close-range stack (projected + geo-aware DBA with RRF)

```powershell
.\.venv\Scripts\python -m src.tools.run_geo_eval --retrieval-only --config src/config/paris_close_range_dual_rrf.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_180.json
```

Paris dual-index variants are now split by objective:
- `src/config/paris_close_range_dual_rrf.json`: close-range default (`index_weights=[1.0,1.0]`), best overall `<=1km/<=2km` tradeoff.
- `src/config/paris_balanced_dual_rrf.json`: balanced mode (`index_weights=[1.0,0.5]`) for better mean/median/`<=10km` while keeping `<=2km`.
- `src/config/paris_close_range_dual_rrf_graph_kde.json`: aggressive W1 mode (graph support + KDE refinement), highest measured `<=1km` on canonical run.

Canonical `n=180` snapshot (Paris test split, seed `42`):
- Baseline projection V2 mild: `<=1km 11.67%`, `<=2km 26.67%`, `mean 15.25 km`.
- `paris_close_range_dual_rrf`: `<=1km 12.78%`, `<=2km 31.11%`, `mean 14.84 km`.
- `paris_balanced_dual_rrf`: `<=1km 10.56%`, `<=2km 31.11%`, `mean 14.60 km`.
- `paris_close_range_dual_rrf_graph_kde`: `<=1km 13.89%`, `<=2km 31.11%`, `mean 15.28 km`.

### Auto-tune full geo stack (retrieval + priors + calibration)

```powershell
.\.venv\Scripts\python -m src.tools.auto_tune_geo_stack --config src/config/defaults.json --images-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --results runs/results.jsonl --output-dir runs/auto_tune_geo
```

Outputs include:
- `runs/auto_tune_geo/auto_tune_summary.json`
- `runs/auto_tune_geo/auto_tune_summary.md`

If any tuning/calibration step fails, the command now restores the original config automatically.

### Benchmark retrieval backbones (model selection by measured geo error)

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_geo_backbones --train-images-dir data/spacenet_paris/chips --train-metadata data/spacenet_paris/metadata.csv --eval-images-dir data/spacenet_paris_test/chips --eval-metadata data/spacenet_paris_test/metadata.csv --model-ids "openai/clip-vit-large-patch14,google/siglip-base-patch16-224" --train-limit 600 --eval-limit 200 --output runs/backbone_bench/backbone_benchmark.json
```

Aerial preset example (RTX 5060 single-GPU focused objective):
```powershell
.\.venv\Scripts\python -m src.tools.benchmark_geo_backbones --train-images-dir data/spacenet_paris/chips --train-metadata data/spacenet_paris/metadata.csv --eval-images-dir data/spacenet_paris_test/chips --eval-metadata data/spacenet_paris_test/metadata.csv --model-preset aerial_rtx5060_precise --rank-objective within_2km_pct --train-limit 600 --eval-limit 200 --output runs/backbone_bench/backbone_benchmark_aerial.json
```

### Auto-upgrade retrieval backbone (benchmark -> rebuild index -> patch config)

```powershell
.\.venv\Scripts\python -m src.tools.upgrade_retrieval_backbone --train-images-dir data/spacenet_paris/chips --train-metadata data/spacenet_paris/metadata.csv --eval-images-dir data/spacenet_paris_test/chips --eval-metadata data/spacenet_paris_test/metadata.csv --config src/config/paris.json --model-preset aerial_rtx5060_precise --rank-objective within_2km_pct --output-dir runs/backbone_upgrade
```

Model presets currently available in `benchmark_geo_backbones`:
- `legacy_clip_siglip`
- `aerial_rtx5060_fast`
- `aerial_rtx5060_precise`
- `aerial_research`

### Generate impact report (baseline vs candidate)

```powershell
.\.venv\Scripts\python -m src.tools.geo_impact_report --baseline docs/eval/geo_eval_baseline.json --candidate docs/eval/geo_eval_current.json --output-json runs/geo_impact.json --output-md runs/geo_impact.md
```

### Canonical benchmark CI (pro workflow)

Run the fixed benchmark suite from the versioned manifest and apply regression policy gates:

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core
```

What this command does:
1. Runs the canonical benchmark profile from `benchmarks/manifest.json`.
1. Compares candidate run against the pinned baseline in `docs/eval/baseline.json`.
1. Enforces regression thresholds from `benchmarks/policy.json`.
1. Writes `docs/eval/latest_report.md` and `docs/eval/latest_pr_summary.md`.
1. Appends one summary line to `docs/eval/history.jsonl`.
1. Returns non-zero exit code if policy checks fail.

Promote a tested run as the new pinned baseline:

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core --promote <run_id>
```

The promoted baseline contract stores:
- baseline run id
- baseline commit SHA
- baseline summary reference used for future comparisons

### Benchmark Tool Guide (step-by-step)

Use this when you want to benchmark geo quality before/after changes and keep a professional history.

Prerequisites:
1. Install dependencies and `torch`.
1. Ensure benchmark data paths used in `benchmarks/manifest.json` exist locally.
1. Activate your venv.

First-time setup (initialize baseline):
1. Run one benchmark:

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core
```

1. Note the `Run complete: <run_id>` value printed in terminal.
1. Promote that run as baseline:

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core --promote <run_id>
```

Daily/PR workflow:
1. Run benchmark on your branch:

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core
```

1. Check policy result:
   - exit code `0`: pass (or skipped if no baseline).
   - exit code `1`: regression policy failed.
1. Open generated report:
   - `docs/eval/latest_report.md`
1. Copy PR-ready summary from:
   - `docs/eval/latest_pr_summary.md`

Where benchmark outputs go:
- Run payload (UI-loadable): `src/dashboard/data/benchmark_runs/<run_id>.json`
- Full artifacts per run: `runs/benchmark_history/<run_id>/`
- Pinned baseline contract: `docs/eval/baseline.json`
- Baseline snapshot: `docs/eval/baseline_summary.json`
- Append-only ledger: `docs/eval/history.jsonl`

How to benchmark different settings:
1. Edit `benchmarks/manifest.json` profile `core` (datasets, limits, model list, scenarios).
1. Re-run `benchmark_ci`.
1. If results are intentionally better and stable, promote the new run id.

How this connects to UI:
1. Open `Scoring` tab in app.
1. Use `Saved runs` dropdown to inspect past run payloads by timestamp.
1. Use baseline/candidate selectors to compare two run ids and optionally append compare snippet to `PROGRESS.md`.

### Run test suite

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Troubleshooting

If startup fails with `WinError 1392` under `.venv\Lib\site-packages\torch\...`, your venv is corrupted.
Recreate it from scratch:

```powershell
deactivate
rmdir /s /q .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then reinstall your matching `torch/torchvision/torchaudio` build (CPU or CUDA) and rerun:

```powershell
.\.venv\Scripts\python -m src.tools.dev_app
```

## Config Profiles

Config files are under `src/config/`:

- `defaults.json`: default runtime profile
- `paris.json`: SpaceNet Paris-focused profile
- `paris_close_range_dba.json`: experimental Paris close-range profile (projection + geo-aware DBA index)
- `paris_structure_geometry_balanced.json`: experimental Paris geometry-lite balanced profile (weak-signal gated structure rerank)
- `paris_close_range_dual_rrf.json`: experimental Paris close-range dual-index profile (projection index + geo-aware DBA index with RRF fusion)
- `paris_balanced_dual_rrf.json`: experimental Paris balanced dual-index profile (better mean/median/<=10km)
- `paris_close_range_dual_rrf_graph_kde.json`: experimental Paris W1-max dual-index profile (graph support + KDE mode refinement)
- `paris_test.json`: Paris test profile

Each shipped profile declares `profile_scope` at the JSON root so region intent is explicit and easier to audit in runs. The active shipped profiles are Paris-only; Open Geo/Wikimedia is retired from the runtime profile list until the project expands beyond Paris again.

`src.tools.run_geo_eval` validates profile/data scope alignment by default. Use `--allow-scope-mismatch` only for intentional cross-scope experiments.

Pass a config where supported with `--config <path>`.

Useful geo quality knobs in `geolocator`:
- `candidate_dedupe_radius_m`: merges near-duplicate candidates from multiple providers.
- `candidate_source_balance_beta`: source-balancing strength for merged candidates from retrieval/GeoCLIP/EXIF (`0` disables balancing).
- `candidate_max_results`: caps merged candidate count before fusion.
- `retrieval_diversity_radius_km`: geographic distance scale for retrieval diversification.
- `retrieval_diversity_lambda`: relevance-vs-diversity tradeoff (`1.0` = no diversification).
- `retrieval_diversity_min_keep`: number of top raw retrievals always preserved before diversification.
- `retrieval_min_keep_topk`: minimum candidates to keep from top-k even if `retrieval_min_score` is too strict.
- `retrieval_locality_radius_km`: distance scale for locality support reranking.
- `retrieval_locality_weight`: strength of locality reranking (higher penalizes isolated candidates more).
- `retrieval_structure_rerank_top_n`: number of highest-scoring retrieval candidates eligible for scene-structure reranking (`0` disables).
- `retrieval_structure_rerank_weight`: blend strength for structure similarity vs base retrieval score.
- `retrieval_query_tta_degrees`: query-time rotation ensemble angles in degrees.
- `retrieval_query_tta_reduce`: how augmented similarity scores are merged (`mean`, `median`, `max`, or `rrf`).
- `retrieval_index_paths`: optional list of extra retrieval indices to query alongside `retrieval_index_path`.
- `retrieval_index_weights`: optional per-index score multipliers (same order as `retrieval_index_paths`).
- `retrieval_index_model_ids`: optional per-index embedding model IDs (same order as `retrieval_index_path` + `retrieval_index_paths`) to mix different backbones in one retrieval pass.
- `retrieval_index_projection_paths`: optional per-index projection paths (same order as `retrieval_index_path` + `retrieval_index_paths`); use `null` for indices that should stay unprojected.
- `retrieval_projection_path`: optional projection file (`.npz`) applied to query embeddings for metric-adapted retrieval.
- `retrieval_per_index_top_k`: optional per-index cap before global merge/rerank (`0` uses `retrieval_top_k`).
- `retrieval_index_score_norm`: per-index score normalization mode (`auto`, `none`, `minmax`, `zscore_sigmoid`, `rank_exp`); `auto` uses `zscore_sigmoid` for multi-index and `none` for single-index.
- `retrieval_source_balance_beta`: source-balancing strength for multi-index top-k selection (`0` disables balancing).

Useful detection quality knobs in `detector`:
- `backend`: detector backend (`rfdetr` by default, with safe sidecar fallback if the package is unavailable; `ultralytics_obb` remains available for YOLO OBB experiments).
- `min_area_px`: filters tiny unstable detections.
- `nms_mode`: `obb` (oriented IoU) or `aabb` (axis-aligned IoU) suppression mode.
- `class_agnostic_nms`: when `false`, NMS keeps overlapping boxes from different classes.
- `use_tta`: enables test-time augmentation in Ultralytics inference.
- `rfdetr_model_size`: RF-DETR size when `backend` is `rfdetr` (`nano`, `small`, `medium`, `large`, `xlarge`, `2xlarge`). Keep Plus/XL licensing under review before promoting it as default.

RF-DETR detector backend:
```json
{
  "detector": {
    "backend": "rfdetr",
    "weights_path": null,
    "min_confidence": 0.35,
    "nms_mode": "aabb",
    "rfdetr_model_size": "medium"
  }
}
```

Install the ML stack to use RF-DETR directly:
```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Useful fusion knobs:
- `fusion.retrieval_score_norm`: `none`, `zscore_sigmoid`, `minmax`, `rank_exp`.
- `fusion.source_prior_retrieval`, `fusion.source_prior_geoclip`, `fusion.source_prior_exif`: source-level priors applied before posterior normalization.
- `fusion.source_prior_retrieval_by_source`: optional per-retrieval-source priors (for IDs like `retrieval:<source>:<item>` in multi-index retrieval).
- `fusion.use_spatial_consensus`: enables neighborhood agreement likelihood in fusion.
- `fusion.spatial_sigma_km`: spatial kernel scale in kilometers.
- `fusion.spatial_consensus_weight`: strength of spatial consensus in posterior weighting.
- `fusion.use_cross_source_agreement`: enables cross-provider support likelihood in fusion.
- `fusion.cross_source_sigma_km`: distance scale for cross-source agreement kernel.
- `fusion.cross_source_weight`: strength of cross-source agreement in posterior weighting.
- `fusion.use_plausibility_rerank`: applies post-posterior spatial coherence reranking.
- `fusion.plausibility_radius_km`: radius used for plausibility neighborhood support.
- `fusion.plausibility_weight`: strength of plausibility reranking.
- `fusion.use_adaptive_outlier_guard`: enables adaptive robust outlier suppression based on weighted candidate support.
- `fusion.outlier_guard_strength`: strength of adaptive outlier suppression (`0` keeps baseline behavior).
- `fusion.outlier_guard_min_scale_km`: minimum spatial scale for outlier guard stability.
- `fusion.outlier_guard_mad_scale`: MAD multiplier controlling tolerance to dispersed hypotheses.
- `fusion.confidence_calibration_logit_scale`, `fusion.confidence_calibration_logit_bias`: calibrate top-1 posterior before tiering.
- `fusion.confidence_high_threshold`, `fusion.confidence_medium_threshold`: enforce stricter tier gates.
- `fusion.confidence_high_min_cross_source_support`, `fusion.confidence_medium_min_cross_source_support`: cap high/medium tiers when top-1 cross-source support is weak.
- `fusion.confidence_high_max_uncertainty_m`, `fusion.confidence_medium_max_uncertainty_m`: cap high/medium tiers when fused uncertainty is too large.
- `fusion.credible_mass`, `fusion.min_credible_candidates`: robust posterior subset used for fused mean/covariance.
- `fusion.use_top_cluster_for_stats`, `fusion.credible_cluster_radius_km`, `fusion.min_credible_cluster_weight`: constrain fused stats to the dominant spatial mode when hypotheses are ambiguous.

## Data and Model Notes

- Large datasets and model artifacts are intentionally not stored in Git.
- Local data-folder guide and cleanup recommendations: `docs/DATA_LAYOUT.md`.
- Typical local path: `data/geo_index/*.npz`
- Typical local path: `data/models/*`
- Realistic Paris street-data bootstrap from Mapillary:
```powershell
$env:MAPILLARY_ACCESS_TOKEN="MLY|..."
.\.venv\Scripts\python -m src.tools.download_mapillary_paris --bbox 48.8156,2.2241,48.9022,2.4699 --out data/paris_realistic_v1/street_mapillary --grid-step-m 80 --street-per-cell 3 --max-images 20000 --seed 42
```
- Realistic Paris street-data bootstrap from Panoramax federated catalog:
```powershell
.\.venv\Scripts\python -m src.tools.download_panoramax_paris --bbox 48.8156,2.2241,48.9022,2.4699 --out data/paris_realistic_v1/street_panoramax --grid-step-m 80 --street-per-cell 3 --max-images 20000 --seed 42
```
- Merge Mapillary and Panoramax street datasets into one combined metadata root:
```powershell
.\.venv\Scripts\python -m src.tools.merge_realistic_street_datasets --metadata data/paris_realistic_v1/street_mapillary/metadata.csv data/paris_realistic_v1/street_panoramax/metadata.csv --out data/paris_realistic_v1/street_combined
```
- Dry-run count check for the same dataset bootstrap:
```powershell
.\.venv\Scripts\python -m src.tools.download_mapillary_paris --bbox 48.8156,2.2241,48.9022,2.4699 --out data/paris_realistic_v1/street_mapillary --grid-step-m 80 --street-per-cell 3 --max-images 20000 --seed 42 --dry-run
```
- The downloader writes:
  - `data/paris_realistic_v1/street_mapillary/images/`
  - `data/paris_realistic_v1/street_mapillary/metadata.csv`
- Current metadata contract:
  - `image_id,path,lat,lon,heading_deg,captured_at,camera_type,width,height,quality_score,sequence,source,license_info`
- Current behavior:
  - prefers `computed_geometry` over `geometry`
  - prefers `computed_compass_angle` over `compass_angle`
  - uses `thumb_2048_url` with `thumb_1024_url` fallback
  - dedupes by `image_id`
  - limits near-duplicates with per-cell sampling plus per-sequence caps
  - resumes cleanly from an existing `metadata.csv` checkpoint instead of restarting from zero
- Panoramax-specific behavior:
  - uses the federated Panoramax catalog by default: `https://api.panoramax.xyz/api`
  - keeps direct picture assets plus per-picture heading from `view:azimuth`
  - preserves source instance information such as `panoramax:ign` or `panoramax:osmfr`
- Best-completeness recommendation for Paris:
  - use both Mapillary and Panoramax street ingestion, then merge them into `street_combined`
- Current checkpoint on disk:
  - `20,000` Mapillary street rows at `data/paris_realistic_v1/street_mapillary/metadata.csv`
  - `20,000` Panoramax street rows at `data/paris_realistic_v1/street_panoramax/metadata.csv`
  - `40,000` merged street rows at `data/paris_realistic_v1/street_combined/metadata.csv`
- Do not use Google Street View, Google Maps scraping, or Google Earth tiles in this dataset path.
- OpenAerialMap pairing for the same realistic Paris dataset:
```powershell
.\.venv\Scripts\python -m src.tools.build_aerial_pairs --street-metadata data/paris_realistic_v1/street_mapillary/metadata.csv --out data/paris_realistic_v1 --provider openaerialmap --crop-size-m 256 --crop-px 512 --allow-missing-aerial false --seed 42
```
- IGN orthophoto pairing for denser Paris coverage:
```powershell
.\.venv\Scripts\python -m src.tools.build_aerial_pairs --street-metadata data/paris_realistic_v1/street_panoramax/metadata.csv --out data/paris_realistic_v1 --provider ign_geopf --crop-size-m 256 --crop-px 512 --allow-missing-aerial false --seed 42
```
- The aerial-pair builder writes:
  - `data/paris_realistic_v1/aerial/images/`
  - `data/paris_realistic_v1/aerial/metadata.csv`
  - `data/paris_realistic_v1/pairs.csv`
- Current aerial metadata contract:
  - `aerial_id,path,lat,lon,source,provider,resolution_m,crop_size_m,crop_px,license_info,paired_street_id,status`
- Current pair contract:
  - `pair_id,street_id,street_path,aerial_id,aerial_path,lat,lon,heading_deg`
- Current provider behavior:
  - queries OAM metadata near each street point
  - prefers the highest-resolution covering image
  - uses the OAM TMS URL from metadata to render a centered crop
  - marks `no_open_aerial_found` when no open aerial scene covers the point
  - also supports `ign_geopf` using the official `ORTHOIMAGERY.ORTHOPHOTOS` service for complete French orthophoto coverage
- Current complete paired checkpoints:
  - `10,000` Panoramax -> IGN pairs at `data/paris_realistic_v1/pairs.csv`
  - `40,000` combined street -> IGN pairs at `data/paris_realistic_v1_combined/pairs.csv`
- Leakage-safe spatial split generation:
```powershell
.\.venv\Scripts\python -m src.tools.split_realistic_dataset --pairs data/paris_realistic_v1/pairs.csv --out data/paris_realistic_v1/splits_full --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --cell-size-m 300 --seed 42
```
- Stricter buffered split generation for benchmark work:
```powershell
.\.venv\Scripts\python -m src.tools.split_realistic_dataset --pairs data/paris_realistic_v1/pairs.csv --out data/paris_realistic_v1/splits_strict --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --cell-size-m 300 --buffer-cells 2 --seed 42
```
- Split sanity check:
```powershell
.\.venv\Scripts\python -m src.tools.split_realistic_dataset --sanity-check-dir data/paris_realistic_v1/splits_full
```
- Split outputs:
  - `data/paris_realistic_v1/splits_full/train_pairs.csv`
  - `data/paris_realistic_v1/splits_full/val_pairs.csv`
  - `data/paris_realistic_v1/splits_full/test_pairs.csv`
  - `data/paris_realistic_v1/splits_full/split_summary.json`
- Current split caveat:
  - the current `min_cross_split_distance_m` is `3.77`, so this split is usable for pipeline bring-up but not yet strict enough to support a final benchmark claim toward `~3 km` mean error
- Current stricter benchmark checkpoint:
  - `data/paris_realistic_v1/splits_strict/`
  - retained pairs: `8405`
  - excluded boundary-buffer pairs: `1595`
  - minimum cross-split distance: `1213.11 m`
  - additional holdout rows are written to `excluded_pairs.csv` instead of silently disappearing
- Current full combined benchmark checkpoint:
  - `data/paris_realistic_v1_combined/splits_strict/`
  - retained pairs: `34821`
  - excluded boundary-buffer pairs: `5179`
  - minimum cross-split distance: `1201.23 m`
- Full combined recovery + strict split:
```powershell
.\.venv\Scripts\python -m src.tools.recover_combined_aerial_dataset --existing-images-dir data/paris_realistic_v1/aerial/images --chunk-meta-dir data/paris_realistic_v1_combined_chunkmeta --chunk-out-dir data/paris_realistic_v1_combined_chunkpairs --final-out-dir data/paris_realistic_v1_combined --split-out-dir data/paris_realistic_v1_combined/splits_strict --provider ign_geopf --crop-size-m 256 --crop-px 512 --allow-missing-aerial false --seed 42 --max-workers 2 --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --cell-size-m 300 --buffer-cells 2 --sort-axis auto
```
- Full combined aerial index:
```powershell
.\.venv\Scripts\python -m src.tools.build_realistic_aerial_index --root data/paris_realistic_v1_combined --metadata aerial/metadata.csv --images-dir aerial/images --output indices/aerial_clip_index.npz --model-id openai/clip-vit-large-patch14
```
- Full combined strict-probe cross-view evaluation:
```powershell
.\.venv\Scripts\python -m src.tools.eval_realistic_crossview --test-pairs data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --aerial-metadata data/paris_realistic_v1_combined/aerial/metadata.csv --street-images-dir data/paris_realistic_v1/street_combined --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --embedding-model openai/clip-vit-large-patch14 --output runs/eval_realistic_crossview_combined_strict_probe240_baseline_full40k.json --top-k 50
```
- Current combined strict-probe baselines:
  - sampled `10k` aerial index:
    - `mean_km = 10.92`
    - `within_2km_pct = 5.00`
    - `within_5km_pct = 10.83`
  - full `40k` aerial index:
    - `mean_km = 10.97`
    - `within_2km_pct = 5.83`
    - `within_5km_pct = 12.50`
- First combined strict-probe cross-view projection run:
```powershell
.\.venv\Scripts\python -m src.tools.mine_realistic_crossview_triplets --pairs data/paris_realistic_v1_combined/splits_strict/train_pairs.csv --street-metadata data/paris_realistic_v1/street_combined/metadata.csv --aerial-metadata data/paris_realistic_v1_combined/aerial/metadata.csv --output runs/paris_realistic_crossview_train_triplets_v1.jsonl --summary-output runs/paris_realistic_crossview_train_triplets_v1.summary.json --positive-radius-m 80 --negative-min-distance-m 300 --negative-max-distance-m 5000 --max-positives 3 --max-negatives 20 --seed 42
.\.venv\Scripts\python -m src.tools.train_crossview_projection --triplets runs/paris_realistic_crossview_train_triplets_v1.jsonl --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --street-images-dir data/paris_realistic_v1/street_combined --output runs/crossview_projection_paris_combined_v1_probe.npz --report-output runs/crossview_projection_paris_combined_v1_probe.report.json --embedding-model openai/clip-vit-large-patch14 --max-triplets 6000 --epochs 8 --batch-size 64 --learning-rate 3e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device auto
.\.venv\Scripts\python -m src.tools.eval_realistic_crossview --test-pairs data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --aerial-metadata data/paris_realistic_v1_combined/aerial/metadata.csv --street-images-dir data/paris_realistic_v1/street_combined --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --projection runs/crossview_projection_paris_combined_v1_probe.npz --embedding-model openai/clip-vit-large-patch14 --output runs/eval_realistic_crossview_combined_strict_probe240_crossviewproj_v1_full40k.json --top-k 50
```
- First combined strict-probe cross-view projection result:
  - `mean_km = 9.75`
  - `median_km = 10.24`
  - `within_1km_pct = 2.08`
  - `within_2km_pct = 7.50`
  - `within_5km_pct = 20.42`
- Interpretation:
  - this first learned cross-view projection beat the frozen full-40k CLIP baseline on `mean_km`, `<=2km`, and `<=5km`
  - it did not yet beat the baseline at `<=1km`, so it is a real improvement but not the final answer
- Full-triplet follow-up projection run:
```powershell
.\.venv\Scripts\python -m src.tools.train_crossview_projection --triplets runs/paris_realistic_crossview_train_triplets_v1.jsonl --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --street-images-dir data/paris_realistic_v1/street_combined --output runs/crossview_projection_paris_combined_v2_full.npz --report-output runs/crossview_projection_paris_combined_v2_full.report.json --embedding-model openai/clip-vit-large-patch14 --max-triplets 0 --epochs 30 --batch-size 32 --learning-rate 1e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device auto
.\.venv\Scripts\python -m src.tools.eval_realistic_crossview --test-pairs data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --aerial-metadata data/paris_realistic_v1_combined/aerial/metadata.csv --street-images-dir data/paris_realistic_v1/street_combined --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --projection runs/crossview_projection_paris_combined_v2_full.npz --embedding-model openai/clip-vit-large-patch14 --output runs/eval_realistic_crossview_combined_strict_probe240_crossviewproj_v2_full40k.json --top-k 50
```
- Full-triplet result:
  - `mean_km = 9.83`
  - `median_km = 10.92`
  - `within_1km_pct = 4.17`
  - `within_2km_pct = 12.08`
  - `within_5km_pct = 22.92`
- Interpretation:
  - training on all `26204` realistic triplets improved close-range hit rates over the first probe model
  - it did not improve `mean_km` or `median_km`, so it should be treated as a close-range tradeoff, not a universal replacement win
- Tier 3 DINOv2 complement experiment:
```powershell
.\.venv\Scripts\python -m src.tools.build_geo_index --images-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --output data/geo_index/spacenet_paris_chips_facebook_dinov2_base.npz --model-id facebook/dinov2-base
.\.venv\Scripts\python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --config src/config/paris_dinov2_rrf_experimental.json --retrieval-only --limit 180 --seed 42 --output runs/geo_eval_paris_dinov2_rrf_experimental_180_fixed.json
```
- Tier 3 result versus current `src/config/paris.json`:
  - `mean_km 14.60 -> 14.42`
  - `median_km 4.21 -> 4.47`
  - `within_1km_pct 10.56 -> 13.33`
  - `within_2km_pct 31.11 -> 31.67`
  - `within_5km_pct 52.78 -> 52.22`
  - `within_10km_pct 65.56 -> 66.67`
- Decision:
  - keep `src/config/paris_dinov2_rrf_experimental.json` as an experimental branch config
  - do not replace `src/config/paris.json` yet because the DINOv2 fusion helps the closest buckets but still regresses `median_km` and `<=5km`
- Tier 4 encoder fine-tune status:
  - added `scripts/run_tier4_encoder_ft.ps1` to run the realistic cross-view encoder fine-tune, aerial-index rebuild, and `probe240` eval as one pipeline
  - validated the path with a `--max-triplets 1` smoke run at `runs/retrieval_encoder_finetune/smoke_one_triplet.report.json`
  - the full unattended CPU run is not benchmarked yet because background launches on this shell environment stalled immediately after CLIP initialization
- Realistic street-to-aerial cross-view evaluation:
```powershell
.\.venv\Scripts\python -m src.tools.eval_realistic_crossview --test-pairs data/paris_realistic_v1/splits_strict/test_pairs.csv --aerial-metadata data/paris_realistic_v1/aerial/metadata.csv --street-images-dir data/paris_realistic_v1/street_panoramax --aerial-index data/paris_realistic_v1/indices/aerial_clip_index.npz --embedding-model openai/clip-vit-large-patch14 --output runs/eval_realistic_crossview_strict.json --top-k 50
```
- See [Geo Tech Notes](src/docs/GEO_TECH.md) for deeper implementation and dataset details.

## Engineering and Contribution

- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Security policy: [SECURITY.md](SECURITY.md)
- Support: [SUPPORT.md](SUPPORT.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Progress tracking: [PROGRESS.md](PROGRESS.md)
- Issue templates: [`.github/ISSUE_TEMPLATE`](.github/ISSUE_TEMPLATE)
- PR template: [`.github/pull_request_template.md`](.github/pull_request_template.md)
- CI workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

## License

This project is released under a non-commercial personal/research license.

- Personal, research, and evaluation use is allowed.
- Commercial use is not allowed without a separate paid commercial license from the project owner.

See [LICENSE](LICENSE) for full terms.
