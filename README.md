# Project Heimdall

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#requirements)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-0A0A0A)](#requirements)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Geospatial perception and analysis platform that combines object detection, geo-candidate retrieval, probabilistic fusion, and operator-facing explainability.

## Project Status Tracking

- Engineering progress log: [PROGRESS.md](PROGRESS.md)
- What it is: append-only record of shipped changes, validation runs, and technical milestones.
- How to use it: check the top snapshot for current status, then review dated entries for full history.
- Benchmark governance: [docs/eval/latest_report.md](docs/eval/latest_report.md), [docs/eval/history.jsonl](docs/eval/history.jsonl), [docs/eval/baseline.json](docs/eval/baseline.json)

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

## Technology Status (As of April 5, 2026)

Current implementation status:

- Geo candidate stack:
  - Multi-provider candidate generation (retrieval index + GeoSpot/GeoCLIP + EXIF/sidecar fallbacks).
  - Candidate validation, near-duplicate merge, and bounded candidate output before fusion.
  - Retrieval candidate diversity control (`retrieval_diversity_radius_km`, `retrieval_diversity_lambda`, `retrieval_diversity_min_keep`).
  - Retrieval minimum-candidate keep policy (`retrieval_min_keep_topk`) to avoid null geo outputs in low-similarity scenes.
  - Retrieval locality reranking (`retrieval_locality_radius_km`, `retrieval_locality_weight`) to suppress geographically isolated false matches.
  - Retrieval query TTA with rotation ensembling (`retrieval_query_tta_degrees`, `retrieval_query_tta_reduce`) for aerial orientation robustness.
  - Multi-index retrieval support with per-index weighting (`retrieval_index_paths`, `retrieval_index_weights`, `retrieval_per_index_top_k`) for scalable dataset expansion.
  - Per-index retrieval model routing (`retrieval_index_model_ids`) so one run can mix indices built by different embedding backbones.
  - Per-index score normalization for multi-index retrieval (`retrieval_index_score_norm`) to reduce cross-dataset score-scale bias.
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

```powershell
.\.venv\Scripts\python -m src.tools.dev_app
```

### 3. Open the app

- Open the URL printed in your terminal (example: `http://127.0.0.1:8000/analysis/`).

## How To Use The App

1. Start the server:

```powershell
.\.venv\Scripts\python -m src.tools.dev_app
```

1. Open the `/analysis/` URL printed in terminal.
1. Select strategy profile.
1. Upload image and click `Analyze Image`.
1. Review detections, geo ranking, and fusion map.
1. Interact with 3D globe: drag to rotate/pan.
1. Scroll wheel to zoom in and zoom out.
1. Use `Zoom In`, `Zoom Out`, and `Reset` buttons.
1. Open the `Scoring` tab for benchmark/eval tools (separate from upload/live analysis).

Benchmark comparison in UI:

1. Go to `Scoring` tab.
1. In `Benchmark Comparison`, click `Run Benchmark Comparison`.
1. Review:
   - scenario table (`leaky_reference`, `realistic_single`, `candidate_multi`)
   - backbone table (model-vs-model metrics and best model).
1. Every benchmark run is saved with a UTC timestamp.
1. Use `Saved runs` dropdown to load a previous run by date/time.
1. Use `Show selected saved run` toggle to switch between viewing latest run output and a selected historical run.
1. Use `Baseline run` + `Candidate run` and click `Compare Runs` to see metric deltas.
1. Use `Append Compare To PROGRESS.md` to write a comparison snippet into `PROGRESS.md`.

Runtime diagnostics:

- `GET /health` returns service readiness summary.
- `GET /health/deps` returns detailed dependency/file/path/write diagnostics.

Safe demo behavior:

- If model dependencies are unavailable, `/analyze/image` and `/analyze/video` now return a realistic mock result instead of a hard error.
- You can also force this mode via query flag: `/analyze/image?safe_demo=1`.

## Common Workflows

### Run single-image inference (CLI)

```powershell
.\.venv\Scripts\python -m src.cli data/samples/real_port_miami.jpg --json
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

### Merge multiple geo retrieval indices (scale data coverage)

```powershell
.\.venv\Scripts\python -m src.tools.merge_geo_indices --inputs data/geo_index/open_geo_clip.npz data/geo_index/spacenet_paris_clip.npz --output data/geo_index/merged_clip.npz --dedupe-radius-m 75
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
.\.venv\Scripts\python -m src.tools.tune_retrieval_geo --config src/config/paris_test.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 300 --output runs/tune_retrieval_geo.json --apply-best-config
```

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
- `paris_test.json`: Paris test profile
- `open_geo.json`: lightweight open-geo profile

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
- `retrieval_query_tta_degrees`: query-time rotation ensemble angles in degrees.
- `retrieval_query_tta_reduce`: how augmented similarity scores are merged (`mean`, `max`, or `rrf`).
- `retrieval_index_paths`: optional list of extra retrieval indices to query alongside `retrieval_index_path`.
- `retrieval_index_weights`: optional per-index score multipliers (same order as `retrieval_index_paths`).
- `retrieval_index_model_ids`: optional per-index embedding model IDs (same order as `retrieval_index_path` + `retrieval_index_paths`) to mix different backbones in one retrieval pass.
- `retrieval_per_index_top_k`: optional per-index cap before global merge/rerank (`0` uses `retrieval_top_k`).
- `retrieval_index_score_norm`: per-index score normalization mode (`auto`, `none`, `minmax`, `zscore_sigmoid`, `rank_exp`); `auto` uses `zscore_sigmoid` for multi-index and `none` for single-index.
- `retrieval_source_balance_beta`: source-balancing strength for multi-index top-k selection (`0` disables balancing).

Useful detection quality knobs in `detector`:
- `min_area_px`: filters tiny unstable detections.
- `nms_mode`: `obb` (oriented IoU) or `aabb` (axis-aligned IoU) suppression mode.
- `class_agnostic_nms`: when `false`, NMS keeps overlapping boxes from different classes.
- `use_tta`: enables test-time augmentation in Ultralytics inference.

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
- Typical local path: `data/geo_index/*.npz`
- Typical local path: `data/models/*`
- Typical local path: `data/dota_v1/*`
- Open geo bootstrap (Wikimedia API, broader coverage):
```powershell
.\.venv\Scripts\python -m src.tools.download_open_geo --mode api --limit 300 --per-anchor 25 --output data/open_geo
.\.venv\Scripts\python -m src.tools.build_geo_index --images-dir data/open_geo/images --metadata data/open_geo/metadata.csv --output data/geo_index/open_geo_clip.npz
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

This project is released under the MIT License. See [LICENSE](LICENSE).
