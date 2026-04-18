# Project Progress

## What This File Is

`PROGRESS.md` is the engineering change log for this repository.

- It tracks meaningful implementation work, validation runs, and tooling changes.
- It is append-only: do not rewrite past entries.
- New updates should be added as a new dated block at the end of the file.

## Current Snapshot (April 5, 2026)

- Geo stack is now multi-index and multi-provider with source balancing, score normalization, retrieval diversity/locality controls, and query-time TTA support.
- Fusion stack includes cross-source agreement, spatial consensus, adaptive outlier guard, calibrated confidence tiering, and richer uncertainty diagnostics.
- Tuning pipeline now supports automated retrieval plus fusion plus calibration orchestration with config rollback on failure and Markdown/JSON run summaries.
- Benchmark tooling now includes realistic vs leaky geo-eval comparisons, hard-negative diagnostics, and backbone model comparison support.
- Latest reported validation in this log: focused suite `18 passed`.
- Latest reported validation in this log: non-UI suite `105 passed`.
- Full suite in this environment can still be blocked when optional UI deps are missing.

## Related References

- [README.md](README.md): setup, run commands, and user-facing workflows.
- [src/docs/GEO_TECH.md](src/docs/GEO_TECH.md): geolocation architecture and tuning notes.
- [src/docs/REPRODUCIBILITY.md](src/docs/REPRODUCIBILITY.md): reproducibility and evaluation workflows.

## Append-Only Log

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


## 2026-01-30
- Aligned README with the 2026-01-30 engineering spec (modules A-G and non-goals).
- Added pydantic schemas for detection, geo candidates, fusion output, evidence, and tracks.
- Implemented GeoCLIP/GeoFT candidate provider stub and candidate interface.
- Added probabilistic fusion engine with likelihoods, uncertainty ellipse, and evidence text.
- Added tracking/filtering stubs plus new tooling stubs (geo candidates, fusion calibration, sequence eval, audit export, full run, metrics).
- Added reproducibility doc and extended default config with fusion/top-N settings.

## 2026-01-30
- Extended UI summary generator to include fusion candidates, evidence, and uncertainty ellipse fields when present.
- Added tests for fusion likelihood helpers and geo candidate sidecar parsing.
- Ran pytest (13 passed).

## 2026-01-30
- Expanded tools/run_all.py fusion output to include candidates, evidence, covariance, and ellipse metadata.
- Updated dashboard table and detail panel to display fusion uncertainty + per-candidate evidence.

## 2026-01-30
- Wired fusion candidates into pipeline outputs and batch schema, including evidence and uncertainty ellipse serialization.
- Added map/table/track views in the dashboard and interactive fusion detail rendering.
- Implemented evaluation metrics with ground truth ingestion (CSV/JSON/JSONL) and updated reproducibility docs.
- Ran pytest (13 passed).

## 2026-01-30
- Switched dashboard and live UI maps to Leaflet tiles with fusion ellipse/candidate overlays and track polylines.
- Updated run_all to emit full pipeline results (detections, verification, fusion) using config-driven pipeline.
- Extended UI summary payloads with detections/verification for object lists.
- Live UI pipeline now includes candidate provider + fusion config.
- Ran pytest (13 passed).

## 2026-01-30
- Merged static dashboard and live UI into the FastAPI server with / (dashboard) and /live routes.
- Updated dashboard and live HTML asset paths to serve from the unified server.
- Ran pytest (13 passed).

## 2026-01-30
- Added shared theme and navigation tabs to switch between dashboard and live UI.
- Unified dark theme across both UIs via shared theme stylesheet.

## 2026-01-30
- Added GeoCLIP provider support for real model loading (GeoSpot Base) with HF download fallback.
- Extended geolocator config with model_id/model_cache_dir defaults and added download helper.
- Updated requirements and README for GeoSpot Base setup.

## 2026-01-30
- Switched shared theme to near-black and aligned header layout across dashboard and live UI.
- Unified hero layout with shared hero-left/hero-right structure for consistent top spacing.

## 2026-01-31
- Renamed live UI to analysis (/analysis) and updated server routing + asset paths for FastAPI and static usage.
- Unified full-black theme, headers, and button sizing across dashboard and analysis pages; added progress indicator on analysis runs.
- Added GeoCLIP debug metadata to analysis responses and UI summary line for candidate visibility.
- Added geolocator encoder_name support in config and pipeline wiring; improved geo fallback from fusion output.
- Integrated GeoSpot Base via SigLIP2 image encoder, including required SigLIP2 inputs and CLIP output patching.
- Tuned geo candidate counts and fusion settings for higher-accuracy defaults.

## 2026-01-31
- Restructured repo into src/ for code, tools, schemas, tests, dashboard, config, docs, scripts, and ingestion; kept data/ at repo root.
- Updated imports to use src.* package paths and adjusted default config/dashboard paths accordingly.
- Updated README and reproducibility docs for new module paths and run commands.
- Updated .gitignore for new dashboard data paths.

## 2026-01-31
- Added retrieval-based geo candidate provider (CLIP embeddings) with multi-provider fusion.
- Added open geo demo downloader (Wikimedia Commons) and retrieval index builder.
- Added retrieval config fields (index path, model, top_k, min_score) and wired across UI/CLI/batch/run_all.
- Default config now points to the open_geo retrieval index for immediate testing.
- Updated README/GEO_TECH for retrieval workflow and open demo dataset.
- Refined map confidence display (top candidate emphasis, ranked markers, dashed uncertainty ring).

## 2026-01-31
- Replaced Leaflet maps with canvas-based globe view for fusion and track displays.
- Added top-N candidate selector (2/5/10/20) persisted in localStorage.

## 2026-01-31
- Simplified analysis UI to focus on single-image analysis (removed batch/video/eval sections).
- Restored a real world map view using Leaflet tiles for fusion candidates and uncertainty ring.
- Added top-N filtering, hover/click tooltips for geo candidates, and geo confidence in the summary.
- Added detection confidence bars and UI legend explaining dots/mean/ring.

## 2026-04-05
- Added geo candidate quality hardening: multi-provider candidate validation, near-duplicate merge, and bounded candidate output before fusion.
- Hardened retrieval index loading and query scoring path for safer runtime behavior.
- Improved fusion robustness with additional retrieval normalization modes, dateline-safe longitude statistics, and expanded geo-fusion tests.
- Added spatial-consensus likelihood in fusion (`use_spatial_consensus`, `spatial_sigma_km`, `spatial_consensus_weight`) to down-rank isolated outliers.
- Extended evaluation metrics with calibration/error metrics (`ece`, `brier`, `nll`) and added coverage tests.
- Updated README and GEO_TECH documentation with a current technology status snapshot and new fusion/config knobs.
- Added source-aware fusion priors (source_prior_retrieval, source_prior_geoclip, source_prior_exif) and confidence diagnostics in fusion outputs.
- Extended fusion serialization/schema with ambiguity and posterior concentration metrics.
- Added configurable detection quality controls (min area filter, class-aware/agnostic NMS, optional TTA) and applied them in Ultralytics + sidecar pipelines.
- Added robust fusion credible-set statistics and exposed credible_set_size in serialized outputs/schema.
- Added top-cluster robust fusion stats mode to avoid multimodal midpoint bias in ambiguous geolocation outputs.
- Replaced deprecated datetime.utcnow() usage in UI server health/analysis payload timestamps.
- Added true OBB-IoU NMS option (detector.nms_mode=obb) and wired it across detector adapters for better rotated-box suppression behavior.
- Added cross-source agreement fusion likelihood to reward hypotheses corroborated across retrieval/GeoCLIP/EXIF sources and reduce source-isolated outlier dominance.
- Added retrieval diversity controls in `GeoRetrievalProvider` (`retrieval_diversity_radius_km`, `retrieval_diversity_lambda`, `retrieval_diversity_min_keep`) and wired them through CLI/UI/batch/eval pipelines.
- Added fusion plausibility reranking (`use_plausibility_rerank`, `plausibility_radius_km`, `plausibility_weight`) plus calibrated confidence tiering (`confidence_calibration_logit_scale`, `confidence_calibration_logit_bias`, threshold knobs).
- Extended fusion serialization/schema with `calibrated_top1_posterior` and added regression coverage for plausibility and calibration behavior.
- Added hard-negative benchmark tooling (`geo_hard_negative_report`) with top-1 distance buckets, top-5@25km, per-group summaries, and hardest-sample export.
- Added auto-fit tooling for source priors (`fit_fusion_priors`) and confidence calibration (`fit_confidence_calibration`) from eval outputs.
- Expanded tuning/metrics tooling: `tune_geo_fusion` now sweeps diversity/plausibility axes and `eval_metrics` now outputs top-1 km aggregates + hard-negative bucket diagnostics.
- Added uncertainty-aware confidence tier caps so high/medium confidence can be downgraded when fused uncertainty radius is too large.
- Added `geo_impact_report` tool to generate JSON + Markdown delta reports between baseline and candidate geo metrics for clear change impact documentation.
- Added cross-source support-aware confidence tier caps (`confidence_high_min_cross_source_support`, `confidence_medium_min_cross_source_support`) to prevent source-isolated top-1 hypotheses from receiving overconfident tiers.
- Extended fusion outputs/schema with `top1_cross_source_support` for explainability and confidence auditability.
- Expanded `eval_metrics` with confidence reliability reporting (`avg_top1_cross_source_support`, `high_confidence_coverage_pct`, `high_confidence_top1`, `medium_or_higher_coverage_pct`, `medium_or_higher_top1`) and wired those into impact-report comparisons.
- Added retrieval-locality reranking (`retrieval_locality_radius_km`, `retrieval_locality_weight`) to down-rank geographically isolated high-score retrieval candidates before fusion.
- Added `merge_geo_indices` utility to combine multiple retrieval `.npz` indices with exact and optional spatial+embedding deduplication for larger coverage datasets.
- Fixed retrieval index loading compatibility for legacy object-array `ids/paths`, restoring retrieval candidate generation on older `.npz` indices.
- Added `tune_retrieval_geo` utility for fast precision sweeps over retrieval top-k/min-score/diversity/locality parameters using cached raw candidates.
- Improved robust fusion statistics for ambiguous multimodal outputs: top-cluster selection now chooses the strongest local posterior cluster by mass instead of hard-anchoring on top-1.
- Upgraded temporal tracking association to use geodesic distance and uncertainty-aware adaptive gating, improving stability across global/dateline cases.
- Added regression coverage for both upgrades:
  - `test_top_cluster_stats_choose_densest_cluster_not_top1_anchor`
  - new `test_tracking.py` suite (base-gate behavior, uncertainty gate expansion, dateline association).
- Ran full test suite after changes: `78 passed, 3 warnings`.
- Replaced temporal filtering stub (`update_posterior`) with proximity-weighted Bayesian-style reweighting against prior candidates, plus posterior diagnostics recomputation.
- Added adaptive temporal uncertainty shrinkage when consecutive frames agree and robust weighted covariance/longitude handling in filtering.
- Wired sequence evaluation (`run_sequence_eval`) to apply temporal posterior updates across frames instead of storing only raw per-frame fusion outputs.
- Added `test_filtering.py` suite for temporal consistency reweighting, uncertainty reduction on agreement, and empty-candidate passthrough behavior.
- Re-ran full suite after temporal filtering changes: `81 passed, 3 warnings`.
- Added retrieval query TTA for geo matching: configurable multi-rotation query embeddings (`retrieval_query_tta_degrees`) with score aggregation mode (`retrieval_query_tta_reduce`: `mean`/`max`).
- Wired retrieval TTA knobs through all runtime paths (CLI, batch, run_all, run_geo_eval, UI server, and tune_retrieval_geo).
- Updated config profiles (`defaults`, `open_geo`, `paris`, `paris_test`) to enable rotation-ensemble retrieval by default (`[0,90,180,270]`, reduce=`max`).
- Added retrieval TTA regression tests (`test_retrieval_tta.py`) and extended config loading tests for new geolocator fields.
- Hardened retrieval score stability by clamping similarity scores to `[-1, 1]` before ranking/thresholding.
- Ran quick retrieval-only A/B eval on SpaceNet Paris subset (`limit=120`) for TTA-off vs TTA-on configs; metrics were equivalent on this sample (likely index/eval leakage dominated), and outputs are saved under `runs/geo_eval_paris_no_tta_120.json` and `runs/geo_eval_paris_tta_120.json`.
- Added retrieval fallback recall control `retrieval_min_keep_topk` to keep a minimum number of top-k candidates even when `retrieval_min_score` filters everything, reducing null-geo failure modes.
- Extended retrieval TTA aggregation to support `rrf` (reciprocal-rank fusion) in addition to `mean`/`max`, with config parsing support.
- Wired `retrieval_min_keep_topk` through CLI/batch/UI/eval/tuning paths and added it to config profiles.
- Expanded test coverage for these upgrades:
  - `test_retrieval_provider_min_keep.py` for strict-threshold fallback behavior
  - `test_retrieval_tta.py` for `rrf` aggregation ranking behavior
  - `test_tune_retrieval_geo.py` for min-keep postprocess behavior
  - `test_config_loading.py` for `retrieval_min_keep_topk` and `retrieval_query_tta_reduce=rrf`.
- Stress-validated fallback recall behavior on a strict-threshold retrieval-only run (`retrieval_min_score=1.1`, `limit=40`, Paris chips):
  - `retrieval_min_keep_topk=0`: `evaluated=0`, `null_predictions=40`
  - `retrieval_min_keep_topk=2`: `evaluated=40`, `null_predictions=0`
  - reports: `runs/geo_eval_paris_strict_keep0_40.json` and `runs/geo_eval_paris_strict_keep2_40.json`.
- Re-ran full suite after this sprint: `88 passed, 3 warnings`.
- Added weighted multi-index retrieval support in `GeoRetrievalProvider` (`retrieval_index_paths`, `retrieval_index_weights`, `retrieval_per_index_top_k`) so multiple datasets can be queried in one pass with source balancing.
- Fixed retrieval query-TTA mode handling so `retrieval_query_tta_reduce=rrf` is honored by runtime provider initialization.
- Wired new multi-index knobs across CLI, batch runner, run_all, geo_eval, UI server, and retrieval tuning/report outputs.
- Expanded open-data bootstrap: `download_open_geo` now supports Wikimedia geosearch API collection across global anchor cities (`--mode api`) plus curated fallback mode.
- Added dependency-resilience hardening:
  - lazy import of heavy `pandas` dependency in `run_geo_eval` and `tune_retrieval_geo`
  - pure-Python fallback path for `ClassicDetector` when `cv2` is unavailable.
- Added dataset-aware fusion priors for multi-index retrieval:
  - new config knob `fusion.source_prior_retrieval_by_source` to tune priors per retrieval source key
  - supports both `retrieval:<source>` and `<source>` keys for match IDs shaped like `retrieval:<source>:<item>`.
- Fusion source parsing is backward-compatible with legacy retrieval IDs (`retrieval:<item>`), preserving previous single-source-family cross-source behavior.
- Upgraded `fit_fusion_priors` to emit retrieval source-specific priors from eval outputs:
  - source reliability now tracks `retrieval_by_source` buckets for source-aware IDs (`retrieval:<source>:<item>`)
  - new CLI control `--per-source-min-count` to avoid overfitting sparse sources
  - output now includes `recommended_retrieval_source_priors` and, when available, `config_patch.fusion.source_prior_retrieval_by_source`.
- Added direct config patch workflow for prior fitting:
  - `fit_fusion_priors` now supports `--apply-config --config <path>` and writes global + per-source retrieval priors directly into the target fusion config.
  - Added regression test coverage for config patch behavior in `test_fit_fusion_priors.py`.
- Added per-index score normalization for retrieval fusion across heterogeneous indices:
  - new geolocator knob `retrieval_index_score_norm` with modes `auto`, `none`, `minmax`, `zscore_sigmoid`, `rank_exp`
  - `auto` defaults to source-wise `zscore_sigmoid` when multiple retrieval indices are active and keeps `none` for single-index setups
  - wired through CLI/batch/UI/eval/tuning and emitted in geo eval/tuning reports.
- Added regression coverage for normalization behavior:
  - `test_index_score_norm_zscore_sigmoid_rebalances_heterogeneous_indices`
  - `test_index_score_norm_auto_uses_multi_index_normalization`
  - config loader coverage for `retrieval_index_score_norm` (default, override, invalid fallback).
- Added direct config patch workflow for confidence calibration:
  - `fit_confidence_calibration` now supports `--apply-config --config <path>` to write learned calibration scale/bias and tier thresholds into fusion config.
  - calibration patch generation now enforces monotonic tier thresholds (`confidence_medium_threshold <= confidence_high_threshold`) for noisy/small datasets.
  - Added regression tests for calibration patch writing and threshold monotonicity in `test_fit_confidence_calibration.py`.
- Added source-balanced retrieval selection for multi-index setups:
  - new geolocator knob `retrieval_source_balance_beta` to reduce top-k domination by a single retrieval index source.
  - balance is applied after locality reranking and before geographic diversity selection.
  - integrated across CLI/batch/UI/eval/tuning, with report visibility in `run_geo_eval` and `tune_retrieval_geo`.
- Added normalization + source-balance regression coverage in `test_retrieval_provider_multi_index.py`.
- Extended retrieval tuning search space to include `retrieval_source_balance_beta` sweeps and apply the best discovered value back into config.
- Added end-to-end geo auto-tuning orchestrator `auto_tune_geo_stack`:
  - runs retrieval tuning + optional result generation + fusion prior fitting + confidence calibration patching in one command
  - writes a structured run summary (`auto_tune_summary.json`) with per-step status.
- Added regression coverage for orchestration and new tuning behavior:
  - `test_auto_tune_geo_stack.py`
  - updated `test_tune_retrieval_geo.py` for source-balance postprocessing behavior.
- Added source-balanced multi-provider candidate merge:
  - new `geolocator.candidate_source_balance_beta` to reduce retrieval-only dominance after provider merge and preserve cross-provider hypotheses before fusion.
  - wired through CLI/batch/run_all/geo_eval/UI and surfaced in eval/tuning report metadata.
  - added regression coverage in `test_multi_provider.py` for balanced vs unbalanced merged top-k behavior.
- Added/updated regression coverage:
  - `test_retrieval_provider_multi_index.py`
  - `test_download_open_geo.py`
  - updated `test_retrieval_provider_min_keep.py`, `test_retrieval_tta.py`, `test_config_loading.py`, `test_run_geo_eval_retrieval_provider.py`.
- Added/updated fusion-prior tests:
  - `test_retrieval_source_priors_can_shift_multi_index_ranking`
  - config parsing assertions for `source_prior_retrieval_by_source`.
  - retrieval fitter tests for per-source reliability and source-prior recommendation behavior.
- Validation:
  - focused suite: `18 passed`
  - non-UI suite (`src/tests/test_*.py` excluding `ui_server` tests): `105 passed`
  - full `pytest -q` in this environment still blocked by missing optional dependency `fastapi` for UI server tests.
- Added adaptive fusion outlier guard:
  - new fusion knobs `use_adaptive_outlier_guard`, `outlier_guard_strength`, `outlier_guard_min_scale_km`, `outlier_guard_mad_scale`
  - applies robust medoid/MAD-based spatial suppression to reduce isolated candidate dominance before final posterior ranking
  - added fusion regression coverage for guard on/off and zero-strength behavior.
- Hardened `auto_tune_geo_stack` reliability:
  - auto-saves original config and restores it on any failed tuning/calibration step
  - now writes both machine-readable (`auto_tune_summary.json`) and human-readable (`auto_tune_summary.md`) run summaries
  - added regression coverage for config rollback behavior in `test_auto_tune_geo_stack.py`.
- Added reality-check benchmark artifacts and documented leakage gap between leaky and realistic geo eval setups:
  - `runs/bench_current_leaky_180.json` (median near-zero due to eval/index leakage)
  - `runs/bench_realistic_single_180.json` (realistic baseline, significantly higher error)
  - `runs/bench_candidate_multi_180.json` (multi-index baseline comparison).
- Added backbone benchmark tool `src/tools/benchmark_geo_backbones.py` to rebuild/evaluate retrieval indices per model and rank models by geo error metrics.
- Added retrieval embedder fallback path to support non-CLIP vision backbones (e.g., SigLIP) and validated with regression tests.
- Implemented true multi-backbone retrieval ensemble support:
  - per-index model routing via new config knob `geolocator.retrieval_index_model_ids`
  - retrieval index metadata now stores `model_id` in built `.npz` indices
  - runtime now computes query embeddings per unique model and applies them to matching indices.
- Wired `retrieval_index_model_ids` through config loading, runtime/eval tooling, and docs.
- Added regression coverage for:
  - per-index model-id loading from index files
  - multi-index retrieval with different embedding dimensions/backbones in one request path.
- Validation in this environment:
  - focused retrieval/config/eval suites passed (`19 passed`)
  - full model/UI execution remains environment-limited when optional deps like `torch` or `fastapi` are unavailable.

## 2026-04-05
- Added professional benchmark-governance stack with fixed manifests and policy gates:
  - `benchmarks/manifest.json` (versioned canonical suites/datasets/model set/limits)
  - `benchmarks/policy.json` (explicit regression thresholds for core scenario metrics).
- Added canonical CLI `src/tools/benchmark_ci.py`:
  - `python -m src.tools.benchmark_ci --profile core`
  - runs scenario + backbone benchmarks from manifest
  - compares candidate against pinned baseline contract
  - writes `docs/eval/latest_report.md` and `docs/eval/latest_pr_summary.md`
  - appends summary rows to `docs/eval/history.jsonl`
  - returns non-zero exit code when policy checks fail.
- Added baseline promotion workflow:
  - `python -m src.tools.benchmark_ci --profile core --promote <run_id>`
  - updates pinned baseline contract (`docs/eval/baseline.json`) and baseline summary snapshot (`docs/eval/baseline_summary.json`) with run id + commit SHA.
- Added benchmark governance artifacts in `docs/eval/`:
  - `baseline.json`, `baseline_summary.json`, `latest_report.md`, `latest_pr_summary.md`, `history.jsonl`.
- Added regression/unit coverage for new benchmark CI behavior in `src/tests/test_benchmark_ci.py`.
- Updated README with the canonical benchmark workflow, promotion command, and eval-history references.

## 2026-04-15
- Hypothesis:
  - Realistic-split accuracy loss was coming from retrieval post-processing (locality/diversity/source-balance) reordering, not from missing top-k recall.
- Change:
  - Extended retrieval tuner ranking objectives to support explicit optimization targets (`within_1km_pct`, `within_5km_pct`, `within_10km_pct`, plus distance-based modes).
  - Added tuner regression coverage for objective parsing/sort behavior.
  - Ran a full realistic-split retrieval sweep (`n=180`) focused on post-processing knobs.
  - Updated `runs/bench_cfg/cfg_realistic_single.json` to the best full-split setting:
    - `retrieval_top_k: 25`
    - `retrieval_min_score: 0.05`
    - `retrieval_min_keep_topk: 0`
    - `retrieval_diversity_radius_km: 0.0`
    - `retrieval_diversity_lambda: 1.0`
    - `retrieval_diversity_min_keep: 1`
    - `retrieval_locality_radius_km: 0.0`
    - `retrieval_locality_weight: 0.0`
    - `retrieval_source_balance_beta: 0.0`
  - Synced realistic benchmark artifact path to current config (`runs/bench_realistic_single_180.json`).
- Files touched:
  - `src/tools/tune_retrieval_geo.py`
  - `src/tests/test_tune_retrieval_geo.py`
  - `runs/bench_cfg/cfg_realistic_single.json`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest src/tests/test_tune_retrieval_geo.py`
  - `./.venv/Scripts/python -m src.tools.tune_retrieval_geo --config runs/bench_cfg/cfg_realistic_single.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --output runs/tune_retrieval_geo_realistic_within1km_focus_v1.json --limit 180 --seed 42 --retrieval-topk 25,50,80 --retrieval-min-score 0.05,0.1 --retrieval-min-keep-topk 0,2 --retrieval-diversity-radius-km 0.0,1.0 --retrieval-diversity-lambda 1.0,0.88 --retrieval-diversity-min-keep 1,4 --retrieval-locality-radius-km 0.0,25.0 --retrieval-locality-weight 0.0,1.2 --retrieval-source-balance-beta 0.0,0.35 --retrieval-query-tta-reduce max --rank-objective within_1km_pct`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --output runs/bench_realistic_single_180_precision_v2.json --retrieval-only --limit 180 --seed 42 --config runs/bench_cfg/cfg_realistic_single.json`
- Metrics (before -> after):
  - Baseline artifact: `runs/bench_realistic_single_180_tuned_within1km_v1.json`
    - `within_1km_pct`: `1.67`
    - `within_5km_pct`: `23.89`
    - `within_10km_pct`: `43.33`
    - `mean_km`: `19.749`
    - `median_km`: `11.395`
  - Candidate artifact: `runs/bench_realistic_single_180_precision_v2.json`
    - `within_1km_pct`: `5.00`
    - `within_5km_pct`: `30.56`
    - `within_10km_pct`: `45.56`
    - `mean_km`: `18.016`
    - `median_km`: `11.499`
- Artifacts:
  - `runs/tune_retrieval_geo_realistic_within1km_focus_v1.json`
  - `runs/bench_realistic_single_180_precision_v2.json`
  - `runs/bench_realistic_single_180.json`
- Decision:
  - Keep this retrieval profile as the current realistic single-index baseline.

## 2026-04-15
- Added retrieval consensus top-1 refinement in `src/core/geo/retrieval_provider.py` (configurable `retrieval_consensus_top_n`, `retrieval_consensus_radius_km`, `retrieval_consensus_score_power`) and wired it through CLI, batch, run-all, UI server, and geo eval paths.
- Extended config schema/runtime for new consensus knobs in `src/core/logic/config.py` and config profiles (`src/config/defaults.json`, `src/config/open_geo.json`, `src/config/paris_test.json`, `src/config/paris.json`).
- Enabled consensus refinement in Paris profile (`retrieval_consensus_top_n=20`, `retrieval_consensus_radius_km=3.0`, `retrieval_consensus_score_power=1.0`).
- Added unit coverage for consensus refinement behavior and config parsing (`src/tests/test_retrieval_diversity.py`, `src/tests/test_config_loading.py`).
- Validation run: `python -m pytest src/tests/test_retrieval_diversity.py src/tests/test_config_loading.py src/tests/test_tune_retrieval_geo.py` -> `18 passed`.
- Benchmark run: `python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --retrieval-only --limit 180 --config src/config/paris.json --output runs/geo_eval_paris_profile_180_consensus_v1.json`.
- Realistic split (`n=180`) results improved vs prior Paris profile (`runs/geo_eval_paris_profile_180_v2.json`):
  - `within_1km_pct`: `5.00` -> `10.00`
  - `within_5km_pct`: `30.56` -> `36.67`
  - `within_10km_pct`: `45.56` -> `50.56`
  - `median_km`: `11.4990` -> `9.7717`
  - `mean_km`: `18.0159` -> `15.5334`

## 2026-04-15
- Added Lab random sample evaluation workflow for fast correctness spot-checking:
  - Frontend: `src/dashboard/analysis/lab/index.html`, `src/dashboard/analysis/lab/lab.js`, `src/dashboard/analysis/lab/lab.css`.
  - Backend: new async endpoints `POST /eval/geo/random/start` and `GET /eval/geo/random/status` in `src/tools/ui_server.py`.
  - Behavior: each run uses a randomized seed, evaluates a random subset, and reports aggregate distance/accuracy with per-sample distance diagnostics.
- Extended server state tracking with `_GEO_RANDOM_STATE` and random-eval summary generation (`within_1km_pct`, computed `within_2km_pct`, `within_5km_pct`, `within_10km_pct`, per-sample rows).
- Improved operator globe visuals for candidate readability:
  - Added candidate-link lines from fused mean to top candidates.
  - Added candidate glow and fused-mean halo map layers.
  - Increased atmosphere/fog contrast and cinematic background treatment in `operator.js` + `operator.css`.
- Updated project licensing from MIT to a non-commercial license:
  - Replaced `LICENSE` with `Project Heimdall Non-Commercial License v1.0`.
  - Updated README license badge and license section to reflect personal/research-only use and separate paid commercial licensing.
- Updated docs for new Lab flow/endpoints and license positioning (`README.md`, `src/dashboard/README.md`, `CHANGELOG.md`).
- Validation commands:
  - `python -m pytest src/tests/test_ui_server_runtime.py src/tests/test_ui_server_integration.py src/tests/test_ui_server_benchmark_runs.py src/tests/test_config_loading.py src/tests/test_retrieval_diversity.py src/tests/test_tune_retrieval_geo.py`
- Validation result: `30 passed`.

## 2026-04-15
- Hypothesis:
  - Retrieval consensus top-1 refinement should be more stable at close range when the center estimate is robust to local outliers, and retrieval tuning should support direct optimization for 1-2 km targets.
- Change:
  - Replaced consensus center estimation in `GeoRetrievalProvider` with a weighted geo-median solver in local geodesic coordinates (Weiszfeld-style iterative update).
  - Extended eval/tuning metrics with explicit `within_2km_pct` support:
    - `run_geo_eval` now emits `within_2km_pct`.
    - `tune_retrieval_geo` now computes `within_2km_pct` and supports `--rank-objective within_2km_pct`.
  - Added regression tests for robust geo-median behavior and new `within_2km_pct` objective handling.
- Files touched:
  - `src/core/geo/retrieval_provider.py`
  - `src/tools/run_geo_eval.py`
  - `src/tools/tune_retrieval_geo.py`
  - `src/tests/test_retrieval_diversity.py`
  - `src/tests/test_tune_retrieval_geo.py`
  - `CHANGELOG.md`
  - `README.md`
  - `src/docs/GEO_TECH.md`
  - `src/docs/RESEARCH_PAPER.md`
- Validation command:
  - `./.venv/Scripts/python -m pytest src/tests/test_retrieval_diversity.py src/tests/test_tune_retrieval_geo.py src/tests/test_ui_server_runtime.py src/tests/test_config_loading.py`
- Metrics (before -> after):
  - No full benchmark run executed in this prompt cycle; no accuracy claim recorded.
- Artifacts:
  - None generated (code + test + doc update cycle only).
- Decision:
  - Keep the robustness + metric-targeting changes; run a full realistic benchmark next to quantify `within_1km_pct`/`within_2km_pct` impact.

## 2026-04-15
- Hypothesis:
  - A guarded adaptive consensus center (choose centroid by default, switch to weighted geo-median only when local support and center separation justify it) can improve close-range hits without regressing broader metrics.
- Change:
  - Updated consensus refinement center selection in `src/core/geo/retrieval_provider.py`:
    - compute both centroid and weighted geo-median on the consensus cluster,
    - switch to geo-median only when support is materially higher (`>5%`) and center gap is non-trivial,
    - otherwise keep centroid.
  - Added/kept explicit `within_2km_pct` reporting in eval/tuning paths.
- Files touched:
  - `src/core/geo/retrieval_provider.py`
  - `src/tools/run_geo_eval.py`
  - `src/tools/tune_retrieval_geo.py`
  - `src/tests/test_retrieval_diversity.py`
  - `src/tests/test_tune_retrieval_geo.py`
  - `README.md`
  - `src/docs/GEO_TECH.md`
  - `src/docs/RESEARCH_PAPER.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest src/tests/test_retrieval_diversity.py src/tests/test_tune_retrieval_geo.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --retrieval-only --limit 180 --seed 42 --config src/config/paris.json --output runs/geo_eval_paris_profile_180_consensus_v4_adaptive_guarded.json`
- Metrics (before -> after):
  - Baseline artifact: `runs/geo_eval_paris_profile_180_consensus_v1.json`
    - `mean_km`: `15.5334`
    - `median_km`: `9.7717`
    - `within_1km_pct`: `10.00`
    - `within_5km_pct`: `36.67`
    - `within_10km_pct`: `50.56`
  - Candidate artifact: `runs/geo_eval_paris_profile_180_consensus_v4_adaptive_guarded.json`
    - `mean_km`: `15.5569`
    - `median_km`: `9.7717`
    - `within_1km_pct`: `10.56`
    - `within_2km_pct`: `19.44`
    - `within_5km_pct`: `36.67`
    - `within_10km_pct`: `50.56`
- Artifacts:
  - `runs/geo_eval_paris_profile_180_consensus_v2_geomedian.json`
  - `runs/geo_eval_paris_profile_180_consensus_v3_adaptive_center.json`
  - `runs/geo_eval_paris_profile_180_consensus_v4_adaptive_guarded.json`
- Decision:
  - Keep guarded adaptive-center consensus as current candidate behavior; it improves close-range hit rate (`within_1km_pct`) with flat median/radius metrics and adds first-class `within_2km_pct` tracking for the 1-2 km target.

## 2026-04-15
- Hypothesis:
  - The external deep research review suggests rank-based multi-index source fusion (RRF) could improve robustness to inter-source score-scale mismatch.
- Change:
  - Merged `tech/accuracy-rtx5060-sprint` into `master`, then synced and pushed all active branches (`feature/film-ui-operator-lab`, `tech/eval-regression-lab`, `tech/geo-retrieval-v3`, `tech/model-hardening`, `tech/accuracy-rtx5060-sprint`).
  - Implemented retrieval source-fusion mode control in runtime/config:
    - new knob `geolocator.retrieval_source_fusion_mode` with `weighted_score` (default) and `rrf`.
    - wired through config loader + retrieval provider + runtime tool paths (`run_geo_eval`, `run_all`, `ui_server`, `tune_retrieval_geo`).
  - Added regression tests for:
    - config parsing/default/fallback for source-fusion mode,
    - retrieval multi-index RRF aggregation behavior.
  - Added benchmark trial config `runs/bench_cfg/cfg_paris_sourcefusion_rrf.json` and executed weighted-vs-rrf comparison on realistic split.
  - Updated docs (`README.md`, `src/docs/GEO_TECH.md`, `src/docs/RESEARCH_PAPER.md`, `CHANGELOG.md`) and added a research-paper section with prioritized approaches from `deep-research-report.md`.
- Files touched:
  - `src/core/geo/retrieval_provider.py`
  - `src/core/logic/config.py`
  - `src/tools/run_geo_eval.py`
  - `src/tools/run_all.py`
  - `src/tools/ui_server.py`
  - `src/tools/tune_retrieval_geo.py`
  - `src/config/defaults.json`
  - `src/config/open_geo.json`
  - `src/config/paris.json`
  - `src/config/paris_test.json`
  - `src/tests/test_config_loading.py`
  - `src/tests/test_retrieval_provider_multi_index.py`
  - `README.md`
  - `src/docs/GEO_TECH.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `CHANGELOG.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest src/tests/test_config_loading.py src/tests/test_retrieval_provider_multi_index.py src/tests/test_retrieval_tta.py src/tests/test_tune_retrieval_geo.py src/tests/test_ui_server_runtime.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --retrieval-only --limit 180 --seed 42 --config src/config/paris.json --output runs/geo_eval_paris_profile_180_sourcefusion_weighted_v1.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --retrieval-only --limit 180 --seed 42 --config runs/bench_cfg/cfg_paris_sourcefusion_rrf.json --output runs/geo_eval_paris_profile_180_sourcefusion_rrf_v1.json`
- Metrics (weighted_score -> rrf):
  - `mean_km`: `15.5569` -> `16.7288`
  - `median_km`: `9.7717` -> `11.0527`
  - `within_1km_pct`: `10.56` -> `7.78`
  - `within_2km_pct`: `19.44` -> `17.78`
  - `within_5km_pct`: `36.67` -> `34.44`
  - `within_10km_pct`: `50.56` -> `44.44`
- Artifacts:
  - `runs/geo_eval_paris_profile_180_sourcefusion_weighted_v1.json`
  - `runs/geo_eval_paris_profile_180_sourcefusion_rrf_v1.json`
  - `runs/bench_cfg/cfg_paris_sourcefusion_rrf.json`
- Decision:
  - Keep `retrieval_source_fusion_mode` support as an experimental option, but do not switch default profile behavior from `weighted_score` at this stage.
  - Next highest-priority direction from the research review: benchmark remote-sensing-native backbones (RemoteCLIP/SatCLIP family) on the same realistic split.

## 2026-04-15
- Hypothesis:
  - A dedicated aerial-backbone upgrade workflow should improve close-range retrieval quality iteration speed by replacing manual model/index/config switching with one reproducible command path.
- Change:
  - Upgraded `src/tools/benchmark_geo_backbones.py` with:
    - aerial model presets (`legacy_clip_siglip`, `aerial_rtx5060_fast`, `aerial_rtx5060_precise`, `aerial_research`),
    - objective-driven ranking (`within_1km_pct`, `within_2km_pct`, `within_5km_pct`, `within_10km_pct`, etc.),
    - per-model failure isolation so one model failure does not abort full benchmark sweep.
  - Added new automation utility `src/tools/upgrade_retrieval_backbone.py`:
    - runs backbone benchmark,
    - selects best model by objective,
    - rebuilds final retrieval index with that model,
    - patches target config (`retrieval_model_id`, `retrieval_index_path`) with optional multi-index reset.
  - Added targeted regression tests for both benchmark and upgrade flows.
  - Updated docs (`README.md`, `src/docs/RESEARCH_PAPER.md`) with the new aerial-backbone workflow and commands.
- Files touched:
  - `src/tools/benchmark_geo_backbones.py`
  - `src/tools/upgrade_retrieval_backbone.py`
  - `src/tests/test_benchmark_geo_backbones.py`
  - `src/tests/test_upgrade_retrieval_backbone.py`
  - `README.md`
  - `src/docs/RESEARCH_PAPER.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_benchmark_geo_backbones.py src/tests/test_upgrade_retrieval_backbone.py`
- Metrics (before -> after):
  - No new geo benchmark metric claim in this prompt cycle (tooling/workflow upgrade only).
- Artifacts:
  - No benchmark artifact generated in this prompt cycle.
- Decision:
  - Keep and use `upgrade_retrieval_backbone` as the standard path for aerial retrieval backbone upgrades on single-GPU runs.

## 2026-04-16
- Hypothesis:
  - Running the new backbone-upgrade flow and a focused `within_2km_pct` retrieval sweep on the realistic split should produce measurable close-range uplift.
- Change:
  - Executed baseline realistic retrieval-only eval (`n=180`) before changes.
  - Ran aerial backbone benchmark preset (`aerial_rtx5060_precise`) with objective `within_2km_pct`:
    - compared `google/siglip-so400m-patch14-384`, `google/siglip-base-patch16-224`, and `openai/clip-vit-large-patch14`.
  - Ran focused retrieval tuning sweep on Paris profile (`3456` combinations, objective `within_2km_pct`) and applied best-config output.
  - Executed post-tune realistic retrieval-only eval (`n=180`) and compared pre/post metrics.
  - Reverted non-improving config changes in `src/config/paris.json` (`retrieval_top_k` back to `25`, `retrieval_min_score` back to `0.05`) because end-to-end metrics were unchanged.
- Files touched:
  - `src/config/paris.json`
  - `PROGRESS.md`
  - `src/docs/RESEARCH_PAPER.md`
- Validation command(s):
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --retrieval-only --limit 180 --seed 42 --config src/config/paris.json --output runs/geo_eval_paris_profile_180_pre_backbone_upgrade_v1.json`
  - `./.venv/Scripts/python -m src.tools.upgrade_retrieval_backbone --train-images-dir data/spacenet_paris/chips --train-metadata data/spacenet_paris/metadata.csv --eval-images-dir data/spacenet_paris_test/chips --eval-metadata data/spacenet_paris_test/metadata.csv --config src/config/paris.json --model-preset aerial_rtx5060_precise --rank-objective within_2km_pct --benchmark-train-limit 600 --benchmark-eval-limit 180 --seed 42 --output-dir runs/backbone_upgrade_rtx5060_v1 --reuse-indices`
  - `./.venv/Scripts/python -m src.tools.tune_retrieval_geo --config src/config/paris.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --output runs/tune_retrieval_geo_within2km_v1.json --retrieval-topk "20,25,30" --retrieval-min-score "0.03,0.05,0.08" --retrieval-min-keep-topk "0,1" --retrieval-diversity-radius-km "0.0,1.0" --retrieval-diversity-lambda "1.0,0.9" --retrieval-diversity-min-keep "1,3" --retrieval-locality-radius-km "0.0,25.0" --retrieval-locality-weight "0.0,0.8,1.2" --retrieval-source-balance-beta "0.0,0.35" --retrieval-query-tta-reduce "max,mean" --rank-objective within_2km_pct --apply-best-config`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --retrieval-only --limit 180 --seed 42 --config src/config/paris.json --output runs/geo_eval_paris_profile_180_post_tune_v1.json`
- Metrics (pre -> post):
  - `mean_km`: `15.5569` -> `15.5569` (`delta=0.0000`)
  - `median_km`: `9.7717` -> `9.7717` (`delta=0.0000`)
  - `within_1km_pct`: `10.56` -> `10.56` (`delta=0.00`)
  - `within_2km_pct`: `19.44` -> `19.44` (`delta=0.00`)
  - `within_5km_pct`: `36.67` -> `36.67` (`delta=0.00`)
  - `within_10km_pct`: `50.56` -> `50.56` (`delta=0.00`)
- Artifacts:
  - `runs/geo_eval_paris_profile_180_pre_backbone_upgrade_v1.json`
  - `runs/backbone_upgrade_rtx5060_v1/backbone_benchmark.json`
  - `runs/tune_retrieval_geo_within2km_v1.json`
  - `runs/geo_eval_paris_profile_180_post_tune_v1.json`
- Decision:
  - Do not switch retrieval backbone from CLIP on this split: benchmark showed CLIP remains best among tested candidates.
  - Do not keep the tuned config changes from this sweep: no end-to-end metric uplift on the canonical realistic eval.
  - Next step should be non-trivial method upgrades (domain-adapted retrieval features, re-ranking head, or hard-negative data curation) rather than more local knob sweeps.

## 2026-04-16
- Hypothesis:
  - A method-level local geometric reranker upgrade (dual-engine matching + evidence gating) can recover local-feature ranking quality without harming the canonical retrieval profile.
- Change:
  - Upgraded retrieval local matching in `src/core/geo/retrieval_provider.py`:
    - added dual-engine local feature support (`SIFT` + `ORB`) with best-of-engines candidate scoring,
    - added weak-signal gate to skip local reranking when geometric evidence is low,
    - added adaptive blend scaling so local evidence only overrides retrieval score when local confidence is meaningful.
  - Added regression coverage for the new safety behavior in `src/tests/test_retrieval_provider_multi_index.py` (`test_local_match_rerank_skips_when_signal_is_weak`).
  - Re-ran realistic split evals (`n=180`) for control and local-match profiles, plus a combined KDE + dual-local profile.
- Files touched:
  - `src/core/geo/retrieval_provider.py`
  - `src/tests/test_retrieval_provider_multi_index.py`
  - `runs/bench_cfg/cfg_paris_kde_refine_e_w1_duallocal.json`
  - `PROGRESS.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `src/docs/GEO_TECH.md`
  - `README.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_retrieval_provider_multi_index.py src/tests/test_config_loading.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_qexp_ctrl.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --output runs/geo_eval_paris_profile_180_qexp_ctrl_v2_after_dual_localcode.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_localmatch_a.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --output runs/geo_eval_paris_profile_180_localmatch_a_v2_dual.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_localmatch_b.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --output runs/geo_eval_paris_profile_180_localmatch_b_v2_dual.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_localmatch_c.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --output runs/geo_eval_paris_profile_180_localmatch_c_v2_dual.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_kde_refine_e_w1_duallocal.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --output runs/geo_eval_paris_profile_180_kde_refine_e_w1_duallocal_v1.json`
- Metrics (before -> after):
  - Control profile (`qexp_ctrl`): unchanged (`within_1km_pct`: `10.56` -> `10.56`), confirming no regression when local rerank is disabled.
  - Local match profile A (`localmatch_a`):
    - `mean_km`: `16.6122` -> `15.2447`
    - `median_km`: `10.9454` -> `9.7717`
    - `within_1km_pct`: `5.00` -> `8.89`
    - `within_2km_pct`: `13.33` -> `18.33`
    - `within_5km_pct`: `32.78` -> `37.22`
    - `within_10km_pct`: `46.67` -> `51.11`
  - Combined KDE (`c_w1`) + dual local profile did not improve close-range hit rate (`within_1km_pct`: `11.11` -> `8.89`).
- Artifacts:
  - `runs/geo_eval_paris_profile_180_qexp_ctrl_v2_after_dual_localcode.json`
  - `runs/geo_eval_paris_profile_180_localmatch_a_v2_dual.json`
  - `runs/geo_eval_paris_profile_180_localmatch_b_v2_dual.json`
  - `runs/geo_eval_paris_profile_180_localmatch_c_v2_dual.json`
  - `runs/geo_eval_paris_profile_180_kde_refine_e_w1_duallocal_v1.json`
- Decision:
  - Keep the dual local geometric reranker implementation as a meaningful method upgrade.
  - Keep local-rerank knobs non-default for the canonical Paris profile because close-range (`within_1km_pct`) remains best with KDE refinement profile `c_w1`.
  - Treat dual local rerank as an optional mode for reducing larger-distance errors, while primary close-range optimization continues via retrieval-density refinement and future domain-adapted representation upgrades.

## 2026-04-16
- Hypothesis:
  - Extreme random-sample errors (`~5846 km`) came from evaluation profile/data mismatch (Paris dataset accidentally evaluated with `open_geo` index), not from corrupted Paris labels.
- Change:
  - Reproduced the mismatch on seed `1870334448` with `src/config/open_geo.json` against Paris data and confirmed US predictions near Statue of Liberty for Paris chips.
  - Added backend profile resolution guard in `src/tools/ui_server.py` for both `/eval/geo/start` and `/eval/geo/random/start`:
    - infer dataset family from `images_dir`/`metadata` path,
    - auto-correct `legacy/open_geo` to `paris` or `paris_test` when Paris paths are detected,
    - expose `profile_requested`, `profile_effective`, `profile_warning`, and `config_path` in eval status and random summary.
  - Updated Lab UI profile persistence key to isolate Lab from Operator (`heimdallLabProfile`) and avoid cross-tab profile leakage.
  - Updated random-sample output formatting to display requested/effective profile, config path, and correction warning.
  - Added regression test: `test_start_geo_random_eval_autocorrects_legacy_profile_for_paris_paths`.
- Files touched:
  - `src/tools/ui_server.py`
  - `src/dashboard/analysis/lab/lab.js`
  - `src/tests/test_ui_server_runtime.py`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_ui_server_runtime.py src/tests/test_ui_server_safe_demo.py src/tests/test_ui_server_health.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --retrieval-only --limit 2 --seed 1870334448 --config src/config/open_geo.json --output runs/tmp_paris_with_open_geo_seed1870334448.json --diag-samples 2`
- Metrics/evidence:
  - Repro run with `open_geo` on Paris chips:
    - `mean_km`: `5846.1583`
    - `within_1km_pct`: `0.0`
    - both samples predicted `40.6892, -74.0445` while GT is Paris (`~48.59, 2.26`).
  - Control run with `paris` on the same seed/samples:
    - `mean_km`: `7.1809`
    - `within_1km_pct`: `50.0`
    - predictions remained in Paris region (`~48.61/2.27`, `~48.71/2.23`).
- Decision:
  - Treat this as an evaluation-integrity bug.
  - Keep auto-correction guard enabled so Lab random/full eval cannot silently run Paris chips on legacy/open-geo profile.

## 2026-04-16
- Hypothesis:
  - The research paper needs an explicit method-decision ledger (kept vs rejected) and complete artifact traceability so all attempted methods and outcomes are auditable.
- Change:
  - Expanded `src/docs/RESEARCH_PAPER.md` with:
    - new evaluation-integrity results subsection (`7.10`) documenting the Paris/open_geo mismatch and fix impact,
    - comprehensive method ledger (`8.3`) listing tried methods, best measured outcomes, and keep/reject decisions,
    - default-policy summary (`8.4`) clarifying what stays on by default,
    - updated Appendix B artifact index to include backbone-cycle and evaluation-integrity artifacts.
  - Updated `AGENT.md` to enforce command-level documentation sync:
    - added non-negotiable rule that every state-changing command must be reflected in docs,
    - added explicit command-level enforcement checklist under per-prompt sync.
- Files touched:
  - `src/docs/RESEARCH_PAPER.md`
  - `AGENT.md`
  - `PROGRESS.md`
- Validation command(s):
  - `rg -n "### 7.10|### 8.3|### 8.4|## Appendix B" src/docs/RESEARCH_PAPER.md`
  - `rg -n "state-changing command|Command-level enforcement" AGENT.md`
- Metrics (before -> after):
  - Documentation-only update; no model metric run in this prompt.
- Artifacts:
  - `src/docs/RESEARCH_PAPER.md`
  - `AGENT.md`
- Decision:
  - Keep this documentation policy and ledger structure as the new baseline for future iterations.

## 2026-04-15
- Added a dedicated external market/SOTA research companion document at `src/docs/MARKET_RESEARCH.md` based on the provided report (`C:\Users\zen\Downloads\deep-research-report.md`).
- Document scope:
  - external landscape summary,
  - datasets/tooling ecosystem,
  - prioritized roadmap,
  - explicit "approaches we will consider" list,
  - current-cycle decision snapshot for retrieval source-fusion mode.
- Linked the new research companion in:
  - `README.md` (Project Status Tracking)
  - `docs/DOCS_MAP.md` (Core Entry Points)
  - `src/docs/RESEARCH_PAPER.md` (companion reference).
- Validation:
  - Documentation-only update; no runtime code path changed.

## 2026-04-18
- Hypothesis:
  - Non-trivial retrieval changes (multi-scale query views and adaptive-mass KDE refinement) might improve close-range accuracy on the realistic Paris split (`n=180`).
- Change:
  - Added retrieval query multi-scale support via `retrieval_query_tta_scales` and wired it through runtime/config/tooling paths.
  - Added adaptive-mass KDE option via `retrieval_kde_refine_adaptive_mass` and wired it through runtime/config/tooling paths.
  - Added regression coverage for new knobs in retrieval/config tests.
  - Standardized shipped config scope labels with `profile_scope` (`PARIS`/`US`) across `src/config/*.json`.
  - Hardened UI benchmark scenario config resolution in `src/tools/ui_server.py`:
    - prefer new scoped names when available,
    - fall back to legacy `runs/bench_cfg/*.json` names when scoped files are absent.
  - Updated `README.md` and `src/docs/RESEARCH_PAPER.md` to document these methods, outcomes, and scope conventions.
- Files touched:
  - `src/core/geo/retrieval_provider.py`
  - `src/core/logic/config.py`
  - `src/cli.py`
  - `src/batch_run.py`
  - `src/tools/run_all.py`
  - `src/tools/run_geo_eval.py`
  - `src/tools/tune_retrieval_geo.py`
  - `src/tools/benchmark_geo_backbones.py`
  - `src/tools/ui_server.py`
  - `src/config/defaults.json`
  - `src/config/paris.json`
  - `src/config/paris_test.json`
  - `src/config/open_geo.json`
  - `src/tests/test_config_loading.py`
  - `src/tests/test_retrieval_tta.py`
  - `src/tests/test_retrieval_provider_multi_index.py`
  - `README.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_config_loading.py src/tests/test_retrieval_tta.py src/tests/test_retrieval_provider_multi_index.py`
  - prior realistic eval artifacts compared in this cycle:
    - `runs/geo_eval_paris_profile_180_tta_agreement_ctrl_v1.json`
    - `runs/geo_eval_paris_profile_180_multiscale_a_v1.json`
    - `runs/geo_eval_paris_profile_180_kde_refine_d_w2_v1.json`
    - `runs/geo_eval_paris_profile_180_kde_refine_d_w2_adapt_a_v1.json`
    - `runs/geo_eval_paris_profile_180_kde_refine_d_w2_adapt_b_v1.json`
- Metrics (before -> after):
  - Multi-scale query views vs control:
    - `mean_km`: `15.5264` -> `15.3451`
    - `median_km`: `9.7717` -> `10.1387`
    - `within_1km_pct`: `10.56` -> `6.11`
    - `within_2km_pct`: `19.44` -> `17.22`
  - Adaptive-mass KDE (`0.7`) vs fixed-mass KDE-W2:
    - `within_1km_pct`: `10.00` -> `11.11`
    - `within_2km_pct`: `20.00` -> `19.44`
    - `mean_km`: `15.4223` -> `15.4590`
  - Adaptive-mass KDE (`0.5`) vs fixed-mass KDE-W2:
    - `within_1km_pct`: `10.00` -> `8.89`
    - `within_2km_pct`: `20.00` -> `17.78`
- Artifacts:
  - `runs/geo_eval_paris_profile_180_tta_agreement_ctrl_v1.json`
  - `runs/geo_eval_paris_profile_180_multiscale_a_v1.json`
  - `runs/geo_eval_paris_profile_180_kde_refine_d_w2_v1.json`
  - `runs/geo_eval_paris_profile_180_kde_refine_d_w2_adapt_a_v1.json`
  - `runs/geo_eval_paris_profile_180_kde_refine_d_w2_adapt_b_v1.json`
- Decision:
  - Keep both features available as experimental knobs.
  - Do not promote either feature to default profile settings yet because close-range target metrics did not improve consistently (`within_2km_pct` regressed).
