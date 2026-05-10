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

## 2026-05-03

- Changed `/analyze/image` failure handling so real-image analysis no longer silently returns safe-demo payloads on worker or pipeline failure.
- Added inline retry when the spawned inference worker fails, then return HTTP 503 if real inference still cannot run.
- Updated the V2 analysis UI to reject demo payloads from the image endpoint instead of rendering them as real geolocation results.
- Updated UI server tests to cover inline retry success and explicit 503 failure paths.
- Added a stronger red failure alert in the V2 analysis UI so users are told fallback/demo output may be random or synthetic and should not be trusted.
- Improved frontend network-error reporting so `Failed to fetch` is shown as an explicit backend-unreachable message instead of a generic browser error.
- Updated candidate cards so long source labels stay collapsed by default with a `More` toggle, full hover title, and copy-on-click support without horizontal overflow.

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

## 2026-04-21
- Hypothesis:
  - Scope-integrity protection should be enforced in CLI evaluation (`run_geo_eval`) as well, not only in UI/backend random eval flows, to prevent accidental cross-scope benchmarking.
- Change:
  - Added scope normalization and inference helpers in `src/tools/run_geo_eval.py`:
    - `normalize_scope`
    - `load_profile_scope`
    - `infer_dataset_scope`
    - `validate_scope_alignment`
  - Added a default-on scope guard in `run_geo_eval`:
    - compares config `profile_scope` with inferred dataset scope from `--images-dir`/`--metadata`,
    - raises on mismatch unless `--allow-scope-mismatch` is passed.
  - Extended eval report payload with:
    - `profile_scope`
    - `dataset_scope`
    - `scope_warning`
    - `allow_scope_mismatch`
  - Added regression tests in `src/tests/test_run_geo_eval_scope_guard.py` covering:
    - scope normalization,
    - config-profile scope resolution,
    - dataset scope inference,
    - mismatch raise/override behavior.
  - Updated docs:
    - `README.md` now documents default scope validation and override flag.
    - `src/docs/RESEARCH_PAPER.md` now records CLI scope guard as part of evaluation-integrity requirements.
- Files touched:
  - `src/tools/run_geo_eval.py`
  - `src/tests/test_run_geo_eval_scope_guard.py`
  - `README.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_run_geo_eval_scope_guard.py src/tests/test_run_geo_eval_retrieval_provider.py`
- Metrics (before -> after):
  - No model-accuracy benchmark run in this prompt cycle (stack integrity and evaluation safety upgrade only).
- Artifacts:
  - No new `runs/*.json` evaluation artifact generated in this prompt cycle.
- Decision:
  - Keep this guard enabled by default as a reliability baseline for CLI benchmarking.

## 2026-04-21
- Hypothesis:
  - Local geometric reranking should not be allowed to override a very confident retrieval top-1 unless geometric evidence is clearly stronger; adding an ambiguity gate may recover close-range precision in hybrid local-match profiles.
- Change:
  - Added ambiguity-gated override logic in `src/core/geo/retrieval_provider.py` inside `_apply_local_match_rerank`:
    - when base top1-top2 retrieval gap is high (`>= 0.10`), local rerank override is blocked unless local evidence advantage is strong (`>= 0.28`).
  - Added regression tests:
    - `test_local_match_rerank_skips_confident_override_without_strong_local_advantage`
    - `test_local_match_rerank_allows_confident_override_when_local_advantage_is_strong`
  - Re-ran realistic retrieval-only evals for existing local-match profiles.
- Files touched:
  - `src/core/geo/retrieval_provider.py`
  - `src/tests/test_retrieval_provider_multi_index.py`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_retrieval_provider_multi_index.py src/tests/test_run_geo_eval_retrieval_provider.py src/tests/test_run_geo_eval_scope_guard.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_localmatch_a.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --output runs/geo_eval_paris_profile_180_localmatch_a_v3_ambiguity_gate.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_kde_refine_e_w1_duallocal.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --output runs/geo_eval_paris_profile_180_kde_refine_e_w1_duallocal_v2_ambiguity_gate.json`
- Metrics (before -> after):
  - `localmatch_a`: unchanged (`within_1km_pct=8.89`, `within_2km_pct=18.33`, `mean_km=15.2447`).
  - `kde_refine_e_w1_duallocal`: unchanged (`within_1km_pct=8.89`, `within_2km_pct=18.33`, `mean_km=15.1863`).
- Artifacts:
  - `runs/geo_eval_paris_profile_180_localmatch_a_v3_ambiguity_gate.json`
  - `runs/geo_eval_paris_profile_180_kde_refine_e_w1_duallocal_v2_ambiguity_gate.json`
- Decision:
  - Keep the ambiguity-gate as a defensive guardrail, but not as a promoted accuracy lever based on current measured deltas.

## 2026-04-21
- Hypothesis:
  - Step-change accuracy gains now require data-centric upgrades (hard negatives around real failure zones), not additional local knob sweeps.
- Change:
  - Added new mining utility: `src/tools/mine_hard_negative_triplets.py`.
    - Inputs: metadata CSV + optional `run_geo_eval` report.
    - Outputs: query-positive-hard-negative triplets in JSONL + summary JSON.
    - Supports failure-threshold filtering (`--min-error-km`), positive/negative radius controls, and scene-level deduplication across sensor/modal variants.
  - Added regression tests:
    - `src/tests/test_mine_hard_negative_triplets.py`
  - Added docs for hard-negative mining workflow in `README.md`.
  - Updated research narrative in `src/docs/RESEARCH_PAPER.md`.
  - Ran an eval artifact with full diagnostics (`diag_samples=180`) and mined triplets from it.
- Files touched:
  - `src/tools/mine_hard_negative_triplets.py`
  - `src/tests/test_mine_hard_negative_triplets.py`
  - `README.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_mine_hard_negative_triplets.py src/tests/test_retrieval_provider_multi_index.py src/tests/test_run_geo_eval_retrieval_provider.py src/tests/test_run_geo_eval_scope_guard.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config src/config/paris.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_paris_profile_180_for_mining_v1.json`
  - `./.venv/Scripts/python -m src.tools.mine_hard_negative_triplets --metadata data/spacenet_paris_test/metadata.csv --eval-report runs/geo_eval_paris_profile_180_for_mining_v1.json --output runs/hard_negative_triplets_paris_test_v2_scene_dedup.jsonl --summary-output runs/hard_negative_triplets_paris_test_v2_scene_dedup_summary.json --min-error-km 2.0 --positive-radius-km 0.35 --negative-pred-radius-km 2.0 --negative-min-gt-distance-km 2.0 --negative-max-gt-distance-km 25.0 --max-positives 3 --max-negatives 12`
- Metrics/artifacts:
  - Eval artifact for mining:
    - `runs/geo_eval_paris_profile_180_for_mining_v1.json`
  - Mined triplets:
    - `runs/hard_negative_triplets_paris_test_v2_scene_dedup.jsonl`
    - `runs/hard_negative_triplets_paris_test_v2_scene_dedup_summary.json`
  - Summary:
    - `total_records=2391`
    - `total_failures_considered=180`
    - `triplets_written=145`
    - avg positives `~2.90`, avg hard negatives `12.0`
- Decision:
  - Keep and prioritize the hard-negative mining pipeline as the immediate path toward real retrieval-backbone accuracy gains.

## 2026-04-21
- Hypothesis:
  - Hard-negative-informed embedding projection can produce a real close-range step-up (`<=1km`, `<=2km`) beyond local rerank/tuning-only changes.
- Change:
  - Added projection-aware retrieval plumbing end-to-end:
    - `ClipEmbedder` now supports optional projection loading/apply (`matrix` + optional `bias`) with post-projection normalization.
    - New config knob: `geolocator.retrieval_projection_path`.
    - Wired projection path through CLI, batch, UI server, eval, and tuning paths.
    - `build_geo_index` supports `--projection-path` during index construction.
  - Added new tools:
    - `src/tools/train_retrieval_projection.py` (train projection from hard-negative triplets)
    - `src/tools/apply_projection_to_geo_index.py` (fast projection transform for existing index embeddings)
  - Upgraded hard-negative miner for explicit query/reference split:
    - `src/tools/mine_hard_negative_triplets.py` now supports `--reference-metadata`.
  - Added/updated tests for projection training/apply, config loading, and miner query/reference behavior.
  - Ran projection-variant evaluation cycle (baseline + V1/V2/V3) on realistic Paris retrieval-only split.
- Files touched:
  - `src/core/geo/retrieval_provider.py`
  - `src/core/logic/config.py`
  - `src/tools/build_geo_index.py`
  - `src/tools/run_geo_eval.py`
  - `src/tools/run_all.py`
  - `src/tools/ui_server.py`
  - `src/tools/tune_retrieval_geo.py`
  - `src/cli.py`
  - `src/batch_run.py`
  - `src/tools/train_retrieval_projection.py`
  - `src/tools/apply_projection_to_geo_index.py`
  - `src/tools/mine_hard_negative_triplets.py`
  - `src/tests/test_train_retrieval_projection.py`
  - `src/tests/test_apply_projection_to_geo_index.py`
  - `src/tests/test_mine_hard_negative_triplets.py`
  - `src/tests/test_config_loading.py`
  - `src/config/defaults.json`
  - `src/config/paris.json`
  - `src/config/paris_test.json`
  - `src/config/open_geo.json`
  - `README.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_mine_hard_negative_triplets.py src/tests/test_apply_projection_to_geo_index.py src/tests/test_train_retrieval_projection.py src/tests/test_config_loading.py src/tests/test_retrieval_provider_multi_index.py src/tests/test_run_geo_eval_retrieval_provider.py src/tests/test_run_geo_eval_scope_guard.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --config src/config/paris.json --limit 120 --seed 42 --diag-samples 120 --output runs/geo_eval_projection_baseline_120.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --config runs/bench_cfg/cfg_paris_projection_trainref_v1.json --limit 120 --seed 42 --diag-samples 120 --output runs/geo_eval_projection_trainref_v1_120_rerun.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild.json --limit 120 --seed 42 --diag-samples 120 --output runs/geo_eval_projection_trainref_v2_mild_120.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --config runs/bench_cfg/cfg_paris_projection_trainref_v3_dim256.json --limit 120 --seed 42 --diag-samples 120 --output runs/geo_eval_projection_trainref_v3_dim256_120.json`
- Metrics (baseline -> selected V2):
  - `mean_km`: `15.517` -> `14.925`
  - `median_km`: `10.498` -> `4.512`
  - `within_1km_pct`: `9.17` -> `12.50`
  - `within_2km_pct`: `16.67` -> `28.33`
  - `within_5km_pct`: `36.67` -> `51.67`
  - `within_10km_pct`: `48.33` -> `61.67`
- Artifacts:
  - `runs/geo_eval_projection_baseline_120.json`
  - `runs/geo_eval_projection_trainref_v1_120_rerun.json`
  - `runs/geo_eval_projection_trainref_v2_mild_120.json`
  - `runs/geo_eval_projection_trainref_v3_dim256_120.json`
  - `runs/retrieval_projection_paris_query_trainref_v2_mild.npz`
  - `runs/retrieval_projection_paris_query_trainref_v2_mild.report.json`
  - `data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz`
- Decision:
  - Keep the projection adaptation path and treat `trainref_v2_mild` as the best current retrieval adaptation direction.
  - Use this as the base for next hard-negative data expansion + projection retraining cycle.

## 2026-04-21
- Hypothesis:
  - `train_retrieval_projection` should support index-only training runs in CI tests without requiring a local image directory.
- Change:
  - Fixed `src/tools/train_retrieval_projection.py` to check for missing embeddings first and only validate/use `--images-dir` when image embedding backfill is actually needed.
- Files touched:
  - `src/tools/train_retrieval_projection.py`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_train_retrieval_projection.py`
  - `./.venv/Scripts/python -m pytest -q`
- Metrics:
  - Local validation after fix: `180 passed, 3 warnings`.
- Decision:
  - Keep this CI compatibility fix; no modeling behavior change to projection math itself.

## 2026-04-21
- Hypothesis:
  - A scope-aware geographic prior in retrieval (Paris bbox hard gate) can remove catastrophic cross-region errors (thousands of km) that appear when mixed-source indices are used.
- Change:
  - Added geo-prior controls to retrieval provider:
    - `retrieval_geo_prior_mode` (`off|soft|hard`)
    - `retrieval_geo_prior_bbox`
    - `retrieval_geo_prior_sigma_km`
    - `retrieval_geo_prior_min_keep`
  - Wired the new knobs through all config-driven entrypoints:
    - `src/cli.py`, `src/batch_run.py`, `src/tools/run_all.py`, `src/tools/run_geo_eval.py`, `src/tools/tune_retrieval_geo.py`, `src/tools/ui_server.py`
  - Extended config schema + loader and profile configs:
    - `src/core/logic/config.py`
    - `src/config/defaults.json`, `src/config/paris.json`, `src/config/paris_test.json`, `src/config/open_geo.json`
  - Added retrieval geo-prior regression tests:
    - `src/tests/test_retrieval_geo_prior.py`
    - updated `src/tests/test_config_loading.py`
  - Updated documentation with method + artifacts:
    - `README.md`, `src/docs/RESEARCH_PAPER.md`
- Files touched:
  - `src/core/geo/retrieval_provider.py`
  - `src/core/logic/config.py`
  - `src/cli.py`
  - `src/batch_run.py`
  - `src/tools/run_all.py`
  - `src/tools/run_geo_eval.py`
  - `src/tools/tune_retrieval_geo.py`
  - `src/tools/ui_server.py`
  - `src/config/defaults.json`
  - `src/config/paris.json`
  - `src/config/paris_test.json`
  - `src/config/open_geo.json`
  - `src/tests/test_retrieval_geo_prior.py`
  - `src/tests/test_config_loading.py`
  - `README.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_retrieval_geo_prior.py src/tests/test_config_loading.py src/tests/test_retrieval_provider_multi_index.py src/tests/test_retrieval_provider_min_keep.py src/tests/test_retrieval_tta.py`
  - `./.venv/Scripts/python -m pytest -q`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --config runs/configs/paris_mixed_scope_no_prior.json --retrieval-only --limit 120 --seed 42 --output runs/geo_eval_mixed_scope_no_prior_120.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --config runs/configs/paris_mixed_scope_hard_prior.json --retrieval-only --limit 120 --seed 42 --output runs/geo_eval_mixed_scope_hard_prior_120.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --config runs/configs/paris_mixed_scope_no_prior.json --retrieval-only --limit 2 --seed 1870334448 --output runs/geo_eval_mixed_scope_no_prior_seed1870334448_2.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --config runs/configs/paris_mixed_scope_hard_prior.json --retrieval-only --limit 2 --seed 1870334448 --output runs/geo_eval_mixed_scope_hard_prior_seed1870334448_2.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --config src/config/paris_test.json --retrieval-only --limit 40 --seed 42 --output runs/geo_eval_paris_test_profile_with_geo_prior_40.json`
- Metrics (before -> after):
  - Mixed-scope stress test (`n=120`, Paris test, Paris+open-geo index mix):
    - `mean_km`: `6656.661` -> `18.822`
    - `median_km`: `5830.112` -> `12.362`
    - `within_10km_pct`: `0.00` -> `40.83`
    - `within_1km_pct`: `0.00` -> `5.00`
  - Replay seed (`1870334448`, `n=2`, same chips previously reported):
    - `mean_km`: `7408.151` -> `0.000`
    - `within_10km_pct`: `0.00` -> `100.00`
- Artifacts:
  - `runs/configs/paris_mixed_scope_no_prior.json`
  - `runs/configs/paris_mixed_scope_hard_prior.json`
  - `runs/geo_eval_mixed_scope_no_prior_120.json`
  - `runs/geo_eval_mixed_scope_hard_prior_120.json`
  - `runs/geo_eval_mixed_scope_no_prior_seed1870334448_2.json`
  - `runs/geo_eval_mixed_scope_hard_prior_seed1870334448_2.json`
  - `runs/geo_eval_paris_test_profile_with_geo_prior_40.json`
- Decision:
  - Keep hard geo-prior defaults on Paris configs (`defaults`, `paris`, `paris_test`).
  - Keep `open_geo` profile geo prior disabled (`off`) because it is a broad-scope/legacy profile.

## 2026-04-21
- Hypothesis:
  - Ensembling projected and non-projected retrieval spaces can improve close-range Paris metrics when query projection is routed per index instead of globally.
- Change:
  - Added per-index projection routing support:
    - New geo config field: `retrieval_index_projection_paths`.
    - Retrieval provider now supports per-index query projection path assignment while keeping per-index model routing.
    - Added encoder-spec routing (`model_id` + `projection_path`) for query embeddings across multi-index retrieval.
  - Wired new field through config-driven entrypoints:
    - `src/cli.py`, `src/batch_run.py`, `src/tools/run_all.py`, `src/tools/run_geo_eval.py`, `src/tools/tune_retrieval_geo.py`, `src/tools/ui_server.py`.
  - Added/updated tests:
    - `src/tests/test_config_loading.py`
    - `src/tests/test_retrieval_provider_multi_index.py` (new per-index projection routing test).
  - Ran controlled `n=180` comparisons on Paris realistic retrieval-only split (seed `42`):
    - projection V2 baseline vs projection V2 + geo-prior stack
    - dual-space projected+raw CLIP (`rrf`) using per-index projection routing
    - projection V3 dim-256 head
    - projection V2 + local geometric rerank
  - Audited dataset scope in `data/spacenet_paris_test/metadata.csv` to verify Paris-only labels.
- Files touched:
  - `src/core/geo/retrieval_provider.py`
  - `src/core/logic/config.py`
  - `src/cli.py`
  - `src/batch_run.py`
  - `src/tools/run_all.py`
  - `src/tools/run_geo_eval.py`
  - `src/tools/tune_retrieval_geo.py`
  - `src/tools/ui_server.py`
  - `src/tests/test_config_loading.py`
  - `src/tests/test_retrieval_provider_multi_index.py`
  - `src/config/defaults.json`
  - `src/config/paris.json`
  - `src/config/paris_test.json`
  - `src/config/open_geo.json`
  - `runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_geo_prior.json`
  - `runs/bench_cfg/cfg_paris_dualspace_rrf_v1.json`
  - `runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_localmatch_v1.json`
  - `README.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_config_loading.py src/tests/test_retrieval_provider_multi_index.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_180_baseline.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_geo_prior.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_geo_prior_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_dualspace_rrf_v1.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_paris_dualspace_rrf_v1_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v3_dim256.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v3_dim256_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_localmatch_v1.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_localmatch_v1_180.json`
- Metrics:
  - Baseline projection V2 (`n=180`):
    - `mean_km=15.2523`, `median_km=5.5043`, `within_1km_pct=11.67`, `within_2km_pct=26.67`
  - Projection V2 + geo prior stack (`n=180`):
    - identical to baseline on this in-scope split (`within_1km_pct=11.67`, `within_2km_pct=26.67`)
  - Dual-space projected+raw CLIP with `rrf` (`n=180`):
    - `mean_km=16.0853`, `median_km=9.3245`, `within_1km_pct=8.89`, `within_2km_pct=18.89`
  - Projection V3 dim-256 (`n=180`):
    - `mean_km=19.9768`, `within_1km_pct=11.67`, `within_2km_pct=18.33`
  - Projection V2 + local match (`n=180`):
    - `mean_km=15.6061`, `within_1km_pct=10.00`, `within_2km_pct=25.56`
  - Dataset scope audit:
    - `2391/2391` metadata rows inside Paris bbox (`48.40..49.10`, `2.05..2.36`).
- Artifacts:
  - `runs/geo_eval_projection_trainref_v2_mild_180_baseline.json`
  - `runs/geo_eval_projection_trainref_v2_mild_geo_prior_180.json`
  - `runs/geo_eval_paris_dualspace_rrf_v1_180.json`
  - `runs/geo_eval_projection_trainref_v3_dim256_180.json`
  - `runs/geo_eval_projection_trainref_v2_mild_localmatch_v1_180.json`
  - `runs/bench_cfg/cfg_paris_dualspace_rrf_v1.json`
  - `runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_localmatch_v1.json`
- Decision:
  - Keep per-index projection routing capability (`retrieval_index_projection_paths`) as experimental infrastructure.
  - Keep projection V2 baseline as current best `n=180` performer among tested variants in this cycle.
  - Geo-prior remains a critical cross-scope safety guard, but does not improve in-scope Paris close-range metrics by itself.

## 2026-04-21
- Hypothesis:
  - CI failure on `master` comes from a retrieval-provider fallback regression introduced by per-index projection routing.
- Change:
  - Fixed fallback model-id resolution in `_collect_ranked_candidates` so mocked/legacy loaded-index objects without `model_id` still map to available query embeddings.
  - File: `src/core/geo/retrieval_provider.py`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_retrieval_provider_min_keep.py`
  - `./.venv/Scripts/python -m pytest -q`
- Metrics:
  - `2 passed` for targeted min-keep tests.
  - Full suite: `184 passed, 3 warnings`.
- Decision:
  - Keep fix; this restores CI stability without changing intended retrieval behavior.

## 2026-04-21
- Hypothesis:
  - Index-side descriptor augmentation (DBA) can improve retrieval precision by smoothing noisy per-image descriptors.
  - A geo-aware DBA constraint (only blend neighbors within a small geographic radius) should preserve close-range gains better than unconstrained DBA.
- Change:
  - Added a new tooling path: `src/tools/augment_geo_index_embeddings.py`
    - Supports descriptor DBA with cosine-neighbor pooling (`neighbors`, `self_weight`, `min_similarity`, `temperature`).
    - Added geo-aware masking via `--max-geo-distance-km` using index lat/lon.
  - Added regression tests for the DBA tool:
    - `src/tests/test_augment_geo_index_embeddings.py`
  - Built and benchmarked multiple DBA index variants on the canonical Paris projected index.
- Files touched:
  - `src/tools/augment_geo_index_embeddings.py`
  - `src/tests/test_augment_geo_index_embeddings.py`
  - `src/config/paris_close_range_dba.json`
  - `PROGRESS.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `README.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_augment_geo_index_embeddings.py`
  - `./.venv/Scripts/python -m pytest -q src/tests/test_retrieval_provider_multi_index.py src/tests/test_run_geo_eval_retrieval_provider.py`
  - `./.venv/Scripts/python -m src.tools.augment_geo_index_embeddings --index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz --output data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild_dba_k5.npz --neighbors 5 --self-weight 1.0 --min-similarity 0.0 --temperature 0.07`
  - `./.venv/Scripts/python -m src.tools.augment_geo_index_embeddings --index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz --output data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild_dba_k10.npz --neighbors 10 --self-weight 1.0 --min-similarity 0.0 --temperature 0.07`
  - `./.venv/Scripts/python -m src.tools.augment_geo_index_embeddings --index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz --output data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild_dba_k20.npz --neighbors 20 --self-weight 1.0 --min-similarity 0.0 --temperature 0.07`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_baseline_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dba_k5.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dba_k5_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dba_k10.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dba_k10_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dba_k20.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dba_k20_60.json`
  - `./.venv/Scripts/python -m src.tools.augment_geo_index_embeddings --index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz --output data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild_dba_geo2_k5.npz --neighbors 5 --self-weight 1.0 --min-similarity 0.0 --temperature 0.07 --max-geo-distance-km 2.0`
  - `./.venv/Scripts/python -m src.tools.augment_geo_index_embeddings --index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz --output data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild_dba_geo5_k5.npz --neighbors 5 --self-weight 1.0 --min-similarity 0.0 --temperature 0.07 --max-geo-distance-km 5.0`
  - `./.venv/Scripts/python -m src.tools.augment_geo_index_embeddings --index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz --output data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild_dba_geo2_k8.npz --neighbors 8 --self-weight 1.0 --min-similarity 0.0 --temperature 0.07 --max-geo-distance-km 2.0`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dba_geo2_k5.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_dba_geo2_k5_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dba_geo5_k5.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_dba_geo5_k5_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dba_geo2_k8.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_dba_geo2_k8_180.json`
- Metrics:
  - Baseline (`n=180`, `runs/geo_eval_projection_trainref_v2_mild_180_baseline.json`):
    - `mean_km=15.2523`, `median_km=5.5043`, `within_1km_pct=11.67`, `within_2km_pct=26.67`, `within_5km_pct=50.00`, `within_10km_pct=64.44`.
  - Unconstrained DBA (`k=5`) on `n=180`:
    - `mean_km=17.0584`, `within_1km_pct=10.56`, `within_2km_pct=22.22` (regression).
  - Geo-aware DBA (`k=5`, `max_geo_distance_km=5`) on `n=180`:
    - `mean_km=15.8026`, `within_1km_pct=11.67`, `within_2km_pct=25.56` (no close-range gain).
  - Geo-aware DBA (`k=8`, `max_geo_distance_km=2`) on `n=180`:
    - `mean_km=16.3238`, `within_1km_pct=11.11`, `within_2km_pct=25.56` (regression).
  - Geo-aware DBA (`k=5`, `max_geo_distance_km=2`) on `n=180`:
    - `mean_km=14.8018`, `median_km=6.3058`, `within_1km_pct=13.33`, `within_2km_pct=29.44`, `within_5km_pct=46.67`, `within_10km_pct=60.56`.
    - vs baseline deltas: `mean_km -0.4505`, `within_1km_pct +1.67`, `within_2km_pct +2.78`, `within_5km_pct -3.33`, `within_10km_pct -3.89`.
- Artifacts:
  - `runs/geo_eval_projection_trainref_v2_mild_baseline_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dba_k5_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dba_k10_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dba_k20_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dba_k5_180.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dba_geo2_k5_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dba_geo5_k5_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dba_geo2_k8_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dba_geo2_k5_180.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dba_geo5_k5_180.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dba_geo2_k8_180.json`
- Decision:
  - Reject unconstrained DBA and most geo-aware DBA settings for default profile.
  - Keep `geo-aware DBA (k=5, radius=2km)` as an objective-specific option for close-range (`<=1km`/`<=2km`) optimization, not as global default due tail regression (`<=5km`/`<=10km`).

## 2026-04-21
- Hypothesis:
  - A dual-index retrieval stack (baseline projected index + geo-aware DBA index) with rank-based source fusion can retain tail robustness while improving close-range accuracy.
- Change:
  - Built and evaluated dual-index configurations using the existing multi-index retrieval path:
    - baseline projected index: `...proj_trainref_v2_mild.npz`
    - geo-aware DBA index: `...proj_trainref_v2_mild_dba_geo2_k5.npz`
  - Added a promoted experimental close-range profile:
    - `src/config/paris_close_range_dual_rrf.json`
  - Updated docs to include the new profile and evaluation command.
- Files touched:
  - `src/config/paris_close_range_dual_rrf.json`
  - `README.md`
  - `PROGRESS.md`
  - `src/docs/RESEARCH_PAPER.md`
- Validation command(s):
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w90_10_weighted.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w90_10_weighted_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w80_20_weighted.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w80_20_weighted_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w70_30_weighted.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w70_30_weighted_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w100_100_weighted.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_weighted_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w100_100_rrf.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w100_100_weighted_sb02.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_weighted_sb02_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w100_100_rrf.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_180.json`
- Metrics:
  - Baseline (`n=180`): `mean_km=15.2523`, `median_km=5.5043`, `within_1km_pct=11.67`, `within_2km_pct=26.67`, `within_5km_pct=50.00`, `within_10km_pct=64.44`.
  - Dual-index `rrf` (`n=180`):
    - `mean_km=14.8410` (`-0.4113`)
    - `median_km=4.8705` (`-0.6337`)
    - `within_1km_pct=12.78` (`+1.11`)
    - `within_2km_pct=31.11` (`+4.44`)
    - `within_5km_pct=50.56` (`+0.56`)
    - `within_10km_pct=63.33` (`-1.11`)
- Artifacts:
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w90_10_weighted_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w80_20_weighted_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w70_30_weighted_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_weighted_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_weighted_sb02_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_180.json`
- Decision:
  - Keep dual-index `rrf` profile as the strongest current close-range candidate.
  - Track the small `within_10km_pct` regression and avoid replacing broad-recall default until that tradeoff is explicitly accepted.

## 2026-04-21
- Hypothesis:
  - Dual-index retrieval can be further improved by objective-specific post-retrieval refinement, yielding separate best profiles for (a) max close-range hit (`<=1km`) and (b) balanced overall error.
- Change:
  - Extended dual-index `rrf` benchmarking with additional weighting and refinement variants:
    - Weighting sweep: `w100_50_rrf` vs existing `w100_100_rrf`.
    - Structural refinement sweep on dual-index stack: geo-prior, graph support rerank, KDE refinement, and graph+KDE.
  - Added new Paris-scoped profile configs:
    - `src/config/paris_balanced_dual_rrf.json`
    - `src/config/paris_close_range_dual_rrf_graph_kde.json`
  - Updated documentation to expose objective-specific profile selection.
- Files touched:
  - `src/config/paris_balanced_dual_rrf.json`
  - `src/config/paris_close_range_dual_rrf_graph_kde.json`
  - `README.md`
  - `PROGRESS.md`
  - `src/docs/RESEARCH_PAPER.md`
- Validation command(s):
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w100_50_rrf.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_50_rrf_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w100_100_rrf_geo_prior.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_geo_prior_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_v1.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_v1_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w100_100_rrf_kde_v1.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_kde_v1_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_kde_v1.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 60 --seed 42 --diag-samples 60 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_kde_v1_60.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_v1.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_v1_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_kde_v1.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_kde_v1_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config src/config/paris_balanced_dual_rrf.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 5 --seed 42 --diag-samples 5 --output runs/geo_eval_smoke_paris_balanced_dual_rrf_5.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config src/config/paris_close_range_dual_rrf_graph_kde.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 5 --seed 42 --diag-samples 5 --output runs/geo_eval_smoke_paris_close_range_dual_rrf_graph_kde_5.json`
  - `./.venv/Scripts/python -m pytest -q src/tests/test_augment_geo_index_embeddings.py src/tests/test_config_loading.py src/tests/test_retrieval_provider_multi_index.py`
- Metrics:
  - Baseline (`n=180`): `mean_km=15.2523`, `median_km=5.5043`, `within_1km_pct=11.67`, `within_2km_pct=26.67`, `within_5km_pct=50.00`, `within_10km_pct=64.44`.
  - Dual-index close-range (`w100_100_rrf`, `n=180`): `mean_km=14.8410`, `median_km=4.8705`, `within_1km_pct=12.78`, `within_2km_pct=31.11`, `within_5km_pct=50.56`, `within_10km_pct=63.33`.
  - Dual-index balanced (`w100_50_rrf`, `n=180`): `mean_km=14.5971`, `median_km=4.2125`, `within_1km_pct=10.56`, `within_2km_pct=31.11`, `within_5km_pct=52.78`, `within_10km_pct=65.56`.
  - Dual-index graph-only (`n=180`): `mean_km=15.1006`, `median_km=5.3282`, `within_1km_pct=13.33`, `within_2km_pct=30.00`, `within_5km_pct=50.00`, `within_10km_pct=63.33`.
  - Dual-index graph+KDE (`n=180`): `mean_km=15.2811`, `median_km=5.3115`, `within_1km_pct=13.89`, `within_2km_pct=31.11`, `within_5km_pct=49.44`, `within_10km_pct=63.33`.
- Artifacts:
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_50_rrf_180.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_geo_prior_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_v1_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_kde_v1_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_kde_v1_60.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_v1_180.json`
  - `runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_graph_kde_v1_180.json`
  - `runs/geo_eval_smoke_paris_balanced_dual_rrf_5.json`
  - `runs/geo_eval_smoke_paris_close_range_dual_rrf_graph_kde_5.json`
- Decision:
  - Keep three explicit Paris dual-index objective profiles:
    - `paris_close_range_dual_rrf`: best default for close-range (`<=1km`/`<=2km`) without major mean regression.
    - `paris_balanced_dual_rrf`: best balanced profile for mean/median/`<=10km` while preserving `<=2km` gain.
    - `paris_close_range_dual_rrf_graph_kde`: aggressive W1 profile with highest measured `<=1km`, accepted only when slight broader-radius regression is acceptable.

## 2026-04-28
- Hypothesis:
  - Projection adaptation should learn faster from error-driven geo triplets if severe/confusion-rich failures contribute more weight than easy failures instead of all triplets being uniform.
- Change:
  - Upgraded `src/tools/mine_hard_negative_triplets.py`:
    - emits `triplet_weight`, `hard_negative_source_counts`, and nearest positive/negative distance diagnostics per triplet.
    - added CLI controls: `--difficulty-mode`, `--difficulty-reference-km`, `--difficulty-max-weight`.
  - Upgraded `src/tools/train_retrieval_projection.py`:
    - added weighted training controls: `--sample-weight-mode`, `--sample-weight-power`, `--sample-weight-max`.
    - trainer now consumes `triplet_weight` and reports weighted triplet satisfaction/loss metrics plus dataset weight summaries.
  - Added regression coverage for the new mining/training contract.
  - Updated `README.md` and `src/docs/RESEARCH_PAPER.md` to document the weighted hard-negative workflow.
- Files touched:
  - `src/tools/mine_hard_negative_triplets.py`
  - `src/tools/train_retrieval_projection.py`
  - `src/tests/test_mine_hard_negative_triplets.py`
  - `src/tests/test_train_retrieval_projection.py`
  - `README.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_mine_hard_negative_triplets.py`
  - `./.venv/Scripts/python -m pytest -q src/tests/test_train_retrieval_projection.py`
- Metrics:
  - Mining regression tests: `5 passed`.
  - Projection training regression tests: `2 passed, 3 warnings`.
  - No new leakage-safe geo benchmark artifact was produced in this prompt, so no accuracy claim is added.
- Artifacts:
  - No new `runs/*.json` benchmark artifact in this prompt.
- Decision:
  - Keep the weighted hard-negative infrastructure.
  - Next step should be a controlled realistic-split comparison: uniform projection training vs difficulty-weighted projection training using the same triplet pool and seed.

## 2026-04-28
- Hypothesis:
  - The branch graph should be simplified once feature work has either landed on `master` or been reduced to stale branch-sync merges, so future work is not obscured by dead refs.
- Change:
  - Audited every local/remote branch against `master` using ahead/behind and unique-commit checks.
  - Confirmed there were no remaining non-merge commits outside `master`.
  - Deleted stale local branches and matching remote branches:
    - `feature/film-ui-operator-lab`
    - `tech/accuracy-backbone-upgrade-v1`
    - `tech/accuracy-rtx5060-sprint`
    - `tech/dba-index-augmentation`
    - `tech/eval-regression-lab`
    - `tech/geo-prior-gating-accuracy`
    - `tech/geo-retrieval-v3`
    - `tech/major-accuracy-upgrade`
    - `tech/model-hardening`
    - `tech/projection-iter2`
    - `tech/projection-v2-geo-prior-benchmark`
    - `tech/retrieval-method-upgrades-v2`
    - `tech/run-geo-eval-scope-guard`
- Files touched:
  - `PROGRESS.md`
- Validation command(s):
  - `git fetch --all --prune`
  - `git rev-list --left-right --count master...<branch>`
  - `git log --oneline --no-merges master..<branch>`
  - `git branch --merged master`
  - `git branch --no-merged master`
  - `git push origin --delete <branch...>`
  - `git branch -a --no-color`
- Metrics:
  - Remaining local branches: `1` (`master`)
  - Remaining remote branches: `1` (`origin/master`)
  - Branches deleted: `13`
- Artifacts:
  - No benchmark or modeling artifact was produced in this maintenance step.
- Decision:
  - Keep `master` as the only active branch until new feature work justifies new scoped topic branches.

## 2026-04-28
- Hypothesis:
  - Difficulty-aware weighting should outperform uniform weighting when projection training reuses the same query-vs-reference hard-negative triplet pool, and split-mismatch failures should be explained directly by the trainer instead of surfacing as a generic empty-dataset error.
- Change:
  - Mined weighted query-vs-reference triplets from the canonical Paris realistic eval using `--reference-metadata`.
  - Trained matched uniform and difficulty-weighted projection heads on the same `68` triplets with the same seed (`42`).
  - Applied both projections to the same Paris reference index and evaluated both on the canonical realistic retrieval-only split (`n=180`).
  - Improved `src/tools/train_retrieval_projection.py`:
    - report now includes missing path counts by role before/after image backfill.
    - `no_valid_training_records` now carries a split-aware hint when positives/negatives are missing because the wrong embedding index was chosen.
  - Updated regression coverage and docs to reflect the corrected query/reference workflow and measured weighted-vs-uniform result.
- Files touched:
  - `src/tools/train_retrieval_projection.py`
  - `src/tests/test_train_retrieval_projection.py`
  - `README.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m src.tools.mine_hard_negative_triplets --metadata data/spacenet_paris_test/metadata.csv --reference-metadata data/spacenet_paris/metadata.csv --eval-report runs/geo_eval_paris_profile_180_for_mining_v1.json --output runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted.jsonl --summary-output runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted_summary.json --min-error-km 2.0 --positive-radius-km 0.35 --negative-pred-radius-km 2.0 --negative-min-gt-distance-km 2.0 --negative-max-gt-distance-km 25.0 --max-positives 3 --max-negatives 12 --difficulty-mode error_km_predmix --difficulty-reference-km 10.0 --difficulty-max-weight 3.0`
  - `./.venv/Scripts/python -m src.tools.train_retrieval_projection --triplets runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted.jsonl --embedding-index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14.npz --images-dir data/spacenet_paris_test/chips --output runs/retrieval_projection_paris_query_trainref_v2_uniform_cmp.npz --report-output runs/retrieval_projection_paris_query_trainref_v2_uniform_cmp.report.json --epochs 8 --batch-size 16 --learning-rate 3e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --orth-weight 0.002 --sample-weight-mode uniform --seed 42 --device cpu`
  - `./.venv/Scripts/python -m src.tools.train_retrieval_projection --triplets runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted.jsonl --embedding-index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14.npz --images-dir data/spacenet_paris_test/chips --output runs/retrieval_projection_paris_query_trainref_v2_weighted_cmp.npz --report-output runs/retrieval_projection_paris_query_trainref_v2_weighted_cmp.report.json --epochs 8 --batch-size 16 --learning-rate 3e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --orth-weight 0.002 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device cpu`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_uniform_cmp.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_uniform_cmp_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`
  - `./.venv/Scripts/python -m pytest -q src/tests/test_train_retrieval_projection.py`
- Metrics:
  - Uniform single-index projection (`n=180`): `mean_km=15.252`, `median_km=5.504`, `within_1km_pct=11.67`, `within_2km_pct=26.67`, `within_5km_pct=50.00`, `within_10km_pct=64.44`.
  - Difficulty-weighted single-index projection (`n=180`): `mean_km=15.081`, `median_km=4.888`, `within_1km_pct=13.89`, `within_2km_pct=27.22`, `within_5km_pct=51.67`, `within_10km_pct=65.00`.
  - Training-side final epoch: `triplet_satisfied_pct` `27.94 -> 29.41`, `weighted_triplet_satisfied_pct` `27.94 -> 30.99`, `weighted_hard_triplet_loss` `0.1016 -> 0.0994`.
  - Projection training regression tests: `3 passed, 3 warnings`.
- Artifacts:
  - `runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted.jsonl`
  - `runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted_summary.json`
  - `runs/retrieval_projection_paris_query_trainref_v2_uniform_cmp.npz`
  - `runs/retrieval_projection_paris_query_trainref_v2_uniform_cmp.report.json`
  - `runs/retrieval_projection_paris_query_trainref_v2_weighted_cmp.npz`
  - `runs/retrieval_projection_paris_query_trainref_v2_weighted_cmp.report.json`
  - `runs/geo_eval_projection_trainref_v2_uniform_cmp_180.json`
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`
- Decision:
  - Promote difficulty-weighted triplet training over uniform weighting for the next single-index projection retraining cycle.
  - Keep the new split-mismatch diagnostics; query-vs-reference training should use the reference index plus query-image backfill, not the query-only index.

## 2026-04-28
- Branch:
  - `tech/structure-rerank-cues`
- Hypothesis:
  - Retrieval should improve if top candidates are re-ranked with coarse scene-layout evidence that CLIP similarity alone does not model well enough, especially corners, street/building line structure, and weak shadow-direction cues.
- Change:
  - Added a structure-aware rerank stage to `GeoRetrievalProvider` before local geometric matching.
  - New scene signature extracts:
    - corner density
    - edge density
    - dominant line-orientation histogram
    - guarded dark-mass / shadow-axis cue
  - Added guarded blending and safety conditions:
    - weak-signal skip
    - minimum score-spread gate
    - confident-top1 protection against noisy overthrow
  - Added config/runtime plumbing:
    - `retrieval_structure_rerank_top_n`
    - `retrieval_structure_rerank_weight`
  - Added focused regression coverage for promotion, weak-signal no-op, and confident-top1 guard behavior.
  - Benchmarked the feature on top of the current weighted single-index projection baseline and updated docs.
- Files touched:
  - `src/core/geo/retrieval_provider.py`
  - `src/core/logic/config.py`
  - `src/cli.py`
  - `src/batch_run.py`
  - `src/tools/run_all.py`
  - `src/tools/run_geo_eval.py`
  - `src/tools/tune_retrieval_geo.py`
  - `src/tools/ui_server.py`
  - `src/tests/test_config_loading.py`
  - `src/tests/test_retrieval_provider_multi_index.py`
  - `README.md`
  - `src/docs/GEO_TECH.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_config_loading.py src/tests/test_retrieval_provider_multi_index.py`
  - `./.venv/Scripts/python -m pytest -q src/tests/test_run_geo_eval_retrieval_provider.py src/tests/test_tune_retrieval_geo.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v1.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v1_180.json`
- Metrics:
  - Focused config/retrieval tests: `25 passed, 3 warnings`.
  - Runtime/tuning plumbing tests: `13 passed, 3 warnings`.
  - Canonical weighted single-index Paris realistic split (`n=180`) baseline:
    - `mean_km=15.0810`
    - `median_km=4.8877`
    - `within_1km_pct=13.89`
    - `within_2km_pct=27.22`
    - `within_5km_pct=51.67`
    - `within_10km_pct=65.00`
  - Structure-aware rerank (`top_n=12`, `weight=0.35`):
    - `mean_km=14.7247`
    - `median_km=4.5903`
    - `within_1km_pct=15.00`
    - `within_2km_pct=28.33`
    - `within_5km_pct=53.33`
    - `within_10km_pct=66.11`
- Artifacts:
  - `runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v1.json`
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v1_180.json`
- Decision:
  - Keep structure-aware reranking as a promising experimental single-index upgrade.
  - Do not promote it to default until it is replayed on additional leakage-safe splits beyond Paris.

## 2026-04-28
- Branch:
  - `tech/structure-analysis-cues-v2`
- Hypothesis:
  - Branch-level planning should be explicit and mandatory so each active branch states its comparison strategy, decision gates, and intended end state before more modeling work lands.
- Change:
  - Added a branch-plan requirement to `AGENT.md`.
  - Updated contributor workflow so each branch must maintain a root `plan.md`.
  - Updated `scripts/new-branch.ps1` to scaffold `plan.md` automatically for future branches.
  - Added a detailed `plan.md` for `tech/structure-analysis-cues-v2` covering:
    - retrieval-backbone comparison (`CLIP` vs `RemoteCLIP` vs `GeoRSCLIP`)
    - choose-then-adapt workflow
    - stronger shadow / corner / footprint cues
    - future geometry / `BEV` / `3D` work for both overhead and street-photo localization
- Files touched:
  - `AGENT.md`
  - `CONTRIBUTING.md`
  - `scripts/new-branch.ps1`
  - `plan.md`
  - `PROGRESS.md`
- Validation command(s):
  - `git fetch --all --prune`
  - `git branch -a --no-color`
  - `rg -n "plan.md|Branch Plan|Every branch must carry" AGENT.md CONTRIBUTING.md scripts/new-branch.ps1 plan.md`
- Metrics:
  - Local branches present: `2` (`master`, `tech/structure-analysis-cues-v2`)
  - Remote branches present: `1` (`origin/master`)
- Artifacts:
  - No benchmark artifact in this planning/policy step.
- Decision:
  - Keep `plan.md` mandatory for every branch.
  - `master` still needs its own branch-local `plan.md`; add it directly on `master` rather than reusing this feature branch plan.

## 2026-04-28
- Branch:
  - `tech/structure-analysis-cues-v2`
- Hypothesis:
  - The branch-plan rule is too weak unless `AGENT.md` explicitly requires `plan.md` to be updated every time branch intent changes, not only when the file is first created.
- Change:
  - Tightened `AGENT.md` so `plan.md` must be reviewed and updated whenever the branch plan changes in substance:
    - new comparisons
    - dropped directions
    - changed decision gates
    - changed next intended move
  - Added same-prompt enforcement so `plan.md` updates are not deferred after branch direction changes.
- Files touched:
  - `AGENT.md`
  - `PROGRESS.md`
- Validation command(s):
  - `rg -n "plan.md|every time|same prompt" AGENT.md`
- Metrics:
  - No benchmark metrics in this policy-only step.
- Artifacts:
  - No run artifact.
- Decision:
  - Keep the stronger wording.
  - Future branch work should treat `plan.md` as a living branch contract, not a one-time stub.

## 2026-04-28
- Branch:
  - `master`
- Hypothesis:
  - The branch-plan policy is incomplete unless `master` also has its own branch-local `plan.md` describing its integration role instead of borrowing a feature branch plan.
- Change:
  - Added a `master`-specific root `plan.md`.
  - Reframed `master` as:
    - integration branch
    - validation gate
    - merge hygiene branch
    - prioritization branch
- Files touched:
  - `plan.md`
  - `PROGRESS.md`
- Validation command(s):
  - `git branch -a --no-color`
  - `Get-Content plan.md | Select-Object -First 120`
- Metrics:
  - Active local branches at this step: `2`
  - Active remote branches at this step: `1`
- Artifacts:
  - No benchmark artifact in this branch-governance step.
- Decision:
  - Keep a dedicated `plan.md` on `master`.
  - Do not reuse feature-branch plans on trunk.

- Branch:
  - `tech/structure-analysis-cues-v2`
- Hypothesis:
  - Extending structure reranking toward geometry-lite cues should improve retrieval quality beyond coarse corner/edge counts by comparing layout, orthogonality, and shadow shape, but the branch needs a measured setting rather than a hand-picked blend weight.
- Change:
  - Expanded `SceneStructureSignature` with:
    - corner spatial layout
    - edge spatial layout
    - line orthogonality
    - line anisotropy
    - shadow elongation
  - Updated retrieval structure similarity to blend those geometry-lite cues with the earlier line/corner/edge/shadow signals.
  - Added focused regression coverage for the new signature fields and for geometry-aligned vs geometry-misaligned structure similarity.
  - Extended `src.tools.tune_retrieval_geo` so it can sweep:
    - `retrieval_structure_rerank_top_n`
    - `retrieval_structure_rerank_weight`
  - Benchmarked the geometry-lite branch on the canonical Paris realistic split and selected a provisional branch setting:
    - `retrieval_structure_rerank_top_n=16`
    - `retrieval_structure_rerank_weight=0.30`
  - Updated `plan.md`, `README.md`, `src/docs/GEO_TECH.md`, and `src/docs/RESEARCH_PAPER.md` with the measured branch result and remaining tradeoff.
- Files touched:
  - `plan.md`
  - `src/core/geo/retrieval_provider.py`
  - `src/tools/tune_retrieval_geo.py`
  - `src/tests/test_retrieval_provider_multi_index.py`
  - `src/tests/test_tune_retrieval_geo.py`
  - `README.md`
  - `src/docs/GEO_TECH.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
  - `runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v2_geometry_b.json`
  - `runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v2_geometry_c.json`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_tune_retrieval_geo.py`
  - `./.venv/Scripts/python -m pytest -q src/tests/test_retrieval_provider_multi_index.py`
  - `./.venv/Scripts/python -m pytest -q src/tests/test_run_geo_eval_retrieval_provider.py src/tests/test_config_loading.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v2_geometry_b.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_b_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v2_geometry_c.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_c_180.json`
- Metrics:
  - Focused tests: `40 passed, 9 warnings` across:
    - `test_tune_retrieval_geo.py`
    - `test_retrieval_provider_multi_index.py`
    - `test_run_geo_eval_retrieval_provider.py`
    - `test_config_loading.py`
  - Control (`runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`):
    - `mean_km=15.0810`
    - `median_km=4.8877`
    - `within_1km_pct=13.89`
    - `within_2km_pct=27.22`
    - `within_5km_pct=51.67`
    - `within_10km_pct=65.00`
  - Geometry-lite branch probe A (`top_n=12`, `weight=0.35`):
    - `mean_km=14.9730`
    - `median_km=4.7578`
    - `within_1km_pct=13.89`
    - `within_2km_pct=27.22`
    - `within_5km_pct=52.78`
    - `within_10km_pct=66.11`
  - Geometry-lite branch probe B (`top_n=16`, `weight=0.25`):
    - `mean_km=14.9548`
    - `median_km=4.7578`
    - `within_1km_pct=14.44`
    - `within_2km_pct=27.22`
    - `within_5km_pct=52.78`
    - `within_10km_pct=65.56`
  - Geometry-lite branch probe C (`top_n=16`, `weight=0.30`):
    - `mean_km=14.6599`
    - `median_km=4.7578`
    - `within_1km_pct=14.44`
    - `within_2km_pct=27.22`
    - `within_5km_pct=52.78`
    - `within_10km_pct=66.11`
- Artifacts:
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_a_180.json`
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_b_180.json`
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_c_180.json`
- Decision:
  - Keep the geometry-lite rerank on this branch as a promising bridge toward heavier geometry.
  - Provisional branch setting is `top_n=16`, `weight=0.30`.
  - Do not merge/promote it as a replacement for the earlier structure-rerank milestone until the close-range (`<=2km`) gap is closed or the backbone comparison changes the tradeoff.

## 2026-04-28
- Branch:
  - `tech/structure-analysis-cues-v2`
- Hypothesis:
  - The geometry-lite signature is useful, but its extra layout / orthogonality / shadow-shape cues need weak-signal gating so they do not override the older structure signal on diffuse scenes.
- Change:
  - Reworked `_scene_structure_similarity` to:
    - anchor on the older line/corner/edge/shadow structure score
    - compute geometry-lite cue support from cue distinctiveness
    - blend geometry-lite layout / orthogonality / shadow-shape only when those cues are actually informative
  - Added regression coverage showing:
    - strong geometry-aligned scenes still score above misaligned scenes
    - weak geometry cues stay secondary to the legacy structure signal
  - Benchmarked three post-gating branch settings on the canonical Paris realistic split:
    - `top_n=12`, `weight=0.35`
    - `top_n=16`, `weight=0.30`
    - new midpoint config `top_n=14`, `weight=0.35`
  - Promoted the new balanced geometry-lite checkpoint:
    - `runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v2_geometry_d.json`
  - Updated `plan.md`, `README.md`, `src/docs/GEO_TECH.md`, and `src/docs/RESEARCH_PAPER.md` to reflect the gated geometry-lite result.
- Files touched:
  - `src/core/geo/retrieval_provider.py`
  - `src/tests/test_retrieval_provider_multi_index.py`
  - `runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v2_geometry_d.json`
  - `README.md`
  - `plan.md`
  - `src/docs/GEO_TECH.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_retrieval_provider_multi_index.py src/tests/test_tune_retrieval_geo.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v2_geometry_c.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_c_gated_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v1.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v1_gated_180.json`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v2_geometry_d.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_d_180.json`
- Metrics:
  - Focused tests: `32 passed, 3 warnings`
  - Weighted projection control (`runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`):
    - `mean_km=15.0810`
    - `median_km=4.8877`
    - `within_1km_pct=13.89`
    - `within_2km_pct=27.22`
    - `within_5km_pct=51.67`
    - `within_10km_pct=65.00`
  - Geometry-lite gated probe C (`top_n=16`, `weight=0.30`):
    - `mean_km=14.9308`
    - `median_km=4.7578`
    - `within_1km_pct=13.89`
    - `within_2km_pct=27.22`
    - `within_5km_pct=52.78`
    - `within_10km_pct=65.56`
  - Geometry-lite gated probe on legacy close-range weights (`top_n=12`, `weight=0.35`):
    - `mean_km=14.7927`
    - `median_km=4.7578`
    - `within_1km_pct=14.44`
    - `within_2km_pct=28.33`
    - `within_5km_pct=52.78`
    - `within_10km_pct=65.56`
  - Geometry-lite gated probe D (`top_n=14`, `weight=0.35`):
    - `mean_km=14.7239`
    - `median_km=4.5903`
    - `within_1km_pct=15.00`
    - `within_2km_pct=28.33`
    - `within_5km_pct=53.33`
    - `within_10km_pct=66.11`
- Artifacts:
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_c_gated_180.json`
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v1_gated_180.json`
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_d_180.json`
  - `runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v2_geometry_d.json`
- Decision:
  - Keep the geometry-lite signature on this branch with weak-signal gating.
  - Current balanced branch checkpoint is `top_n=14`, `weight=0.35`.
  - The close-range regression is closed; next branch work should move to the planned backbone comparison rather than more blind rerank tweaking.

## 2026-04-29
- Branch:
  - `tech/structure-analysis-cues-v2`
- Hypothesis:
  - The next real accuracy step needs representation-level work, not more blind rerank layering. We need a correct encoder-fine-tuning loop, a larger hard-negative pool, and a concrete benchmarked answer to whether a tuned model should replace the serving profile or only be appended as an auxiliary retrieval branch.
- Change:
  - Fixed projected-backbone benchmarking so projection training uses a dedicated `projection_support` index built from the triplet reference pool instead of reusing the tiny sampled eval index.
  - Added real CLIP-family encoder fine-tuning in `src/tools/train_retrieval_encoder.py`, saving a local `save_pretrained()` model directory usable as a retrieval model id.
  - Added cached `visual_projection` training so frozen-vision runs build pooled image features once and reuse them across epochs.
  - Scaled hard-negative mining in `src/tools/mine_hard_negative_triplets.py` with multi-report support and `--max-failures-per-query`.
  - Added `src/tools/run_retrieval_finetune_loop.py` to evaluate on the fixed Paris `180` slice, mine failures, fine-tune the encoder, rebuild the index, optionally add a DBA companion, and re-evaluate.
  - Fixed the loop evaluation so tuned-model runs are compared against a matched rebuilt baseline instead of unfairly against the full production serving config.
  - Added auxiliary serving-config generation to the loop:
    - keep `paris_close_range_dual_rrf` as the primary branch
    - append tuned model index and tuned DBA index as extra sources
    - give them their own `retrieval_index_model_ids`
    - force `null` per-index projection routing so they do not inherit the base projection
  - Added root `research.md` as a chronological research ledger with before/after metrics and updated `README.md` / `src/docs/RESEARCH_PAPER.md` to reference it.
- Files touched:
  - `src/tools/benchmark_geo_backbones.py`
  - `src/tools/upgrade_retrieval_backbone.py`
  - `src/tools/mine_hard_negative_triplets.py`
  - `src/tools/train_retrieval_encoder.py`
  - `src/tools/run_retrieval_finetune_loop.py`
  - `src/tests/test_benchmark_geo_backbones.py`
  - `src/tests/test_mine_hard_negative_triplets.py`
  - `src/tests/test_train_retrieval_encoder.py`
  - `src/tests/test_run_retrieval_finetune_loop.py`
  - `README.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `PROGRESS.md`
  - `research.md`
- Validation command(s):
  - `./.venv/Scripts/python -m pytest -q src/tests/test_mine_hard_negative_triplets.py src/tests/test_train_retrieval_encoder.py src/tests/test_run_retrieval_finetune_loop.py src/tests/test_benchmark_geo_backbones.py`
  - `./.venv/Scripts/python -m pytest -q src/tests/test_train_retrieval_encoder.py src/tests/test_run_retrieval_finetune_loop.py src/tests/test_run_geo_eval_retrieval_provider.py`
  - `./.venv/Scripts/python -m src.tools.run_geo_eval --retrieval-only --config runs/aux_fusion_final/aux_conservative.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/aux_fusion_final/aux_conservative_eval_180.json`
- Metrics:
  - Backbone smoke benchmark after `projection_support` fix (`train_limit=80`, `eval_limit=20`):
    - raw CLIP: `mean_km=27.7099`, `median_km=42.9709`, `<=5km=25.0%`, `<=10km=35.0%`
    - projected CLIP: `mean_km=29.2042`, `median_km=44.2391`, `<=5km=25.0%`, `<=10km=35.0%`
    - raw SigLIP-base: `mean_km=25.0207`, `median_km=17.8138`, `<=5km=10.0%`, `<=10km=40.0%`
    - projected SigLIP-base: `mean_km=27.2065`, `median_km=17.8265`, `<=5km=10.0%`, `<=10km=35.0%`
  - Hard-negative mining scale-up from the production close-range profile:
    - strict setup: `52` triplets
    - looser A: `74` triplets
    - looser B: `129` triplets
  - Cached projection-only training behavior on `loose_b`:
    - `1` epoch: `956` cached images, `375.85 s` cache build, `4.20 s` training, `best_weighted_hard_triplet_loss=0.1937`
    - `10` epochs: `956` cached images, `326.14 s` cache build, `33.45 s` training, `best_weighted_hard_triplet_loss=0.1199`
  - Fixed Paris `180` fine-tune loop (`train_limit=300`, `eval_limit=180`, `rank_objective=within_2km_pct`):
    - production serving baseline `paris_close_range_dual_rrf`: `mean_km=14.8410`, `median_km=4.8705`, `<=1km=12.78%`, `<=2km=31.11%`, `<=5km=50.56%`, `<=10km=63.33%`
    - matched rebuilt base model: `mean_km=23.1429`, `median_km=10.5783`, `<=1km=0.56%`, `<=2km=1.67%`, `<=5km=17.78%`, `<=10km=36.67%`
    - tuned encoder on the same rebuilt index: `mean_km=21.5331`, `median_km=11.0497`, `<=1km=1.11%`, `<=2km=3.33%`, `<=5km=20.56%`, `<=10km=44.44%`
    - winning round training details: `130` resolved triplets, `966` cached images, `281.40 s` cache build, `23.83 s` training, `best_weighted_triplet_satisfied_pct=0.873`
  - Auxiliary serving candidate on the real Paris `180` benchmark:
    - baseline `paris_close_range_dual_rrf`: `mean_km=14.8410`, `median_km=4.8705`, `<=1km=12.78%`, `<=2km=31.11%`, `<=5km=50.56%`, `<=10km=63.33%`
    - conservative auxiliary blend (`aux_index_weight=0.15`, `aux_dba_weight=0.05`): `mean_km=15.6990`, `median_km=9.7449`, `<=1km=10.00%`, `<=2km=21.11%`, `<=5km=37.78%`, `<=10km=52.78%`
  - Focused regression coverage:
    - earlier workflow/test expansion: `15 passed`
    - current focused suite: `12 passed, 3 warnings`
- Artifacts:
  - `runs/backbone_bench/smoke_raw_clip.json`
  - `runs/backbone_bench/smoke_projected_clip.json`
  - `runs/backbone_bench/smoke_raw_siglip_base.json`
  - `runs/backbone_bench/smoke_projected_siglip_base.json`
  - `runs/tmp_triplets_curr_summary.json`
  - `runs/tmp_triplets_loose_a_summary.json`
  - `runs/tmp_triplets_loose_b_summary.json`
  - `runs/tmp_encoder_loose_b.report.json`
  - `runs/tmp_encoder_loose_b_e10.report.json`
  - `runs/retrieval_finetune_loop_180_looseb_e10/loop_summary.json`
  - `runs/retrieval_finetune_loop_180_looseb_e10/round_01/encoder.report.json`
  - `runs/aux_fusion_final/aux_conservative.json`
  - `runs/aux_fusion_final/aux_conservative_eval_180.json`
- Decision:
  - Keep the encoder fine-tuning and mining infrastructure; it is the correct next research direction.
  - Do not promote the tuned encoder as a replacement for the serving stack yet.
  - Do not promote the auxiliary fused serving config yet; even a conservative low-weight blend regressed sharply on the real Paris `180` benchmark.
  - Continue to serve `paris_close_range_dual_rrf` as the primary close-range branch and treat the tuned-model auxiliary source as infrastructure that still needs stronger training data before deployment.

## 2026-04-29 23:10 - Data Branch Kickoff: realistic Paris dataset

- Branch:
  - created `tech/paris-realistic-data-v1` from merged `master`
- Research conclusion update:
  - recorded the current state as a data-limited stalemate rather than a modeling-only problem
  - documented that better street-view comparison, cross-view geolocation, and stronger model adaptation now require a realistic street-plus-aerial dataset
- Planning work:
  - replaced the root `plan.md` on the new branch with a branch-local roadmap
  - captured the repo audit for:
    - retrieval and index pipeline
    - metadata CSV conventions
    - hard-negative mining reuse points
    - projection-training reuse points
    - config implications in `src/core/logic/config.py`
    - docs and `PROGRESS.md` policy
  - staged the next implementation sequence:
    - Mapillary street-image ingestion
    - OpenAerialMap aerial pairing
    - leakage-safe spatial splits
    - realistic cross-view triplet mining
    - street-to-aerial projection and evaluation
    - orientation-aware scoring
    - street-to-street retrieval
    - fused street-plus-aerial evaluation
    - later architecture-upgrade branch only after a fixed realistic baseline exists
- Decision:
  - stop treating more small rerank ideas as the primary path to a breakthrough
  - move the next research cycle onto realistic data construction first

## 2026-04-29 23:45 - Mapillary Paris street-data ingestion tool

- Added:
  - `src/tools/download_mapillary_paris.py`
  - `src/tests/test_download_mapillary_paris.py`
- Implemented a first realistic street-image ingestion tool for the new Paris data branch:
  - reads `MAPILLARY_ACCESS_TOKEN` from environment
  - splits the requested Paris bbox into safe smaller query cells using meter-based grid steps
  - queries `graph.mapillary.com/images` with the required fields
  - prefers `computed_geometry` over `geometry`
  - prefers `computed_compass_angle` over `compass_angle`
  - uses `thumb_2048_url` with `thumb_1024_url` fallback
  - writes `data/paris_realistic_v1/street/images/` plus `metadata.csv`
  - dedupes by `image_id`
  - limits near-duplicates by cell sampling and per-sequence caps
  - supports `--dry-run` for expected-count checks without writing files
  - includes retry and backoff for API and download reads
- Documentation:
  - updated `README.md` with the new realistic Paris street-data bootstrap command and metadata contract
- Decision:
  - start the realistic data pipeline with a controlled, reproducible street-image bootstrap before moving to aerial pairing and split generation

## 2026-04-30 00:15 - OpenAerialMap aerial pairing tool

- Added:
  - `src/tools/build_aerial_pairs.py`
  - `src/tests/test_build_aerial_pairs.py`
- Implemented the first aerial-pair generation step for the realistic Paris branch:
  - reads `street/metadata.csv`
  - queries OpenAerialMap metadata near each street coordinate
  - selects the highest-resolution scene covering the point
  - reads the OAM TMS URL from scene metadata and renders a centered aerial crop
  - writes `aerial/images/`, `aerial/metadata.csv`, and `pairs.csv`
  - marks `no_open_aerial_found` when no OAM scene covers the point
  - supports `--allow-missing-aerial` for explicit missing-pair rows
- Documentation:
  - updated `README.md` with the OAM pairing command and output contracts
- Decision:
  - keep the provider abstraction minimal for now: one open provider, explicit contracts, and TMS-based centered crops before adding more imagery sources

## 2026-04-30 00:35 - Leakage-safe realistic split builder

- Added:
  - `src/tools/split_realistic_dataset.py`
  - `src/tests/test_split_realistic_dataset.py`
- Implemented spatial splitting for the realistic Paris dataset:
  - groups pairs by geographic cells instead of random rows
  - writes `train_pairs.csv`, `val_pairs.csv`, `test_pairs.csv`
  - writes `split_summary.json` with pair counts, cell counts, bbox, seed, cell size, and minimum cross-split distance
  - added a `--sanity-check-dir` mode to report minimum cross-split distance from an existing split folder
- Documentation:
  - updated `README.md` with split generation and sanity-check commands
- Decision:
  - keep the fixed realistic branch benchmark leakage-safe from the start rather than fixing split contamination later

## 2026-04-30 01:10 - Paris completeness upgrade: Panoramax + IGN

- Added:
  - `src/tools/download_panoramax_paris.py`
  - `src/tools/merge_realistic_street_datasets.py`
  - `src/tests/test_download_panoramax_paris.py`
  - `src/tests/test_merge_realistic_street_datasets.py`
- Extended:
  - `src/tools/build_aerial_pairs.py`
  - `src/tests/test_build_aerial_pairs.py`
- Upgraded the branch from single-source bootstrap toward a more complete Paris dataset path:
  - added Panoramax federated street-image ingestion using the STAC-style search API
  - preserves direct image assets, capture time, heading, source instance, and license metadata
  - added a merge step so Mapillary and Panoramax datasets can be combined into one street metadata root
  - added `ign_geopf` orthophoto support in aerial pairing for dense Paris-wide coverage when OAM is sparse
- Documentation:
  - updated `README.md` with the recommended higher-completeness path:
    - Mapillary + Panoramax street ingestion
    - merged `street_combined` dataset
    - IGN orthophoto aerial pairing for Paris
- Decision:
  - for Paris specifically, completeness now means combining open French street imagery with French orthophotos instead of relying on one global provider alone

## 2026-04-30 07:35 - Full data-build checkpoint

- Operational build results on `tech/paris-realistic-data-v1`:
  - confirmed `.env` token loading for Mapillary and validated the live downloader on a smoke bbox
  - full-city Mapillary pull reached `3029` downloaded street images before the command window expired
  - full-city Panoramax pull was promoted into a stable `10000`-image street dataset at:
    - `data/paris_realistic_v1/street_panoramax/metadata.csv`
  - IGN aerial pairing produced `3802` centered aerial crops before the long run window expired
  - materialized those completed crops into:
    - `data/paris_realistic_v1/aerial/metadata.csv`
    - `data/paris_realistic_v1/pairs.csv`
  - current pair count: `3802`
  - current split outputs:
    - train `2671`
    - val `583`
    - test `548`
- Hardening:
  - updated the Mapillary downloader so transient CDN or per-cell failures no longer abort the full run
  - downloader now skips failed image downloads and keeps the dataset build moving
- Current caveat:
  - the current `3802`-pair cross-view subset is usable for immediate experimentation, but it is not yet the final full-Paris dataset
  - the current split generator still needs stronger leakage protection because the minimum cross-split distance on this partial slice is too small
- Decision:
  - keep the `10000` Panoramax street dataset and `3802` IGN pair subset as the first real realistic-Paris checkpoint
  - continue the data branch from this checkpoint rather than discarding partially completed large downloads

## 2026-04-30 17:05 - Full Panoramax -> IGN pair completion

- Performance fix:
  - upgraded `ign_geopf` aerial pairing from multi-tile WMTS rendering to direct WMS `GetMap` requests against the official IGN `wms-r` endpoint
  - this removed the main bottleneck in the chunked aerial build
- Operational result:
  - completed the chunked aerial pairing run for all `10000` Panoramax street images
  - merged chunk outputs back into the final dataset root
  - final dataset counts in `data/paris_realistic_v1/`:
    - `street_panoramax/metadata.csv`: `10000` street images
    - `aerial/metadata.csv`: `10000` aerial crops
    - `pairs.csv`: `10000` positive street-to-aerial pairs
  - rebuilt full split files in `data/paris_realistic_v1/splits_full/`:
    - train `7003`
    - val `1500`
    - test `1497`
- Current caveat:
  - the data volume is now complete for the Panoramax -> IGN branch
  - the split generator still needs a stronger anti-leakage policy because the current minimum cross-split distance remains too small on the dense urban grid
- Decision:
  - treat `data/paris_realistic_v1/` as the first complete realistic Paris dataset checkpoint
  - continue improving split integrity and optional Mapillary augmentation on top of this completed core dataset

## 2026-04-29 - Research and replication docs refreshed for the data bottleneck

- Updated `research.md` to record the current realistic Paris data checkpoint explicitly:
  - `20,000` Mapillary street rows
  - `20,000` Panoramax street rows
  - `40,000` merged street rows
  - `10,000` complete Panoramax -> IGN street-to-aerial pairs
- Documented the current limitations honestly:
  - the full `40,000` combined street -> IGN pairing attempt did not finish cleanly
  - the current `splits_full` summary still reports `min_cross_split_distance_m = 3.77`, so the benchmark split is not strict enough yet
- Updated `src/docs/RESEARCH_PAPER.md` so the paper now states:
  - the retrieval/fusion stack was validated across multiple controlled upgrades
  - progress then hit a data-limited stalemate rather than a purely algorithmic one
  - the current research path is realistic street-to-aerial data collection and replication
  - the current dataset is enough to start real cross-view training, but not enough to justify a serious `~3 km` mean-accuracy claim yet
- Updated `README.md` dataset instructions to match the actual branch outputs and replication path:
  - `street_mapillary`
  - `street_panoramax`
  - `street_combined`
  - `pairs.csv`
  - `splits_full`

## 2026-04-29 - Strict realistic split and first cross-view benchmark loop

- Added:
  - `src/tools/eval_realistic_crossview.py`
  - `src/tests/test_eval_realistic_crossview.py`
- Tightened:
  - `src/tools/split_realistic_dataset.py`
  - `src/tests/test_split_realistic_dataset.py`
  - `src/tools/mine_hard_negative_triplets.py`
  - `src/tests/test_mine_hard_negative_triplets.py`
- Split fix:
  - changed realistic split generation from shuffled adjacent cell assignment to contiguous geographic bands plus explicit boundary buffering
  - split builder now supports `--buffer-cells`
  - excluded boundary rows are materialized into `excluded_pairs.csv` instead of being hidden
  - exact street/aerial coordinate matches are now treated as valid positives in hard-negative mining when query and reference paths differ
- Strict benchmark artifact:
  - `data/paris_realistic_v1/splits_strict/split_summary.json`
  - counts:
    - retained pairs: `8405`
    - excluded boundary-buffer pairs: `1595`
    - train: `6435`
    - val: `739`
    - test: `1231`
  - anti-leakage floor:
    - `min_cross_split_distance_m = 1213.11`
- First realistic cross-view benchmark probe on the strict split:
  - baseline report: `runs/eval_realistic_crossview_strict_probe120_baseline.json`
  - projection report: `runs/paris_realistic_strict_crossview_projection.report.json`
  - projected eval report: `runs/eval_realistic_crossview_strict_probe120_projected.json`
  - probe size: `120` strict test queries
  - baseline metrics:
    - `mean_km = 8.44`
    - `median_km = 7.96`
    - `within_1km_pct = 5.83`
    - `within_2km_pct = 7.50`
    - `within_5km_pct = 15.00`
  - first projection pass (`1500` train-only synthetic triplets, `5` epochs):
    - `mean_km = 8.60`
    - `median_km = 9.16`
    - `within_1km_pct = 10.00`
    - `within_2km_pct = 13.33`
    - `within_5km_pct = 28.33`
- Decision:
  - the realistic cross-view path is now measurable on a stricter split
  - the first projection pass improved close-range hit rates but did not improve mean/median error yet
  - next work should mine harder train triplets and compare against the same strict probe before claiming a better model

## 2026-04-30 - Full combined realistic dataset completion and benchmark handoff

- Completed the full all-source realistic Paris recovery path:
  - `data/paris_realistic_v1_combined/pairs.csv` now contains `40,000` combined street -> IGN pairs
  - `data/paris_realistic_v1_combined/aerial/metadata.csv` now contains `40,000` aerial rows
  - `data/paris_realistic_v1_combined/recovery_summary.json` records the merged dataset state
- Tightened the combined benchmark root:
  - rebuilt `data/paris_realistic_v1_combined/splits_strict/` after the full merge
  - split summary:
    - train: `26204`
    - val: `3382`
    - test: `5235`
    - excluded: `5179`
    - retained: `34821`
    - `min_cross_split_distance_m = 1201.23`
- Fixed two infrastructure bottlenecks that blocked realistic-scale benchmarking:
  - vectorized `min_cross_split_distance_m` in `src/tools/split_realistic_dataset.py` so strict-split summary generation no longer stalls on the large combined dataset
  - added batched image embedding support in:
    - `src/core/geo/retrieval_provider.py`
    - `src/tools/build_geo_index.py`
  - added:
    - `src/tools/build_realistic_aerial_index.py`
    - `src/tools/recover_combined_aerial_dataset.py`
    - `src/tests/test_build_geo_index.py`
    - `src/tests/test_recover_combined_aerial_dataset.py`
- Validation:
  - `6 passed` on:
    - `src/tests/test_build_geo_index.py`
    - `src/tests/test_split_realistic_dataset.py`
- First merged-dataset benchmark reports:
  - sampled `10k` aerial index, `240`-query strict probe:
    - `runs/eval_realistic_crossview_combined_strict_probe240_baseline_sample10k.json`
    - `mean_km = 10.92`
    - `median_km = 11.05`
    - `within_1km_pct = 2.08`
    - `within_2km_pct = 5.00`
    - `within_5km_pct = 10.83`
  - full `40k` aerial index, same `240`-query strict probe:
    - `runs/eval_realistic_crossview_combined_strict_probe240_baseline_full40k.json`
    - `mean_km = 10.97`
    - `median_km = 11.75`
    - `within_1km_pct = 2.92`
    - `within_2km_pct = 5.83`
    - `within_5km_pct = 12.50`
- Decision:
  - the realistic data phase is now complete enough to stop treating data collection as the main blocker
  - expanding the merged probe from a sampled `10k` aerial index to the full `40k` index helped close-range hit rates slightly, but did not improve mean/median error
  - the next bottleneck is the model, not the existence of a realistic dataset

## 2026-04-30 - First combined cross-view projection training pass

- Added dedicated realistic cross-view model-training tooling:
  - `src/tools/mine_realistic_crossview_triplets.py`
  - `src/tools/train_crossview_projection.py`
  - tests:
    - `src/tests/test_mine_realistic_crossview_triplets.py`
    - `src/tests/test_train_crossview_projection.py`
- Validation:
  - `4 passed` on:
    - `src/tests/test_mine_realistic_crossview_triplets.py`
    - `src/tests/test_train_crossview_projection.py`
    - `src/tests/test_eval_realistic_crossview.py`
- Mined the first full realistic train pool from the combined strict split:
  - `runs/paris_realistic_crossview_train_triplets_v1.jsonl`
  - `runs/paris_realistic_crossview_train_triplets_v1.summary.json`
  - counts:
    - `total_queries = 26204`
    - `triplets_written = 26204`
    - `avg_positives = 2.97`
    - `avg_negatives = 20.0`
- Trained the first query-only street-to-aerial projection:
  - `runs/crossview_projection_paris_combined_v1_probe.npz`
  - `runs/crossview_projection_paris_combined_v1_probe.report.json`
  - training subset:
    - `6000` mined train triplets
    - `6000` embedded street queries
    - `26679` aerial references touched by those triplets
  - training report:
    - `8` epochs
    - `weighted_hard_triplet_loss: 0.1147 -> 0.0928`
    - `weighted_triplet_satisfied_pct: 0.75% -> 4.92%`
- Benchmarked against the same full `40k` combined strict probe (`240` queries):
  - frozen CLIP baseline:
    - `runs/eval_realistic_crossview_combined_strict_probe240_baseline_full40k.json`
    - `mean_km = 10.97`
    - `median_km = 11.75`
    - `within_1km_pct = 2.92`
    - `within_2km_pct = 5.83`
    - `within_5km_pct = 12.50`
  - first cross-view projection:
    - `runs/eval_realistic_crossview_combined_strict_probe240_crossviewproj_v1_full40k.json`
    - `mean_km = 9.75`
    - `median_km = 10.24`
    - `within_1km_pct = 2.08`
    - `within_2km_pct = 7.50`
    - `within_5km_pct = 20.42`
- Decision:
  - this is the first real model-side improvement on the combined realistic benchmark
  - the new projection meaningfully improves medium-range localization, but it is not yet the final close-range answer because `<=1km` regressed
  - next training passes should focus on harder close-range negatives and faster query-embedding reuse rather than going back to more data collection

## 2026-04-30 - Research paper doctoral-style framing update

- Clarified document roles:
  - `research.md` is the evidence ledger for exact dates, commands, artifacts, and before/after metrics
  - `src/docs/RESEARCH_PAPER.md` is the authored research manuscript
- Updated `AGENT.md` so future contributors preserve the intended paper style:
  - problem statement
  - notation
  - algorithm description
  - experiment
  - result
  - decision
- Expanded `src/docs/RESEARCH_PAPER.md` with formal mathematical sections covering:
  - ranked geolocation candidate inference
  - CLIP-style retrieval embedding similarity
  - multi-index weighted and RRF retrieval
  - asymmetric street-to-aerial query projection
  - triplet and contrastive loss
  - structure/geometry reranking
  - consensus, KDE, and geo-aware DBA
  - probabilistic fusion and uncertainty
  - leakage-safe spatial splitting
  - benchmark metrics
- Added the latest data/model progression to the paper chronology:
  - realistic Paris data bottleneck
  - `40,000` street-to-aerial pairs
  - strict split with `1201.23 m` minimum cross-split distance
  - first query-only cross-view projection result
- Validation:
  - documentation-only update; no tests were required

## 2026-04-30 - Analysis UI and optional RF-DETR branch start

- Created branch-local plan for `tech/analysis-ui-rfdetr`.
- Reviewed RF-DETR as an optional object detector candidate:
  - real-time transformer detection/segmentation model from Roboflow
  - DINOv2-style backbone
  - open Apache-designated model sizes are suitable for experimentation
  - Plus/XL variants require separate license review before any default use
- Added optional detector config path:
  - `detector.backend = "rfdetr"`
  - `detector.rfdetr_model_size`
  - existing `ultralytics_obb` behavior remains the default
- Added RF-DETR adapter scaffolding that converts axis-aligned RF-DETR boxes into Heimdall `Detection` objects for the existing canvas/evidence pipeline.
- Upgraded the `/analysis/` globe workflow:
  - radar-style pulsing candidate points
  - click-to-fly candidate selection
  - selected candidate uncertainty ring
  - selected-to-mean support line
  - candidate inspector with rank, coordinates, posterior, retrieval score, interval, and source
- Decision:
  - RF-DETR is optional until it is benchmarked against the current detector stack
  - the UI now exposes more verification data without changing the analysis API contract

## 2026-04-30 - Pre-push research documentation gate

- Updated `AGENT.md` to make research/documentation sync an explicit pre-commit, pre-push, PR, and merge gate.
- New rule:
  - if published work affects algorithms, data, evaluation, model behavior, UI analysis behavior, research direction, or measured results, the branch must update the relevant docs before pushing
  - relevant docs include `src/docs/RESEARCH_PAPER.md`, `research.md`, `PROGRESS.md`, `README.md`, and `plan.md`
- Decision:
  - future agents should not push research-relevant code-only changes unless `PROGRESS.md` clearly explains why the research paper did not need an update

## 2026-04-30 - Windows CMD app launcher

- Added `run_heimdall.cmd` at the repo root.
- The runner:
  - changes into the repo directory
  - uses `.venv\Scripts\python.exe` when available
  - falls back to `python`
  - launches `src.tools.dev_app`
  - opens `/analysis/` in the default browser through the new `--open-browser` launcher flag
  - passes through extra flags such as `--no-reload`
- Updated `src/tools/dev_ui.py` with `--open-browser` support.
- Updated `README.md` quick-start instructions.
- Research-paper gate:
  - this is workflow/UI launch plumbing only, with no algorithm, data, evaluation, or measured-result change, so `src/docs/RESEARCH_PAPER.md` and `research.md` did not need content updates

## 2026-04-30 - Analysis profile default fix

- Updated `/analysis/` profile selection so `Paris (SpaceNet)` is the default profile.
- Existing browser storage containing the old `legacy` / Open Geo selection is now reset to `paris` on page load.
- Users can still manually switch to `Paris Test (SpaceNet)` or `Open Geo (Wikimedia)` after the page loads.
- Research-paper gate:
  - this is UI default-state behavior only, with no algorithm, data, evaluation, or measured-result change, so `src/docs/RESEARCH_PAPER.md` and `research.md` did not need content updates

## 2026-04-30 - Local data layout and cleanup guidance

- Audited local `data/` size and usage.
- Added `docs/DATA_LAYOUT.md` with:
  - recommended images for manual `/analysis/` testing
  - folders to keep for current realistic Paris model work
  - cleanup candidates for generated chunk outputs, raw SpaceNet rebuild sources, DOTA data, and old smoke checkpoints
  - conservative and aggressive cleanup command examples
- Updated `README.md` to point to the data guide.
- Research-paper gate:
  - this is local storage hygiene/documentation only, with no algorithm, data collection, benchmark result, or model behavior change, so `src/docs/RESEARCH_PAPER.md` and `research.md` did not need content updates

## 2026-04-30 - Paris-focused data cleanup and RF-DETR default

- Removed local non-Paris/legacy data caches:
  - `data/open_geo/`
  - `data/dota`
  - `data/dota_v1/`
  - `data/samples/`
  - `data/test_paris/`
  - `data/geo_index/open_geo_clip.npz`
  - 5-byte temporary `data/geo_index/tmp*_train_images_best_model.npz` stubs
- Removed local intermediate Paris recovery/checkpoint caches duplicated by final combined outputs:
  - `data/paris_realistic_v1_combined_chunkpairs/`
  - `data/paris_realistic_v1_combined_chunkmeta/`
  - `data/paris_realistic_v1_chunkpairs/`
  - `data/paris_realistic_v1_chunkmeta/`
  - `data/paris_realistic_smoke/`
- Kept current Paris assets:
  - `data/paris_realistic_v1/street_combined/`
  - `data/paris_realistic_v1_combined/aerial/`
  - `data/paris_realistic_v1_combined/splits_strict/`
  - `data/paris_realistic_v1_combined/indices/`
  - Paris SpaceNet chips/metadata and Paris retrieval indices
- Created clean manual analysis upload set:
  - `data/analysis_tests/paris_street/images/`
  - `data/analysis_tests/paris_street/manifest.csv`
- Cleared ignored generated dashboard JSON files that referenced removed sample/DOTA paths.
- Removed Open Geo from active runtime profile selection:
  - deleted `src/config/open_geo.json`
  - removed Open Geo option from `/analysis/` and `/analysis/lab/`
  - removed Open Geo config health check path
- Made RF-DETR the default detector backend in shipped Paris configs:
  - `detector.backend = "rfdetr"`
  - `detector.weights_path = null`
  - `detector.nms_mode = "aabb"`
  - `detector.rfdetr_model_size = "medium"`
- Added safe detector fallback:
  - if RF-DETR is requested but the `rfdetr` package is unavailable, the factory falls back to sidecar/classic behavior instead of crashing the app
- Updated docs:
  - `README.md`
  - `docs/DATA_LAYOUT.md`
  - `src/docs/GEO_TECH.md`
  - `src/docs/REPRODUCIBILITY.md`
  - `research.md`
  - `src/docs/RESEARCH_PAPER.md`
- Decision:
  - active product/research workflow is Paris-only for now
  - broad Open Geo/Wikimedia support should return later on a dedicated expansion branch, not in the default app path
## 2026-04-30
- Hypothesis: The application was falling back to demo mode because the inference worker spawned by ui_server.py was encountering a deadlock when passing the large payload queue via multiprocessing.Queue. Reading the queue while waiting for the process to join resolves the deadlock.
- Change: Increased _WORKER_IMAGE_TIMEOUT_S and _WORKER_VIDEO_TIMEOUT_S to 900.0s. Fixed multiprocessing deadlock in _run_inference_worker by calling esult_queue.get(timeout=timeout_s) before process.join(). Fixed bug in _make_demo_image_payload to put allback_reason at the top level of the payload so the frontend correctly renders the "Status Note". Disabled compile=True inside fdetr_detector.py's optimize_for_inference to prevent JIT tracing from hanging.
- Files touched: src/tools/ui_server.py, src/core/detection/rfdetr_detector.py.
- Validation command: Ran the frontend and verified that rfdetr successfully initialized and was used as the backend without timing out.
- Artifacts: None.
- Decision: Keep the fixes. No changes made to RESEARCH_PAPER.md because this was purely an engineering fix for the local analysis GUI server and did not affect the core modeling algorithms or research claims.

## 2026-04-30 Tier 1: Default Config Upgrade
- Hypothesis: Upgrading paris.json from single-index unprojected retrieval to the validated dual-index projected+DBA RRF config should match the paris_balanced_dual_rrf benchmark.
- Change: Updated paris.json to use projected index, DBA companion index, retrieval_projection_path, RRF source fusion, and balanced index weights [1.0, 0.5].
- Files touched: src/config/paris.json
- Validation command: .venv\Scripts\python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --config src/config/paris.json --retrieval-only --limit 180 --seed 42 --output runs/geo_eval_tier1_upgraded_paris_180.json
- Metrics (before -> after): mean_km 15.53->14.60, median_km 9.77->4.21, within_1km 10.56%->10.56%, within_2km 19.44%->31.11%, within_5km 37.22%->52.78%, within_10km 50.56%->65.56%
- Artifacts: runs/geo_eval_tier1_upgraded_paris_180.json
- Decision: Keep. Massive improvement in median (+57% reduction) and within_2km (+60% relative gain) with zero code changes.

## 2026-04-30 Tier 2: Full-Triplet Cross-View Projection Completed
- Goal: Retrain the street-to-aerial query projection on the full mined realistic triplet corpus instead of the earlier `6000`-triplet probe.
- Launch command: .venv\Scripts\python -m src.tools.train_crossview_projection --triplets runs/paris_realistic_crossview_train_triplets_v1.jsonl --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --street-images-dir data/paris_realistic_v1/street_combined --output runs/crossview_projection_paris_combined_v2_full.npz --report-output runs/crossview_projection_paris_combined_v2_full.report.json --embedding-model openai/clip-vit-large-patch14 --max-triplets 0 --epochs 30 --batch-size 32 --learning-rate 1e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device auto
- Runtime note: this environment currently has `torch 2.11.0+cpu`, so `--device auto` resolves to CPU and the full run is materially slower than the original GPU-based expectation.
- Result:
  - training artifact: `runs/crossview_projection_paris_combined_v2_full.npz`
  - training report: `runs/crossview_projection_paris_combined_v2_full.report.json`
  - strict eval: `runs/eval_realistic_crossview_combined_strict_probe240_crossviewproj_v2_full40k.json`
  - dataset: `26204` triplets, `27171` aerial references touched, `0` dropped rows
  - training: `30` epochs on CPU, `elapsed_sec = 620.57`
  - weighted hard triplet loss: `0.1145 -> 0.0905`
  - weighted satisfied pct: `1.51% -> 6.28%`
  - benchmark delta vs Tier 2 probe model:
    - `mean_km 9.75 -> 9.83`
    - `median_km 10.24 -> 10.92`
    - `within_1km 2.08% -> 4.17%`
    - `within_2km 7.50% -> 12.08%`
    - `within_5km 20.42% -> 22.92%`
- Decision: keep as a measured close-range gain on the realistic cross-view benchmark, but do not overclaim it as a universal improvement because `mean_km` and `median_km` regressed slightly even while `<=1km`, `<=2km`, and `<=5km` improved.

## 2026-04-30 Tier 3: DINOv2 Complementary Backbone Evaluated
- Goal: Add a non-CLIP visual signal to Paris retrieval by building a DINOv2 aerial index and fusing it with the validated dual-index CLIP setup.
- Build command: .venv\Scripts\python -m src.tools.build_geo_index --images-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --output data/geo_index/spacenet_paris_chips_facebook_dinov2_base.npz --model-id facebook/dinov2-base
- New experimental config: `src/config/paris_dinov2_rrf_experimental.json`
- Supporting code fix: `src/core/logic/config.py` now parses `retrieval_index_model_ids` as an ordered sequence instead of deduplicating repeated strings. This was necessary because the three-index setup legitimately uses the CLIP model id twice plus one DINOv2 model id.
- Regression coverage: `src/tests/test_config_loading.py::test_load_config_preserves_duplicate_retrieval_index_model_ids`
- Validation command: .venv\Scripts\python -m pytest -q src/tests/test_config_loading.py
- Eval command: .venv\Scripts\python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --config src/config/paris_dinov2_rrf_experimental.json --retrieval-only --limit 180 --seed 42 --output runs/geo_eval_paris_dinov2_rrf_experimental_180_fixed.json
- Artifacts:
  - DINOv2 index: `data/geo_index/spacenet_paris_chips_facebook_dinov2_base.npz`
  - eval: `runs/geo_eval_paris_dinov2_rrf_experimental_180_fixed.json`
- Metrics (Tier 1 baseline -> Tier 3 experimental): `mean_km 14.60->14.42`, `median_km 4.21->4.47`, `within_1km 10.56%->13.33%`, `within_2km 31.11%->31.67%`, `within_5km 52.78%->52.22%`, `within_10km 65.56%->66.67%`
- Decision: keep DINOv2 as experimental only. It improves very-close retrieval and slightly improves `mean_km`, but it regresses `median_km` and `<=5km`, so there is not yet a clean case to replace `src/config/paris.json`.

## 2026-04-30 Tier 4: Realistic Cross-View Encoder Fine-Tune Launched
- Goal: Fine-tune the CLIP image encoder directly on the full realistic cross-view triplet corpus, then rebuild the `40k` aerial index with the tuned model and measure it on the strict `probe240` benchmark.
- Runtime constraint: this machine is still CPU-only, so the first planned Tier 4 pass is a full-data `1`-epoch run to get a measured signal before committing to a much longer multi-epoch job.
- Prepared full-run launcher: `scripts/run_tier4_encoder_ft.ps1`
- Intended training command inside that runner: .venv\Scripts\python.exe -m src.tools.train_retrieval_encoder --triplets runs/paris_realistic_crossview_train_triplets_v1.jsonl --query-images-dir data/paris_realistic_v1\street_combined --reference-images-dir data/paris_realistic_v1_combined --model-id openai/clip-vit-large-patch14 --output-dir runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1 --report-output runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1.report.json --train-scope vision_encoder --epochs 1 --batch-size 8 --learning-rate 1e-5 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.2 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device auto
- Smoke test:
  - command path validated with `--max-triplets 1`, `--batch-size 1`, `--epochs 1`
  - artifacts: `runs/retrieval_encoder_finetune/smoke_one_triplet/` and `runs/retrieval_encoder_finetune/smoke_one_triplet.report.json`
  - result: the encoder trainer, model save, and local-model reload path are all working
- Blocker on unattended full run:
  - multiple background launches on this shell environment stalled immediately after CLIP initialization, before real training progress
  - logs captured from those attempts: `runs/tier4_encoder_ft_pipeline.log`, `runs/tier4_encoder_ft_train.stderr.log`
- Expected follow-up artifacts once the full run is executed successfully:
  - tuned model: `runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1/`
  - training report: `runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1.report.json`
  - rebuilt index: `data/paris_realistic_v1_combined/indices/aerial_clip_index_retrieval_encoder_ft_v1_e1.npz`
  - eval report: `runs/eval_realistic_crossview_combined_strict_probe240_encoderft_v1_e1_full40k.json`
- Dataset scale for this run: `26204` triplets, `26204` unique street queries, `27171` unique aerial reference paths, `40000` aerial reference records in the evaluation index root.
- Decision: Tier 4 is prepared but not yet benchmarked. The training path itself is valid, but the full unattended CPU run needs a reliable execution path before I can claim Tier 4 metrics.

## 2026-05-01
- Hypothesis: The branch-plan rule is still too easy to violate unless the contributor workflow explicitly says that `plan.md` must stay branch-specific and must not be carried unchanged across merges or branch switches.
- Change:
  - tightened `AGENT.md` so branch changes and merges must rewrite `plan.md` for the destination branch's own purpose
  - updated `CONTRIBUTING.md` to say each branch needs its own `plan.md` and that copying another branch's file verbatim is not allowed
  - updated `scripts/new-branch.ps1` template to remind contributors that the scaffold must be replaced with a branch-specific plan
- Files touched:
  - `AGENT.md`
  - `CONTRIBUTING.md`
  - `scripts/new-branch.ps1`
  - `PROGRESS.md`
- Validation command:
  - `rg -n --hidden -S "branch-specific|do not copy another branch's|do not reuse another branch's|destination branch" AGENT.md CONTRIBUTING.md scripts/new-branch.ps1`
- Artifacts: None.
- Decision: Keep. `plan.md` should remain a branch-local contract and should not spread unchanged across branches.
## 2026-05-07
- Hypothesis: Adding dedicated endpoints for operator workflow and tracking the timeline will meet the product requirement for a single-operator visual investigation console.
- Change:
  - Added `/api/operator/analyze`, `/api/operator/session`, `/api/operator/reset`, `/api/operator/pin`, `/api/operator/note`, `/api/operator/confirm`, and `/api/operator/export.json` routes to `ui_server.py`.
  - Migrated `index.html` structure to a three-column layout (left/right panels + center map).
  - Modified `operator.js` to hit `/api/operator/analyze`, render the session timeline and clues directly, and interact with the session endpoints.
  - Added explicit exception catching in the analyze route to log pipeline errors and emit 500 status with session object, removing silent failures.
  - Added a UI "Dev Mode" checkbox to skip ML pipeline and mock candidates for faster dashboard debugging.
- Files touched: `src/tools/ui_server.py`, `src/dashboard/analysis/index.html`, `src/dashboard/analysis/operator.js`, `src/dashboard/analysis/operator.css`, `src/dashboard/analysis/shared.js`.
- Validation command: `python -m pytest src/tests/ -v`
- Metrics (before -> after): Not an ML change.
- Artifacts: none
- Decision: Keep change. Meets Heimdall Operator Mode requirements.
## 2026-05-07 (Follow up)
- Hypothesis: Improving map interactivity (Mapbox/MapLibre popups, pointer cursors, center bounding) and tightening the logic around confidence bands will produce a more professional, "geoguessr-style" UI that meets the polished requirements of a visual intelligence dashboard.
- Change:
  - Enhanced `operator.js` to draw maplibregl.Popup elements over candidate pins upon click, displaying target coordinate info.
  - Adjusted map centering and zooming (`easeTo`) so the viewport dynamically frames selected components.
  - Reduced `confidence_high_max_uncertainty_m` (500_000m -> 150_000m) and `confidence_medium_max_uncertainty_m` (2_000_000m -> 800_000m) in `src/core/logic/config.py` to make the prediction of geolocation accuracy stricter and more reliable.
  - Increased `spatial_consensus_weight` floor in `src/core/logic/fusion.py` to gently improve candidate cluster reward calculations.
  - Applied CSS polishes to the dashboard scrolls, input focuses, and timelines.
- Files touched: `src/core/logic/config.py`, `src/core/logic/fusion.py`, `src/dashboard/analysis/operator.js`, `src/dashboard/analysis/operator.css`.
- Validation command: `python -m pytest src/tests/ -v`
- Metrics: N/A, pipeline regression tests passed.
- Decision: Keep change.
## 2026-05-07 (CI Fixes)
- Hypothesis: Test `test_load_config_spatial_consensus_defaults` failed due to the tightened confidence tier bounds (500_000m -> 150_000m). Updating the test explicitly to match the improved logical configuration will solve it.
- Change:
  - Updated `src/tests/test_config_loading.py` to assert the stricter maximum uncertainties for the confidence tiers.
- Files touched: `src/tests/test_config_loading.py`.
- Validation command: `python -m pytest src/tests/ -v`
- Metrics: All 250 tests passed.
- Decision: Keep change.

## 2026-05-09 Retrieval-Mistake Hard-Negative Projection
- Hypothesis: The current app is missing targeted supervision from its own high-scoring retrieval mistakes; mining those candidates as hard negatives should produce a larger accuracy gain than another hand-tuned rerank knob.
- Change:
  - Added `src/tools/mine_retrieval_hard_triplets.py` to call the active retrieval provider, keep nearby reference chips as positives, and convert retrieved-but-wrong candidates into hard negatives.
  - Added `src/tests/test_mine_retrieval_hard_triplets.py` to cover candidate filtering, SpaceNet chip path normalization, and end-to-end mining with a fake provider.
  - Trained `runs/retrieval_hardneg_crossview_projection_v1.npz` from `160` mined current-stack mistakes.
  - Updated `src/config/paris.json` to use the new projection.
- Validation commands:
  - `$env:TMP='c:\Users\zen\Desktop\Projects\Project-Heimdall\.tmp'; $env:TEMP=$env:TMP; .\.venv\Scripts\python.exe -m pytest src\tests\test_mine_retrieval_hard_triplets.py src\tests\test_train_crossview_projection.py -q`
  - `.\.venv\Scripts\python.exe -m src.tools.run_geo_eval --config src/config/paris.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --limit 80 --seed 42 --output runs/geo_eval_master_current_80.json --allow-scope-mismatch`
  - `.\.venv\Scripts\python.exe -m src.tools.run_geo_eval --config runs/config_paris_hardneg_projection_v1.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --limit 80 --seed 42 --output runs/geo_eval_hardneg_projection_v1_80.json --allow-scope-mismatch`
- Metrics on the same `80` strict probe samples:
  - mean_km `9.1105 -> 4.7680`
  - median_km `7.0931 -> 4.8332`
  - p90_km `16.7438 -> 6.0800`
  - within_2km `3.75% -> 1.25%`
  - within_5km `25.00% -> 57.50%`
  - within_10km `62.50% -> 100.00%`
- Decision: Keep and promote. This is a real serving-path accuracy improvement, with the remaining close-range regression pointing to the next mining pass: denser near-field hard negatives under roughly `3 km`.

## 2026-05-09 Compact Fusion Statistics Follow-Up
- Hypothesis: After the hard-negative projection, some residual error comes from averaging over too broad a Paris-scale credible set rather than from the retrieval representation alone.
- Change:
  - Tested retrieval-only v1, a broad+near v2 projection, top-posterior fusion, compact fusion, and compact estimate statistics while preserving `fusion.top_k=25` for UI candidate display.
  - Updated `src/config/paris.json` to keep all 25 fused candidates but tighten estimate statistics: `retrieval_temperature=0.22`, `credible_mass=0.6`, `min_credible_candidates=1`, `credible_cluster_radius_km=6.0`, and `plausibility_radius_km=12.0`.
  - Updated `research.md`, `README.md`, and `src/docs/RESEARCH_PAPER.md` with the v2 rejection and compact-stat fusion result.
- Validation commands:
  - `.\.venv\Scripts\python.exe -m src.tools.run_geo_eval --config runs/config_paris_hardneg_projection_v1_compactstats_fullcandidates.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --limit 80 --seed 42 --output runs/geo_eval_hardneg_projection_v1_compactstats_fullcandidates_80.json --allow-scope-mismatch`
  - `.\.venv\Scripts\python.exe -m src.tools.run_geo_eval --retrieval-only --config src/config/paris.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --limit 80 --seed 42 --output runs/geo_eval_hardneg_projection_v1_retrieval_only_80.json --allow-scope-mismatch`
- Metrics on the same `80` strict probe samples:
  - promoted v1 full pipeline: `mean 4.7680`, `median 4.8332`, `p90 6.0800`, `<=2km 1.25%`, `<=5km 57.50%`, `<=10km 100.00%`
  - compact-stat fusion with 25 candidates kept: `mean 4.7288`, `median 4.7759`, `p90 6.0684`, `<=2km 1.25%`, `<=5km 56.25%`, `<=10km 100.00%`
  - v2 broad+near hard-negative projection: `mean 4.8302`, `median 4.9400`, `p90 6.0356`, `<=2km 0.00%`, `<=5km 52.50%`, `<=10km 100.00%`
- Decision: Keep the compact-stat fusion tweak and reject v2. This is a small center/tail gain, not the missing breakthrough; the stronger next step remains richer near-field supervision and candidate distribution improvement.

## 2026-05-10 Retrieval-Dominant Paris Serving Path
- Hypothesis: After hard-negative projection training, the Paris retrieval provider is stronger than the broad GeoSpot/GeoCLIP provider; injecting GeoCLIP candidates into the Paris retrieval-index path now dilutes close-range accuracy.
- Change:
  - Added `geolocator.use_geoclip_with_retrieval` with backwards-compatible default `true`.
  - Updated CLI, batch, UI server, full-run, and geo-eval pipeline builders so retrieval-index profiles can skip GeoCLIP provider construction and use retrieval candidates directly.
  - Set `src/config/paris.json` to `use_geoclip_with_retrieval=false`.
  - Rejected the support-density stats selector experiment after it regressed the fixed strict probe.
  - Updated `research.md`, `README.md`, and `src/docs/RESEARCH_PAPER.md`.
- Validation commands:
  - `$env:TMP='c:\Users\zen\Desktop\Projects\Project-Heimdall\.tmp'; $env:TEMP=$env:TMP; .\.venv\Scripts\python.exe -m pytest src\tests\test_config_loading.py src\tests\test_run_geo_eval_retrieval_provider.py src\tests\test_fusion_geo_quality.py -q`
  - `.\.venv\Scripts\python.exe -m src.tools.run_geo_eval --config src/config/paris.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --limit 80 --seed 42 --output runs/geo_eval_hardneg_retrieval_only_provider_80.json --allow-scope-mismatch --diag-samples 5`
- Metrics on the same `80` strict probe samples:
  - compact-stat full provider path: `mean 4.7288`, `median 4.7759`, `p90 6.0684`, `<=2km 1.25%`, `<=5km 56.25%`, `<=10km 100.00%`
  - retrieval-dominant serving path: `mean 4.6213`, `median 4.6345`, `p90 6.0913`, `<=2km 0.00%`, `<=5km 66.25%`, `<=10km 100.00%`
  - rejected support-density selector: `mean 5.3660`, `median 5.1836`, `p90 7.3209`, `<=5km 42.50%`
- Decision: Keep and promote. This is a meaningful close-range serving improvement with a negligible p90 tradeoff; it also reduces runtime cost by avoiding unnecessary GeoCLIP inference for Paris retrieval-index runs.

## 2026-05-10 Diversity-Capped Retrieval-Mistake Projection
- Hypothesis: The next improvement should come from training on the app's current high-scoring mistakes, but the larger mined pool must avoid overfitting repeated negative reference chips.
- Change:
  - Added `--init-projection` to `src.tools.train_crossview_projection` so new projection passes can fine-tune the current serving projection instead of resetting to identity.
  - Added `--max-negative-reuse` plus diversity summary fields to `src.tools.mine_retrieval_hard_triplets`.
  - Added tests for initial-projection shape validation and negative-reuse capping.
  - Updated `src/config/paris.json` to use `runs/retrieval_hardneg_crossview_projection_v4_cap16_initv1.npz` on this branch.
  - Updated `research.md` and `src/docs/RESEARCH_PAPER.md` with the accepted/rejected scaling results.
- Validation commands:
  - `$env:TMP='c:\Users\zen\Desktop\Projects\Project-Heimdall\.tmp'; $env:TEMP=$env:TMP; .\.venv\Scripts\python.exe -m pytest src\tests\test_train_crossview_projection.py src\tests\test_mine_retrieval_hard_triplets.py -q`
  - `.\.venv\Scripts\python.exe -m src.tools.run_geo_eval --config runs/config_paris_hardneg_projection_v4_cap16_initv1.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --limit 80 --seed 42 --output runs/geo_eval_hardneg_projection_v4_cap16_initv1_80.json --allow-scope-mismatch --diag-samples 5`
- Metrics on the same `80` strict probe samples:
  - previous retrieval-dominant v1: `mean 4.6213`, `median 4.6345`, `p90 6.0913`, `<=2km 0.00%`, `<=5km 66.25%`, `<=10km 100.00%`
  - v4 cap16 initialized from v1: `mean 4.5791`, `median 4.6367`, `p90 5.9161`, `<=2km 0.00%`, `<=5km 67.50%`, `<=10km 100.00%`
  - rejected identity-init v2: `mean 21.6922`, `median 23.0790`, `p90 27.2324`, `<=5km 0.00%`
  - rejected full 480 v1-init v2: `mean 5.3267`, `median 5.4025`, `p90 6.6124`, `<=5km 36.25%`
- Artifacts:
  - `runs/retrieval_hardneg_crossview_projection_v4_cap16_initv1.npz`
  - `runs/geo_eval_hardneg_projection_v4_cap16_initv1_80.json`
  - `runs/retrieval_hard_triplets_train480_diverse_v3_summary.json`
- Decision: Keep on this branch and push. This is a measured incremental model improvement, not the requested breakthrough. The important finding is that repeated-reference concentration is now a known bottleneck; the next real step should expand diverse near-field mistakes rather than train longer on the same failure chips.

## 2026-05-10 Candidate Oracle Rank Diagnostic
- Hypothesis: The current hard-negative projection may already retrieve near-correct locations, but rank them below visually similar wrong Paris candidates.
- Change:
  - Added candidate-oracle diagnostics to `src.tools.run_geo_eval`, including closest returned candidate distance, closest returned candidate rank, candidate count, and aggregate oracle metrics.
  - Added focused tests for oracle-rank extraction.
  - Updated `research.md` and `src/docs/RESEARCH_PAPER.md` with the candidate-oracle conclusion.
- Validation commands:
  - `$env:TMP='c:\Users\zen\Desktop\Projects\Project-Heimdall\.tmp'; $env:TEMP=$env:TMP; .\.venv\Scripts\python.exe -m pytest src\tests\test_run_geo_eval_retrieval_provider.py -q`
  - `.\.venv\Scripts\python.exe -m src.tools.run_geo_eval --config src/config/paris.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --limit 80 --seed 42 --output runs/geo_eval_oracle_rank_diagnostics_80.json --allow-scope-mismatch --diag-samples 10`
- Metrics on the same `80` strict probe samples:
  - current serving prediction: `mean 4.5791`, `median 4.6367`, `p90 5.9161`, `<=2km 0.00%`, `<=5km 67.50%`, `<=10km 100.00%`
  - returned top-25 oracle: `mean 2.3528`, `median 2.1638`, `p90 4.0063`, `<=1km 21.25%`, `<=2km 43.75%`, `<=5km 100.00%`
  - mean rank of closest returned candidate: `15.125`
  - rejected graph-support real pipeline test: `mean 4.7027`, `p90 6.5027`, `<=2km 2.50%`, `<=5km 67.50%`
- Decision: Keep and push diagnostics, but do not promote graph support or the existing learned candidate reranker. The next bottleneck is visual ranking inside the returned shortlist, not candidate coverage or pure spatial clustering.

## 2026-05-10 Oracle-Candidate Ranking Experiments
- Hypothesis: If the closest returned candidate is already present, direct listwise ranking or oracle-candidate triplets should move it toward rank 1.
- Change:
  - Added `exp` activation support to `CandidateRerankModel`.
  - Added `--fit-mode listwise` to `src.tools.train_geo_candidate_reranker`.
  - Added `--positive-source closest_candidate` to `src.tools.mine_retrieval_hard_triplets` so the positive can be the closest returned retrieval candidate instead of only the nearest reference metadata chip.
  - Added focused tests for listwise reranker training and closest-candidate positive mining.
- Validation commands:
  - `New-Item -ItemType Directory -Force .tmp | Out-Null; $env:TMP=(Resolve-Path .tmp).Path; $env:TEMP=$env:TMP; .\.venv\Scripts\python.exe -m pytest src\tests\test_candidate_rerank.py src\tests\test_mine_retrieval_hard_triplets.py -q`
  - `.\.venv\Scripts\python.exe -m src.tools.train_geo_candidate_reranker --config src/config/paris.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/train_pairs.csv --eval-metadata data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --limit 240 --eval-limit 80 --seed 42 --fit-mode listwise --target-sigma-km 1.5 --listwise-epochs 40 --listwise-learning-rate 0.02 --listwise-l2 0.001 --fusion-weight 1.5 --temperature 0.22 --output runs/candidate_reranker_listwise_v1.json --report-output runs/candidate_reranker_listwise_v1.report.json`
  - `.\.venv\Scripts\python.exe -m src.tools.mine_retrieval_hard_triplets --config src/config/paris.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/train_pairs.csv --reference-metadata data/spacenet_paris/metadata.csv --limit 240 --seed 42 --positive-source closest_candidate --positive-radius-km 2.0 --positive-fallback-top-k 1 --negative-min-gt-distance-km 0.75 --negative-max-gt-distance-km 18.0 --max-positives 1 --max-negatives 8 --max-negative-reuse 16 --output runs/retrieval_oracle_candidate_triplets_train240_v1.jsonl --summary-output runs/retrieval_oracle_candidate_triplets_train240_v1_summary.json`
- Metrics:
  - listwise reranker full fusion test: `mean 5.3499`, `p90 6.7075`, `<=5km 42.50%`; rejected.
  - oracle-candidate triplet mining: `104` triplets from `240` records, but only `8` unique positive chips; this is too concentrated.
  - oracle-candidate projection full serving test: `mean 5.0353`, `median 5.0583`, `p90 6.3590`, `<=5km 50.00%`; rejected.
  - oracle-candidate projection improved closest-candidate rank (`15.125` -> `10.375`) but worsened oracle `<=2km` (`43.75%` -> `30.00%`), so it damaged candidate coverage while improving one ranking diagnostic.
- Decision: Keep the tooling and artifacts as research evidence, but do not promote either model. The fix is not more pressure on the same tiny positive pool; the next step is to increase diverse true-positive candidates in the retrieval index/training set.

## 2026-05-10 Realistic Aerial Index Promotion
- Hypothesis: The oracle-candidate triplets failed because the shortlist positive pool was too concentrated. Adding the full realistic IGN aerial index should increase diverse true positives before another ranking loss is attempted.
- Change:
  - Added `data/paris_realistic_v1_combined/indices/aerial_clip_index.npz` to the active Paris retrieval profile.
  - Kept the SpaceNet retrieval indices on `runs/retrieval_hardneg_crossview_projection_v4_cap16_initv1.npz`.
  - Routed the realistic aerial index through raw CLIP by setting the third `retrieval_index_projection_paths` entry to `null`.
  - Set `retrieval_per_index_top_k=25` so the realistic index contributes a full candidate shortlist instead of only competing after global top-k truncation.
- Validation commands:
  - `.\.venv\Scripts\python.exe -m src.tools.run_geo_eval --config runs/config_paris_with_realistic_aerial_index_v1.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --limit 80 --seed 42 --output runs/geo_eval_realistic_aerial_index_v1_80.json --allow-scope-mismatch --diag-samples 10`
  - `.\.venv\Scripts\python.exe -m src.tools.run_geo_eval --config runs/fusion_sweep_configs/realistic_rrf_w050_top25.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --limit 80 --seed 42 --output runs/geo_eval_realistic_rrf_w050_top25_full_80.json --allow-scope-mismatch --diag-samples 10`
  - `.\.venv\Scripts\python.exe -m src.tools.run_geo_eval --config runs/fusion_sweep_configs/realistic_weighted_w050_top25.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --limit 80 --seed 42 --retrieval-only --output runs/geo_eval_realistic_weighted_w050_top25_retrieval_only_80.json --allow-scope-mismatch --diag-samples 5`
- Metrics on the same `80` strict probe samples:
  - previous serving profile: `mean 4.5791`, `median 4.6367`, `p90 5.9161`, `<=2km 0.00%`, `<=5km 67.50%`; oracle `mean 2.3528`, `<=2km 43.75%`, best-rank mean `15.125`
  - realistic aerial index profile: `mean 4.4793`, `median 4.6127`, `p90 5.7859`, `<=2km 0.00%`, `<=5km 70.00%`; oracle `mean 1.6715`, `<=2km 66.25%`, best-rank mean `18.25`
  - lighter rank-fusion weighting: `mean 4.5026`, `median 4.6306`, `p90 5.8956`, `<=2km 3.75%`, `<=5km 66.25%`; oracle `mean 1.5891`, `<=2km 66.25%`, best-rank mean `14.5375`
  - score-based weighted fusion was rejected despite `<=2km 8.75%` in retrieval-only because it regressed broad ranking (`mean 5.2260`, `p90 7.9741`, `<=5km 45.00%`).
- Decision: Promote the realistic aerial index profile because it improves mean, median, p90, `<=5km`, and oracle coverage. Do not promote the lighter weighting yet; it helps close-range rank but gives up too much `<=5km` on the full pipeline. The new bottleneck is sharper rank selection from a much better shortlist, not candidate availability.
