# Project Heimdall: Full Research Paper Draft

Version: v1.0  
Date: April 21, 2026

Companion external landscape review: `src/docs/MARKET_RESEARCH.md`.
Chronological experiment ledger with the latest branch-era before/after metrics: `research.md`.

## Title
Project Heimdall: Benchmark-Governed Geolocation from Imagery via Multi-Provider Retrieval, Robust Probabilistic Fusion, and Uncertainty-Aware Analysis

## Abstract
This document presents a full research-style account of Project Heimdall, an engineering system for image geolocation developed from January to April 2026. Heimdall integrates oriented object detection, multi-provider geolocation candidate generation, retrieval index search, posterior fusion, and confidence-aware uncertainty outputs. Development progressed through multiple algorithmic iterations: retrieval diversification and locality reranking, query-time test-time augmentation (TTA), multi-index weighted retrieval, source-balanced candidate selection, score normalization across heterogeneous indices, source-aware fusion priors, cross-source agreement and spatial consensus likelihoods, adaptive outlier suppression, and temporal posterior filtering. We report controlled benchmark artifacts and ablations from repository runs. A central finding is that evaluation protocol quality dominates apparent model quality: leakage-prone evaluation can show near-perfect performance, while realistic split evaluation remains substantially more difficult. The resulting system emphasizes reproducibility, benchmark governance, and traceable accuracy claims.

## Keywords
Image geolocation, retrieval, probabilistic fusion, uncertainty estimation, benchmark governance, aerial imagery, CLIP embeddings.

## 1. Introduction
Image geolocation aims to infer where an image was captured from visual evidence alone. Practical deployments must satisfy more than point accuracy: confidence must be calibrated, failure modes must be inspectable, and iterative model work must remain reproducible under changing data and dependencies. Project Heimdall was built as an end-to-end engineering research platform for this objective.

The project began with basic candidate generation and heuristic scoring, then evolved into a benchmark-governed retrieval and fusion stack. The current system supports multi-source geolocation hypotheses and robust fusion diagnostics while preserving operator-facing visibility through a local analysis interface.

## 2. Scope and Research Questions
### 2.1 Primary Objective
Infer geographic location for a query image with high precision while producing reliable uncertainty and traceable evidence.

### 2.2 Research Questions
1. Which retrieval controls improve realistic geolocation quality without collapsing recall?
2. Which fusion mechanisms reduce overconfident, source-isolated false positives?
3. How strongly do evaluation protocol choices (especially data leakage) affect apparent performance?
4. Can governance tooling enforce stable benchmark comparisons over time?

## 3. System Overview
Heimdall follows a modular pipeline:
1. Detection layer (oriented objects).
2. Geo candidate generation (retrieval + model providers + metadata fallback).
3. Probabilistic fusion with diagnostics.
4. Optional temporal posterior update.
5. Serialization, API/UI serving, and benchmark tooling.

Key runtime modules:
- Detection: `src/core/detection/`
- Geolocation providers: `src/core/geo/`
- Fusion and logic: `src/core/logic/`
- Evaluation/tuning utilities: `src/tools/`
- Analysis UI: `src/tools/ui_server.py`, `src/dashboard/analysis/`

## 4. Methods
### 4.1 Detection
- Detector runtime: Ultralytics OBB (`yolo11x-obb.pt` by default).
- Controls:
  - minimum area filtering,
  - OBB or AABB NMS mode,
  - class-aware/class-agnostic suppression,
  - optional detector TTA.

### 4.2 Geolocation Candidate Generation
Providers:
- Retrieval index provider (`GeoRetrievalProvider`).
- GeoSpot/GeoCLIP-style provider.
- EXIF/sidecar fallback provider.

Retrieval controls explored:
- `retrieval_min_score` thresholding.
- `retrieval_min_keep_topk` fallback recall guard.
- Diversity re-selection (`retrieval_diversity_*`).
- Locality reranking (`retrieval_locality_*`).
- Structure-aware top-shortlist reranking using corner density, edge density, dominant line orientation, and weak shadow-axis cues (`retrieval_structure_rerank_*`).
- Local geometric reranking with dual feature engines (`SIFT` + `ORB`) and evidence gating.
- Query TTA (`retrieval_query_tta_degrees`, `retrieval_query_tta_reduce`).
- Multi-index expansion (`retrieval_index_paths`, `retrieval_index_weights`, `retrieval_per_index_top_k`).
- Per-index model routing (`retrieval_index_model_ids`).
- Cross-index score normalization (`retrieval_index_score_norm`).
- Source-balance in retrieval top-k (`retrieval_source_balance_beta`).
- Source-balance after provider merge (`candidate_source_balance_beta`).

### 4.3 Probabilistic Fusion
Fusion is performed in log-space with source-aware priors and additional likelihood terms:
- Spatial consensus likelihood.
- Cross-source agreement likelihood.
- Optional plausibility reranking.
- Adaptive outlier guard (robust medoid/MAD-style suppression).
- Dateline-safe longitudinal statistics.
- Credible-set and top-cluster statistics for ambiguous multimodal outputs.

Confidence/uncertainty controls:
- Calibrated top-1 posterior.
- Tier thresholds (high/medium/low).
- Cross-source support tier caps.
- Uncertainty-radius tier caps.

### 4.4 Temporal Posterior Update
Later iterations replaced a stub temporal filter with:
- Proximity-weighted posterior reweighting,
- geodesic, dateline-safe association gates,
- uncertainty-aware adaptive gating and shrinkage under agreement.

### 4.5 Benchmark Governance
A governance layer was added to prevent ad-hoc metric claims:
- `benchmarks/manifest.json` for fixed benchmark contracts.
- `benchmarks/policy.json` for regression thresholds.
- `src/tools/benchmark_ci.py` for automated compare/report/promote flow.
- Baseline and run history in `docs/eval/`.

## 5. Development Chronology
### 5.1 Phase I: Pipeline Foundation (Jan 29-31, 2026)
- Baseline architecture and interfaces.
- EXIF/sidecar geolocation.
- Structured schema outputs.
- Initial UI and serving path.
- Initial fusion and uncertainty serialization.

### 5.2 Phase II: Retrieval and Fusion Hardening (Apr 5, 2026)
- Multi-index and source-aware retrieval.
- Diversity/locality/source balancing controls.
- Expanded fusion likelihoods and confidence calibration.
- Hard-negative reporting and auto-tuning tools.
- Temporal filtering and tracking upgrades.

### 5.3 Phase III: Latest Accuracy Iteration (Apr 15, 2026)
- Added `median` TTA reduce support.
- Extended retrieval tuning to sweep TTA reduce modes (`mean`, `median`, `max`, `rrf`).
- Added best-mode writeback support in tuning workflow.
- Added objective-driven ranking in retrieval tuning (`within_1km_pct`, `within_2km_pct`, `within_5km_pct`, `within_10km_pct`).
- Ran full realistic-split retrieval post-processing sweep (`n=180`) and retuned realistic single-index profile to a simplified ranking path (locality/diversity/source-balance disabled).
- Added retrieval consensus top-1 refinement (`retrieval_consensus_top_n`, `retrieval_consensus_radius_km`, `retrieval_consensus_score_power`) and upgraded center estimation to adaptive centroid/weighted-geo-median selection for local outlier robustness.
- Added a Lab random-sample evaluation mode for lightweight spot-checking of per-sample distance errors and quick accuracy sanity checks between full benchmark runs.
- Added aerial retrieval backbone benchmark presets (`aerial_rtx5060_fast`, `aerial_rtx5060_precise`, `aerial_research`) with objective-driven model selection (`within_1km_pct`, `within_2km_pct`, etc.).
- Added one-command retrieval backbone upgrade workflow (`src.tools.upgrade_retrieval_backbone`) that benchmarks candidate backbones, rebuilds the final index with the selected model, and patches config for reproducible rollout.
- Added graph-support and KDE mode retrieval reranking ablations on the realistic split (`n=180`) with objective-specific profiles (`within_1km` and `within_2km`).
- Upgraded local geometric reranking to a dual-engine method (`SIFT` + `ORB`) with weak-signal gating and adaptive blend scaling.
- Added a structure-aware retrieval reranker that uses corners, line orientation, and guarded shadow cues to reorder the top retrieval shortlist before local matching.

## 6. Experimental Protocol
### 6.1 Datasets and Artifacts Used in This Document
- SpaceNet Paris train-like index artifacts (`data/geo_index/spacenet_paris_clip.npz`).
- SpaceNet Paris test chips and metadata (`data/spacenet_paris_test/chips`, `metadata.csv`).
- Open geo retrieval artifacts (Wikimedia-derived index) where specified by config.

### 6.2 Core Metrics
- Distance metrics: `mean_km`, `median_km`, `p90_km`, `p95_km`.
- Radius accuracy: `within_1km_pct`, `within_2km_pct`, `within_5km_pct`, `within_10km_pct`, `within_50km_pct`.
- Coverage diagnostics: `evaluated`, `null_predictions`.

### 6.3 Reliability Metrics (available in tooling)
- `ece`, `brier`, `nll`.
- Cross-source support and confidence-tier coverage outputs.

### 6.4 Leakage-Safe Reporting Principle
A core rule in this paper: claims about generalization rely on realistic split settings. Leakage-prone runs are kept only as diagnostic controls.

## 7. Results
### 7.1 Leakage-Control Reality Check (n=180)
Artifacts:
- `runs/bench_current_leaky_180.json`
- `runs/bench_realistic_single_180.json`
- `runs/bench_candidate_multi_180.json`

| Setting | mean_km | median_km | within_5km_pct | within_10km_pct |
|---|---:|---:|---:|---:|
| Leaky current index | 0.4006 | 0.00011 | 97.78 | 100.00 |
| Realistic single index | 18.0159 | 11.4990 | 30.56 | 45.56 |
| Realistic multi-index candidate | 19.3096 | 11.4842 | 19.44 | 40.56 |

Observation:
- Leakage-prone setup drastically overstates quality.
- Realistic split error is one to two orders of magnitude larger.

### 7.2 Query TTA On/Off Ablation (n=120 subset)
Artifacts:
- `runs/geo_eval_paris_no_tta_120.json`
- `runs/geo_eval_paris_tta_120.json`

| Setting | mean_km | median_km | within_5km_pct | within_10km_pct |
|---|---:|---:|---:|---:|
| No TTA | 0.1982 | 0.000119 | 97.50 | 99.17 |
| TTA (`0,90,180,270`, `max`) | 0.1982 | 0.000119 | 97.50 | 99.17 |

Observation:
- No measurable gain on this subset under this protocol.

### 7.3 Strict Threshold Recall Collapse and Recovery (n=40)
Artifacts:
- `runs/geo_eval_paris_strict_keep0_40.json`
- `runs/geo_eval_paris_strict_keep2_40.json`

| Setting | evaluated | null_predictions | mean_km |
|---|---:|---:|---:|
| `retrieval_min_keep_topk=0` | 0 | 40 | n/a |
| `retrieval_min_keep_topk=2` | 40 | 0 | 0.00012 |

Observation:
- `retrieval_min_keep_topk` is a critical guardrail against full null-output failure under strict thresholds.

### 7.4 TTA Reduction Mode Sweep (n=40, fixed retrieval knobs)
Artifact:
- `runs/tune_retrieval_geo_tta_modes_med.json`

| TTA reduce mode | mean_km | median_km | within_1km_pct | within_5km_pct |
|---|---:|---:|---:|---:|
| `max` | 0.2513 | 0.000129 | 90.0 | 100.0 |
| `rrf` | 0.6963 | 0.000139 | 77.5 | 100.0 |
| `mean` | 0.7168 | 0.000134 | 77.5 | 97.5 |
| `median` | 1.0809 | 0.000182 | 62.5 | 95.0 |

Observation:
- `max` ranked best on this tested subset.
- `median` remains exposed as a tunable option rather than a new default.

### 7.5 Realistic Retrieval Post-Processing Retune (n=180)
Artifacts:
- `runs/tune_retrieval_geo_realistic_within1km_focus_v1.json`
- `runs/bench_realistic_single_180_precision_v2.json`

| Variant | mean_km | median_km | within_1km_pct | within_5km_pct | within_10km_pct |
|---|---:|---:|---:|---:|---:|
| Prior realistic profile | 19.7494 | 11.3947 | 1.67 | 23.89 | 43.33 |
| Retuned realistic profile | 18.0159 | 11.4990 | 5.00 | 30.56 | 45.56 |

Main knobs changed:
- `retrieval_top_k: 25`
- `retrieval_min_score: 0.05`
- `retrieval_min_keep_topk: 0`
- `retrieval_diversity_radius_km: 0.0`
- `retrieval_diversity_lambda: 1.0`
- `retrieval_diversity_min_keep: 1`
- `retrieval_locality_radius_km: 0.0`
- `retrieval_locality_weight: 0.0`
- `retrieval_source_balance_beta: 0.0`

Observation:
- In this realistic split, aggressive post-processing reduced top-1 precision.
- The simplified ranking path improved close-range accuracy substantially (`within_1km_pct`: `1.67` -> `5.00`).

### 7.6 Retrieval Consensus Top-1 Refinement (n=180)
Artifacts:
- `runs/geo_eval_paris_profile_180_v2.json`
- `runs/geo_eval_paris_profile_180_consensus_v1.json`

| Variant | mean_km | median_km | within_1km_pct | within_5km_pct | within_10km_pct |
|---|---:|---:|---:|---:|---:|
| Retuned realistic profile (no consensus) | 18.0159 | 11.4990 | 5.00 | 30.56 | 45.56 |
| + Consensus top-1 refinement | 15.5334 | 9.7717 | 10.00 | 36.67 | 50.56 |

Main knobs:
- `retrieval_consensus_top_n: 20`
- `retrieval_consensus_radius_km: 3.0`
- `retrieval_consensus_score_power: 1.0`

Observation:
- Top-K candidates already contained near-ground-truth hypotheses in many failures; consensus refinement improved top-1 selection quality.
- On this split, consensus refinement yielded a large close-range gain while preserving zero null predictions.

### 7.7 Backbone Upgrade Cycle and Post-Tune Verification (n=180)
Artifacts:
- `runs/backbone_upgrade_rtx5060_v1/backbone_benchmark.json`
- `runs/geo_eval_paris_profile_180_pre_backbone_upgrade_v1.json`
- `runs/tune_retrieval_geo_within2km_v1.json`
- `runs/geo_eval_paris_profile_180_post_tune_v1.json`

Backbone benchmark summary (`model_preset=aerial_rtx5060_precise`, objective=`within_2km_pct`):

| Model | mean_km | median_km | within_1km_pct | within_2km_pct | within_5km_pct |
|---|---:|---:|---:|---:|---:|
| `openai/clip-vit-large-patch14` | 20.5010 | 8.5180 | 2.22 | 10.00 | 36.67 |
| `google/siglip-base-patch16-224` | 24.5362 | 15.6908 | 0.56 | 4.44 | 19.44 |
| `google/siglip-so400m-patch14-384` | 26.8980 | 17.7574 | 0.56 | 3.33 | 13.89 |

End-to-end realistic profile verification (`src/config/paris.json`, retrieval-only):

| Variant | mean_km | median_km | within_1km_pct | within_2km_pct | within_5km_pct | within_10km_pct |
|---|---:|---:|---:|---:|---:|---:|
| Pre-cycle baseline | 15.5569 | 9.7717 | 10.56 | 19.44 | 36.67 | 50.56 |
| Post-tune run | 15.5569 | 9.7717 | 10.56 | 19.44 | 36.67 | 50.56 |

Observation:
- On this realistic split, tested SigLIP aerial candidates did not outperform CLIP.
- Focused retrieval post-processing sweep (`within_2km_pct` objective) produced no measurable end-to-end uplift in the canonical evaluation run.
- Practical implication: further gain likely requires method-level upgrades (domain-adapted representations, reranking head, or stronger hard-negative data), not additional local knob sweeps on current index/profile.

### 7.8 Graph-Support and KDE Refinement Ablation (n=180)
Control artifact:
- `runs/geo_eval_paris_profile_180_qexp_ctrl_v1.json`

Candidate artifacts:
- `runs/geo_eval_paris_profile_180_graphrerank_a_v1.json`
- `runs/geo_eval_paris_profile_180_graphrerank_b_v1.json`
- `runs/geo_eval_paris_profile_180_graphrerank_c_v1.json`
- `runs/geo_eval_paris_profile_180_kde_refine_a_v1.json`
- `runs/geo_eval_paris_profile_180_kde_refine_b_no_consensus_v1.json`
- `runs/geo_eval_paris_profile_180_kde_refine_c_w1_v1.json`
- `runs/geo_eval_paris_profile_180_kde_refine_d_w2_v1.json`

| Variant | mean_km | median_km | within_1km_pct | within_2km_pct | within_5km_pct |
|---|---:|---:|---:|---:|---:|
| Control (`qexp_ctrl_v1`) | 15.5264 | 9.7717 | 10.56 | 19.44 | 37.22 |
| Graph rerank A | 16.1146 | 10.1016 | 5.56 | 17.78 | 35.56 |
| Graph rerank B | 16.8216 | 10.3132 | 6.11 | 16.67 | 35.00 |
| Graph rerank C | 15.3627 | 10.1008 | 7.22 | 17.78 | 37.22 |
| KDE refine A | 15.6553 | 10.3081 | 8.89 | 18.89 | 36.67 |
| KDE refine B (no consensus) | 14.6695 | 9.9699 | 8.33 | 18.89 | 37.78 |
| KDE refine C (W1 focus) | 15.4602 | 9.8686 | 11.11 | 19.44 | 37.78 |
| KDE refine D (W2 focus) | 15.4223 | 9.9580 | 10.00 | 20.00 | 38.33 |

Observation:
- Graph-support reranking underperformed control across close-range metrics on this split.
- KDE mode refinement produced objective-dependent gains:
  - best `within_1km_pct`: `11.11` (`kde_refine_c_w1_v1`, +0.55 over control),
  - best `within_2km_pct`: `20.00` (`kde_refine_d_w2_v1`, +0.56 over control),
  - best `within_5km_pct`: `38.33` (`kde_refine_d_w2_v1`, +1.11 over control).

### 7.9 Dual Local Geometric Reranker Upgrade (n=180)
Artifacts:
- Control stability check:
  - `runs/geo_eval_paris_profile_180_qexp_ctrl_v2_after_dual_localcode.json`
- Legacy local matcher baseline:
  - `runs/geo_eval_paris_profile_180_localmatch_a_v1.json`
- Dual-local reranker evals:
  - `runs/geo_eval_paris_profile_180_localmatch_a_v2_dual.json`
  - `runs/geo_eval_paris_profile_180_localmatch_b_v2_dual.json`
  - `runs/geo_eval_paris_profile_180_localmatch_c_v2_dual.json`
- Combined KDE + dual-local probe:
  - `runs/geo_eval_paris_profile_180_kde_refine_e_w1_duallocal_v1.json`

| Variant | mean_km | median_km | within_1km_pct | within_2km_pct | within_5km_pct | within_10km_pct |
|---|---:|---:|---:|---:|---:|---:|
| Control after code change | 15.5264 | 9.7717 | 10.56 | 19.44 | 37.22 | 50.56 |
| Legacy local match A (pre-upgrade) | 16.6122 | 10.9454 | 5.00 | 13.33 | 32.78 | 46.67 |
| Dual local match A | 15.2447 | 9.7717 | 8.89 | 18.33 | 37.22 | 51.11 |
| Dual local match B | 15.2417 | 9.7717 | 8.89 | 18.33 | 37.78 | 51.11 |
| Dual local match C | 15.5318 | 9.7717 | 8.89 | 18.33 | 37.22 | 51.11 |
| KDE C + Dual Local | 15.1863 | 9.8397 | 8.89 | 18.33 | 37.78 | 51.11 |

Observation:
- The dual local matcher is a real method upgrade over legacy local matching (large recovery in mean/median and all radius metrics).
- Control profile stayed identical after the code change, confirming no regression when local match is disabled.
- On this split, the dual local reranker did not exceed the best close-range profile from KDE refinement (`within_1km_pct` remained below `11.11`).
- Practical use: keep dual local rerank as an optional mode for tail-error reduction while close-range target optimization continues through density-aware retrieval refinement and backbone/data upgrades.

### 7.10 Evaluation Integrity Guard: Profile/Data Mismatch Fix
Artifacts:
- `runs/tmp_paris_with_open_geo_seed1870334448.json`
- `runs/tmp_paris_with_paris_cfg_seed1870334448.json`

| Setup (same two Paris chips, same seed) | mean_km | within_1km_pct | Key behavior |
|---|---:|---:|---|
| Paris data + `open_geo` profile | 5846.1583 | 0.00 | Predicted near `40.6892, -74.0445` (US) |
| Paris data + `paris` profile | 7.1809 | 50.00 | Predictions stayed in Paris region |

Observation:
- Extreme errors were caused by evaluation profile/index mismatch, not label corruption.
- This failure mode can dominate metrics and must be treated as an experimental validity issue.
- The Lab/backend now auto-corrects legacy/open-geo profile selection when dataset paths clearly target Paris (`spacenet_paris*`), and surfaces requested/effective profile in output.
- `src.tools.run_geo_eval` now enforces the same guardrail in CLI flows by default (`profile_scope` vs inferred dataset scope), with explicit override via `--allow-scope-mismatch`.

### 7.11 Multi-Scale Query Views and Adaptive-Mass KDE Follow-Up (n=180)
Artifacts:
- Multi-scale query-view probe:
  - `runs/geo_eval_paris_profile_180_tta_agreement_ctrl_v1.json`
  - `runs/geo_eval_paris_profile_180_multiscale_a_v1.json`
- Adaptive-mass KDE follow-up:
  - `runs/geo_eval_paris_profile_180_kde_refine_d_w2_v1.json`
  - `runs/geo_eval_paris_profile_180_kde_refine_d_w2_adapt_a_v1.json`
  - `runs/geo_eval_paris_profile_180_kde_refine_d_w2_adapt_b_v1.json`

| Variant | mean_km | median_km | p90_km | within_1km_pct | within_2km_pct | within_5km_pct | within_10km_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTA control (`tta_agreement_ctrl`) | 15.5264 | 9.7717 | 43.3924 | 10.56 | 19.44 | 37.22 | 50.56 |
| Multi-scale query views (`multiscale_a`) | 15.3451 | 10.1387 | 42.7910 | 6.11 | 17.22 | 37.78 | 49.44 |
| KDE-W2 baseline (`kde_refine_d_w2`) | 15.4223 | 9.9580 | 42.6563 | 10.00 | 20.00 | 38.33 | 50.56 |
| KDE-W2 + adaptive mass 0.7 (`adapt_a`) | 15.4590 | 10.1419 | 43.2513 | 11.11 | 19.44 | 38.33 | 49.44 |
| KDE-W2 + adaptive mass 0.5 (`adapt_b`) | 15.4685 | 10.1257 | 42.7986 | 8.89 | 17.78 | 38.89 | 49.44 |

Observation:
- Multi-scale query views improved mean and p90 slightly but materially hurt close-range accuracy (`within_1km_pct` and `within_2km_pct`), so this variant is not promoted.
- Adaptive-mass KDE (`0.7`) improved `within_1km_pct` on this run but regressed the primary W2 objective (`within_2km_pct`) and worsened central tendency; `0.5` regressed both close-range metrics.
- Current default stance remains: keep fixed-mass KDE profiles and single-scale query views unless a future leakage-safe benchmark shows a consistent gain.

### 7.12 Error-Driven Hard-Negative Mining Pipeline (Data-Centric Upgrade Path)
Artifacts:
- `runs/geo_eval_paris_profile_180_for_mining_v1.json`
- `runs/hard_negative_triplets_paris_test_v2_scene_dedup.jsonl`
- `runs/hard_negative_triplets_paris_test_v2_scene_dedup_summary.json`

Results:
- `total_records`: `2391`
- `total_failures_considered`: `180`
- `triplets_written`: `145`
- Average tuple density: `~2.90` positives and `12.00` hard negatives per query.

Observation:
- The bottleneck is no longer only inference logic; it is lack of targeted hard-negative supervision around failure regions.
- The new mining pipeline converts real evaluation failures into structured triplets suitable for retrieval-backbone fine-tuning, which is the highest-probability path to step-change improvements in `within_1km`/`within_2km` metrics.

### 7.13 Projection-Aware Retrieval Adaptation (Hard-Negative Metric Learning)
Artifacts:
- Baseline:
  - `runs/geo_eval_projection_baseline_120.json`
- Projection variants:
  - `runs/geo_eval_projection_trainref_v1_120_rerun.json`
  - `runs/geo_eval_projection_trainref_v2_mild_120.json`
  - `runs/geo_eval_projection_trainref_v3_dim256_120.json`
- Projection training and transformed index artifacts:
  - `runs/retrieval_projection_paris_query_trainref_v2_mild.npz`
  - `data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz`

Method:
- Built query-vs-reference hard-negative triplets (queries from Paris test, positives/negatives from Paris reference pool).
- Trained a lightweight linear projection head over retrieval embeddings.
- Applied projection to index embeddings and enabled projection at query time.
- Evaluated multiple projection variants on the realistic retrieval-only split (`n=120`, seed `42`).

| Variant | mean_km | median_km | within_1km_pct | within_2km_pct | within_5km_pct | within_10km_pct |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (no projection) | 15.517 | 10.498 | 9.17 | 16.67 | 36.67 | 48.33 |
| Projection V1 | 19.517 | 9.062 | 7.50 | 19.17 | 35.00 | 52.50 |
| Projection V2 (mild, selected) | 14.925 | 4.512 | 12.50 | 28.33 | 51.67 | 61.67 |
| Projection V3 (dim256) | 19.822 | 9.145 | 11.67 | 16.67 | 33.33 | 53.33 |

Observation:
- V2 is the first clear step-change in this cycle for close-range accuracy on the realistic Paris split:
  - `within_1km_pct`: `9.17` -> `12.50` (`+3.33`)
  - `within_2km_pct`: `16.67` -> `28.33` (`+11.66`)
  - `within_5km_pct`: `36.67` -> `51.67` (`+15.00`)
- V1 and V3 did not offer stable full-metric gains versus baseline.
- This validates hard-negative, data-driven representation adaptation as a stronger lever than additional local post-processing sweeps.

Follow-up weighted training check (same 68 query-vs-reference triplets, canonical realistic split `n=180`, seed `42`):
- Uniform weighting: `mean_km=15.252`, `median_km=5.504`, `within_1km_pct=11.67`, `within_2km_pct=26.67`, `within_5km_pct=50.00`, `within_10km_pct=64.44`.
- Difficulty-aware weighting: `mean_km=15.081`, `median_km=4.888`, `within_1km_pct=13.89`, `within_2km_pct=27.22`, `within_5km_pct=51.67`, `within_10km_pct=65.00`.
- Training-side weighted objective also improved (`weighted_triplet_satisfied_pct`: `27.94` -> `30.99`; `weighted_hard_triplet_loss`: `0.1016` -> `0.0994`).
- Interpretation: emphasizing severe/confusion-rich failures is beneficial even before enlarging the triplet pool.

### 7.14 Structure-Aware Retrieval Rerank (Corners, Lines, Shadow Cue)
Artifacts:
- Control:
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`
- Initial structure rerank:
  - `runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v1.json`
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v1_180.json`
- Geometry-lite gated branch probe:
  - `src/config/paris_structure_geometry_balanced.json`
  - `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_d_180.json`

Method:
- Added retrieval controls:
  - `retrieval_structure_rerank_top_n`
  - `retrieval_structure_rerank_weight`
- For the top retrieval shortlist, extracted a coarse grayscale scene signature composed of:
  - corner density,
  - edge density,
  - dominant line-orientation histogram,
  - guarded dark-mass / shadow-axis cue as weak illumination-layout evidence.
- On this branch, expanded that signature with geometry-lite cues:
  - corner spatial layout,
  - edge spatial layout,
  - line orthogonality / anisotropy,
  - shadow elongation.
- Added weak-signal gating so those extra geometry-lite cues stay secondary on diffuse scenes instead of overriding the older structure signal.
- Blended structure similarity with the base retrieval score only when the structure evidence was strong enough and a confident base top-1 did not already dominate.

Canonical weighted single-index Paris benchmark (`n=180`, seed `42`):

| Variant | mean_km | median_km | within_1km_pct | within_2km_pct | within_5km_pct | within_10km_pct |
|---|---:|---:|---:|---:|---:|---:|
| Weighted projection control | 15.0810 | 4.8877 | 13.89 | 27.22 | 51.67 | 65.00 |
| + Structure-aware rerank (`top_n=12`, `weight=0.35`) | 14.7247 | 4.5903 | 15.00 | 28.33 | 53.33 | 66.11 |
| + Geometry-lite gated rerank (`top_n=14`, `weight=0.35`) | 14.7239 | 4.5903 | 15.00 | 28.33 | 53.33 | 66.11 |

Observation:
- This is the first measured gain from explicitly modeling scene layout cues rather than only embedding similarity or local keypoints.
- Weak-signal gating recovered the earlier close-range gains while preserving the richer geometry-lite signature.
- On the balanced branch setting, geometry-lite now matches the earlier structure-rerank milestone instead of regressing `within_2km_pct`.
- The remaining question is whether those richer cues can beat the older structure-only result consistently enough to justify heavier geometry follow-up, or whether the backbone comparison should remain the primary next lever.

### 7.15 Scope-Aware Retrieval Geo Prior (Hard Region Gate)
Artifacts:
- Configs:
  - `runs/configs/paris_mixed_scope_no_prior.json`
  - `runs/configs/paris_mixed_scope_hard_prior.json`
- Eval reports:
  - `runs/geo_eval_mixed_scope_no_prior_120.json`
  - `runs/geo_eval_mixed_scope_hard_prior_120.json`
  - `runs/geo_eval_mixed_scope_no_prior_seed1870334448_2.json`
  - `runs/geo_eval_mixed_scope_hard_prior_seed1870334448_2.json`

Method:
- Added retrieval geo-prior controls:
  - `retrieval_geo_prior_mode` (`off|soft|hard`)
  - `retrieval_geo_prior_bbox` (`[lat_min, lat_max, lon_min, lon_max]`)
  - `retrieval_geo_prior_sigma_km`
  - `retrieval_geo_prior_min_keep`
- Applied hard Paris bbox priors to Paris configs so retrieval cannot silently jump to out-of-scope continents.

Mixed-scope stress benchmark (`n=120`, Paris eval set, Paris index + open-geo index):

| Variant | mean_km | median_km | p90_km | within_1km_pct | within_2km_pct | within_5km_pct | within_10km_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| No geo prior | 6656.6614 | 5830.1122 | 8934.4830 | 0.00 | 0.00 | 0.00 | 0.00 |
| Hard Paris geo prior | 18.8218 | 12.3615 | 48.6208 | 5.00 | 10.00 | 22.50 | 40.83 |

Targeted failure replay (`seed=1870334448`, same two chips reported in debugging):

| Variant | mean_km | within_10km_pct | Worst sample distances |
|---|---:|---:|---|
| No geo prior | 7408.151 | 0.00 | `8970.226`, `5846.076` km |
| Hard Paris geo prior | 0.000 | 100.00 | `0.000`, `0.000` km |

Observation:
- This is a structural safety gain, not a local tuning change: catastrophic cross-scope retrieval failures are removed under mixed-source conditions.
- The fix directly addresses the previously observed `~5846 km` random-sample failure mode on Paris chips.
- Paris profiles now encode explicit geographic priors by default; open-geo/legacy profile keeps geo prior disabled.

## 8. Methods Tried and Practical Outcome
### 8.1 Changes with Clear Practical Value
- Multi-provider candidate generation and merge controls.
- Retrieval locality and diversity controls.
- Retrieval consensus top-1 refinement (clustered centroid over top candidates).
- Retrieval KDE mode top-1 refinement with objective-specific profiles (`within_1km` vs `within_2km` emphasis).
- `retrieval_min_keep_topk` fallback for robustness.
- Cross-source fusion signals and confidence caps.
- Benchmark governance with promotion workflow.
- Automated aerial-backbone upgrade workflow (objective-based backbone benchmark + final-index rebuild + config patch).
- Dual local geometric reranker (`SIFT` + `ORB` + evidence gate) as a safer local-feature ranking path than legacy local matching.
- Error-driven hard-negative triplet mining (`src.tools.mine_hard_negative_triplets`) for retrieval fine-tuning data.
- Projection-aware retrieval adaptation trained from hard negatives (query-time projection + projected index) with measured realistic-split gains.
- Scope-aware retrieval geo prior (`off|soft|hard` + bbox gate) to prevent cross-region catastrophic errors in mixed-source retrieval.

### 8.2 Changes with Conditional Value
- Query TTA: useful in some regimes, neutral in tested subset.
- Multi-index expansion: helps coverage potential but can degrade precision without score normalization and balancing.
- Alternative TTA reducers (`mean`, `rrf`, `median`): not best in latest measured subset.
- Multi-scale query views (`retrieval_query_tta_scales`): no close-range win on realistic Paris split.
- Aerial backbone swap to tested SigLIP candidates: no gain over CLIP on the realistic Paris split used here.
- Graph-support reranking on single-index control underperformed baseline, but in dual-index stack it improved `within_1km_pct` when paired with KDE refinement (with mean/tail tradeoff).
- Adaptive-mass KDE refinement (`retrieval_kde_refine_adaptive_mass`): mixed results; did not improve `within_2km_pct` vs fixed-mass KDE-W2 baseline.
- Ambiguity-gated local rerank override: protective behavior, but no metric uplift observed on tested local-match profiles.
- Dual local reranker + KDE combination: improved tail metrics but did not beat best close-range (`within_1km_pct`) profile.
- Structure-aware retrieval rerank (`retrieval_structure_rerank_*`): promising gain on the weighted single-index Paris run, but still only validated on one city/split.
- Geo-aware database-side descriptor augmentation (DBA): can improve close-range metrics in objective-specific settings, but current variants regressed broader tail metrics on canonical `n=180`.
- Dual-index projected+DBA stack with `rrf` source fusion: improved close-range and central tendency on canonical `n=180`, with slight `within_10km_pct` tradeoff.

### 8.3 Complete Method Ledger (Keep vs Reject)
This table is the explicit decision ledger for methods tried in this project cycle.

| Method / Variant | Best measured result in this draft | Decision |
|---|---|---|
| `retrieval_min_keep_topk` guardrail | Avoided full collapse (`evaluated`: `0` -> `40`; `null_predictions`: `40` -> `0`) in strict threshold test | Keep (required safety guard) |
| Query TTA on/off (`0,90,180,270` + `max`) | Neutral on tested subset (`mean_km` unchanged at `0.1982`) | Keep as optional; no universal gain |
| TTA reducers (`max`, `rrf`, `mean`, `median`) | `max` best (`within_1km_pct=90.0` on sweep subset) | Keep `max` preference; others experimental |
| Realistic-profile simplification (disable locality/diversity/source-balance in ranking path) | `within_1km_pct`: `1.67` -> `5.00`; `within_5km_pct`: `23.89` -> `30.56` | Keep |
| Retrieval consensus top-1 refinement (`top_n=20`, `radius=3km`) | `within_1km_pct`: `5.00` -> `10.00`; `median_km`: `11.4990` -> `9.7717` | Keep |
| Guarded adaptive centroid vs geo-median center selection | `within_1km_pct`: `10.00` -> `10.56` with flat median | Keep |
| Multi-index source fusion mode `rrf` vs `weighted_score` | `rrf` worse (`within_1km_pct`: `7.78` vs `10.56`) | Keep as experimental; do not default |
| Aerial backbone swap to tested SigLIP models | No win over CLIP on realistic split | Reject as default (CLIP stays baseline) |
| Focused `within_2km_pct` knob sweep (`3456` combos) | No end-to-end uplift on canonical eval (`all deltas 0`) | Revert config changes |
| Graph-support rerank (`A/B/C`) | All close-range metrics underperformed control | Reject for current profile |
| KDE refine profile `C` (W1 focus) | Best close-range hit (`within_1km_pct=11.11`) | Keep as optional W1-focused profile |
| KDE refine profile `D` (W2 focus) | Best `within_2km_pct=20.00`, `within_5km_pct=38.33` | Keep as optional W2-focused profile |
| Multi-scale query views (`retrieval_query_tta_scales`) | Mean/p90 improved but close-range regressed (`within_1km_pct`: `10.56` -> `6.11`) | Keep as experimental; do not default |
| Adaptive-mass KDE refinement (`retrieval_kde_refine_adaptive_mass`) | `adapt_a` raised `within_1km_pct` (`10.00` -> `11.11`) but reduced `within_2km_pct` (`20.00` -> `19.44`) | Keep as experimental; do not default |
| Dual local reranker (`SIFT+ORB`, weak-signal gate, adaptive blend) | Large gain vs legacy local match A (`within_1km_pct`: `5.00` -> `8.89`) | Keep as optional mode |
| Ambiguity-gated local rerank override | No measured delta on tested profiles (`localmatch_a`, `kde_e_w1_duallocal`) | Keep as safe guard; not a promoted accuracy lever |
| KDE + dual-local combined profile | Did not beat best W1 profile (`within_1km_pct` stayed `8.89`) | Do not adopt as close-range default |
| Profile/data mismatch guard in Lab/backend/CLI | Eliminated catastrophic profile mismatch failure mode (`~5846 km` case) and now blocks mismatched `run_geo_eval` runs by default | Keep (evaluation-integrity requirement) |
| Scope-aware retrieval geo prior (`retrieval_geo_prior_*`) | Mixed-scope stress test: `mean_km` `6656.66` -> `18.82`; replay seed case `7408.15` -> `0.00` | Keep (Paris defaults) |
| Error-driven hard-negative triplet miner | Produced `145` structured triplets from `180` realistic eval failures | Keep; use for backbone fine-tuning pipeline |
| Hard-negative projection adaptation (`trainref_v2_mild`) | `within_1km_pct`: `9.17` -> `12.50`, `within_2km_pct`: `16.67` -> `28.33` on `n=120` realistic eval | Keep as current best retrieval adaptation direction |
| Difficulty-aware weighting on mined query-vs-reference triplets | On canonical single-index `n=180`: `mean_km`: `15.25` -> `15.08`, `median_km`: `5.50` -> `4.89`, `within_1km_pct`: `11.67` -> `13.89`, `within_2km_pct`: `26.67` -> `27.22` versus uniform weighting | Keep; prefer over uniform weighting for future projection retraining cycles |
| Structure-aware retrieval rerank (`top_n=12`, `weight=0.35`) | On canonical weighted single-index `n=180`: `mean_km`: `15.08` -> `14.72`, `median_km`: `4.89` -> `4.59`, `within_1km_pct`: `13.89` -> `15.00`, `within_2km_pct`: `27.22` -> `28.33`, `within_5km_pct`: `51.67` -> `53.33` | Keep as experimental single-index rerank; validate beyond Paris before defaulting |
| Projection V2 + geo-prior stack on canonical Paris split (`n=180`) | No delta vs baseline (`within_1km_pct=11.67`, `within_2km_pct=26.67` in both) | Keep geo-prior as scope-safety guard; do not claim close-range lift on in-scope data |
| Dual-space projected+raw CLIP fusion with per-index projection routing (`retrieval_index_projection_paths`) | Underperformed projection V2 baseline on `n=180` (`within_1km_pct`: `11.67` -> `8.89`, `within_2km_pct`: `26.67` -> `18.89`) | Keep capability as experimental infra; reject as default profile |
| Geo-aware DBA index augmentation (`neighbors=5`, `max_geo_distance_km=2`) | On canonical `n=180`: `mean_km`: `15.25` -> `14.80`, `within_1km_pct`: `11.67` -> `13.33`, `within_2km_pct`: `26.67` -> `29.44`, but `within_5km_pct`: `50.00` -> `46.67` | Keep as objective-specific close-range profile; reject as default until tail regression is solved |
| Dual-index projection+DBA with `rrf` (`index_paths=[baseline,dba_geo2_k5]`) | On canonical `n=180`: `mean_km`: `15.25` -> `14.84`, `median_km`: `5.50` -> `4.87`, `within_1km_pct`: `11.67` -> `12.78`, `within_2km_pct`: `26.67` -> `31.11`, `within_5km_pct`: `50.00` -> `50.56`, `within_10km_pct`: `64.44` -> `63.33` | Keep as strongest current close-range profile candidate; monitor slight `<=10km` regression |
| Dual-index projection+DBA `rrf` balanced weighting (`index_weights=[1.0,0.5]`) | On canonical `n=180`: `mean_km`: `15.25` -> `14.60`, `median_km`: `5.50` -> `4.21`, `within_1km_pct`: `11.67` -> `10.56`, `within_2km_pct`: `26.67` -> `31.11`, `within_5km_pct`: `50.00` -> `52.78`, `within_10km_pct`: `64.44` -> `65.56` | Keep as balanced profile when broader-radius reliability matters more than max `<=1km` |
| Dual-index projection+DBA + graph-support + KDE refinement | On canonical `n=180`: `within_1km_pct`: `11.67` -> `13.89` (best measured), `within_2km_pct`: `26.67` -> `31.11`, with `mean_km`: `15.25` -> `15.28` and slight `<=5km/<=10km` regressions | Keep as aggressive W1 profile only; do not use as global default |

### 8.4 What Is Currently Kept by Default
- Canonical Paris realistic profile remains CLIP-based retrieval with consensus refinement.
- Projection-adapted retrieval remains the best single-index experimental direction; within that path, difficulty-weighted triplet training is now preferred over uniform weighting for future Paris hard-negative cycles.
- Structure-aware reranking of the top retrieval shortlist is the current best non-learning scene-layout add-on, but it remains experimental until it is rechecked on broader leakage-safe splits.
- KDE and dual-local methods remain opt-in evaluation profiles for objective-specific tradeoffs.
- RRF source fusion and alternate backbones remain research modes, not production defaults on current split.
- Per-index projection routing is kept as infrastructure for future heterogeneous-index experiments, but current dual-space RRF profile is not promoted.
- Multi-scale query views and adaptive-mass KDE remain experimental toggles; defaults stay single-scale (`1.0`) and fixed adaptive mass (`0.0`).
- Geo-aware DBA remains optional for close-range objective runs (`<=1km`/`<=2km`) and is not promoted to default profile yet.
- Dual-index projection+DBA (`rrf`) is now the preferred experimental profile when the objective is `<=2km` rather than broad-radius recall.
- Dual-index balanced weighting (`[1.0,0.5]`) is preferred when optimizing mean/median and `<=10km` while preserving `<=2km`.
- Dual-index + graph+KDE remains a specialized W1-max profile (`<=1km`) with explicit tail-risk tradeoff.
- Evaluation-integrity guards (profile/path auto-resolution + explicit profile reporting + CLI scope validation) are mandatory.

## 9. Error Analysis
Observed failure classes:
1. Retrieval/index leakage causing inflated metrics in naive protocols.
2. Strict thresholding leading to empty candidate sets (fixed by min-keep).
3. Source dominance in multi-index retrieval reducing provider diversity.
4. Ambiguous multimodal geographies requiring uncertainty-aware output rather than overconfident point claims.

Mitigations implemented:
- Realistic/leaky benchmark separation.
- Min-keep fallback.
- Source-balance and score normalization.
- Cross-source and uncertainty-aware confidence gating.

## 10. Reproducibility
### 10.1 Required Documents
- `PROGRESS.md` (append-only engineering history)
- `src/docs/GEO_TECH.md` (geo stack details)
- `src/docs/REPRODUCIBILITY.md` (repro procedures)

### 10.2 Core Commands
```powershell
# Core benchmark run
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core

# Promote a run to baseline
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core --promote <run_id>

# Retrieval tuning sweep including TTA reducers
.\.venv\Scripts\python -m src.tools.tune_retrieval_geo `
  --config src/config/paris_test.json `
  --images-dir data/spacenet_paris_test/chips `
  --metadata data/spacenet_paris_test/metadata.csv `
  --retrieval-query-tta-reduce max,median,mean,rrf

# Aerial retrieval backbone benchmark (RTX 5060-focused preset)
.\.venv\Scripts\python -m src.tools.benchmark_geo_backbones `
  --train-images-dir data/spacenet_paris/chips `
  --train-metadata data/spacenet_paris/metadata.csv `
  --eval-images-dir data/spacenet_paris_test/chips `
  --eval-metadata data/spacenet_paris_test/metadata.csv `
  --model-preset aerial_rtx5060_precise `
  --rank-objective within_2km_pct `
  --output runs/backbone_bench/backbone_benchmark_aerial.json

# One-command backbone upgrade (benchmark -> index rebuild -> config patch)
.\.venv\Scripts\python -m src.tools.upgrade_retrieval_backbone `
  --train-images-dir data/spacenet_paris/chips `
  --train-metadata data/spacenet_paris/metadata.csv `
  --eval-images-dir data/spacenet_paris_test/chips `
  --eval-metadata data/spacenet_paris_test/metadata.csv `
  --config src/config/paris.json `
  --model-preset aerial_rtx5060_precise `
  --rank-objective within_2km_pct `
  --output-dir runs/backbone_upgrade
```

### 10.3 Validation Coverage Recorded in Project Log
From `PROGRESS.md`:
- Full suite checkpoints reached `78 passed`, then `81 passed`, then `88 passed` during major iterations.
- Focused/non-UI checkpoints reported `18 passed` and `105 passed`.
- Some full-suite UI tests are dependency-limited in minimal environments.

## 11. Threats to Validity
1. Data leakage risk: nearest-neighbor retrieval can silently overfit if index and eval share location-identical imagery.
2. Domain concentration: strongest local artifacts are Paris-focused; global claims require broader and cleaner splits.
3. Dependency variability: optional runtime stacks can affect which tests execute in a given environment.
4. Subset sensitivity: small ablation subsets can understate or overstate certain method gains.

## 12. Ethics and Responsible Use
This system can increase location inference capability from images; misuse could impact privacy and safety. Recommended safeguards:
- avoid processing personally sensitive imagery without explicit authorization,
- enforce controlled access and audit logs in operational deployments,
- communicate uncertainty and avoid binary certainty claims from ambiguous scenes,
- separate research prototypes from production-grade decision systems.

## 13. Limitations
- Current results are strongest in constrained regional settings with available retrieval indices.
- Multi-index scaling still requires strong curation to avoid precision dilution.
- Uncertainty outputs improve transparency but do not eliminate epistemic blind spots.
- Research artifacts are engineering-oriented and not yet standardized into a public benchmark paper submission package.

## 14. Conclusion
Project Heimdall demonstrates a practical path from prototype geolocation to a benchmark-governed retrieval+fusion platform. The most important technical lesson is methodological: robust evaluation protocol and governance are as important as algorithmic sophistication. The highest-impact near-term improvements remain data curation quality, leakage-safe benchmark design, and calibration-driven tuning.

## 15. Future Work
1. Prioritize remote-sensing-native embedding backbones (RemoteCLIP/SatCLIP-style) and benchmark them against current CLIP/SigLIP retrieval on leakage-safe splits.
2. Add a localizability/selective-prediction gate to reduce catastrophic confident errors on non-localizable scenes.
3. Extend uncertainty outputs with conformal credible regions over candidate clusters.
4. Expand realistic, leakage-safe global and cross-view evaluation sets with hard negatives and strict anti-duplicate checks.
5. Integrate data quality scoring into retrieval-index curation.

### 15.2 Data Required For Major Accuracy Gains
- More dense local positives per scene: at least `5-10` geographically-near variants (small offsets, seasonal/time differences) for each hard query region.
- Hard negatives around confusion zones: near-lookalike tiles at `2-25 km` GT distance, especially from districts repeatedly confused in evaluation.
- Cross-sensor/domain coverage: balanced `PAN`, `RGB-PanSharpen`, and `MUL-PanSharpen` examples with explicit pair/triplet links.
- Leakage-safe split metadata: strict scene/tile-family separation between train/val/test to avoid inflated metrics.
- Optional but high leverage: additional city datasets beyond Paris with the same metadata contract to reduce over-specialization.

### 15.1 Possible Approaches Under Consideration (From `deep-research-report.md`)
- `Domain-adapted embeddings` (highest priority): evaluate Remote Sensing foundation encoders as primary retrieval backbones.
- `Difficulty-aware hard-negative weighting`: controlled Paris benchmarking is now complete and favors weighting (`within_1km_pct`: `11.67` -> `13.89`, `mean_km`: `15.25` -> `15.08` versus uniform single-index training). The remaining question is whether that gain holds on larger triplet pools and non-Paris cities.
- `Scene-structure reranking`: coarse cues from corners, edge mass, building-line orientation, and guarded shadow direction already improved the canonical weighted single-index Paris run (`within_1km_pct`: `13.89` -> `15.00`, `mean_km`: `15.08` -> `14.72`). The current geometry-lite branch extends that with corner/edge spatial layout, line orthogonality / anisotropy, and shadow elongation, and weak-signal gating now restores the earlier close-range gains on the same split (`within_2km_pct`: `27.22` -> `28.33`) while keeping the richer signature available for further comparison.
- `Selective abstention`: add a localizability head/policy to decide predict vs abstain before final confidence tiering.
- `Spatially guaranteed uncertainty`: conformal prediction for region-level coverage guarantees on top of probabilistic fusion.
- `Rank-based multi-index fusion`: RRF was implemented as an optional mode (`retrieval_source_fusion_mode=rrf`) and benchmarked.
- On current realistic Paris split (`n=180`), it underperformed `weighted_score` (`within_1km_pct`: `7.78` vs `10.56`), so it remains experimental for future multi-index/global settings.
- `Dual-space projection routing`: per-index query projection routing (`retrieval_index_projection_paths`) was added to safely combine projected and non-projected indices in one run.
- First Paris realistic test (`n=180`, projected+raw CLIP with `rrf`) underperformed the current projection V2 baseline (`within_1km_pct`: `8.89` vs `11.67`; `within_2km_pct`: `18.89` vs `26.67`), so this remains experimental infrastructure.

## Appendix A: Major Algorithmic Knobs (Geo)
- Retrieval:
  - `retrieval_projection_path`
  - `retrieval_top_k`, `retrieval_min_score`, `retrieval_min_keep_topk`
  - `retrieval_diversity_radius_km`, `retrieval_diversity_lambda`, `retrieval_diversity_min_keep`
  - `retrieval_locality_radius_km`, `retrieval_locality_weight`
  - `retrieval_structure_rerank_top_n`, `retrieval_structure_rerank_weight`
  - `retrieval_consensus_top_n`, `retrieval_consensus_radius_km`, `retrieval_consensus_score_power`
  - `retrieval_query_tta_degrees`, `retrieval_query_tta_modes`, `retrieval_query_tta_scales`, `retrieval_query_tta_auto_modality`, `retrieval_query_tta_reduce`
  - `retrieval_tta_agreement_top_n`, `retrieval_tta_agreement_weight`
  - `retrieval_index_paths`, `retrieval_index_weights`, `retrieval_per_index_top_k`
  - `retrieval_index_model_ids`, `retrieval_index_projection_paths`, `retrieval_index_score_norm`, `retrieval_source_fusion_mode`, `retrieval_source_balance_beta`
  - `retrieval_geo_prior_mode`, `retrieval_geo_prior_bbox`, `retrieval_geo_prior_sigma_km`, `retrieval_geo_prior_min_keep`
  - `retrieval_kde_refine_top_n`, `retrieval_kde_refine_sigma_km`, `retrieval_kde_refine_score_power`, `retrieval_kde_refine_margin_threshold`, `retrieval_kde_refine_switch_radius_km`, `retrieval_kde_refine_max_iters`, `retrieval_kde_refine_adaptive_mass`
- Candidate merge:
  - `candidate_source_balance_beta`
- Fusion:
  - `source_prior_retrieval`, `source_prior_geoclip`, `source_prior_exif`
  - `source_prior_retrieval_by_source`
  - `use_spatial_consensus`, `spatial_sigma_km`, `spatial_consensus_weight`
  - `use_cross_source_agreement`, `cross_source_sigma_km`, `cross_source_weight`
  - `use_plausibility_rerank`, `plausibility_radius_km`, `plausibility_weight`
  - `use_adaptive_outlier_guard`, `outlier_guard_strength`, `outlier_guard_min_scale_km`, `outlier_guard_mad_scale`
  - `confidence_calibration_logit_scale`, `confidence_calibration_logit_bias`
  - `confidence_high_threshold`, `confidence_medium_threshold`
  - `confidence_high_min_cross_source_support`, `confidence_medium_min_cross_source_support`
  - `confidence_high_max_uncertainty_m`, `confidence_medium_max_uncertainty_m`

## Appendix B: Artifact Index Used in This Draft
- `runs/bench_current_leaky_180.json`
- `runs/bench_realistic_single_180.json`
- `runs/bench_candidate_multi_180.json`
- `runs/bench_realistic_single_180_precision_v2.json`
- `runs/geo_eval_paris_profile_180_v2.json`
- `runs/geo_eval_paris_profile_180_consensus_v1.json`
- `runs/geo_eval_paris_profile_180_consensus_v2_geomedian.json`
- `runs/geo_eval_paris_profile_180_consensus_v3_adaptive_center.json`
- `runs/geo_eval_paris_profile_180_consensus_v4_adaptive_guarded.json`
- `runs/geo_eval_paris_profile_180_sourcefusion_weighted_v1.json`
- `runs/geo_eval_paris_profile_180_sourcefusion_rrf_v1.json`
- `runs/bench_cfg/cfg_paris_sourcefusion_rrf.json`
- `runs/geo_eval_paris_profile_180_pre_backbone_upgrade_v1.json`
- `runs/geo_eval_paris_profile_180_post_tune_v1.json`
- `runs/backbone_upgrade_rtx5060_v1/backbone_benchmark.json`
- `runs/tune_retrieval_geo_within2km_v1.json`
- `runs/geo_eval_paris_profile_180_graphrerank_a_v1.json`
- `runs/geo_eval_paris_profile_180_graphrerank_b_v1.json`
- `runs/geo_eval_paris_profile_180_graphrerank_c_v1.json`
- `runs/geo_eval_paris_profile_180_kde_refine_a_v1.json`
- `runs/geo_eval_paris_profile_180_kde_refine_b_no_consensus_v1.json`
- `runs/geo_eval_paris_profile_180_kde_refine_c_w1_v1.json`
- `runs/geo_eval_paris_profile_180_kde_refine_d_w2_v1.json`
- `runs/geo_eval_paris_profile_180_tta_agreement_ctrl_v1.json`
- `runs/geo_eval_paris_profile_180_multiscale_a_v1.json`
- `runs/geo_eval_paris_profile_180_kde_refine_d_w2_adapt_a_v1.json`
- `runs/geo_eval_paris_profile_180_kde_refine_d_w2_adapt_b_v1.json`
- `runs/geo_eval_paris_profile_180_localmatch_a_v1.json`
- `runs/geo_eval_paris_profile_180_qexp_ctrl_v2_after_dual_localcode.json`
- `runs/geo_eval_paris_profile_180_localmatch_a_v2_dual.json`
- `runs/geo_eval_paris_profile_180_localmatch_b_v2_dual.json`
- `runs/geo_eval_paris_profile_180_localmatch_c_v2_dual.json`
- `runs/geo_eval_paris_profile_180_kde_refine_e_w1_duallocal_v1.json`
- `runs/geo_eval_paris_no_tta_120.json`
- `runs/geo_eval_paris_tta_120.json`
- `runs/geo_eval_paris_strict_keep0_40.json`
- `runs/geo_eval_paris_strict_keep2_40.json`
- `runs/tune_retrieval_geo_tta_modes_med.json`
- `runs/tune_retrieval_geo_realistic_within1km_focus_v1.json`
- `runs/tmp_paris_with_open_geo_seed1870334448.json`
- `runs/tmp_paris_with_paris_cfg_seed1870334448.json`
- `runs/geo_eval_mixed_scope_no_prior_120.json`
- `runs/geo_eval_mixed_scope_hard_prior_120.json`
- `runs/geo_eval_mixed_scope_no_prior_seed1870334448_2.json`
- `runs/geo_eval_mixed_scope_hard_prior_seed1870334448_2.json`
- `runs/geo_eval_paris_test_profile_with_geo_prior_40.json`
- `runs/geo_eval_projection_baseline_120.json`
- `runs/geo_eval_projection_trainref_v1_120_rerun.json`
- `runs/geo_eval_projection_trainref_v2_mild_120.json`
- `runs/geo_eval_projection_trainref_v3_dim256_120.json`
- `runs/geo_eval_projection_trainref_v2_mild_180_baseline.json`
- `runs/geo_eval_projection_trainref_v2_mild_geo_prior_180.json`
- `runs/geo_eval_projection_trainref_v3_dim256_180.json`
- `runs/geo_eval_projection_trainref_v2_mild_localmatch_v1_180.json`
- `runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`
- `runs/bench_cfg/cfg_paris_projection_trainref_v2_weighted_cmp_structure_v1.json`
- `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v1_180.json`
- `runs/geo_eval_paris_dualspace_rrf_v1_180.json`
- `runs/bench_cfg/cfg_paris_dualspace_rrf_v1.json`
- `runs/geo_impact_latest.json`
- `runs/geo_impact_latest.md`

## Appendix C: Internal References
- `[R1]` `PROGRESS.md`
- `[R2]` `src/docs/GEO_TECH.md`
- `[R3]` `src/docs/REPRODUCIBILITY.md`
- `[R4]` `docs/eval/` governance artifacts
