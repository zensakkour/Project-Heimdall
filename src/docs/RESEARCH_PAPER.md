# Project Heimdall: Full Research Paper Draft

Version: v1.0  
Date: April 16, 2026

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

## 8. Methods Tried and Practical Outcome
### 8.1 Changes with Clear Practical Value
- Multi-provider candidate generation and merge controls.
- Retrieval locality and diversity controls.
- Retrieval consensus top-1 refinement (clustered centroid over top candidates).
- `retrieval_min_keep_topk` fallback for robustness.
- Cross-source fusion signals and confidence caps.
- Benchmark governance with promotion workflow.
- Automated aerial-backbone upgrade workflow (objective-based backbone benchmark + final-index rebuild + config patch).

### 8.2 Changes with Conditional Value
- Query TTA: useful in some regimes, neutral in tested subset.
- Multi-index expansion: helps coverage potential but can degrade precision without score normalization and balancing.
- Alternative TTA reducers (`mean`, `rrf`, `median`): not best in latest measured subset.
- Aerial backbone swap to tested SigLIP candidates: no gain over CLIP on the realistic Paris split used here.

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

### 15.1 Possible Approaches Under Consideration (From `deep-research-report.md`)
- `Domain-adapted embeddings` (highest priority): evaluate Remote Sensing foundation encoders as primary retrieval backbones.
- `Selective abstention`: add a localizability head/policy to decide predict vs abstain before final confidence tiering.
- `Spatially guaranteed uncertainty`: conformal prediction for region-level coverage guarantees on top of probabilistic fusion.
- `Rank-based multi-index fusion`: RRF was implemented as an optional mode (`retrieval_source_fusion_mode=rrf`) and benchmarked.
: On current realistic Paris split (`n=180`), it underperformed `weighted_score` (`within_1km_pct`: `7.78` vs `10.56`), so it remains experimental for future multi-index/global settings.

## Appendix A: Major Algorithmic Knobs (Geo)
- Retrieval:
  - `retrieval_top_k`, `retrieval_min_score`, `retrieval_min_keep_topk`
  - `retrieval_diversity_radius_km`, `retrieval_diversity_lambda`, `retrieval_diversity_min_keep`
  - `retrieval_locality_radius_km`, `retrieval_locality_weight`
  - `retrieval_consensus_top_n`, `retrieval_consensus_radius_km`, `retrieval_consensus_score_power`
  - `retrieval_query_tta_degrees`, `retrieval_query_tta_reduce`
  - `retrieval_index_paths`, `retrieval_index_weights`, `retrieval_per_index_top_k`
  - `retrieval_index_model_ids`, `retrieval_index_score_norm`, `retrieval_source_fusion_mode`, `retrieval_source_balance_beta`
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
- `runs/geo_eval_paris_profile_180_sourcefusion_weighted_v1.json`
- `runs/geo_eval_paris_profile_180_sourcefusion_rrf_v1.json`
- `runs/geo_eval_paris_no_tta_120.json`
- `runs/geo_eval_paris_tta_120.json`
- `runs/geo_eval_paris_strict_keep0_40.json`
- `runs/geo_eval_paris_strict_keep2_40.json`
- `runs/tune_retrieval_geo_tta_modes_med.json`
- `runs/tune_retrieval_geo_realistic_within1km_focus_v1.json`
- `runs/geo_impact_latest.json`
- `runs/geo_impact_latest.md`

## Appendix C: Internal References
- `[R1]` `PROGRESS.md`
- `[R2]` `src/docs/GEO_TECH.md`
- `[R3]` `src/docs/REPRODUCIBILITY.md`
- `[R4]` `docs/eval/` governance artifacts
