# Project Heimdall: Research Paper Draft

Version: v1.2
Date: May 10, 2026
Author: Zein Sakkour

Companion external landscape review: `src/docs/MARKET_RESEARCH.md`.
Chronological experiment ledger with the latest branch-era before/after metrics: `research.md`.

## Title
Project Heimdall: Benchmark-Governed Geolocation from Imagery via Multi-Provider Retrieval, Robust Probabilistic Fusion, and Uncertainty-Aware Analysis

## Author
Zein Sakkour

## Abstract
This paper presents my research and engineering work on Project Heimdall, an image geolocation system developed from January to April 2026. Heimdall integrates oriented object detection, multi-provider geolocation candidate generation, retrieval index search, posterior fusion, and confidence-aware uncertainty outputs. The project progressed through multiple algorithmic iterations: retrieval diversification and locality reranking, query-time test-time augmentation (TTA), multi-index weighted retrieval, source-balanced candidate selection, score normalization across heterogeneous indices, source-aware fusion priors, cross-source agreement and spatial consensus likelihoods, adaptive outlier suppression, and temporal posterior filtering. I report controlled benchmark artifacts and ablations from repository runs rather than isolated anecdotal examples. A central finding is that evaluation protocol quality can dominate apparent model quality: leakage-prone evaluation can show near-perfect performance, while realistic split evaluation remains substantially more difficult. A second major finding is that retrieval-side tuning plateaued near `14-15 km` mean error on the canonical realistic Paris benchmark; the work therefore shifted into realistic street-to-aerial supervision from Mapillary, Panoramax, and IGN orthophotos. The current combined Paris dataset contains `40,000` street-to-aerial pairs, and the first query-only street-to-aerial projection improves the full combined strict probe from `10.97 km` mean error to `9.75 km`, while still falling short of the target `~3 km` mean accuracy. The resulting system emphasizes reproducibility, benchmark governance, traceable accuracy claims, and explicit separation between validated improvements and open research gaps.

## Keywords
Image geolocation, retrieval, probabilistic fusion, uncertainty estimation, benchmark governance, aerial imagery, CLIP embeddings.

## 1. Introduction
Image geolocation aims to infer where an image was captured from visual evidence alone. Practical deployments must satisfy more than point accuracy: confidence must be calibrated, failure modes must be inspectable, and iterative model work must remain reproducible under changing data and dependencies. Project Heimdall was built as an end-to-end engineering research platform for this objective.

I began Project Heimdall with basic candidate generation and heuristic scoring, then evolved it into a benchmark-governed retrieval and fusion stack. The current system supports multi-source geolocation hypotheses and robust fusion diagnostics while preserving operator-facing visibility through a local analysis interface. This paper is written as a research account of that process: what I built, what improved, what failed, and what evidence supports the current direction.

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
- Optional sun/shadow consistency likelihood when capture time and observed shadow direction are available.
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

### 4.6 Mathematical Formulation
This section defines the mathematical objects used throughout the project. The notation is intentionally explicit because many apparent improvements in geolocation can be explained by evaluation leakage, index composition, or ranking protocol rather than by a genuinely better model.

#### 4.6.1 Image Geolocation as Ranked Candidate Inference
Let a query image be denoted by `q`. The geolocation system returns a ranked set of candidates:

```text
C(q) = {c_1, ..., c_K}
c_i = (lat_i, lon_i, s_i, source_i, metadata_i)
```

where `(lat_i, lon_i)` is the candidate coordinate, `s_i` is the candidate score, and `source_i` records the provider or retrieval index that produced the candidate. The final point estimate is either the highest posterior candidate or a spatially aggregated estimate over a local candidate cluster:

```text
y_hat(q) = argmax_c P(c | q)
```

For metric reporting, the ground-truth coordinate is `y = (lat, lon)`, and the prediction error is the geodesic distance:

```text
e(q) = d_haversine(y_hat(q), y)
```

#### 4.6.2 Retrieval Embeddings and Similarity Search
For image retrieval, each image is embedded by an encoder `f_theta`:

```text
z_q = normalize(f_theta(q))
z_i = normalize(f_theta(x_i))
```

The base retrieval score is cosine similarity:

```text
sim(q, x_i) = z_q^T z_i
```

The index returns the `K` highest scoring reference images. In a single-index setup:

```text
TopK(q) = argsort_i(z_q^T z_i)[1:K]
```

For multiple indices, each index `j` has its own reference pool, optional model ID, optional query projection, and weight. Weighted-score fusion uses a normalized score:

```text
score_ij = w_j * norm_j(z_q^T z_ij)
```

Rank reciprocal fusion (RRF) instead converts per-source ranks to a rank score:

```text
rrf(c) = sum_j w_j / (k_rrf + rank_j(c))
```

RRF was kept as infrastructure because it can reduce score-scale mismatch, but it did not universally improve the Paris realistic benchmarks.

#### 4.6.3 Query-Time Projection and Cross-View Projection
The earlier aerial-only projection experiments used a shared embedding-space projection. For realistic street-to-aerial retrieval, the better formulation is asymmetric: street queries are projected into the aerial reference embedding space while the aerial index stays fixed.

For a street query `q_s` and aerial reference `a_i`:

```text
u_q = normalize(f_theta(q_s))
v_i = normalize(f_theta(a_i))
u'_q = normalize(W u_q + b)
sim_cross(q_s, a_i) = u'_q^T v_i
```

The first combined realistic model pass trained only `(W, b)` and kept the base encoder frozen. For each training row, the triplet contains one street query, one or more positives, and multiple geographic hard negatives:

```text
T = (q_s, P(q_s), N(q_s))
P(q_s) = {a_p : d(a_p, q_s) <= r_pos}
N(q_s) = {a_n : r_min <= d(a_n, q_s) <= r_max}
```

The triplet objective optimizes the margin between the hardest positive and hardest negative:

```text
s_pos = min_{a_p in P(q_s)} u'_q^T v_p
s_neg = max_{a_n in N(q_s)} u'_q^T v_n
L_triplet = max(0, m + s_neg - s_pos)
```

The optional contrastive cross-entropy term treats the positive set as the correct class against sampled negatives:

```text
L_ce = -log( exp(s_pos / tau) / (exp(s_pos / tau) + sum_n exp(s_n / tau)) )
L = alpha(q_s) * (L_triplet + lambda_ce * L_ce)
```

where `alpha(q_s)` is the mined triplet weight. In the current implementation, harder triplets receive larger weights when the negatives are geographically close and dense. This gives the model more pressure on cases that are most likely to affect `within_1km`, `within_2km`, and `within_5km` metrics.

#### 4.6.4 Structure and Geometry Reranking
The structure-aware reranker augments embedding similarity with a low-level scene-layout similarity. For each image, a layout descriptor `g(x)` is extracted from edge density, corner density, dominant line orientation, and guarded shadow/dark-mass cues. The candidate score becomes:

```text
score_final(q, x_i) = (1 - beta) * score_embed(q, x_i) + beta * score_layout(g(q), g(x_i))
```

The value of `beta` is gated by evidence strength. If the image lacks strong structural signal, the system reduces the layout contribution so weak edge noise does not override the embedding score. This explains why the geometry branch is treated as an experimental reranker rather than a replacement for learned retrieval.

#### 4.6.5 Consensus, KDE, and Database-Side Augmentation
Several top-K refinement methods were tested after initial retrieval. Consensus refinement assumes that a dense local cluster among top candidates is more trustworthy than an isolated high-similarity outlier:

```text
support(c_i) = sum_{c_j in TopK} 1[d(c_i, c_j) <= r] * score(c_j)^gamma
score_consensus(c_i) = score(c_i) * support(c_i)
```

KDE refinement estimates a spatial mode over candidates:

```text
p(y | q) proportional sum_i score(c_i)^gamma * exp(-d(y, y_i)^2 / (2 sigma^2))
```

Geo-aware database-side augmentation (DBA) smooths reference embeddings using nearby reference neighbors:

```text
z'_i = normalize(w_self z_i + sum_{j in N_geo(i)} w_ij z_j)
```

DBA helped some close-range objectives but also produced tail-risk tradeoffs, so it remains objective-specific rather than a universal default.

#### 4.6.6 Probabilistic Fusion and Uncertainty
Fusion combines candidate evidence in log space. For a candidate location `y_i` from source `s`, the posterior score is:

```text
log P(y_i | q) = log pi_s + log L_retrieval(y_i | q) + log L_spatial(y_i | C) + log L_cross_source(y_i | C) - penalty_outlier(y_i)
```

Here `pi_s` is a source prior, `L_spatial` rewards agreement with nearby candidates, and `L_cross_source` rewards support from independent providers. The adaptive outlier guard uses robust distance statistics so a single high-score but spatially isolated candidate cannot dominate without support.

Uncertainty is estimated from the posterior mass distribution. A credible region is the smallest candidate cluster whose cumulative posterior exceeds a target level:

```text
R_alpha = smallest region such that sum_{y_i in R_alpha} P(y_i | q) >= alpha
```

Confidence tiers then combine posterior mass, cross-source support, and uncertainty radius. This is why the system reports both a point prediction and diagnostics rather than a single unqualified coordinate.

#### 4.6.7 Sun/Shadow Consistency as Physical Evidence
The May 2026 iteration added a benchmarkable path for using capture time as physical evidence. If a query has a timestamp and the detector or sidecar provides an observed shadow azimuth, each candidate location can be checked against an approximate solar position model:

```text
shadow_expected(y_i, t) = (sun_azimuth(y_i, t) + 180 deg) mod 360 deg
delta_shadow = angular_distance(shadow_expected, shadow_observed)
L_shadow(y_i | q, t) = exp(-0.5 * (delta_shadow / sigma_shadow)^2)
```

The candidate log posterior receives `log L_shadow` only when both timestamp and shadow evidence are present. Missing timestamps or missing shadow observations leave the ranking unchanged. This avoids penalizing ordinary images while allowing time-aware disambiguation when the evidence exists.

#### 4.6.8 Candidate Reranking and Local Visual Verification
The May 2026 runtime branch also tested a second-stage candidate reranker because candidate-oracle analysis showed that useful candidates are often present but not top-ranked. Two paths were separated:

1. A lightweight learned candidate-feature scorer using rank, normalized score, source type, local support density, and distance-to-centroid features.
2. A local visual verifier over the top retrieved aerial chips, using the existing dual local feature stack to rescore visually compatible chips before fusion.

The learned feature scorer was kept as infrastructure but not enabled by default because the first leakage-safe probe did not move held-out top-rank metrics. The local visual verifier did improve the full runtime fusion path, so the Paris runtime profile now enables a limited top-12 local match pass.

#### 4.6.9 Leakage-Safe Spatial Splitting
The realistic Paris dataset is split by geographic cells rather than random rows. Let each pair be assigned to a spatial cell:

```text
cell(y) = (floor(lat_m / h), floor(lon_m / h))
```

where `h = 300 m` in the strict combined split. Boundary-buffer cells are excluded when they are too close to another split. The goal is:

```text
min_{i in split A, j in split B} d(y_i, y_j) >= d_min
```

The current combined strict split reports `min_cross_split_distance_m = 1201.23`, which makes it materially safer than the earlier permissive split with meter-scale leakage.

#### 4.6.10 Evaluation Metrics
For a benchmark set `Q`, mean and median error are:

```text
mean_km = (1 / |Q|) sum_{q in Q} e(q)
median_km = median({e(q) : q in Q})
```

Radius accuracy is:

```text
within_R_pct = 100 * |{q : e(q) <= R}| / |Q|
```

The project reports multiple radii because a method can improve medium-range localization while regressing exact close-range localization. The first combined cross-view projection is a concrete example: it improved `mean_km`, `median_km`, `within_2km_pct`, and `within_5km_pct`, but regressed `within_1km_pct`.

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

### 5.4 Phase IV: Data Bottleneck and Realistic Cross-View Training (Apr 29-30, 2026)
- After the retrieval-side plateau near `14-15 km` mean error, I shifted the research direction from additional handcrafted reranking toward realistic paired supervision.
- Built a Paris street-image dataset from Mapillary and Panoramax metadata and imagery.
- Paired street images with IGN orthophoto aerial crops to produce a `40,000` pair street-to-aerial dataset.
- Rebuilt leakage-safe spatial splits with a buffered minimum cross-split distance of `1201.23 m`.
- Built a full `40,000` aerial CLIP index for the combined realistic benchmark.
- Added realistic cross-view triplet mining where the query is a street image, the positive is the paired aerial crop, and negatives are geographically close but wrong aerial crops.
- Added a query-only street-to-aerial projection trainer that maps street embeddings into the fixed aerial embedding space.
- Ran the first combined strict-probe model adaptation: the learned projection improved mean error from `10.97 km` to `9.75 km` and `within_5km_pct` from `12.50%` to `20.42%`, while regressing `within_1km_pct` from `2.92%` to `2.08%`.

### 5.5 Phase V: Physical-Feature Fusion Path (May 9, 2026)
- Wired sidecar capture-time metadata into the shared image metadata helper so EXIF is no longer the only way to activate sun/shadow scoring.
- Added support for ISO timestamps, EXIF-style timestamps, Unix seconds, and Mapillary-style epoch milliseconds.
- Added explicit capture-time injection through `fuse_candidates()` and `HeimdallPipeline.run()` so offline benchmark rows can exercise the same physical feature without mutating image files.
- Updated `run_geo_eval.py` and `tune_geo_fusion.py` to normalize realistic CSV schemas (`street_path`/`lat`/`lon`) and enrich pair rows from street image metadata when the pair CSV does not carry `captured_at`.
- Added regression tests showing that shadow evidence can promote the physically plausible candidate over a higher raw retrieval-score candidate when capture time is available.
- Ran strict-probe fusion sweeps on the Paris runtime profile. The promoted setting increases retrieval temperature (`0.08` -> `0.28`), disables the current spatial-consensus term for this close-range Paris profile, weakens cross-source agreement, keeps a lighter plausibility rerank, disables the adaptive outlier guard for this profile, and enables timestamp-gated shadow scoring. On a fixed 40-sample strict probe, mean error improved from `9.65 km` to `9.26 km`, median from `7.14 km` to `7.10 km`, `<=1 km` from `0.00%` to `2.50%`, `<=5 km` from `25.00%` to `32.50%`, and `<=10 km` from `62.50%` to `67.50%`.
- Added candidate-level reranking infrastructure and a supervised training tool (`src/tools/train_geo_candidate_reranker.py`). The first learned feature scorer was rejected for default use because it did not improve the fixed held-out 40-sample probe, but the infrastructure remains available for larger feature/model experiments.
- Re-enabled local visual verification for the Paris runtime profile with a bounded top-12 candidate pass (`retrieval_local_match_top_n=12`, `retrieval_local_match_weight=0.6`, `retrieval_local_match_max_features=1800`). On the same fixed 40-sample strict probe, full runtime fusion improved from the Phase V retune (`9.26 km` mean, `7.10 km` median, `19.66 km` p90, `67.50% <=10 km`) to `8.61 km` mean, `6.77 km` median, `17.63 km` p90, and `72.50% <=10 km`. The tradeoff is a small `<=5 km` decrease (`32.50%` -> `30.00%`), so this is promoted as a medium/tail-error improvement rather than a close-range breakthrough.

### 5.6 Phase VI: Retrieval-Mistake Supervision and Diversity Control (May 10, 2026)
- Added a retrieval-mistake hard-negative miner that trains on the app's own high-scoring wrong candidates rather than generic geographic negatives.
- The first promoted projection (`runs/retrieval_hardneg_crossview_projection_v1.npz`) cut the fixed 80-sample strict Paris serving probe from `9.1105 km` mean / `16.7438 km` p90 to `4.7680 km` mean / `6.0800 km` p90.
- The next serving update disabled GeoCLIP candidate injection when a Paris retrieval index is active, improving the same probe to `4.6213 km` mean, `4.6345 km` median, `6.0913 km` p90, `66.25% <=5 km`, and `100.00% <=10 km`.
- A larger 480-query hard-negative pass exposed an overconcentration failure: naive training from identity regressed to `21.6922 km` mean, and v1-initialized training on the full concentrated set regressed to `5.3267 km` mean.
- The current branch adds diversity-capped hard-negative mining and initialized projection fine-tuning. The best completed cap-16 run (`152` triplets, `49` unique negative chips) improves the retrieval-dominant baseline to `4.5791 km` mean, `5.9161 km` p90, `67.50% <=5 km`, and `100.00% <=10 km`.
- Interpretation: this is a measured incremental model improvement, not a solved close-range model. The important research finding is that repeated-reference concentration is now a bottleneck; the next step must create broader near-field hard negatives or adapt the visual representation itself.

## 6. Experimental Protocol
### 6.1 Datasets and Artifacts Used in This Document
- SpaceNet Paris train-like index artifacts (`data/geo_index/spacenet_paris_clip.npz`).
- SpaceNet Paris test chips and metadata (`data/spacenet_paris_test/chips`, `metadata.csv`).
- Historical Open Geo/Wikimedia retrieval artifacts were used in earlier mismatch/debugging work, but the active runtime profiles are now Paris-only.
- Realistic Paris Panoramax -> IGN checkpoint (`data/paris_realistic_v1/pairs.csv`) with `10,000` paired street-to-aerial examples.
- Full realistic Paris combined dataset (`data/paris_realistic_v1_combined/pairs.csv`) with `40,000` paired street-to-aerial examples.
- Full realistic Paris combined strict split (`data/paris_realistic_v1_combined/splits_strict/`) with `34,821` retained pairs and `5,179` excluded boundary pairs.
- Full realistic Paris aerial index (`data/paris_realistic_v1_combined/indices/aerial_clip_index.npz`).
- Realistic street metadata includes timestamp fields in source metadata tables; the current evaluation tooling can now pass those timestamps into fusion when present.
- Pair-level benchmark CSVs may omit `captured_at`; evaluator and tuner tooling now auto-discover `metadata.csv` under the supplied street image directory and use it to fill missing capture-time fields by `street_id`, `image_id`, or normalized image path.

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
Project Heimdall demonstrates the path I took from prototype image geolocation to a benchmark-governed retrieval and fusion platform. The most important technical lesson is methodological: robust evaluation protocol and governance are as important as algorithmic sophistication. The latest work sharpens a second lesson: after validating a large portion of the retrieval and fusion stack, the project hit a data bottleneck before it hit a clean architecture ceiling. I addressed that bottleneck by building a realistic Paris street-to-aerial dataset and strict spatial split, then began model adaptation on that benchmark. The first cross-view projection is a real improvement, but it is not the final answer. The remaining research problem is now clearer: train stronger cross-view representations and validate them on the fixed combined split before making any serious claim toward a `~3 km` mean target.

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

### 15.3 Realistic Paris Data Program: Current Status and Reproduction

After the retrieval-side plateau, the project moved to a dedicated data branch to build realistic cross-view supervision instead of continuing to stack small reranking heuristics. The core claim from this phase is not that the `~3 km` target is solved. The claim is that the technology stack is now validated well enough to identify the next bottleneck honestly: realistic paired data.

Current local dataset checkpoint:

| Artifact | Count | Local path | Interpretation |
|---|---:|---|---|
| Mapillary street metadata | 20,000 | `data/paris_realistic_v1/street_mapillary/metadata.csv` | street-view crawl checkpoint |
| Panoramax street metadata | 20,000 | `data/paris_realistic_v1/street_panoramax/metadata.csv` | street-view crawl checkpoint |
| Combined street metadata | 40,000 | `data/paris_realistic_v1/street_combined/metadata.csv` | merged street corpus |
| Panoramax -> IGN aerial pairs | 10,000 | `data/paris_realistic_v1/pairs.csv` | first complete cross-view training/eval checkpoint |
| Full combined street -> IGN pairs | 40,000 | `data/paris_realistic_v1_combined/pairs.csv` | merged full realistic cross-view dataset |
| Full combined aerial metadata | 40,000 | `data/paris_realistic_v1_combined/aerial/metadata.csv` | merged IGN aerial crop metadata |
| Full combined strict split | 34,821 retained / 5,179 excluded | `data/paris_realistic_v1_combined/splits_strict/split_summary.json` | leakage-buffered benchmark split |

Important qualification:

- The older `data/paris_realistic_v1/` checkpoint remains useful as the first complete Panoramax -> IGN branch, but the benchmark-ready root is now `data/paris_realistic_v1_combined/`.
- The strict combined split reports `min_cross_split_distance_m = 1201.23`, which is materially safer than the earlier permissive split.
- Therefore the dataset phase is now complete enough to support realistic model-training work, even though the current frozen CLIP baseline is still far from a serious `~3 km` mean-accuracy claim.

Replication path used in this repo:

```powershell
$env:MAPILLARY_ACCESS_TOKEN="..."
.\.venv\Scripts\python -m src.tools.download_mapillary_paris --bbox 48.8156,2.2241,48.9022,2.4699 --out data/paris_realistic_v1/street_mapillary --grid-step-m 80 --street-per-cell 3 --max-images 20000 --seed 42
.\.venv\Scripts\python -m src.tools.download_panoramax_paris --bbox 48.8156,2.2241,48.9022,2.4699 --out data/paris_realistic_v1/street_panoramax --grid-step-m 80 --street-per-cell 3 --max-images 20000 --seed 42
.\.venv\Scripts\python -m src.tools.merge_realistic_street_datasets --metadata data/paris_realistic_v1/street_mapillary/metadata.csv data/paris_realistic_v1/street_panoramax/metadata.csv --out data/paris_realistic_v1/street_combined
.\.venv\Scripts\python -m src.tools.build_aerial_pairs --street-metadata data/paris_realistic_v1/street_panoramax/metadata.csv --out data/paris_realistic_v1 --provider ign_geopf --crop-size-m 256 --crop-px 512 --allow-missing-aerial false --seed 42
.\.venv\Scripts\python -m src.tools.recover_combined_aerial_dataset --existing-images-dir data/paris_realistic_v1/aerial/images --chunk-meta-dir data/paris_realistic_v1_combined_chunkmeta --chunk-out-dir data/paris_realistic_v1_combined_chunkpairs --final-out-dir data/paris_realistic_v1_combined --split-out-dir data/paris_realistic_v1_combined/splits_strict --provider ign_geopf --crop-size-m 256 --crop-px 512 --allow-missing-aerial false --seed 42 --max-workers 2 --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --cell-size-m 300 --buffer-cells 2 --sort-axis auto
.\.venv\Scripts\python -m src.tools.build_realistic_aerial_index --root data/paris_realistic_v1_combined --metadata aerial/metadata.csv --images-dir aerial/images --output indices/aerial_clip_index.npz --model-id openai/clip-vit-large-patch14
.\.venv\Scripts\python -m src.tools.eval_realistic_crossview --test-pairs data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --aerial-metadata data/paris_realistic_v1_combined/aerial/metadata.csv --street-images-dir data/paris_realistic_v1/street_combined --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --embedding-model openai/clip-vit-large-patch14 --output runs/eval_realistic_crossview_combined_strict_probe240_baseline_full40k.json --top-k 50
```

Why this phase matters:

1. It makes the research replicable by collaborators instead of relying on ad-hoc local imagery.
2. It aligns training data with the real street-to-aerial task rather than the older aerial-only retrieval proxy.
3. It makes it possible to test whether encoder adaptation, orientation-aware scoring, street-to-street retrieval, and fusion actually improve the task we care about.

Research decision from the current checkpoint:

- I should not claim that the current frozen CLIP baseline is already enough to deliver `~3 km` mean error.
- I now treat the data phase as complete enough to shift into model training on `data/paris_realistic_v1_combined/`.
- The immediate next move is to mine harder triplets from the strict combined train split and compare projection/encoder training against the current merged-index baseline.

Merged-dataset benchmark snapshots:

| Benchmark | Query set | Reference set | Mean km | Median km | <=1 km | <=2 km | <=5 km | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| Old strict realistic baseline | `120`-query probe on `data/paris_realistic_v1/splits_strict` | `10,000` Panoramax -> IGN pairs | 8.44 | 7.96 | 5.83% | 7.50% | 15.00% | Panoramax-only core dataset |
| Combined strict probe, sampled aerial index | `240`-query probe on `data/paris_realistic_v1_combined/splits_strict` | sampled `10,000` combined aerial index | 10.92 | 11.05 | 2.08% | 5.00% | 10.83% | `runs/eval_realistic_crossview_combined_strict_probe240_baseline_sample10k.json` |
| Combined strict probe, full aerial index | `240`-query probe on `data/paris_realistic_v1_combined/splits_strict` | full `40,000` combined aerial index | 10.97 | 11.75 | 2.92% | 5.83% | 12.50% | `runs/eval_realistic_crossview_combined_strict_probe240_baseline_full40k.json` |
| Combined strict probe, first query-only cross-view projection | `240`-query probe on `data/paris_realistic_v1_combined/splits_strict` | full `40,000` combined aerial index | 9.75 | 10.24 | 2.08% | 7.50% | 20.42% | `runs/eval_realistic_crossview_combined_strict_probe240_crossviewproj_v1_full40k.json` |

Interpretation:

- The larger combined benchmark is harder than the older Panoramax-only core benchmark, so direct mean-km comparisons should be made cautiously.
- Expanding from the sampled `10k` aerial index to the full `40k` index slightly improved close-range hit rates, but not mean or median error.
- The first query-only street-to-aerial projection run is the first real model-side gain on the merged benchmark: `mean_km` improved from `10.97` to `9.75`, `<=2km` improved from `5.83%` to `7.50%`, and `<=5km` improved from `12.50%` to `20.42%`, but `<=1km` regressed from `2.92%` to `2.08%`.
- The system is therefore no longer blocked on missing realistic data; it is now blocked on the strength of the cross-view model trained on that data.

Runtime fusion benchmark snapshot:

| Runtime profile | Query set | Mean km | Median km | p90 km | <=2 km | <=5 km | <=10 km | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Paris profile before Phase V fusion retune | strict combined pair probe, fixed 40-sample seed slice | 9.65 | 7.14 | 19.66 | 2.50% | 25.00% | 62.50% | `runs/fusion_sweep_paris_runtime_40.json` baseline |
| Paris profile after Phase V fusion retune | same fixed 40-sample seed slice | 9.26 | 7.10 | 19.66 | 5.00% | 32.50% | 67.50% | `runs/geo_eval_branch_paris_tuned2_40.json` |
| Paris profile with bounded local visual verification | same fixed 40-sample seed slice | 8.61 | 6.77 | 17.63 | 5.00% | 30.00% | 72.50% | `runs/geo_eval_branch_local_match_40.json` |

This is a runtime profile improvement, not a replacement for the larger `240`-query cross-view benchmark. The result is still useful because it validates the production inference path used by the app: RF-DETR detections, retrieval candidates, metadata-enriched capture time, and probabilistic fusion all execute together under `run_geo_eval.py`.

First combined cross-view training workflow used in this repo:

```powershell
.\.venv\Scripts\python -m src.tools.mine_realistic_crossview_triplets --pairs data/paris_realistic_v1_combined/splits_strict/train_pairs.csv --street-metadata data/paris_realistic_v1/street_combined/metadata.csv --aerial-metadata data/paris_realistic_v1_combined/aerial/metadata.csv --output runs/paris_realistic_crossview_train_triplets_v1.jsonl --summary-output runs/paris_realistic_crossview_train_triplets_v1.summary.json --positive-radius-m 80 --negative-min-distance-m 300 --negative-max-distance-m 5000 --max-positives 3 --max-negatives 20 --seed 42
.\.venv\Scripts\python -m src.tools.train_crossview_projection --triplets runs/paris_realistic_crossview_train_triplets_v1.jsonl --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --street-images-dir data/paris_realistic_v1/street_combined --output runs/crossview_projection_paris_combined_v1_probe.npz --report-output runs/crossview_projection_paris_combined_v1_probe.report.json --embedding-model openai/clip-vit-large-patch14 --max-triplets 6000 --epochs 8 --batch-size 64 --learning-rate 3e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device auto
.\.venv\Scripts\python -m src.tools.eval_realistic_crossview --test-pairs data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --aerial-metadata data/paris_realistic_v1_combined/aerial/metadata.csv --street-images-dir data/paris_realistic_v1/street_combined --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --projection runs/crossview_projection_paris_combined_v1_probe.npz --embedding-model openai/clip-vit-large-patch14 --output runs/eval_realistic_crossview_combined_strict_probe240_crossviewproj_v1_full40k.json --top-k 50
```

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

### 15.2 Apr 30, 2026 Geolocation Improvement Execution

The April 30 execution converted the earlier strategy memo into a measured Tier 1 to Tier 4 rollout on the current Paris stack.

- Tier 1 was the immediate serving win.
  - `src/config/paris.json` was promoted from the older single-index setup to the validated dual-index projected+DBA `rrf` profile.
  - On `runs/geo_eval_tier1_upgraded_paris_180.json`, the default serving path improved from `mean_km 15.53` to `14.60`, from `median_km 9.77` to `4.21`, and from `<=2km 19.44%` to `31.11%`, while `<=1km` stayed flat at `10.56%`.
- Tier 2 scaled the realistic cross-view projection training to the full mined corpus.
  - Training command used all `26204` triplets for `30` epochs and produced `runs/crossview_projection_paris_combined_v2_full.npz` plus `runs/crossview_projection_paris_combined_v2_full.report.json`.
  - On the strict `probe240` benchmark, close-range results improved versus the first `6000`-triplet probe model:
    - `<=1km 2.08% -> 4.17%`
    - `<=2km 7.50% -> 12.08%`
    - `<=5km 20.42% -> 22.92%`
  - The tradeoff is that `mean_km 9.75 -> 9.83` and `median_km 10.24 -> 10.92` regressed slightly, so Tier 2 is a real but mixed gain.
- Tier 3 added a DINOv2 aerial branch as a complementary retrieval source.
  - Built `data/geo_index/spacenet_paris_chips_facebook_dinov2_base.npz`.
  - Added `src/config/paris_dinov2_rrf_experimental.json`.
  - Discovered and fixed a real infrastructure bug: `src/core/logic/config.py` had been deduplicating `retrieval_index_model_ids`, which broke positional routing whenever the same model id appeared more than once in a multi-index config.
  - The fixed Tier 3 eval in `runs/geo_eval_paris_dinov2_rrf_experimental_180_fixed.json` showed:
    - `mean_km 14.60 -> 14.42`
    - `median_km 4.21 -> 4.47`
    - `<=1km 10.56% -> 13.33%`
    - `<=2km 31.11% -> 31.67%`
    - `<=5km 52.78% -> 52.22%`
  - Interpretation: DINOv2 contributes a real complementary signal, but the effect is still mixed, so it stays experimental instead of replacing `paris.json`.
- Tier 4 prepared the full realistic cross-view encoder fine-tune path.
  - Added `scripts/run_tier4_encoder_ft.ps1` to run the encoder fine-tune, realistic aerial-index rebuild, and strict `probe240` eval in one pipeline.
  - Validated the end-to-end path with a one-triplet smoke run recorded in `runs/retrieval_encoder_finetune/smoke_one_triplet.report.json`.
  - On this CPU-only workspace, repeated unattended background launches stalled after CLIP initialization, so no full Tier 4 benchmark is claimed yet.

Operationally, the main outcome of this execution is that the default Paris serving path improved immediately through Tier 1, while the deeper architecture experiments are now better separated into three categories: measured close-range tradeoffs (Tier 2), mixed complementary fusion (Tier 3), and prepared-but-not-yet-benchmarked encoder adaptation (Tier 4).

### 15.3 May 9, 2026 Retrieval-Mistake Hard-Negative Projection

The next experiment targeted the active application failure mode rather than the offline cross-view benchmark alone. The working hypothesis was that the production Paris profile was not mainly missing another fusion heuristic; it was missing supervised pressure against the retrieval candidates it already confuses with the correct location.

I added `src.tools.mine_retrieval_hard_triplets`, which runs the configured retrieval provider on real street queries, keeps geographically nearby reference chips as positives, and records the provider's own high-ranking wrong chips as hard negatives. This creates triplets from current inference mistakes:

```text
T_live(q_s) = (q_s, P_near(y_q), N_retrieved_wrong(q_s))
```

where `N_retrieved_wrong` contains candidates returned by the active retrieval stack whose coordinates are outside the configured ground-truth exclusion radius but still close enough to represent plausible Paris confusions.

The first run mined `160` valid triplets from `160` training queries with no missing files, empty candidate sets, or dropped triplets. Training a query-only cross-view projection for `6` epochs produced `runs/retrieval_hardneg_crossview_projection_v1.npz`. The fair serving-path evaluation used the same `80` strict probe samples, same seed, same full `run_geo_eval.py` path, and changed only the query projection file:

| Projection | Mean km | Median km | p90 km | <=2 km | <=5 km | <=10 km |
|---|---:|---:|---:|---:|---:|---:|
| Current master projection | 9.1105 | 7.0931 | 16.7438 | 3.75% | 25.00% | 62.50% |
| Retrieval-mistake hard-negative projection | 4.7680 | 4.8332 | 6.0800 | 1.25% | 57.50% | 100.00% |

This is a clean medium-range serving-path improvement: mean error fell by approximately `47.7%`, p90 error fell by approximately `63.7%`, and every evaluated sample landed within `10 km`. The remaining regression is exact close-range precision (`<=2 km`), which indicates that the next mining pass should deliberately oversample sub-`3 km` confusions instead of using only the broader `1-25 km` negative window.

The decision from this experiment was to promote `runs/retrieval_hardneg_crossview_projection_v1.npz` in `src/config/paris.json` and keep retrieval-mistake mining as the next main model-improvement loop.

### 15.4 May 9, 2026 Compact Fusion Statistics and Negative Near-Field Result

After promoting the retrieval-mistake projection, I ran a follow-up ablation to test whether the residual error came from the representation or from the final fusion statistic. The relevant diagnostic was retrieval-only v1: on the same `80` strict probe samples it reached `mean 4.6143 km`, `median 4.6345 km`, and `<=5 km 66.25%`, better than the full fused v1 result on mean and `<=5 km`. This indicates that late fusion can still dilute a useful candidate ranking.

I then tested two candidate remedies:

| Variant | Mean km | Median km | p90 km | <=2 km | <=5 km | <=10 km | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| v1 full pipeline | 4.7680 | 4.8332 | 6.0800 | 1.25% | 57.50% | 100.00% | reference |
| v1 retrieval-only | 4.6143 | 4.6345 | 6.0913 | 0.00% | 66.25% | 100.00% | diagnostic |
| v2 broad+near hard-negative projection | 4.8302 | 4.9400 | 6.0356 | 0.00% | 52.50% | 100.00% | rejected |
| v1 compact-stat fusion, 25 candidates retained | 4.7288 | 4.7759 | 6.0684 | 1.25% | 56.25% | 100.00% | promoted as minor improvement |
| v1 retrieval-dominant serving path, GeoCLIP gated off when retrieval index exists | 4.6213 | 4.6345 | 6.0913 | 0.00% | 66.25% | 100.00% | promoted as close-range serving improvement |

The near-field mixed v2 projection was rejected. Although it slightly improved p90, it regressed mean, median, `<=2 km`, and `<=5 km`, so it did not provide a credible improvement. The compact-stat fusion setting was kept because it improves mean, median, and p90 while preserving all `25` displayed fusion candidates for inspection; the cost is one fewer sample inside `5 km` on this `n=80` slice.

The May 10 serving-path update then tested whether the weaker global GeoSpot/GeoCLIP provider was diluting the now-stronger Paris retrieval index. The result supported that hypothesis. Setting `geolocator.use_geoclip_with_retrieval=false` keeps GeoCLIP available for profiles without retrieval indices, but uses the hard-negative retrieval provider alone when the Paris index is present. Compared with compact-stat fusion, mean error improved from `4.7288 km` to `4.6213 km`, median from `4.7759 km` to `4.6345 km`, and `<=5 km` from `56.25%` to `66.25%`; p90 moved slightly from `6.0684 km` to `6.0913 km`. A support-density selector was also tested and rejected (`mean 5.3660 km`, `<=5 km 42.50%`), showing that candidate clustering alone cannot replace learned cross-view ranking. Methodologically, this is a serving-path correction: once the specialized Paris retrieval model became stronger than the broad global provider, the system needed a way to stop adding lower-quality global hypotheses by default.

### 15.5 May 10, 2026 Candidate Oracle Rank Diagnostic

After promoting the diversity-capped hard-negative projection, I added candidate-oracle reporting to `src.tools.run_geo_eval`. The diagnostic asks a stricter question than top-1 accuracy: if the system is allowed to choose the closest candidate already returned in the top-25 shortlist, how good could it be without collecting more reference imagery?

On the fixed `80` strict Paris probe (`seed=42`), the answer is that candidate coverage is not the main bottleneck:

| Variant | Mean km | Median km | p90 km | <=1 km | <=2 km | <=5 km | <=10 km |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current serving prediction | 4.5791 | 4.6367 | 5.9161 | 0.00% | 0.00% | 67.50% | 100.00% |
| Candidate oracle over returned top-25 | 2.3528 | 2.1638 | 4.0063 | 21.25% | 43.75% | 100.00% | 100.00% |

The closest returned candidate has mean rank `15.125`, which means the correct local evidence is often present but buried under higher-scoring confusions. I retested the existing learned candidate reranker and rejected it because it left the serving metrics unchanged. I also tested a graph-support rerank inspired by the offline shortlist sweep. It did move a few samples into `<=2 km`, but the real pipeline regressed (`mean 4.7027 km`, `p90 6.5027 km`), so it was rejected as a default.

This changes the research diagnosis. The model is no longer primarily missing Paris candidate coverage on this probe; it is missing a visual ranking function strong enough to identify the correct street-level candidate inside a local cluster. Pure spatial support is insufficient because nearby wrong candidates are also spatially coherent. The next serious improvement should therefore target learned reranking or encoder adaptation with direct supervision over the returned shortlist, not more hand-tuned clustering.

I then tested that hypothesis directly in two ways. First, I replaced the scalar ridge candidate reranker with a listwise softmax trainer and an exponential rank-score activation. This added a more appropriate loss for shortlist ranking, but it still used only aggregate candidate features such as rank, retrieval score, support, and centroid distance. It failed in the full pipeline (`mean 5.3499 km`, `p90 6.7075 km`, `<=5 km 42.50%`), confirming that the aggregate features do not contain enough visual discrimination.

Second, I modified `src.tools.mine_retrieval_hard_triplets` so positives can be mined from the closest returned candidate itself (`--positive-source closest_candidate`). This is a direct attempt to train the projection against the oracle-rank gap. The mined set produced `104` triplets from `240` train records, but only `8` unique positive chips. A conservative projection update from the current serving projection improved the closest-candidate rank diagnostic (`15.125` to `10.375`) but regressed the actual serving result (`mean 5.0353 km`, `<=5 km 50.00%`) and reduced oracle quality (`<=2 km 43.75%` to `30.00%`). This is an important negative result: direct oracle-positive training has the right objective, but the current positive pool is too concentrated to generalize.

The resulting model-improvement plan is therefore more specific. The next bottleneck is not simply "add a ranking loss"; it is "create enough diverse correct visual positives inside the shortlist for a ranking loss to learn from." The retrieval index and training mining path need more local positive variety before another projection or encoder pass is likely to produce a large improvement.

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
- `runs/geo_eval_oracle_rank_diagnostics_80.json`
- `runs/candidate_reranker_current_stack_v1.report.json`
- `runs/geo_eval_graph_support_v1_80.json`
- `runs/candidate_reranker_listwise_v1.report.json`
- `runs/geo_eval_candidate_reranker_listwise_v1_80.json`
- `runs/retrieval_oracle_candidate_triplets_train240_v1_summary.json`
- `runs/retrieval_oracle_candidate_projection_v1.report.json`
- `runs/geo_eval_oracle_candidate_projection_v1_80.json`
- `runs/geo_impact_latest.json`
- `runs/geo_impact_latest.md`

## Appendix C: Internal References
- `[R1]` `PROGRESS.md`
- `[R2]` `src/docs/GEO_TECH.md`
- `[R3]` `src/docs/REPRODUCIBILITY.md`
- `[R4]` `docs/eval/` governance artifacts

## 9. Operator Dashboard and Local Visualization
Beyond benchmarking, a crucial aspect of Heimdall's development was providing a transparent, local-first interface for analyzing individual images. Early versions of the interface provided basic candidate visualization, but the requirements evolved to necessitate a "serious visual investigation console" known as Heimdall Operator Mode.

The operator mode (`/api/operator/*`) introduces an isolated session architecture. It executes the Heimdall pipeline chronologically, capturing stage transitions, explicit warnings if candidate providers fail or degrade, and structural clues from the RF-DETR detections. To support transparent human-in-the-loop interaction, the operator UI renders the fused probability estimate alongside the generated evidence, and records user-added notes and pins. This interface explicitly replaces "silent failures" with verbose failure states and timeline reporting, establishing the framework for future multi-operator and workflow-oriented investigations.
