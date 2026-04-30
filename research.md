# Project Heimdall Research Ledger

Last updated: April 30, 2026

This file is the compact research-facing ledger for the repo. It is not the paper itself; it is the evidence log behind the paper. It is different from:

- `PROGRESS.md`: append-only engineering work log
- `src/docs/RESEARCH_PAPER.md`: narrative paper-style draft

Use this file when you need the exact sequence of major geolocation changes, measured before/after performance, commands, and artifact paths without digging through every run output. Use `src/docs/RESEARCH_PAPER.md` when you want the authored narrative research write-up.

## Current Status

Best currently kept Paris realistic profiles on the canonical `n=180` split (`seed=42`):

| Profile | Mean km | Median km | <=1 km | <=2 km | <=5 km | <=10 km | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Weighted single-index projection control | 15.0810 | 4.8877 | 13.89% | 27.22% | 51.67% | 65.00% | single-index control |
| Geometry-lite balanced single-index (`top_n=14`, `weight=0.35`) | 14.7239 | 4.5903 | 15.00% | 28.33% | 53.33% | 66.11% | best single-index rerank branch result |
| `paris_close_range_dual_rrf` | 14.8410 | 4.87 | 12.78% | 31.11% | 50.56% | 63.33% | best close-range serving profile |
| `paris_balanced_dual_rrf` | 14.5970 | 4.21 | 10.56% | 31.11% | 52.78% | 65.56% | best balanced mean/median profile |
| `paris_close_range_dual_rrf_graph_kde` | 15.28 | n/a | 13.89% | 31.11% | lower than balanced | lower than balanced | aggressive `<=1km` profile |

Current hard conclusion:

- Heuristics helped, but only incrementally.
- Geometry-lite reranking is real and worth keeping as an experimental branch result.
- The best close-range gains came from multi-index projection + geo-aware DBA, not from more handcrafted scene cues alone.
- The current stack is in a data-limited stalemate: the repo now has enough retrieval, projection, and eval machinery to measure progress honestly, but not enough realistic street-to-aerial supervision to unlock a major accuracy jump.
- The path toward a major jump is model/data work: larger hard-negative sets, realistic street-view versus aerial pairs, and encoder adaptation, not more blind rerank knobs.

## Realistic Paris Data Checkpoint

Current data collected on the dedicated data branch:

| Dataset artifact | Count | Path | Status |
|---|---:|---|---|
| Mapillary street metadata | 20,000 | `data/paris_realistic_v1/street_mapillary/metadata.csv` | complete street crawl checkpoint |
| Panoramax street metadata | 20,000 | `data/paris_realistic_v1/street_panoramax/metadata.csv` | complete street crawl checkpoint |
| Combined street metadata | 40,000 | `data/paris_realistic_v1/street_combined/metadata.csv` | merged street-view corpus |
| Panoramax -> IGN aerial pairs | 10,000 | `data/paris_realistic_v1/pairs.csv` | first complete paired cross-view checkpoint |
| Full all-source street -> IGN pairs | 40,000 | `data/paris_realistic_v1_combined/pairs.csv` | merged full realistic cross-view dataset |
| Full all-source aerial metadata | 40,000 | `data/paris_realistic_v1_combined/aerial/metadata.csv` | merged IGN aerial crop metadata |
| Full all-source strict split | 34,821 retained / 5,179 excluded | `data/paris_realistic_v1_combined/splits_strict/split_summary.json` | leakage-buffered benchmark split |

Important caveats:

- The older `data/paris_realistic_v1/splits_full/split_summary.json` is still too permissive (`min_cross_split_distance_m = 3.77`) and should not be used for final claims.
- The benchmark-ready split is now `data/paris_realistic_v1_combined/splits_strict/` with `min_cross_split_distance_m = 1201.23`.
- This means the repo now has a real full realistic Paris dataset and a leakage-buffered benchmark split, but not yet a trained model that reaches the `~3 km` target on that split.

Replication commands used in the repo:

```powershell
.\.venv\Scripts\python -m src.tools.download_mapillary_paris --bbox 48.8156,2.2241,48.9022,2.4699 --out data/paris_realistic_v1/street_mapillary --grid-step-m 80 --street-per-cell 3 --max-images 20000 --seed 42
.\.venv\Scripts\python -m src.tools.download_panoramax_paris --bbox 48.8156,2.2241,48.9022,2.4699 --out data/paris_realistic_v1/street_panoramax --grid-step-m 80 --street-per-cell 3 --max-images 20000 --seed 42
.\.venv\Scripts\python -m src.tools.merge_realistic_street_datasets --metadata data/paris_realistic_v1/street_mapillary/metadata.csv data/paris_realistic_v1/street_panoramax/metadata.csv --out data/paris_realistic_v1/street_combined
.\.venv\Scripts\python -m src.tools.build_aerial_pairs --street-metadata data/paris_realistic_v1/street_panoramax/metadata.csv --out data/paris_realistic_v1 --provider ign_geopf --crop-size-m 256 --crop-px 512 --allow-missing-aerial false --seed 42
.\.venv\Scripts\python -m src.tools.recover_combined_aerial_dataset --existing-images-dir data/paris_realistic_v1/aerial/images --chunk-meta-dir data/paris_realistic_v1_combined_chunkmeta --chunk-out-dir data/paris_realistic_v1_combined_chunkpairs --final-out-dir data/paris_realistic_v1_combined --split-out-dir data/paris_realistic_v1_combined/splits_strict --provider ign_geopf --crop-size-m 256 --crop-px 512 --allow-missing-aerial false --seed 42 --max-workers 2 --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --cell-size-m 300 --buffer-cells 2 --sort-axis auto
.\.venv\Scripts\python -m src.tools.build_realistic_aerial_index --root data/paris_realistic_v1_combined --metadata aerial/metadata.csv --images-dir aerial/images --output indices/aerial_clip_index.npz --model-id openai/clip-vit-large-patch14
.\.venv\Scripts\python -m src.tools.eval_realistic_crossview --test-pairs data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --aerial-metadata data/paris_realistic_v1_combined/aerial/metadata.csv --street-images-dir data/paris_realistic_v1/street_combined --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --embedding-model openai/clip-vit-large-patch14 --output runs/eval_realistic_crossview_combined_strict_probe240_baseline_full40k.json --top-k 50
```

Current merged-dataset benchmark snapshots:

| Benchmark | Query set | Reference set | Mean km | Median km | <=1 km | <=2 km | <=5 km | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| Old strict realistic baseline | `120` query probe on `data/paris_realistic_v1/splits_strict` | `10,000` Panoramax -> IGN pairs | 8.44 | 7.96 | 5.83% | 7.50% | 15.00% | Panoramax-only core dataset |
| Combined strict probe, sampled aerial index | `240` query probe on `data/paris_realistic_v1_combined/splits_strict` | sampled `10,000` combined aerial index | 10.92 | 11.05 | 2.08% | 5.00% | 10.83% | `runs/eval_realistic_crossview_combined_strict_probe240_baseline_sample10k.json` |
| Combined strict probe, full aerial index | `240` query probe on `data/paris_realistic_v1_combined/splits_strict` | full `40,000` combined aerial index | 10.97 | 11.75 | 2.92% | 5.83% | 12.50% | `runs/eval_realistic_crossview_combined_strict_probe240_baseline_full40k.json` |
| Combined strict probe, first query-only cross-view projection | `240` query probe on `data/paris_realistic_v1_combined/splits_strict` | full `40,000` combined aerial index | 9.75 | 10.24 | 2.08% | 7.50% | 20.42% | `runs/eval_realistic_crossview_combined_strict_probe240_crossviewproj_v1_full40k.json` |

Interpretation:

- The realistic data program is complete enough to benchmark honestly on a much larger and stricter street-to-aerial dataset.
- Expanding the combined probe from the sampled `10k` aerial index to the full `40k` index modestly improved close-range hit rates (`<=1km`, `<=2km`, `<=5km`) but did not improve mean or median error yet.
- The first query-only street-to-aerial projection run is the first real model-side gain on the harder combined benchmark: `mean_km` improved from `10.97` to `9.75`, `<=2km` improved from `5.83%` to `7.50%`, and `<=5km` improved from `12.50%` to `20.42%`, but `<=1km` regressed from `2.92%` to `2.08%`.
- The bottleneck has therefore shifted from "insufficient realistic data exists" to "how well we train the cross-view model on the harder combined benchmark."

First combined cross-view training workflow used in the repo:

```powershell
.\.venv\Scripts\python -m src.tools.mine_realistic_crossview_triplets --pairs data/paris_realistic_v1_combined/splits_strict/train_pairs.csv --street-metadata data/paris_realistic_v1/street_combined/metadata.csv --aerial-metadata data/paris_realistic_v1_combined/aerial/metadata.csv --output runs/paris_realistic_crossview_train_triplets_v1.jsonl --summary-output runs/paris_realistic_crossview_train_triplets_v1.summary.json --positive-radius-m 80 --negative-min-distance-m 300 --negative-max-distance-m 5000 --max-positives 3 --max-negatives 20 --seed 42
.\.venv\Scripts\python -m src.tools.train_crossview_projection --triplets runs/paris_realistic_crossview_train_triplets_v1.jsonl --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --street-images-dir data/paris_realistic_v1/street_combined --output runs/crossview_projection_paris_combined_v1_probe.npz --report-output runs/crossview_projection_paris_combined_v1_probe.report.json --embedding-model openai/clip-vit-large-patch14 --max-triplets 6000 --epochs 8 --batch-size 64 --learning-rate 3e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device auto
.\.venv\Scripts\python -m src.tools.eval_realistic_crossview --test-pairs data/paris_realistic_v1_combined/splits_strict/test_pairs_probe240.csv --aerial-metadata data/paris_realistic_v1_combined/aerial/metadata.csv --street-images-dir data/paris_realistic_v1/street_combined --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --projection runs/crossview_projection_paris_combined_v1_probe.npz --embedding-model openai/clip-vit-large-patch14 --output runs/eval_realistic_crossview_combined_strict_probe240_crossviewproj_v1_full40k.json --top-k 50
```

Research recommendation from this checkpoint:

- The current data is enough to start realistic street-to-aerial training, hard-negative mining, and baseline cross-view evaluation on the full combined dataset.
- The current frozen CLIP baseline on the combined strict probe is still far from a serious `~3 km` mean result.
- The next priority should now shift from data collection to model training on this new benchmark root: mine harder triplets from `data/paris_realistic_v1_combined/splits_strict/train_pairs.csv`, train the cross-view projection / encoder path, and compare against the current full-index baseline.

## Timeline

## 1. Foundation Phase

### Jan 29 to Jan 31, 2026

Main changes:

- Built the repo structure, pipeline skeleton, schemas, CLI, batch runner, UI server, serialization, EXIF/sidecar geo input, and early fusion plumbing.
- Added Ultralytics OBB, GeoSpot/GeoCLIP integration, and the first live analysis UI.
- Moved the repo into `src/` and added the first retrieval-based geo provider with CLIP embeddings.

Measured state:

- Early work established capability and tests, not a stable leakage-safe benchmark yet.
- Test checkpoints advanced from `4 passed` to `13 passed` during this phase.

Decision:

- No serious geo-accuracy claim should be made from this phase alone because benchmark protocol was not mature yet.

## 2. Benchmark and Fusion Hardening

### Apr 5, 2026

Main changes:

- Added candidate validation, near-duplicate merge, retrieval diversity/locality controls, score normalization, source balancing, spatial consensus, cross-source agreement, adaptive outlier suppression, confidence calibration, plausibility rerank, temporal filtering, and benchmark governance tooling.

Measured state:

- This phase mostly hardened reliability and evaluation integrity.
- Validation checkpoints reached `78 passed`, then `81 passed`, later `105 passed` in focused non-UI coverage.

Decision:

- This was the phase that made later metric claims trustworthy.

## 3. Measured Accuracy Milestones

### Apr 15, 2026: Realistic single-index retune

Change:

- Simplified the realistic single-index profile by removing over-aggressive post-processing on the canonical Paris realistic split.

Performance:

| Metric | Before | After |
|---|---:|---:|
| Mean km | 19.7494 | 18.0159 |
| Median km | 11.3947 | 11.4990 |
| <=1 km | 1.67% | 5.00% |
| <=5 km | 23.89% | 30.56% |
| <=10 km | 43.33% | 45.56% |

Decision:

- Simpler ranking beat the previous overprocessed profile on the realistic split.

Artifacts:

- `runs/tune_retrieval_geo_realistic_within1km_focus_v1.json`
- `runs/bench_realistic_single_180_precision_v2.json`

### Apr 15, 2026: Consensus top-1 refinement

Change:

- Added `retrieval_consensus_top_n=20` and `retrieval_consensus_radius_km=3.0` with adaptive center selection.

Performance:

| Metric | Before | After |
|---|---:|---:|
| Mean km | 18.0159 | 15.5334 |
| Median km | 11.4990 | 9.7717 |
| <=1 km | 5.00% | 10.00% |
| <=5 km | 30.56% | 36.67% |
| <=10 km | 45.56% | 50.56% |

Decision:

- Strong win. Consensus refinement became one of the highest-leverage retrieval-side changes.

Artifacts:

- `runs/geo_eval_paris_profile_180_v2.json`
- `runs/geo_eval_paris_profile_180_consensus_v1.json`

### Apr 15 to Apr 21, 2026: Graph and KDE reranking ablations

Change:

- Tested graph-support reranking and KDE mode refinement as retrieval-side selection upgrades.

Performance highlights versus control (`mean_km=15.5264`, `<=1km=10.56%`, `<=2km=19.44%`, `<=5km=37.22%`):

| Variant | Mean km | <=1 km | <=2 km | <=5 km | Result |
|---|---:|---:|---:|---:|---|
| Graph rerank A | 16.1146 | 5.56% | 17.78% | 35.56% | regression |
| Graph rerank B | 16.8216 | 6.11% | 16.67% | 35.00% | regression |
| Graph rerank C | 15.3627 | 7.22% | 17.78% | 37.22% | still worse close-range |
| KDE refine C | 15.4602 | 11.11% | 19.44% | 37.78% | best `<=1km` |
| KDE refine D | 15.4223 | 10.00% | 20.00% | 38.33% | best `<=2km` / `<=5km` |

Decision:

- Graph rerank was rejected.
- KDE remained experimental and objective-specific rather than a universal default.

### Apr 21, 2026: Dual local geometric reranker (`SIFT + ORB`)

Change:

- Replaced the legacy local matcher with a dual-engine local reranker plus weak-signal gating.

Performance:

| Metric | Legacy local match A | Dual local match A |
|---|---:|---:|
| Mean km | 16.6122 | 15.2447 |
| Median km | 10.9454 | 9.7717 |
| <=1 km | 5.00% | 8.89% |
| <=2 km | 13.33% | 18.33% |
| <=5 km | 32.78% | 37.22% |
| <=10 km | 46.67% | 51.11% |

Decision:

- Real upgrade over the legacy local matcher, but not the main answer for the close-range target.

### Apr 21, 2026: Scope guard and geo prior integrity fix

Change:

- Added profile/data scope enforcement and scope-aware geo prior to stop catastrophic cross-region retrieval failures.

Performance:

| Scenario | Before | After |
|---|---:|---:|
| Mixed-scope stress test mean km | 6656.66 | 18.82 |
| Mixed-scope stress test <=10 km | 0.00% | 40.83% |
| Failure seed replay mean km | 7408.15 | 0.00 |

Decision:

- Mandatory evaluation-integrity guard. This did not make in-scope Paris retrieval smarter, but it eliminated invalid-profile disaster cases.

Artifacts:

- `runs/geo_eval_mixed_scope_no_prior_120.json`
- `runs/geo_eval_mixed_scope_hard_prior_120.json`
- `runs/geo_eval_mixed_scope_no_prior_seed1870334448_2.json`
- `runs/geo_eval_mixed_scope_hard_prior_seed1870334448_2.json`

### Apr 21 to Apr 22, 2026: Projection adaptation from hard negatives

Change:

- Added mined hard-negative projection training against a train-reference index.

Performance on realistic `n=120`:

| Metric | Baseline | `trainref_v2_mild` |
|---|---:|---:|
| <=1 km | 9.17% | 12.50% |
| <=2 km | 16.67% | 28.33% |

Decision:

- Projection adaptation became the strongest single-index learned direction.

### Apr 27 to Apr 28, 2026: Difficulty-weighted triplet training

Change:

- Kept the projection workflow, but weighted mined triplets by difficulty instead of training uniformly.

Performance on canonical single-index Paris realistic split (`n=180`):

| Metric | Uniform | Difficulty-weighted |
|---|---:|---:|
| Mean km | 15.252 | 15.081 |
| Median km | 5.504 | 4.888 |
| <=1 km | 11.67% | 13.89% |
| <=2 km | 26.67% | 27.22% |
| <=5 km | 50.00% | 51.67% |
| <=10 km | 64.44% | 65.00% |

Decision:

- Weighted triplets beat uniform training and became the kept single-index control.

Artifacts:

- `runs/geo_eval_projection_trainref_v2_uniform_cmp_180.json`
- `runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`

### Apr 28, 2026: Structure-aware rerank v1

Change:

- Added top-shortlist reranking from corner density, edge density, dominant line orientation, and guarded shadow-axis cues.

Performance versus weighted single-index control:

| Metric | Control | Structure v1 |
|---|---:|---:|
| Mean km | 15.0810 | 14.7247 |
| Median km | 4.8877 | 4.5903 |
| <=1 km | 13.89% | 15.00% |
| <=2 km | 27.22% | 28.33% |
| <=5 km | 51.67% | 53.33% |
| <=10 km | 65.00% | 66.11% |

Decision:

- Real improvement. Kept as the best non-learning single-index rerank upgrade.

Artifacts:

- `runs/geo_eval_projection_trainref_v2_weighted_cmp_180.json`
- `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v1_180.json`

### Apr 28, 2026: Geometry-lite extension, then weak-signal gating

Change:

- Extended the scene signature with corner/edge spatial layout, line orthogonality, anisotropy, and shadow elongation.
- Then added weak-signal gating so these cues only matter when distinctive enough.

Measured progression:

| Variant | Mean km | <=1 km | <=2 km | <=5 km | <=10 km | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Weighted control | 15.0810 | 13.89% | 27.22% | 51.67% | 65.00% | baseline |
| Geometry-lite A (`12`, `0.35`) | 14.9730 | 13.89% | 27.22% | 52.78% | 66.11% | modest |
| Geometry-lite B (`16`, `0.25`) | 14.9548 | 14.44% | 27.22% | 52.78% | 65.56% | modest |
| Geometry-lite C (`16`, `0.30`) | 14.6599 | 14.44% | 27.22% | 52.78% | 66.11% | best ungated mean |
| Gated C (`16`, `0.30`) | 14.9308 | 13.89% | 27.22% | 52.78% | 65.56% | safer, weaker |
| Gated legacy weights (`12`, `0.35`) | 14.7927 | 14.44% | 28.33% | 52.78% | 65.56% | closes close-range gap |
| Gated D (`14`, `0.35`) | 14.7239 | 15.00% | 28.33% | 53.33% | 66.11% | balanced kept setting |

Decision:

- Keep the gated geometry-lite variant as the current best single-index branch checkpoint.
- It matches the earlier structure milestone rather than blowing past it, so it is useful but not a breakthrough.

Artifacts:

- `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_c_180.json`
- `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_c_gated_180.json`
- `runs/geo_eval_projection_trainref_v2_weighted_cmp_structure_v2_geometry_d_180.json`

### Apr 28, 2026: Dual-index projection + geo-aware DBA

Change:

- Added an auxiliary geo-aware DBA index and fused it with the projection index using `rrf`.

Performance versus the weighted single-index control:

| Profile | Mean km | Median km | <=1 km | <=2 km | <=5 km | <=10 km |
|---|---:|---:|---:|---:|---:|---:|
| Weighted single-index control | 15.25 | 5.50 | 11.67% | 26.67% | 50.00% | 64.44% |
| `paris_close_range_dual_rrf` | 14.84 | 4.87 | 12.78% | 31.11% | 50.56% | 63.33% |
| `paris_balanced_dual_rrf` | 14.60 | 4.21 | 10.56% | 31.11% | 52.78% | 65.56% |
| `paris_close_range_dual_rrf_graph_kde` | 15.28 | n/a | 13.89% | 31.11% | below balanced | below balanced |

Decision:

- `paris_close_range_dual_rrf` is the best serving profile when the objective is close-range precision.
- `paris_balanced_dual_rrf` is the best broad tradeoff profile.
- The graph+KDE dual profile is a specialized `<=1km` mode, not the general default.

### Apr 28 to Apr 29, 2026: Extra rerank ideas that did not help

Change:

- Tried stronger dual-index follow-ups, including geometry on top of the dual stack and extra local matching.

Performance:

| Variant | Mean km | <=2 km | Result |
|---|---:|---:|---|
| `paris_balanced_dual_rrf` | 14.597 | 31.11% | incumbent |
| dual balanced + geometry rerank | 14.835 | 30.56% | worse |
| `paris_close_range_dual_rrf` | 14.841 | 31.11% | incumbent |
| dual close-range + local match | 14.936 | 30.56% | worse |

Decision:

- Rejected. More rerank layers on the current dual stack were not buying meaningful gains.

### Apr 29, 2026: Backbone benchmark infrastructure fix

Change:

- Fixed projected-backbone benchmarking so projection training uses a dedicated `projection_support` index built from the triplet reference pool instead of incorrectly reusing the tiny eval sample index.

Performance on the smoke 80/20 split:

| Model | Mean km | Median km | <=5 km | <=10 km |
|---|---:|---:|---:|---:|
| Raw CLIP | 27.7099 | 42.9709 | 25.0% | 35.0% |
| Projected CLIP | 29.2042 | 44.2391 | 25.0% | 35.0% |
| Raw SigLIP-base | 25.0207 | 17.8138 | 10.0% | 40.0% |
| Projected SigLIP-base | 27.2065 | 17.8265 | 10.0% | 35.0% |

Decision:

- The infrastructure fix is important and correct.
- The first adapted smoke runs are not good enough; projected variants regressed on this smoke setup.

Artifacts:

- `runs/backbone_bench/smoke_raw_clip.json`
- `runs/backbone_bench/smoke_projected_clip.json`
- `runs/backbone_bench/smoke_raw_siglip_base.json`
- `runs/backbone_bench/smoke_projected_siglip_base.json`

### Apr 29, 2026: Hard-negative mining scale-up

Change:

- Relaxed mining filters to grow the training set before moving from projection-only adaptation to real encoder tuning.

Performance:

| Mining setting | Triplets written |
|---|---:|
| Current strict setup | 52 |
| Looser A | 74 |
| Looser B | 129 |

Decision:

- The stack needed more domain-matched hard negatives.
- `loose_b` became the working direction for encoder fine-tuning experiments.

Artifacts:

- `runs/tmp_triplets_curr_summary.json`
- `runs/tmp_triplets_loose_a_summary.json`
- `runs/tmp_triplets_loose_b_summary.json`

### Apr 29, 2026: Real encoder fine-tuning path and cached projection training

Change:

- Added `src/tools/train_retrieval_encoder.py` to fine-tune a CLIP-family image encoder directly from mined query/positive/negative images.
- Added a cached `visual_projection` fast path so frozen-vision training reuses pooled vision features instead of recomputing them every epoch.

Measured training behavior on `loose_b`:

| Run | Unique cached images | Cache time | Train time | Best weighted hard triplet loss | Weighted satisfied pct |
|---|---:|---:|---:|---:|---:|
| `tmp_encoder_loose_b` (`1` epoch) | 956 | 375.85 s | 4.20 s | 0.1937 | 0.0 |
| `tmp_encoder_loose_b_e10` (`10` epochs) | 956 | 326.14 s | 33.45 s | 0.1199 | 0.0 |

Decision:

- The cached path materially reduced recurring training cost.
- Better triplet satisfaction during isolated `loose_b` training was still not enough to prove retrieval improvement by itself.

Artifacts:

- `runs/tmp_encoder_loose_b.report.json`
- `runs/tmp_encoder_loose_b_e10.report.json`

### Apr 29, 2026: Fixed Paris-180 fine-tune loop

Change:

- Added `src/tools/run_retrieval_finetune_loop.py` to evaluate on the fixed Paris-180 split, mine failures, fine-tune the encoder, rebuild the train index, optionally build a DBA companion, and re-evaluate.
- Fixed the loop so it compares the tuned model against a matched rebuilt baseline rather than unfairly against the full production serving config.

Measured result:

| Setting | Mean km | Median km | <=1 km | <=2 km | <=5 km | <=10 km |
|---|---:|---:|---:|---:|---:|---:|
| Production serving baseline `paris_close_range_dual_rrf` | 14.8410 | 4.87 | 12.78% | 31.11% | 50.56% | 63.33% |
| Matched rebuilt base model (`train_limit=300`) | 23.1429 | 10.5783 | 0.56% | 1.67% | 17.78% | 36.67% |
| Tuned encoder on same rebuilt index | 21.5331 | 11.0497 | 1.11% | 3.33% | 20.56% | 44.44% |

Training details from the winning round:

- `130` resolved triplets
- `966` unique cached images
- `281.40 s` feature-cache build
- `23.83 s` actual training
- `best_weighted_triplet_satisfied_pct = 0.873`

Decision:

- Encoder fine-tuning is doing something real on a matched evaluation setup.
- It is still far behind the full production dual-index serving stack.
- This supports the conclusion that bigger train indices and stronger adaptation data are required before the tuned encoder can replace or beat the current serving profile.

Artifacts:

- `runs/retrieval_finetune_loop_180_looseb_e10/loop_summary.json`
- `runs/retrieval_finetune_loop_180_looseb_e10/round_01/encoder.report.json`

### Apr 29, 2026: Auxiliary tuned-branch serving support

Change:

- Extended the fine-tune loop so a tuned model can be appended as an auxiliary retrieval source on top of the existing serving profile instead of replacing it.
- Added `--aux-index-weight` and `--aux-dba-weight` plus per-index projection isolation so the tuned sources do not inherit the base projection path incorrectly.

Measured result on the real Paris-180 serving benchmark:

| Setting | Mean km | Median km | <=1 km | <=2 km | <=5 km | <=10 km |
|---|---:|---:|---:|---:|---:|---:|
| Serving baseline `paris_close_range_dual_rrf` | 14.8410 | 4.8705 | 12.78% | 31.11% | 50.56% | 63.33% |
| Auxiliary fused serving candidate (`aux_index_weight=0.15`, `aux_dba_weight=0.05`) | 15.6990 | 9.7449 | 10.00% | 21.11% | 37.78% | 52.78% |

Decision:

- The deployment shape is implemented and benchmarked.
- It is not promoted, because even a conservative auxiliary blend regressed badly on the real serving benchmark.
- A higher-weight blend was not promoted after this result because the tuned source is still too weak relative to the primary branch.

Artifacts:

- `runs/aux_fusion_final/aux_conservative.json`
- `runs/aux_fusion_final/aux_conservative_eval_180.json`

## What We Learned

1. Benchmark quality matters as much as model quality.
   Leakage-prone setups can make a weak system look solved.

2. Small heuristics can help, but they plateau.
   Geometry-lite cues improved the single-index path, but they did not create the major jump.

3. The strongest current serving gains came from better retrieval structure, not from a brand-new backbone.
   Projection adaptation, then dual-index projection + DBA, delivered the most credible close-range improvements so far.

4. Major future gains are more likely to come from targeted data and encoder adaptation.
   The repo now has the right scaffolding for that:
   - scaled hard-negative mining
   - real encoder fine-tuning
   - fixed-loop evaluation on Paris-180
   - auxiliary serving-branch fusion support

5. The next bottleneck is realistic data, not another round of scene heuristics.
   Better street-view comparisons, cross-view geolocation, and stronger models now depend on building a leakage-safe Paris street-plus-aerial dataset with enough coverage to train and test on the same problem we actually care about.

## Current Best Artifacts To Reuse

- Best close-range serving profile:
  - `src/config/paris_close_range_dual_rrf.json`
- Best balanced serving profile:
  - `src/config/paris_balanced_dual_rrf.json`
- Best single-index geometry-lite profile:
  - `src/config/paris_structure_geometry_balanced.json`
- Best fixed-loop encoder experiment:
  - `runs/retrieval_finetune_loop_180_looseb_e10/`

## Next Recommended Work

1. Build a realistic Paris street-image dataset from an allowed source with GPS, heading, capture metadata, and sequence information.
2. Pair those street images with open aerial imagery and create leakage-safe spatial train, validation, and test splits.
3. Mine realistic cross-view hard negatives from that dataset and retrain the street-to-aerial projection under the fixed benchmark protocol.
4. Stand up realistic baseline evaluations for street-to-aerial, street-to-street, and fused retrieval before any new architecture claims.
5. Only after those baselines exist, benchmark stronger remote-sensing-native encoders and other architecture upgrades on the same fixed realistic split.
