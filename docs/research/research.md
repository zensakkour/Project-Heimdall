# Project Heimdall Research Ledger

Last updated: May 10, 2026

This file is the compact research-facing ledger for the repo. It is not the paper itself; it is the evidence log behind the paper. It is different from:

- `docs/engineering/PROGRESS.md`: append-only engineering work log
- `src/docs/RESEARCH_PAPER.md`: narrative paper-style draft

Use this file when you need the exact sequence of major geolocation changes, measured before/after performance, commands, and artifact paths without digging through every run output. Use `src/docs/RESEARCH_PAPER.md` when you want the authored narrative research write-up.

## Current Status

Active scope:

- Project Heimdall is currently Paris-focused.
- Open Geo/Wikimedia runtime profiles and local Open Geo data caches are retired from the active app path until broad-scope geolocation is resumed on a dedicated branch.
- Manual analysis uploads should use `data/analysis_tests/paris_street/images/` or the full `data/paris_realistic_v1/street_combined/images/` corpus.

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
- The full realistic aerial index is now useful in the active Paris profile: it improves top-1 serving error and substantially improves the returned-candidate oracle.
- The current bottleneck has moved from "not enough diverse positive candidates" to "not enough visual ranking strength to select the best candidate from a much better shortlist."
- The path toward a major jump is still model/data work: larger diverse realistic positives, stronger shortlist ranking, and encoder adaptation, not more blind rerank knobs.

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

## May 10, 2026: Realistic Aerial Index in the Active Paris Profile

After the candidate-oracle diagnostic showed that direct oracle-positive training had only `8` unique positive chips, I tested whether the active app profile needed a richer candidate source before another ranking loss. The promoted change adds the full realistic IGN aerial index (`data/paris_realistic_v1_combined/indices/aerial_clip_index.npz`) to `src/config/paris.json`. The SpaceNet indices keep the current hard-negative projection, while the realistic aerial index is queried in raw CLIP space through per-index projection routing.

Fixed strict probe comparison (`80` samples, `seed=42`, full app path):

| Variant | Mean km | Median km | p90 km | <=2 km | <=5 km | Oracle mean km | Oracle <=2 km | Oracle best-rank mean | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Previous serving profile | 4.5791 | 4.6367 | 5.9161 | 0.00% | 67.50% | 2.3528 | 43.75% | 15.125 | replaced |
| Realistic aerial index, RRF | 4.4793 | 4.6127 | 5.7859 | 0.00% | 70.00% | 1.6715 | 66.25% | 18.250 | promoted |
| Lighter rank-fusion weighting | 4.5026 | 4.6306 | 5.8956 | 3.75% | 66.25% | 1.5891 | 66.25% | 14.538 | kept as diagnostic |
| Score-based weighted fusion | 5.2260 | 5.3401 | 7.9741 | 8.75% | 45.00% | 1.6658 | 67.50% | 9.588 | rejected |

Decision:

- Promote the realistic aerial index profile because it improves mean, median, p90, `<=5km`, and candidate-oracle coverage.
- Do not promote lighter weighting yet: it creates the first `<=2km` hits in this branch, but gives up too much broad `<=5km` accuracy.
- Reject score-based weighted fusion because it over-optimizes close hits and damages the general ranking.
- The next serious model step is shortlist ranking on this richer candidate pool, not more spatial clustering.

Post-promotion mining check:

- Pre-index oracle-positive mining had `104` triplets but only `8` unique positive chips.
- With the realistic index active and no positive fallback, mining produced `70` strict near-positive triplets with `46` unique positive paths.
- `67/70` positives came from the realistic aerial index, mean positive distance was `1.0336 km`, and mean positive rank was `22.3`.
- A listwise aggregate-feature candidate reranker trained on the improved shortlist still regressed held-out retrieval-only evaluation (`mean 4.5748 -> 4.8177`, `<=5km 66.25% -> 60.00%`), so the next ranker needs visual pair evidence, not only rank/score/source/cluster features.
- I then added source-filtered mining so positives and negatives can be constrained to `aerial_clip_index`. This produced a coherent realistic-only set (`75` triplets, `45` unique positives, `98` unique negatives), but a source-specific query projection trained on it regressed held-out retrieval-only evaluation (`mean 4.5748 -> 4.6198`, oracle `<=2km 66.25% -> 45.00%`). The source-filtered miner is useful infrastructure; the projection itself is rejected.

## May 10, 2026: Candidate-Cloud Consensus Promotion

After the realistic aerial index improved oracle coverage, I tested whether a visual pair reranker could select better candidates directly. The new `src/tools/train_visual_pair_reranker.py` trainer uses frozen CLIP query/candidate embeddings, elementwise product and absolute-difference features, retrieval score, rank, and source flags. This is kept as research infrastructure, but the first leakage-safe run did not improve the held-out fixed probe.

Visual pair reranker result (`80` strict probe samples, retrieval-only):

| Variant | Mean km | Median km | p90 km | <=5 km | Decision |
|---|---:|---:|---:|---:|---|
| Base retrieval shortlist | 4.5748 | 4.6205 | 5.9087 | 66.25% | baseline |
| Visual pair reranker, best fusion weight | 4.5748 | 4.6205 | 5.9087 | 66.25% | rejected because best weight was `0.0` |
| Positive reranker weights | worse | worse | worse | 58.75-62.50% | rejected |

I then extended `src/tools/tune_retrieval_geo.py` so the same tuner can sweep graph-support reranking, consensus refinement, and KDE mode refinement on realistic pair CSVs. The graph-support sweep looked strong inside the tuner (`mean 3.6347`, `<=5km 90.00%`), but the real evaluator path only reached `mean 4.5127`, `<=5km 67.50%`, so graph-support was not promoted.

The validated improvement came from candidate-cloud consensus over the richer mixed shortlist:

| Variant | Mean km | Median km | p90 km | <=2 km | <=5 km | Decision |
|---|---:|---:|---:|---:|---:|---|
| Previous promoted full profile | 4.4793 | 4.6127 | 5.7859 | 0.00% | 70.00% | replaced |
| Raw retrieval-only baseline for this probe | 4.5748 | 4.6205 | 5.9087 | n/a | 66.25% | diagnostic |
| Candidate-cloud consensus (`top_n=25`, `radius=2km`, `score_power=0`) | 3.9904 | 4.0641 | 5.7860 | 8.75% | 76.25% | promoted |

Decision:

- Promote `retrieval_consensus_top_n=25`, `retrieval_consensus_radius_km=2.0`, and `retrieval_consensus_score_power=0.0` in `src/config/paris.json`.
- Keep `src/tools/train_visual_pair_reranker.py` as a negative ablation/infrastructure path, but do not enable its model.
- Keep the expanded tuner support because it made the graph/consensus/KDE comparison reproducible and exposed a tuner-vs-runtime mismatch that must be checked before future promotions.
- The candidate oracle still remains much better than top-1 (`oracle <=2km 66.25%`, `oracle <=5km 100%`), so the next major jump still needs visual ranking/encoder improvement rather than only spatial consensus.

Validation:

```powershell
$env:TMP='c:\Users\zen\Desktop\Projects\Project-Heimdall\.tmp'; $env:TEMP=$env:TMP; .\.venv\Scripts\python.exe -m pytest src\tests\test_config_loading.py src\tests\test_retrieval_diversity.py src\tests\test_retrieval_provider_multi_index.py src\tests\test_tune_retrieval_geo.py -q
```

Result: `46 passed`.

## May 9, 2026: Retrieval-Mistake Hard-Negative Projection

This branch tested a more targeted answer to the "what is the model missing?" question. Instead of adding another heuristic reranker, it mined hard-negative triplets from the current production retrieval stack's own high-scoring wrong candidates. The training examples therefore encode the mistakes the app actually makes at inference time.

New tool:

```powershell
.\.venv\Scripts\python.exe -m src.tools.mine_retrieval_hard_triplets --config src/config/paris.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/train_pairs.csv --reference-metadata data/spacenet_paris/metadata.csv --limit 160 --output runs/retrieval_hard_triplets_train160.jsonl --summary-output runs/retrieval_hard_triplets_train160_summary.json
```

Mining result:

- `160/160` train records produced valid triplets.
- No missing query files, no empty candidate sets, no dropped triplets.
- Each query used nearby SpaceNet Paris reference chips as positives and the retrieval provider's own wrong returned chips as negatives.

Training command:

```powershell
.\.venv\Scripts\python.exe -m src.tools.train_crossview_projection --triplets runs/retrieval_hard_triplets_train160.jsonl --aerial-index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz --street-images-dir data/paris_realistic_v1/street_combined --output runs/retrieval_hardneg_crossview_projection_v1.npz --max-triplets 160 --epochs 6 --batch-size 16 --learning-rate 3e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --sample-weight-mode triplet_weight --sample-weight-max 4 --device auto
```

Fair serving-path comparison on the same `80` strict probe samples (`seed=42`, `run_geo_eval.py`, full app path, same config except projection file):

| Model | Mean km | Median km | p90 km | <=2 km | <=5 km | <=10 km |
|---|---:|---:|---:|---:|---:|---:|
| Current master projection | 9.1105 | 7.0931 | 16.7438 | 3.75% | 25.00% | 62.50% |
| Retrieval-mistake hard-negative projection | 4.7680 | 4.8332 | 6.0800 | 1.25% | 57.50% | 100.00% |

Decision:

- Promote `runs/retrieval_hardneg_crossview_projection_v1.npz` in `src/config/paris.json`.
- Keep the mining tool because it turns live retrieval failures into supervised training data.
- The result is a clear runtime serving-path improvement in mean error, p90 error, `<=5km`, and `<=10km`; the only measured regression is `<=2km`, so the next pass should mine more near-field hard negatives below `3 km` to recover close-range precision.

Follow-up fusion diagnostics on the same `80` strict probe samples separated representation quality from estimate aggregation:

| Variant | Mean km | Median km | p90 km | <=2 km | <=5 km | <=10 km | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Promoted v1 full pipeline | 4.7680 | 4.8332 | 6.0800 | 1.25% | 57.50% | 100.00% | baseline after v1 |
| v1 retrieval-only diagnostic | 4.6143 | 4.6345 | 6.0913 | 0.00% | 66.25% | 100.00% | diagnostic only |
| v2 broad+near hard-negative projection | 4.8302 | 4.9400 | 6.0356 | 0.00% | 52.50% | 100.00% | rejected |
| v1 compact-stat fusion, 25 candidates kept | 4.7288 | 4.7759 | 6.0684 | 1.25% | 56.25% | 100.00% | promoted as a small tail/center tweak |
| v1 retrieval-dominant serving path, GeoCLIP gated off when retrieval index exists | 4.6213 | 4.6345 | 6.0913 | 0.00% | 66.25% | 100.00% | promoted as close-range serving improvement |

Interpretation:

- Near-field mixed v2 training was not promoted because it regressed mean, median, and `<=5 km` despite a tiny p90 gain.
- Retrieval-only v1 is still better on `<=5 km`, which suggests the remaining close-range failure is mostly candidate distribution and candidate ranking rather than late fusion alone.
- `src/config/paris.json` now keeps all `25` fusion candidates for UI/map inspection, but tightens the fusion estimate statistics with `retrieval_temperature=0.22`, `credible_mass=0.6`, `min_credible_candidates=1`, `credible_cluster_radius_km=6.0`, and `plausibility_radius_km=12.0`.
- The May 10 serving-path update adds `geolocator.use_geoclip_with_retrieval=false` for the Paris profile. This keeps GeoSpot/GeoCLIP available for profiles without retrieval indices, but prevents the weaker global provider from diluting the hard-negative retrieval model when the Paris index is active.
- A support-density candidate selector was tested and rejected: `mean 5.3660 km`, `median 5.1836 km`, `p90 7.3209 km`, and `<=5 km 42.50%` on the same `80` strict probe samples. The result confirms that local candidate density alone is not a safe replacement for learned cross-view ranking.

## May 10, 2026: Diversity-Capped Retrieval-Mistake Projection

This branch tested the next hard-negative scaling step. A naive `480`-query mining pass produced valid triplets, but training exposed a concentration failure: the trainer touched only `84` unique references, and projections trained directly on that pool regressed the fixed strict probe. The fix was to make hard-negative mining diversity-aware and to support conservative fine-tuning from the current projection instead of always starting from identity.

New tooling:

- `src.tools.mine_retrieval_hard_triplets` now supports `--max-negative-reuse` and reports `unique_positive_paths`, `unique_negative_paths`, and `top_negative_reuse`.
- `src.tools.train_crossview_projection` now supports `--init-projection`, allowing a new projection pass to fine-tune the current serving projection instead of resetting the learned mapping.

Completed benchmark on the same `80` strict probe samples (`seed=42`):

| Variant | Mean km | Median km | p90 km | <=2 km | <=5 km | <=10 km | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Current retrieval-dominant v1 | 4.6213 | 4.6345 | 6.0913 | 0.00% | 66.25% | 100.00% | previous baseline |
| 480 mined triplets, identity init | 21.6922 | 23.0790 | 27.2324 | 0.00% | 0.00% | 2.50% | rejected |
| 480 mined triplets, v1 init | 5.3267 | 5.4025 | 6.6124 | 0.00% | 36.25% | 100.00% | rejected |
| 480 mined triplets, tiny v1 update | 4.6347 | 4.6503 | 6.1219 | 0.00% | 67.50% | 100.00% | rejected: mixed |
| Diversity cap 8, v1 init (`104` triplets, `37` unique negatives) | 4.5974 | 4.6376 | 5.9181 | 0.00% | 68.75% | 100.00% | kept as diagnostic |
| Offline cap 16, v1 init (`152` triplets, `49` unique negatives) | 4.5791 | 4.6367 | 5.9161 | 0.00% | 67.50% | 100.00% | promoted on this branch |

Decision:

- Promote `runs/retrieval_hardneg_crossview_projection_v4_cap16_initv1.npz` on the branch because it improves mean error and p90 while preserving `<=10 km`.
- Do not claim a breakthrough: the gain is real but modest, and `<=2 km` remains at `0.00%`.
- The stronger research conclusion is diagnostic: the current Paris error set is extremely concentrated. Further large gains need broader near-field hard negatives or a stronger cross-view representation, not simply more epochs on repeated wrong chips.

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

## May 10, 2026: Candidate Oracle Rank Diagnostic

This branch added explicit top-k candidate oracle metrics to `src.tools.run_geo_eval`. The purpose was to separate two different failure modes:

1. The right location is missing from the retrieved candidates.
2. The right/near location is present but ranked below visually similar wrong candidates.

On the same fixed `80` strict Paris probe samples (`seed=42`) with the promoted diversity-capped hard-negative projection, the full serving path remains:

| Variant | Mean km | Median km | p90 km | <=1 km | <=2 km | <=5 km | <=10 km |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current serving prediction | 4.5791 | 4.6367 | 5.9161 | 0.00% | 0.00% | 67.50% | 100.00% |
| Candidate oracle over returned top-25 | 2.3528 | 2.1638 | 4.0063 | 21.25% | 43.75% | 100.00% | 100.00% |

Additional diagnostic:

- Mean returned candidate count: `25.0`.
- Mean rank of the closest returned candidate: `15.125`.
- Existing learned candidate reranker retest was rejected: base and reranked metrics were identical (`mean 4.5722 km`, `<=5 km 67.50%`), while the oracle remained much better.
- Graph-support reranking was also rejected for default serving despite an offline shortlist improvement. The real pipeline run worsened mean and p90 (`mean 4.7027 km`, `p90 6.5027 km`) while only moving `<=2 km` to `2.50%`.

Decision:

- Keep the oracle-rank diagnostics because they expose the remaining bottleneck directly.
- Do not promote the tested graph-support config or the current learned reranker.
- The next model work should target a stronger street-to-aerial visual reranker or encoder adaptation objective that can lift the oracle candidate from rank ~15 toward rank 1. Pure spatial clustering is not enough.

Follow-up on the same branch:

- Added an experimental listwise candidate-reranker trainer (`--fit-mode listwise`) with an exponential rank-score activation. It learned a small offline signal but failed in the full fusion path: `mean 5.3499 km`, `p90 6.7075 km`, `<=5 km 42.50%`. Rejected.
- Added `--positive-source closest_candidate` to `src.tools.mine_retrieval_hard_triplets`, allowing direct oracle-candidate supervision from the returned shortlist.
- Mined `104` oracle-candidate triplets from `240` train records. The run exposed a severe positive-diversity bottleneck: only `8` unique positive chips.
- Trained `runs/retrieval_oracle_candidate_projection_v1.npz` from the current projection. It improved closest-candidate mean rank (`15.125` -> `10.375`) but damaged serving accuracy and oracle quality (`mean 5.0353 km`, `<=5 km 50.00%`, oracle `<=2 km 30.00%`). Rejected.

Updated decision:

- Do not train harder on the current oracle-positive pool; it is too concentrated.
- The next useful work is data/index work: increase positive diversity inside the returned shortlist, then repeat oracle-positive training. The model is not lacking a ranking loss alone; it is lacking enough varied correct candidate examples for that loss to generalize.

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

## Apr 30, 2026: Tier 1 kept, Tier 2 completed, Tier 3 evaluated

- Promoted `src/config/paris.json` from the older single-index setup to the validated dual-index projected+DBA `rrf` profile.
- Verified on `runs/geo_eval_tier1_upgraded_paris_180.json` that the default Paris serving config improved from `mean_km 15.53` to `14.60`, from `median_km 9.77` to `4.21`, and from `<=2km 19.44%` to `31.11%`, while `<=1km` stayed flat at `10.56%`.
- Ran the next realistic cross-view projection training pass on the full `26,204` mined triplets:

```powershell
.\.venv\Scripts\python -m src.tools.train_crossview_projection --triplets runs/paris_realistic_crossview_train_triplets_v1.jsonl --aerial-index data/paris_realistic_v1_combined/indices/aerial_clip_index.npz --street-images-dir data/paris_realistic_v1/street_combined --output runs/crossview_projection_paris_combined_v2_full.npz --report-output runs/crossview_projection_paris_combined_v2_full.report.json --embedding-model openai/clip-vit-large-patch14 --max-triplets 0 --epochs 30 --batch-size 32 --learning-rate 1e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device auto
```

- Important runtime constraint: this workspace currently has `torch 2.11.0+cpu`, so `--device auto` resolves to CPU. That made Tier 2 materially slower here than the original GPU-based expectation.
- Tier 2 result on the same strict `probe240` benchmark:

| Model | Mean km | Median km | <=1 km | <=2 km | <=5 km |
|---|---:|---:|---:|---:|---:|
| First probe projection (`6000` triplets, `8` epochs) | 9.75 | 10.24 | 2.08% | 7.50% | 20.42% |
| Full-triplet projection (`26204` triplets, `30` epochs) | 9.83 | 10.92 | 4.17% | 12.08% | 22.92% |

- Interpretation:
  - Scaling projection training to all `26k` triplets improved the close-range success rates that matter most for this benchmark.
  - The tradeoff is that `mean_km` and `median_km` regressed slightly, so this is a real but mixed gain rather than a clean Pareto improvement.
- Tier 3 DINOv2 follow-up:

```powershell
.\.venv\Scripts\python -m src.tools.build_geo_index --images-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --output data/geo_index/spacenet_paris_chips_facebook_dinov2_base.npz --model-id facebook/dinov2-base
.\.venv\Scripts\python -m src.tools.run_geo_eval --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --config src/config/paris_dinov2_rrf_experimental.json --retrieval-only --limit 180 --seed 42 --output runs/geo_eval_paris_dinov2_rrf_experimental_180_fixed.json
```

- Supporting fix during Tier 3:
  - `src/core/logic/config.py` had been deduplicating `retrieval_index_model_ids`, which silently broke the positional mapping for a three-index config that legitimately repeats the CLIP model id twice.
  - The loader now preserves order and duplicates for `retrieval_index_model_ids`, and `src/tests/test_config_loading.py` covers that case.
- Tier 3 result versus the Tier 1 serving baseline:

| Config | Mean km | Median km | <=1 km | <=2 km | <=5 km | <=10 km |
|---|---:|---:|---:|---:|---:|---:|
| Tier 1 default Paris config | 14.60 | 4.21 | 10.56% | 31.11% | 52.78% | 65.56% |
| Tier 3 DINOv2 experimental fusion | 14.42 | 4.47 | 13.33% | 31.67% | 52.22% | 66.67% |

- Interpretation:
  - DINOv2 added a complementary signal and helped the very-close buckets, especially `<=1km`.
  - The regression in `median_km` and `<=5km` means the gain is still mixed, so the DINOv2 fusion should stay experimental rather than replacing the default Paris serving config.
- Tier 4 encoder fine-tune kickoff:

```powershell
.\.venv\Scripts\python.exe -m src.tools.train_retrieval_encoder --triplets runs/paris_realistic_crossview_train_triplets_v1.jsonl --query-images-dir data/paris_realistic_v1\street_combined --reference-images-dir data/paris_realistic_v1_combined --model-id openai/clip-vit-large-patch14 --output-dir runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1 --report-output runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1.report.json --train-scope vision_encoder --epochs 1 --batch-size 8 --learning-rate 1e-5 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.2 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device auto
```

- Rationale:
  - Tier 4 should test real encoder adaptation on the realistic cross-view corpus, not just another projection layer.
  - Because this workspace is CPU-only, the first pass is intentionally `1` epoch over the full `26204`-triplet dataset so we get a measured outcome before scaling to a slower multi-epoch run.
- Execution status:
  - Added `scripts/run_tier4_encoder_ft.ps1` so the training, aerial-index rebuild, and `probe240` eval can run as one reproducible Tier 4 pipeline.
  - Validated the path with a `--max-triplets 1` smoke run: `runs/retrieval_encoder_finetune/smoke_one_triplet/` and `runs/retrieval_encoder_finetune/smoke_one_triplet.report.json`.
  - The full unattended background launch was not healthy in this shell environment: it stalled after CLIP initialization and did not enter measurable training.
  - Captured logs from the stalled attempt: `runs/tier4_encoder_ft_pipeline.log` and `runs/tier4_encoder_ft_train.stderr.log`.
  - Expected full-run outputs once the execution path is stable:
    - `runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1/`
    - `runs/retrieval_encoder_finetune/paris_realistic_crossview_v1_e1.report.json`
    - `data/paris_realistic_v1_combined/indices/aerial_clip_index_retrieval_encoder_ft_v1_e1.npz`
    - `runs/eval_realistic_crossview_combined_strict_probe240_encoderft_v1_e1_full40k.json`
