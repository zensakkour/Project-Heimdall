# Branch Plan: `tech/paris-realistic-data-v1`

## Branch Purpose

This branch exists to break the current data bottleneck.

The repo now has a reasonably mature retrieval, mining, projection-training, and evaluation stack, but the measured work to date shows a stalemate: more reranking and more hand-built scene cues are not producing the major geolocation jump we want. The next step is to build a realistic, leakage-safe Paris street-to-aerial dataset and use it to train and evaluate on the actual cross-view problem.

## Audit Summary

Prompt 1 asked for a repo audit before changing implementation. That audit is complete enough to guide this branch.

Current retrieval, index, and eval pipeline:

- The repo already builds embedding indices as `.npz` bundles with `embeddings`, coordinates, ids, paths, and `model_id`.
- Retrieval serving already supports single-index and multi-index search, per-index weights, optional per-index projection paths, and downstream fusion/reranking.
- Existing evaluation work has already converged on leakage-safe fixed-split benchmarking as the required standard.

Current metadata CSV expectations:

- Existing tools already tolerate metadata columns like `path` or `image`, plus `lat` or `latitude`, and `lon` or `longitude`.
- The next dataset should stay close to those conventions so the new data pipeline can reuse existing index and eval code with minimal glue.

Existing hard-negative mining and projection training reuse points:

- `src/tools/mine_hard_negative_triplets.py` already emits weighted JSONL triplets for query, positives, and hard negatives.
- `src/tools/train_retrieval_projection.py` already trains a lightweight projection head from those mined triplets with sample weighting.
- The new realistic cross-view tools should reuse those ideas, but explicitly model street-query versus aerial-reference training rather than same-view retrieval only.

Existing config implications:

- `src/core/logic/config.py` already has the retrieval-source and projection hooks needed for multi-source experiments.
- New work should add only the minimum extra knobs needed for realistic cross-view scoring, especially heading-aware retrieval and future street-plus-aerial fusion.

Docs and progress policy:

- `PROGRESS.md` is append-only.
- Branches are expected to keep a root `plan.md`.
- Research claims should be documented with measured before/after metrics and artifact paths.
- No improvement claim should be made unless it beats the fixed realistic baseline on a leakage-safe split.

## Implementation Plan

The branch plan is:

1. Build the realistic Paris street-image dataset from an allowed provider with reliable metadata.
2. Pair those street images with open aerial imagery and persist explicit pair files.
3. Create leakage-safe spatial train, validation, and test splits.
4. Mine cross-view hard negatives and train a street-to-aerial projection baseline.
5. Evaluate street-to-aerial, street-to-street, and fused retrieval on the same fixed test split.
6. Only after those baselines exist, open a separate model-upgrade research branch for larger architectural experiments.

## Prompt Backlog

### Prompt 1: Audit current repo before changing anything

Read the whole Project-Heimdall repo before making changes.

Focus on:
- current retrieval/index/eval pipeline
- existing metadata CSV formats
- existing hard-negative mining tools
- existing projection training tools
- existing config schema under `src/core/logic/config.py`
- existing docs and `PROGRESS.md` policy

Then produce a short implementation plan for adding a realistic Paris cross-view dataset pipeline.

Do not modify files yet.

Status:
- Completed at the planning level for this branch.
- The audit findings are captured above and should govern the implementation order below.

### Prompt 2: Build Mapillary street-image dataset

Mapillary has metadata fields like `computed_geometry`, `computed_compass_angle`, `thumb_2048_url`, `quality_score`, and `sequence`, so use those.

Implement a Mapillary street-image ingestion tool for Project-Heimdall.

Create:
`src/tools/download_mapillary_paris.py`

CLI:

```powershell
python -m src.tools.download_mapillary_paris `
  --bbox 48.8156,2.2241,48.9022,2.4699 `
  --out data/paris_realistic_v1/street `
  --grid-step-m 80 `
  --street-per-cell 3 `
  --max-images 20000 `
  --seed 42
```

Requirements:
1. Read Mapillary token from environment variable `MAPILLARY_ACCESS_TOKEN`.
2. Split the Paris bbox into safe smaller query cells.
3. Query Mapillary Graph API for images in each cell.
4. Request these fields:
   `id, geometry, computed_geometry, compass_angle, computed_compass_angle, captured_at, camera_type, width, height, thumb_1024_url, thumb_2048_url, quality_score, sequence`
5. Prefer `computed_geometry` over `geometry` when available.
6. Prefer `computed_compass_angle` over `compass_angle` when available.
7. Download images using `thumb_2048_url`, fallback to `thumb_1024_url`.
8. Save images to:
   `data/paris_realistic_v1/street/images/`
9. Write metadata:
   `data/paris_realistic_v1/street/metadata.csv`
10. Columns:
    `image_id,path,lat,lon,heading_deg,captured_at,camera_type,width,height,quality_score,sequence,source,license_info`
11. Deduplicate by `image_id`.
12. Avoid near-duplicates by limiting images per sequence and per grid cell.
13. Add `--dry-run` that only reports expected count.
14. Add retry/backoff for API and download errors.
15. Add tests with mocked API responses.
16. Update `README.md` and `PROGRESS.md`.
17. Do not use Google Street View or scrape Google Maps.

### Prompt 3: Build aerial pairs

Use OpenAerialMap first because it is open imagery. Its API exposes metadata and TMS endpoints, and OAM imagery is publicly licensed through OIN under CC-BY 4.0.

Implement aerial-pair generation for the Mapillary Paris street dataset.

Create:
`src/tools/build_aerial_pairs.py`

CLI:

```powershell
python -m src.tools.build_aerial_pairs `
  --street-metadata data/paris_realistic_v1/street/metadata.csv `
  --out data/paris_realistic_v1 `
  --provider openaerialmap `
  --crop-size-m 256 `
  --crop-px 512 `
  --allow-missing-aerial false `
  --seed 42
```

Requirements:
1. Read street metadata CSV.
2. For each street image lat/lon, query OpenAerialMap metadata and TMS endpoints for imagery covering that coordinate.
3. Prefer highest-resolution available imagery.
4. Fetch an aerial crop centered on the street image coordinate.
5. Save aerial crops to:
   `data/paris_realistic_v1/aerial/images/`
6. Write:
   `data/paris_realistic_v1/aerial/metadata.csv`
7. Columns:
   `aerial_id,path,lat,lon,source,provider,resolution_m,crop_size_m,crop_px,license_info,paired_street_id,status`
8. Write positive pairs:
   `data/paris_realistic_v1/pairs.csv`
9. Pair columns:
   `pair_id,street_id,street_path,aerial_id,aerial_path,lat,lon,heading_deg`
10. If OAM has no imagery for a point:
    - mark `status=no_open_aerial_found`
    - skip the pair unless `--allow-missing-aerial` is true
11. Add a provider abstraction so future providers can be added later.
12. Add tests with mocked OAM metadata and TMS responses.
13. Update `README.md` and `PROGRESS.md`.
14. Do not download Google Maps, Google Earth, or Google Street View tiles.

### Prompt 4: Create leakage-safe splits

Implement spatial train, validation, and test splitting for the realistic Paris dataset.

Create:
`src/tools/split_realistic_dataset.py`

CLI:

```powershell
python -m src.tools.split_realistic_dataset `
  --pairs data/paris_realistic_v1/pairs.csv `
  --out data/paris_realistic_v1/splits `
  --train-ratio 0.70 `
  --val-ratio 0.15 `
  --test-ratio 0.15 `
  --cell-size-m 300 `
  --seed 42
```

Requirements:
1. Split by geographic cells, not random rows.
2. Ensure nearby images from the same area do not leak across train, validation, and test.
3. Write:
   - `train_pairs.csv`
   - `val_pairs.csv`
   - `test_pairs.csv`
4. Write `split_summary.json` with:
   - total pairs
   - split counts
   - cell counts
   - bbox
   - seed
   - cell size
5. Add a sanity-check command that reports minimum cross-split distance.
6. Add tests.
7. Update `README.md` and `PROGRESS.md`.

### Prompt 5: Mine hard negatives

Implement realistic cross-view hard-negative mining.

Create:
`src/tools/mine_realistic_crossview_triplets.py`

CLI:

```powershell
python -m src.tools.mine_realistic_crossview_triplets `
  --pairs data/paris_realistic_v1/splits/train_pairs.csv `
  --street-metadata data/paris_realistic_v1/street/metadata.csv `
  --aerial-metadata data/paris_realistic_v1/aerial/metadata.csv `
  --output data/paris_realistic_v1/triplets/train_triplets.jsonl `
  --summary-output data/paris_realistic_v1/triplets/train_triplets_summary.json `
  --positive-radius-m 80 `
  --negative-min-distance-m 300 `
  --negative-max-distance-m 5000 `
  --max-positives 3 `
  --max-negatives 20 `
  --mine-visual-negatives `
  --embedding-model openai/clip-vit-large-patch14 `
  --seed 42
```

Requirements:
1. Query equals street image.
2. Positive equals paired aerial crop.
3. Extra positives equals aerial crops within `positive-radius-m`.
4. Geographic hard negatives equals aerial crops between `negative-min-distance-m` and `negative-max-distance-m`.
5. If `--mine-visual-negatives` is enabled:
   - compute CLIP embeddings for street and aerial images
   - find visually similar but geographically wrong aerial crops
   - add them as hard negatives
6. Output JSONL:

```json
{
  "query_id": "...",
  "query_path": "...",
  "query_lat": 0.0,
  "query_lon": 0.0,
  "query_heading_deg": 0.0,
  "positive_ids": ["..."],
  "positive_paths": ["..."],
  "negative_ids": ["..."],
  "negative_paths": ["..."],
  "negative_distances_m": [0.0],
  "triplet_weight": 0.0
}
```

7. Harder triplets should get higher `triplet_weight`:
   - higher visual similarity
   - closer but wrong location
   - same arrondissement or cell if available
8. Write summary JSON:
   `total_queries, triplets_written, avg_positives, avg_negatives, skipped_no_positive, skipped_no_negative`
9. Cache embeddings so repeated runs are fast.
10. Add tests.
11. Update `README.md` and `PROGRESS.md`.

### Prompt 6: Train street to aerial projection head

Extend the existing projection-training pipeline to support street-query to aerial-reference cross-view training.

Use:
`data/paris_realistic_v1/triplets/train_triplets.jsonl`

Add or extend:
`src/tools/train_crossview_projection.py`

CLI:

```powershell
python -m src.tools.train_crossview_projection `
  --triplets data/paris_realistic_v1/triplets/train_triplets.jsonl `
  --output runs/crossview_projection_paris_v1.npz `
  --report-output runs/crossview_projection_paris_v1.report.json `
  --embedding-model openai/clip-vit-large-patch14 `
  --epochs 10 `
  --batch-size 32 `
  --learning-rate 3e-4 `
  --weight-decay 1e-4 `
  --margin 0.08 `
  --temperature 0.07 `
  --sample-weight-mode triplet_weight `
  --seed 42
```

Requirements:
1. Query side is street image embedding.
2. Reference side is aerial image embedding.
3. Train a lightweight projection head from street embedding space into aerial embedding space.
4. Keep base encoder frozen.
5. Support triplet loss and optional contrastive CE loss.
6. Consume `triplet_weight` from JSONL.
7. Save projection as NPZ with matrix, bias if used, `model_id`, and metadata.
8. Write report JSON:
   `final loss, triplet satisfied pct, weighted triplet satisfied pct, train count, skipped count, seed, config`
9. Add tests using tiny fake embeddings.
10. Update `README.md` and `PROGRESS.md`.

### Prompt 7: Build projected aerial index and evaluate

Add a full evaluation workflow for the realistic Paris dataset.

Goal:
Evaluate street query images against aerial reference index using the trained cross-view projection.

Create or extend:
`src/tools/eval_realistic_crossview.py`

CLI:

```powershell
python -m src.tools.eval_realistic_crossview `
  --test-pairs data/paris_realistic_v1/splits/test_pairs.csv `
  --aerial-metadata data/paris_realistic_v1/aerial/metadata.csv `
  --projection runs/crossview_projection_paris_v1.npz `
  --embedding-model openai/clip-vit-large-patch14 `
  --output runs/eval_realistic_crossview_paris_v1.json `
  --top-k 50 `
  --seed 42
```

Requirements:
1. Build or load aerial reference embeddings.
2. Embed street query image.
3. Apply trained street-to-aerial projection to the query embedding.
4. Retrieve top-k aerial candidates.
5. Compute distance error in meters and kilometers.
6. Report:
   `mean_km, median_km, p90_km, within_100m_pct, within_250m_pct, within_500m_pct, within_1km_pct, within_2km_pct, within_5km_pct, top1/top5/top10 recall by distance threshold`
7. Save per-sample diagnostics:
   `query path, GT coordinate, predicted coordinate, distance, top candidates`
8. Add tests.
9. Update `README.md` and `PROGRESS.md`.

### Prompt 8: Add orientation-aware scoring

This corresponds to the orientation-aware research: orientation information and rotation or viewpoint handling are central in cross-view geolocalization. EGS also explicitly targets robustness to rotation and FOV shifts.

Implement orientation-aware cross-view retrieval scoring.

Goal:
Use Mapillary `heading_deg` to improve street-to-aerial matching.

Add config knobs:
- `geolocator.use_heading_scoring`
- `geolocator.heading_score_weight`
- `geolocator.aerial_rotation_tta_degrees`

Implementation:
1. During retrieval, if query `heading_deg` is available:
   - score aerial candidates with rotated aerial embeddings or rotated query or aerial crops
   - support rotation TTA degrees: `0, 90, 180, 270` by default
2. Add a simple heading compatibility score:
   - if candidate has orientation metadata, compare heading
   - otherwise use best score over aerial rotations
3. Blend:
   `final_score = base_embedding_score + heading_score_weight * heading_score`
4. Add evaluation flag:
   `--use-heading-scoring`
5. Benchmark on:
   `data/paris_realistic_v1/splits/test_pairs.csv`
6. Save before and after reports:
   - `runs/eval_realistic_crossview_no_heading.json`
   - `runs/eval_realistic_crossview_heading.json`
7. Add tests.
8. Update `README.md` and `PROGRESS.md`.

### Prompt 9: Build street to street retrieval

This is the biggest practical product feature after street-to-aerial.

Implement street-to-street retrieval for realistic Paris geolocation.

Goal:
Use Mapillary street images as a street-level reference index. A query street photo can match nearby or visually similar street photos directly, then use their GPS.

Create:
- `src/tools/build_street_index.py`
- `src/tools/eval_street_retrieval.py`

Build CLI:

```powershell
python -m src.tools.build_street_index `
  --metadata data/paris_realistic_v1/street/metadata.csv `
  --split data/paris_realistic_v1/splits/train_pairs.csv `
  --output data/paris_realistic_v1/indices/street_clip_index.npz `
  --embedding-model openai/clip-vit-large-patch14
```

Eval CLI:

```powershell
python -m src.tools.eval_street_retrieval `
  --query-pairs data/paris_realistic_v1/splits/test_pairs.csv `
  --street-index data/paris_realistic_v1/indices/street_clip_index.npz `
  --output runs/eval_street_retrieval_paris_v1.json `
  --top-k 50
```

Requirements:
1. Build a street reference index from train split only.
2. Query with test street images.
3. Exclude exact same `image_id` if present.
4. Predict GPS from top1 and also weighted top-k average.
5. Report the same distance metrics as cross-view eval.
6. Add optional sequence or cell deduplication to avoid leakage.
7. Add tests.
8. Update `README.md` and `PROGRESS.md`.

### Prompt 10: Fuse street to street and street to aerial

Implement fusion of street-to-street and street-to-aerial candidates.

Goal:
For a query street photo, retrieve:
1. street candidates from the street index
2. aerial candidates from the aerial index using cross-view projection

Then fuse both candidate lists into one final prediction.

Create:
`src/tools/eval_realistic_fusion.py`

CLI:

```powershell
python -m src.tools.eval_realistic_fusion `
  --test-pairs data/paris_realistic_v1/splits/test_pairs.csv `
  --street-index data/paris_realistic_v1/indices/street_clip_index.npz `
  --aerial-index data/paris_realistic_v1/indices/aerial_projected_index.npz `
  --projection runs/crossview_projection_paris_v1.npz `
  --output runs/eval_realistic_fusion_paris_v1.json `
  --street-weight 0.6 `
  --aerial-weight 0.4 `
  --top-k 50
```

Requirements:
1. Convert street retrieval results and aerial retrieval results into common `GeoCandidate` objects.
2. Add source labels:
   - `retrieval:street`
   - `retrieval:aerial_crossview`
3. Use the existing fusion engine when possible.
4. Add source priors for street and aerial separately.
5. Report:
   - street-only metrics
   - aerial-only metrics
   - fused metrics
6. Save per-sample diagnostics showing whether street or aerial won.
7. Add tests.
8. Update `README.md` and `PROGRESS.md`.

### Prompt 11: Only after metrics improve, create a model-upgrade research branch

Create a research branch for architecture upgrades only after the realistic Paris dataset and baseline metrics exist.

Do not replace the current pipeline. Add experimental modes only.

Investigate and implement in this order:
1. Remote-sensing-native aerial encoder support.
2. TransGeo-style transformer retrieval head for street-to-aerial matching.
3. Graph reranker over top-k candidates.
4. SuperPoint or LightGlue or upgraded local feature matching.
5. Localizability gate or abstention model.

For each method:
- add a config flag
- add tests
- run `eval_realistic_fusion` on the same fixed test split
- write before and after metrics to `runs/`
- update `src/docs/RESEARCH_PAPER.md`
- update `PROGRESS.md`

Do not claim improvement unless it beats the fixed realistic baseline.
