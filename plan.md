# Branch Plan: `tech/analysis-ui-rfdetr`

## Branch Purpose

This branch no longer represents only the original RF-DETR and analysis-dashboard work. It now carries the current Paris geolocation improvement execution on top of that UI/runtime base, and it is the last branch not yet merged into `master`.

The effective branch goal is:

1. Preserve the already-integrated analysis UI and RF-DETR runtime work.
2. Land the April 30, 2026 Paris geolocation improvements, research updates, and merge this branch into `master`.

## Current State

- `master` already contains:
  - `tech/paris-realistic-data-v1`
  - `tech/structure-analysis-cues-v2`
- This branch is the only local branch not yet merged into `master`.
- The default Paris serving config `src/config/paris.json` has already been upgraded to the validated dual-index projected+DBA RRF profile.
- The remaining branch-specific work is the Apr 30 geolocation execution record, the DINOv2 experimental path, the config-loader fix, and the Tier 4 encoder fine-tune runner.

## Geolocation Execution Status

1. Tier 1 complete and kept.
   - `paris.json` now uses the validated dual-index projected RRF configuration.
   - Measured on `runs/geo_eval_tier1_upgraded_paris_180.json`: `mean_km 15.53 -> 14.60`, `median_km 9.77 -> 4.21`, `<=2km 19.44% -> 31.11%`, `<=5km 37.22% -> 52.78%`.
2. Tier 2 complete and measured.
   - Full `26204`-triplet cross-view projection run finished on CPU.
   - Close-range hit rates improved, but `mean_km` and `median_km` regressed slightly, so this remains a mixed gain rather than a clean default replacement.
3. Tier 3 complete and kept experimental.
   - Added `src/config/paris_dinov2_rrf_experimental.json`.
   - Added a config-loader fix so repeated `retrieval_index_model_ids` preserve order and duplicates.
   - DINOv2 improved the closest buckets and slightly improved `mean_km`, but regressed `median_km` and `<=5km`, so it does not replace `paris.json`.
4. Tier 4 prepared, not benchmarked.
   - Added `scripts/run_tier4_encoder_ft.ps1` to run full realistic cross-view encoder fine-tune, aerial-index rebuild, and `probe240` eval in one pipeline.
   - The training path is smoke-tested with `runs/retrieval_encoder_finetune/smoke_one_triplet.report.json`.
   - Full unattended CPU execution still needs a reliable run mode before metrics can be claimed.
5. Tier 5 not started in this branch.
   - No CosPlace or MegaLoc integration has been landed yet.

## Files That Must Ship With This Branch

- Docs:
  - `PROGRESS.md`
  - `research.md`
  - `README.md`
  - `src/docs/RESEARCH_PAPER.md`
  - `plan.md`
- Config/runtime:
  - `src/config/paris_dinov2_rrf_experimental.json`
  - `src/core/logic/config.py`
  - `src/tests/test_config_loading.py`
  - `scripts/run_tier4_encoder_ft.ps1`
- Existing local branch change also in scope:
  - `src/tools/train_crossview_projection.py`

## Publish Plan

1. Finish doc updates so the branch ledger matches the measured Tier 1 to Tier 4 state.
2. Run targeted validation:
   - `pytest -q src/tests/test_config_loading.py`
   - smoke-check the Tier 4 encoder trainer path
3. Commit the branch intentionally.
4. Push `tech/analysis-ui-rfdetr`.
5. Merge `tech/analysis-ui-rfdetr` into local `master`.
6. Push `master`.
7. Confirm `git branch --no-merged master` is empty.

## Decision Criteria

- Merge this branch if:
  - the Tier 1 default config upgrade remains the kept serving default
  - Tier 2 and Tier 3 are documented honestly as mixed outcomes
  - Tier 4 is documented as prepared but not benchmarked
  - the config-loader fix and its test pass
  - `master` ends up containing every currently relevant branch
