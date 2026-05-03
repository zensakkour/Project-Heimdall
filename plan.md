# Branch Plan: `master`

## Purpose

`master` is now the integration branch for the April 2026 Paris geolocation rollout. The immediate goal is no longer to merge feature branches. That work is already complete. The goal is to keep a correct view of shipped tier status and make the next experimental step explicit.

## Current Repository State

- Current branch: `master`
- Working tree: clean
- All local branches are already merged into `master`:
  - `tech/analysis-ui-rfdetr`
  - `tech/paris-realistic-data-v1`
  - `tech/structure-analysis-cues-v2`
- `git branch --no-merged master` is empty.

## Tier Status

1. Tier 1 complete and kept as the default serving path.
   - `src/config/paris.json` is the active kept config.
   - Measured artifact: `runs/geo_eval_tier1_upgraded_paris_180.json`
   - Result: `mean_km 15.53 -> 14.60`, `median_km 9.77 -> 4.21`, `<=2km 19.44% -> 31.11%`, `<=5km 37.22% -> 52.78%`.
2. Tier 2 complete and measured.
   - Full realistic cross-view projection retrain finished.
   - Status: mixed gain, especially for close-range buckets, but not a clean replacement.
3. Tier 3 complete and kept experimental.
   - Experimental config: `src/config/paris_dinov2_rrf_experimental.json`
   - Status: slightly better `mean_km` and very-close hits, but weaker `median_km` and `<=5km`; do not promote over Tier 1.
4. Tier 4 prepared but not benchmarked.
   - Runner: `scripts/run_tier4_encoder_ft.ps1`
   - Smoke test passed for the encoder fine-tune path.
   - Blocker: unattended full CPU execution stalled before usable training progress, so no benchmark artifact exists yet.
5. Tier 5 not started.
   - No CosPlace or MegaLoc integration is landed in this repo state.

## Next Intended Move

1. Either get Tier 4 to run reliably end to end on available hardware, or explicitly defer it until GPU-capable execution is available.
2. Keep Tier 1 as the serving baseline unless a later tier beats it on both close-range and overall error, not just isolated buckets.
3. Treat Tier 3 as an experimental side path until it shows a cleaner benchmark win.

## Decision Criteria

- Keep `src/config/paris.json` as default unless a candidate improves both practical close-range accuracy and broader error metrics.
- Do not claim Tier 4 progress beyond "prepared" until a full benchmark artifact exists.
- Do not start Tier 5 integration until Tier 4 is either measured or deliberately postponed.
