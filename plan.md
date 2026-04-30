# Branch Plan: `tech/analysis-ui-rfdetr`

## Branch Purpose

This branch exists to improve the operator-facing analysis experience and add an optional RF-DETR detector backend without disturbing the validated data/model milestone on `master`.

The user goal is twofold:

1. Make the site feel like an interactive geolocation analysis dashboard: globe/map candidates, confidence interval, candidate uncertainty, radar-like pulsing points, click-to-focus behavior, and enough supporting data to verify the predicted location.
2. Evaluate RF-DETR as a stronger object-detection option for street/source-photo analysis while keeping the current detector stack intact.

## Current Direction

- Preserve the existing MapLibre globe and fusion output payload.
- Add interaction and display layers around existing candidate/fusion data rather than inventing a separate map pipeline.
- Make RF-DETR the default detector backend for active Paris configs.
- Keep a sidecar/classic fallback so the app still opens if `rfdetr` is not installed yet.
- Keep the runtime app Paris-focused; remove Open Geo/Wikimedia from active profile selection until a future expansion branch.
- Keep local `data/` organized around current Paris work: final realistic street/aerial corpora, Paris indices, model cache, and a small manual analysis-test folder.

## RF-DETR Notes

- Repository reviewed: `https://github.com/roboflow/rf-detr`.
- RF-DETR is a real-time transformer object detection and instance segmentation model built around a DINOv2-style vision backbone.
- Apache-designated RF-DETR model sizes are compatible with open experimentation; Plus/XL variants have separate licensing and should not be made default without a license review.
- Initial integration should convert RF-DETR axis-aligned boxes into Heimdall `Detection` objects so the rest of the pipeline and canvas overlay keep working.

## Implementation Plan

1. Add RF-DETR detector adapter and config parsing.
2. Add tests with a mocked `rfdetr` module so CI does not require downloading model weights.
3. Upgrade the `/analysis/` map:
   - pulsing candidate points
   - click-to-fly candidate focus
   - selected candidate uncertainty ring
   - selected-to-mean support line
   - compact candidate inspector with rank, coordinates, posterior, retrieval score, interval, and source
4. Update README and `PROGRESS.md`.
5. Remove non-Paris local data caches and stale Open Geo runtime profile.
6. Create `data/analysis_tests/paris_street/` from real Paris street images for manual UI testing.
7. Run focused detector tests plus frontend syntax checks.
8. Push the branch separately.

## Decision Criteria

- The UI change is acceptable if it preserves the existing analysis endpoint and improves verification without hiding raw evidence.
- RF-DETR is acceptable as the default detector backend if missing-package fallback keeps the app usable and tests pass without downloading model weights.
- Open Geo should not appear in `/analysis/` or `/analysis/lab/` while the product is Paris-focused.
- Local cleanup is acceptable only if it removes non-Paris or duplicated intermediate artifacts while preserving final Paris realistic datasets and Paris benchmark assets.
