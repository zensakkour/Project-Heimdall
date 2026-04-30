# Branch Plan: `tech/analysis-ui-rfdetr`

## Branch Purpose

This branch exists to improve the operator-facing analysis experience and add an optional RF-DETR detector backend without disturbing the validated data/model milestone on `master`.

The user goal is twofold:

1. Make the site feel like an interactive geolocation analysis dashboard: globe/map candidates, confidence interval, candidate uncertainty, radar-like pulsing points, click-to-focus behavior, and enough supporting data to verify the predicted location.
2. Evaluate RF-DETR as a stronger object-detection option for street/source-photo analysis while keeping the current detector stack intact.

## Current Direction

- Preserve the existing MapLibre globe and fusion output payload.
- Add interaction and display layers around existing candidate/fusion data rather than inventing a separate map pipeline.
- Add RF-DETR behind `detector.backend = "rfdetr"` so the default configs still work without the `rfdetr` package installed.
- Keep RF-DETR as optional until it is benchmarked against the current detector path on the same UI/server workflow.

## RF-DETR Notes

- Repository reviewed: `https://github.com/roboflow/rf-detr`.
- RF-DETR is a real-time transformer object detection and instance segmentation model built around a DINOv2-style vision backbone.
- Apache-designated RF-DETR model sizes are compatible with open experimentation; Plus/XL variants have separate licensing and should not be made default without a license review.
- Initial integration should convert RF-DETR axis-aligned boxes into Heimdall `Detection` objects so the rest of the pipeline and canvas overlay keep working.

## Implementation Plan

1. Add optional RF-DETR detector adapter and config parsing.
2. Add tests with a mocked `rfdetr` module so CI does not require downloading model weights.
3. Upgrade the `/analysis/` map:
   - pulsing candidate points
   - click-to-fly candidate focus
   - selected candidate uncertainty ring
   - selected-to-mean support line
   - compact candidate inspector with rank, coordinates, posterior, retrieval score, interval, and source
4. Update README and `PROGRESS.md`.
5. Run focused detector tests plus frontend syntax checks.
6. Push the branch separately.

## Decision Criteria

- The UI change is acceptable if it preserves the existing analysis endpoint and improves verification without hiding raw evidence.
- RF-DETR is acceptable as an optional backend if existing configs keep behaving the same and tests pass without RF-DETR installed.
- RF-DETR should not replace the default detector until detector quality and geolocation impact are benchmarked.
