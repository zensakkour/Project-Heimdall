# Branch Plan: `feat/analysis-ui-v2`

## Purpose

Modernize and enhance the Analysis Dashboard UI. This branch focuses on improving the visualization of geolocation results, RF-DETR detections, and overall user experience for analyzing model performance.

## Objectives

1.  **UI Refresh**: Update `index.html` and `styles.css` for a more modern, responsive layout.
2.  **Performance Visualization**: Improve how geolocation candidates and ground truth are displayed in the dashboard.
3.  **Interactive Analysis**: Add features to filter and drill down into specific detection/geolocation cases.
4.  **Backend Integration**: Ensure `dev_ui.py` and `dev_app.py` support any new frontend requirements.

## Current State

-   Dashboard exists in `src/dashboard/`.
-   Uses `app.js`, `index.html`, `styles.css`.
-   Basic functionality for viewing analysis results is present but needs polish and features.

## Next Steps

1.  Audit current `src/dashboard/` implementation.
2.  Identify specific UI/UX bottlenecks.
3.  Implement layout improvements.
4.  Enhance data visualization components in `app.js`.

## Decision Criteria

-   Keep changes if they improve the clarity of model analysis.
-   Ensure compatibility with existing data schemas (e.g., `batch_result.schema.json`).
-   Verify that the dev server (`dev_ui.py`) remains functional.
