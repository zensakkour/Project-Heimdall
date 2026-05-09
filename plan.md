# Branch Plan: `feat/operator-dashboard`

## Purpose

Implement Heimdall Operator Mode: a polished single-operator visual intelligence dashboard for one image/video task at a time. Do not build multi-user case management or surveillance workflows. Focus on one operator, one uploaded source, one analysis session, with strong privacy and safety guardrails.

## Objectives

1.  **Backend Operator Endpoints**: Implement session management, map pins, notes, confirmation endpoints, and robust error bubbling (no silent fails).
2.  **UI Layout**: Restructure the operator mode into a serious visual investigation console with three panels: left (source/ingest), center (map), right (intelligence/results).
3.  **Timeline & Evidence**: Display extracted clues explicitly and visualize the chronological progress of the backend pipeline on a timeline.
4.  **Dev Mode**: Provide a frontend-facing toggle to mock analysis output for rapid UI styling testing.

## Current State

-   Added `ui_server.py` Operator API endpoints (`/api/operator/*`).
-   Configured session timeline and error capturing during the pipeline run.
-   Refined `operator.js`, `index.html`, and `operator.css` for a panel-based layout and interactive timeline/clues.

## Next Steps

1. Submit changes for review and merge.

## Decision Criteria

-   Keep if tests pass and operator workflow supports end-to-end ingest to export loop.
