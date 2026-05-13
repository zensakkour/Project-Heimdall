# Branch Plan: `tech/operator-session-logs-streetwalk-v1`

## Purpose

Improve the Heimdall Operator UI while keeping the existing visual style. Add practical app-like functionality around local session persistence and investigate/implement a street-view navigation workflow.

## Objectives

1.  **Feature 1: Local Operator Session Logs**: Add automatic local session logging for Operator Mode into `operator_sessions/`. Keep it local only. Provide a UI to load past sessions.
2.  **Feature 2: Street Walk / Street View Investigation**: Implement a local UI flow where selecting a map candidate can open a street-level view and navigate nearby street imagery based on local data.

## Current State

-   Added `_save_operator_session()` to persist state to `operator_sessions/`.
-   Created `/api/operator/sessions` and `/api/operator/sessions/{session_id}`.
-   Updated frontend (`index.html`, `operator.js`) to list and reload past sessions.
-   Implemented `LocalStreetViewProvider` abstraction in `src/core/geo/street_view.py`.
-   Created `/api/operator/street_view` endpoint.
-   Added "Street View" button to candidate cards and created a modal in `index.html`.
-   Added unit tests for both features.

## Next Steps

1. Complete pre-commit instructions.
2. Submit changes for review.

## Decision Criteria

-   Keep. The local session logs effectively retain investigations, and the street-view provider correctly falls back to local data smoothly without faking unavailable APIs.
