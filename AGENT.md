# AGENT.md

## Purpose
This file is the operational guide for future contributors and coding agents working on Project Heimdall.
Use it to keep algorithm history, experiment intent, and evaluation standards consistent.

## Canonical Documents
- Engineering change log: `PROGRESS.md` (append-only)
- Research narrative: `src/docs/RESEARCH_PAPER.md`
- Geo stack technical reference: `src/docs/GEO_TECH.md`
- Repro and eval workflow: `src/docs/REPRODUCIBILITY.md`
- Benchmark contracts and outputs: `benchmarks/` and `docs/eval/`

## Non-Negotiable Rules
1. Do not rewrite historical entries in `PROGRESS.md`; only append.
2. Do not claim accuracy improvements without citing a concrete run artifact in `runs/` or `docs/eval/`.
3. Prefer realistic split evaluation over leakage-prone setups when making quality claims.
4. Keep config changes explicit and traceable; mention exact knobs changed.
5. If a change touches scoring/fusion/retrieval logic, add or update regression tests.

## Experiment Logging Protocol
For every meaningful modeling change, record these fields in `PROGRESS.md`:
- Date (YYYY-MM-DD)
- Hypothesis (what should improve and why)
- Implementation summary (files and knobs)
- Validation command(s)
- Result summary with numeric metrics
- Artifact paths (`runs/*.json`, `docs/eval/*.md`, etc.)
- Decision (keep/revert/follow-up)

## Suggested Experiment Record Template
```text
## YYYY-MM-DD
- Hypothesis:
- Change:
- Files touched:
- Validation command:
- Metrics (before -> after):
- Artifacts:
- Decision:
```

## Accuracy Claim Standard
A performance claim should include:
- Dataset/split identity
- Metric set (`mean_km`, `median_km`, `within_5km_pct`, etc.)
- Sample count (`evaluated`, `total`, `null_predictions`)
- Baseline and candidate artifact paths
- Whether results are leakage-safe or potentially leakage-prone

## Branch and Merge Practice
- Keep model work on dedicated branches (for example `tech/*`).
- Before merge, run focused tests first, then broader suites when dependencies allow.
- Keep commits scoped: one conceptual change per commit when possible.

## Current Research Direction (As of 2026-04-15)
- Continue benchmark-driven improvement of realistic geolocation quality.
- Expand and curate retrieval data with hard negatives.
- Tune retrieval and fusion jointly with reproducible sweeps.
- Improve confidence calibration and reliability under ambiguity.

## When Updating This File
Update only when process or standards change. Keep entries concise, concrete, and enforceable.
