# Documentation Map

This file is the index of Markdown documentation in Project Heimdall.
Use it as the first stop before adding new `.md` files.

## Core Entry Points
- `README.md`: front-page overview, quick start, and documentation navigation.
- `docs/WORKFLOWS.md`: command recipes, benchmark flow, tuning/training entry points, and demo-maintenance steps.
- `docs/engineering/PROGRESS.md`: append-only engineering/research history and validation snapshots.
- `src/docs/RESEARCH_PAPER.md`: full research-style narrative of methods, experiments, results, and conclusions.
- `src/docs/MARKET_RESEARCH.md`: external market/SOTA landscape synthesis and prioritized approaches considered.
- `src/docs/GEO_TECH.md`: geolocation architecture, knobs, and technical behavior.
- `src/docs/REPRODUCIBILITY.md`: reproducibility and evaluation procedures.
- `docs/research/research.md`: chronological evidence ledger with before/after metrics.

## Governance and Community
- `docs/governance/CONTRIBUTING.md`
- `docs/governance/CHANGELOG.md`
- `docs/governance/CODE_OF_CONDUCT.md`
- `docs/governance/SECURITY.md`
- `docs/governance/SUPPORT.md`

## Component-Specific Docs
- `src/dashboard/README.md`: dashboard/analysis UI structure and endpoints.
- `docs/DATA_LAYOUT.md`: datasets, model artifacts, and local directory expectations.
- `branches/README.md`: git worktree usage guidance.
- `docs/CLEANUP_REPORT.md`: repository cleanup/refactor inventory and removal rationale.

## Generated / Run-Dependent Docs
- `docs/eval/latest_report.md`: benchmark run report (generated).
- `docs/eval/latest_pr_summary.md`: benchmark PR summary (generated).

## Documentation Hygiene Rules
1. Prefer updating an existing document over creating a new one.
2. If adding a new document, add it to this index in the same commit.
3. Keep `docs/engineering/PROGRESS.md` append-only.
4. Keep research outcomes synchronized in `src/docs/RESEARCH_PAPER.md`.
5. Keep `README.md` focused on user-facing overview, quick start, and navigation; move long command recipes into `docs/WORKFLOWS.md`.
