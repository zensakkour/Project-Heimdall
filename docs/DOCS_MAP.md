# Documentation Map

This file is the index of Markdown documentation in Project Heimdall.
Use it as the first stop before adding new `.md` files.

## Core Entry Points
- `README.md`: user-facing setup, run commands, and high-level platform overview.
- `PROGRESS.md`: append-only engineering/research history and validation snapshots.
- `src/docs/RESEARCH_PAPER.md`: full research-style narrative of methods, experiments, results, and conclusions.
- `src/docs/MARKET_RESEARCH.md`: external market/SOTA landscape synthesis and prioritized approaches considered.
- `src/docs/GEO_TECH.md`: geolocation architecture, knobs, and technical behavior.
- `src/docs/REPRODUCIBILITY.md`: reproducibility and evaluation procedures.
- `AGENT.md`: operational rules for coding agents and per-prompt documentation sync.

## Governance and Community
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `SUPPORT.md`

## Component-Specific Docs
- `src/dashboard/README.md`: dashboard/analysis UI structure and endpoints.
- `branches/README.md`: git worktree usage guidance.

## Generated / Run-Dependent Docs
- `docs/eval/latest_report.md`: benchmark run report (generated).
- `docs/eval/latest_pr_summary.md`: benchmark PR summary (generated).

## Documentation Hygiene Rules
1. Prefer updating an existing document over creating a new one.
2. If adding a new document, add it to this index in the same commit.
3. Keep `PROGRESS.md` append-only.
4. Keep research outcomes synchronized in `src/docs/RESEARCH_PAPER.md`.
5. Keep `README.md` focused on user-facing workflows and quick navigation.
