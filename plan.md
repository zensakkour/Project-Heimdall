# Branch Plan: `master`

## Status
- Branch purpose: integration, validation, documentation sync, and merge hygiene.
- Completion policy: `master` is not the place for unfinished branch experiments; it should only collect work that has a clear branch plan, validation trail, and merge rationale.

## Problem
The repo needs a stable trunk branch that reflects the current accepted state while still making it obvious:
- which experiment branches are active
- which directions are validated enough to merge
- which findings remain experimental

## Direction
### Stage 1: Track Active Branch Intent
- Ensure active branches have a root `plan.md`.
- Reject merges from branches that do not explain:
  - goal
  - planned comparisons
  - decision gates
  - next intended move

### Stage 2: Merge Only Validated Work
- Prefer merging branches that include:
  - focused tests
  - benchmark artifacts where claims are made
  - documentation updates in the same cycle

### Stage 3: Keep the Trunk Clean
- Delete stale branches after merge or explicit rejection.
- Keep `master` aligned with `origin/master`.
- Avoid leaving branch-specific experimental narratives in trunk docs without saying they are experimental.

### Stage 4: Prioritize Next Research Moves
- Favor compare -> choose -> deepen over parallel speculative expansion.
- Current likely priorities:
  - retrieval backbone comparisons
  - realistic-split validation
  - structure / geometry cues only after the backbone decision is justified

## Benchmarks and Evidence Rules
- No accuracy claim lands on `master` without a concrete artifact path.
- Prefer leakage-safe evaluation over easier but inflated setups.
- Keep `PROGRESS.md`, `README.md`, and `src/docs/RESEARCH_PAPER.md` aligned with what has actually been merged.

## Immediate Next Actions
1. Keep branch-plan enforcement active for all future branches.
2. Merge only branches whose `plan.md` and validation evidence are current.
3. Remove stale branches quickly after they are merged or abandoned.
