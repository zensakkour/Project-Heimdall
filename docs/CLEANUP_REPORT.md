# Cleanup Report: `refactor/cleanup-modernize`

## Repository Inventory

Tracked source/config/UI files are primarily:

- Python: `167` tracked `.py` files under `src/`, with CLI entry points in `src/cli.py`, batch tooling in `src/batch_run.py`, domain modules under `src/core/`, schemas under `src/schemas/`, tests under `src/tests/`, and operational tools under `src/tools/`.
- Frontend: vanilla HTML/CSS/JavaScript under `src/dashboard/`, including the analysis UI in `src/dashboard/analysis/`.
- Configuration and workflows: JSON configs in `src/config/`, schemas in `src/schemas/`, benchmark contracts in `benchmarks/`, GitHub Actions in `.github/workflows/`, shell helpers in `src/scripts/`, PowerShell helpers in `scripts/`, and `run_heimdall.cmd`.
- Documentation: `README.md`, `docs/`, `src/docs/`, governance files, and dashboard-specific docs.
- Research/eval artifacts: selected tracked JSON/JSONL/NPZ artifacts under `runs/` and `docs/eval/`.

Languages and frameworks:

- Python 3.10+ with FastAPI, pytest, Pillow, NumPy/ML tooling, and the project core pipeline.
- Vanilla HTML/CSS/JavaScript with MapLibre GL in the dashboard.
- GitHub Actions YAML, PowerShell, Bash, JSON Schema, and Windows command scripts.

## Duplication and Refactor Findings

The scan found no safe three-or-more-copy function/class duplication that should be abstracted immediately. Most repeated code is either:

- test setup with intentionally local fixtures,
- small CLI argument boilerplate,
- generated research/eval records that should remain explicit,
- or live UI/server code still changing in the working tree.

Following the project goal of avoiding premature abstractions, this cleanup removed concrete duplication and stale artifacts instead of introducing broad abstractions over unstable code.

## Removed Items

- `src/dashboard/analysis/operator.js.orig`: tracked backup copy of the analysis UI script. The live source is `src/dashboard/analysis/operator.js`.
- `src/dashboard/analysis/operator.css.orig`: tracked backup copy of the analysis UI stylesheet. The live source is `src/dashboard/analysis/operator.css`.
- `fix_research.py`: one-off script that appended an old research-paper section and was not referenced by code, docs, tests, or workflows.
- `test_image_save.py`: root-level ad-hoc smoke test outside the configured pytest suite. The covered behavior is represented by `src/tests/test_ui_server_operator_session.py`.

## Preserved Items

- `docs/engineering/PROGRESS.md`, `docs/research/research.md`, `src/docs/RESEARCH_PAPER.md`, `src/docs/GEO_TECH.md`, `src/docs/REPRODUCIBILITY.md`, and `docs/DATA_LAYOUT.md` are actively referenced and remain canonical.
- `docs/eval/` and tracked `runs/` artifacts are preserved because README/research docs use them as benchmark evidence.
- `branches/README.md` is preserved as repository workflow documentation.

## Ignore Policy Updates

`.gitignore` is organized into logical groups and now explicitly ignores:

- Python caches/build output and coverage artifacts,
- virtual environments,
- local secrets,
- pytest/runtime scratch directories,
- operator sessions and local case files,
- generated dashboard data,
- large data/model/run outputs,
- OS/editor files.

This prevents local analysis state such as `cases/` and `.pytest_tmp/` from entering future commits.

## Validation

- `.\.venv\Scripts\python.exe -m compileall -q src`
- `.\.venv\Scripts\python.exe -m src.tools.check_readme_links README.md src/dashboard/README.md`
- `.\.venv\Scripts\python.exe -m pytest src\tests\test_ui_server_operator_session.py -q`
- `$env:TMP=(Resolve-Path .pytest_tmp).Path; $env:TEMP=$env:TMP; .\.venv\Scripts\python.exe -m pytest -q`

Result: `285 passed, 3 warnings` with repo-local `.pytest_tmp` as the pytest temp base.

The full suite first failed before tests executed for affected cases because pytest could not scan the default Windows temp directory `C:\Users\zen\AppData\Local\Temp\pytest-of-zen`. The repo-local temp rerun passed.
