# Reproducibility

This document captures exact evaluation steps for Project Heimdall as of 2026-01-30.

## Environment
- Python 3.10+ (Windows 10/11 or WSL2)
- Install dependencies: `pip install -r requirements.txt`

## Full evaluation run (stub pipeline)
1. Prepare a list of images (manual Paris upload samples live in `data/analysis_tests/paris_street/images`).
2. Run the full pipeline:
   ```powershell
   python -m src.tools.run_all data/analysis_tests/paris_street/images/mapillary__1021055432583866.jpg
   ```
3. Compute metrics (requires ground truth to be meaningful):
   ```powershell
   python -m src.tools.eval_metrics runs/results.jsonl --ground-truth data/ground_truth.jsonl
   ```
4. Generate UI data:
   ```powershell
   python -m src.tools.generate_ui_data --jsonl runs/results.jsonl
   ```

## Notes
- Metrics are placeholders until ground truth labels are wired in.
- Capture the git commit hash in run metadata when executing production evaluations.

### Ground truth format
Provide a CSV/JSON/JSONL file with `image`, `latitude`, `longitude` fields. Examples:
- JSONL line: `{"image": "sample.jpg", "latitude": 35.0, "longitude": -120.0}`
- CSV header: `image,latitude,longitude`
