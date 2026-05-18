# Workflows

Command-oriented usage guide for Project Heimdall.

Use this file for:

- common operational commands
- benchmark and evaluation recipes
- tuning and training entry points
- demo asset maintenance

For architecture and configuration details, see:

- [../src/docs/GEO_TECH.md](../src/docs/GEO_TECH.md)
- [../src/docs/REPRODUCIBILITY.md](../src/docs/REPRODUCIBILITY.md)

## App Workflows

### Start the local app

CMD:

```cmd
run_heimdall.cmd
```

PowerShell:

```powershell
.\.venv\Scripts\python -m src.tools.dev_app
```

### Generate dashboard summary data

```powershell
.\.venv\Scripts\python -m src.tools.generate_ui_data --jsonl runs/results.jsonl
```

### Rebuild dashboard payload from batch output

```powershell
.\.venv\Scripts\python -m src.tools.generate_ui_data --jsonl runs/results.jsonl
```

## CLI Inference

### Run single-image inference

```powershell
.\.venv\Scripts\python -m src.cli data/analysis_tests/paris_street/images/mapillary__1021055432583866.jpg --json
```

### Run batch inference

```powershell
.\.venv\Scripts\python -m src.batch_run data/university-1652/images/train --output runs/results.jsonl
```

## Testing and Validation

### Run the full test suite

```powershell
.\.venv\Scripts\python -m pytest -q
```

### Geo baseline gate

```powershell
.\.venv\Scripts\python -m src.tools.check_geo_regression --baseline docs/eval/geo_eval_baseline.json --candidate docs/eval/geo_eval_current.json
```

Update workflow:

1. Run geo eval and save the latest report to `docs/eval/geo_eval_current.json`.
1. Run the regression gate command above.
1. If metric changes are intentional, update `docs/eval/geo_eval_baseline.json` in a dedicated PR with rationale.

### Generate impact report

```powershell
.\.venv\Scripts\python -m src.tools.geo_impact_report --baseline docs/eval/geo_eval_baseline.json --candidate docs/eval/geo_eval_current.json --output-json runs/geo_impact.json --output-md runs/geo_impact.md
```

### Hard-negative benchmark report

```powershell
.\.venv\Scripts\python -m src.tools.geo_hard_negative_report --results runs/results.jsonl --ground-truth data/spacenet_paris/metadata.csv --output runs/hard_negative_report.json
```

## Benchmarking

### Canonical benchmark CI

Run the fixed benchmark suite from the versioned manifest and apply regression policy gates:

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core
```

What this command does:

1. Runs the canonical benchmark profile from `benchmarks/manifest.json`.
1. Compares the candidate run against the pinned baseline in `docs/eval/baseline.json`.
1. Enforces regression thresholds from `benchmarks/policy.json`.
1. Writes `docs/eval/latest_report.md` and `docs/eval/latest_pr_summary.md`.
1. Appends one summary line to `docs/eval/history.jsonl`.
1. Returns non-zero exit code if policy checks fail.

Promote a tested run as the new pinned baseline:

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core --promote <run_id>
```

### Benchmark Tool Guide

Use this when you want to benchmark geo quality before and after changes and keep a professional history.

Prerequisites:

1. Install dependencies and `torch`.
1. Ensure benchmark data paths used in `benchmarks/manifest.json` exist locally.
1. Activate your venv.

First-time setup:

1. Run one benchmark:

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core
```

1. Note the `Run complete: <run_id>` value printed in terminal.
1. Promote that run as baseline:

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core --promote <run_id>
```

Daily or PR workflow:

1. Run benchmark on your branch:

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_ci --profile core
```

1. Check policy result:
   - exit code `0`: pass or skipped if no baseline
   - exit code `1`: regression policy failed
1. Open generated report:
   - `docs/eval/latest_report.md`
1. Copy PR-ready summary from:
   - `docs/eval/latest_pr_summary.md`

Where benchmark outputs go:

- Run payload: `src/dashboard/data/benchmark_runs/<run_id>.json`
- Full artifacts per run: `runs/benchmark_history/<run_id>/`
- Pinned baseline contract: `docs/eval/baseline.json`
- Baseline snapshot: `docs/eval/baseline_summary.json`
- Append-only ledger: `docs/eval/history.jsonl`

How this connects to the UI:

1. Open the scoring tab in the app.
1. Use the saved-runs dropdown to inspect past run payloads by timestamp.
1. Use baseline and candidate selectors to compare two run ids.

### Benchmark retrieval backbones

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_geo_backbones --train-images-dir data/spacenet_paris/chips --train-metadata data/spacenet_paris/metadata.csv --eval-images-dir data/spacenet_paris_test/chips --eval-metadata data/spacenet_paris_test/metadata.csv --model-ids "openai/clip-vit-large-patch14,google/siglip-base-patch16-224" --train-limit 600 --eval-limit 200 --output runs/backbone_bench/backbone_benchmark.json
```

Aerial preset example:

```powershell
.\.venv\Scripts\python -m src.tools.benchmark_geo_backbones --train-images-dir data/spacenet_paris/chips --train-metadata data/spacenet_paris/metadata.csv --eval-images-dir data/spacenet_paris_test/chips --eval-metadata data/spacenet_paris_test/metadata.csv --model-preset aerial_rtx5060_precise --rank-objective within_2km_pct --train-limit 600 --eval-limit 200 --output runs/backbone_bench/backbone_benchmark_aerial.json
```

### Auto-upgrade retrieval backbone

```powershell
.\.venv\Scripts\python -m src.tools.upgrade_retrieval_backbone --train-images-dir data/spacenet_paris/chips --train-metadata data/spacenet_paris/metadata.csv --eval-images-dir data/spacenet_paris_test/chips --eval-metadata data/spacenet_paris_test/metadata.csv --config src/config/paris.json --model-preset aerial_rtx5060_precise --rank-objective within_2km_pct --output-dir runs/backbone_upgrade
```

Model presets currently available:

- `legacy_clip_siglip`
- `aerial_rtx5060_fast`
- `aerial_rtx5060_precise`
- `aerial_research`

## Retrieval and Fusion Tuning

### Fit fusion source priors

```powershell
.\.venv\Scripts\python -m src.tools.fit_fusion_priors --results runs/results.jsonl --ground-truth data/spacenet_paris/metadata.csv --per-source-min-count 5 --output runs/fusion_priors.json
```

Apply learned priors directly to config:

```powershell
.\.venv\Scripts\python -m src.tools.fit_fusion_priors --results runs/results.jsonl --ground-truth data/spacenet_paris/metadata.csv --per-source-min-count 5 --apply-config --config src/config/defaults.json --output runs/fusion_priors.json
```

### Fit confidence calibration

```powershell
.\.venv\Scripts\python -m src.tools.fit_confidence_calibration --results runs/results.jsonl --ground-truth data/spacenet_paris/metadata.csv --output runs/confidence_calibration.json
```

Apply learned calibration directly to config:

```powershell
.\.venv\Scripts\python -m src.tools.fit_confidence_calibration --results runs/results.jsonl --ground-truth data/spacenet_paris/metadata.csv --apply-config --config src/config/defaults.json --output runs/confidence_calibration.json
```

### Tune retrieval precision

```powershell
.\.venv\Scripts\python -m src.tools.tune_retrieval_geo --config src/config/paris_test.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 300 --output runs/tune_retrieval_geo.json --rank-objective within_2km_pct --apply-best-config
```

### Auto-tune full geo stack

```powershell
.\.venv\Scripts\python -m src.tools.auto_tune_geo_stack --config src/config/defaults.json --images-dir data/spacenet_paris/chips --metadata data/spacenet_paris/metadata.csv --results runs/results.jsonl --output-dir runs/auto_tune_geo
```

Outputs include:

- `runs/auto_tune_geo/auto_tune_summary.json`
- `runs/auto_tune_geo/auto_tune_summary.md`

If any tuning or calibration step fails, the command restores the original config automatically.

## Training and Mining

### Mine hard-negative triplets

```powershell
.\.venv\Scripts\python -m src.tools.run_geo_eval --retrieval-only --config src/config/paris.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_paris_profile_180_for_mining_v1.json
.\.venv\Scripts\python -m src.tools.mine_hard_negative_triplets --metadata data/spacenet_paris_test/metadata.csv --reference-metadata data/spacenet_paris/metadata.csv --eval-report runs/geo_eval_paris_profile_180_for_mining_v1.json --output runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted.jsonl --summary-output runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted_summary.json --min-error-km 2.0 --positive-radius-km 0.35 --negative-pred-radius-km 2.0 --negative-min-gt-distance-km 2.0 --negative-max-gt-distance-km 25.0 --max-positives 3 --max-negatives 12 --difficulty-mode error_km_predmix --difficulty-reference-km 10.0 --difficulty-max-weight 3.0
```

This produces query-positive-hard-negative tuples from real failure cases and supports merged reports plus per-query failure caps.

### Fine-tune the retrieval encoder

```powershell
.\.venv\Scripts\python -m src.tools.train_retrieval_encoder --triplets runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted.jsonl --query-images-dir data/spacenet_paris_test/chips --reference-images-dir data/spacenet_paris/chips --model-id openai/clip-vit-large-patch14 --output-dir runs/retrieval_encoder_finetune/paris_round1_model --report-output runs/retrieval_encoder_finetune/paris_round1_model.report.json --train-scope vision_encoder --epochs 4 --batch-size 8 --learning-rate 1e-5 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.2 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device auto
```

### Train and apply retrieval projection

```powershell
.\.venv\Scripts\python -m src.tools.train_retrieval_projection --triplets runs/hard_negative_triplets_paris_test_query_train_ref_v2_weighted.jsonl --embedding-index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14.npz --images-dir data/spacenet_paris_test/chips --output runs/retrieval_projection_paris_query_trainref_v2_weighted_cmp.npz --report-output runs/retrieval_projection_paris_query_trainref_v2_weighted_cmp.report.json --epochs 8 --batch-size 16 --learning-rate 3e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --orth-weight 0.002 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --seed 42 --device cpu
.\.venv\Scripts\python -m src.tools.apply_projection_to_geo_index --index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14.npz --projection-path runs/retrieval_projection_paris_query_trainref_v2_weighted_cmp.npz --output data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_weighted_cmp.npz
```

Then point:

- `geolocator.retrieval_index_path` to the projected index
- `geolocator.retrieval_projection_path` to the same projection file

### Mine cross-view hard negatives

```powershell
.\.venv\Scripts\python.exe -m src.tools.mine_retrieval_hard_triplets --config src/config/paris.json --images-dir data/paris_realistic_v1/street_combined --metadata data/paris_realistic_v1_combined/splits_strict/train_pairs.csv --reference-metadata data/spacenet_paris/metadata.csv --limit 160 --output runs/retrieval_hard_triplets_train160.jsonl --summary-output runs/retrieval_hard_triplets_train160_summary.json
.\.venv\Scripts\python.exe -m src.tools.train_crossview_projection --triplets runs/retrieval_hard_triplets_train160.jsonl --aerial-index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz --street-images-dir data/paris_realistic_v1/street_combined --output runs/retrieval_hardneg_crossview_projection_v1.npz --max-triplets 160 --epochs 6 --batch-size 16 --learning-rate 3e-4 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.3 --sample-weight-mode triplet_weight --sample-weight-max 4 --device auto
```

### Iterative Paris retrieval fine-tune loop

```powershell
.\.venv\Scripts\python -m src.tools.run_retrieval_finetune_loop --train-images-dir data/spacenet_paris/chips --train-metadata data/spacenet_paris/metadata.csv --eval-images-dir data/spacenet_paris_test/chips --eval-metadata data/spacenet_paris_test/metadata.csv --bootstrap-config src/config/paris_close_range_dual_rrf.json --base-model-id openai/clip-vit-large-patch14 --rounds 1 --train-limit 600 --eval-limit 180 --eval-seed 42 --rank-objective within_2km_pct --use-dba --dba-neighbors 5 --dba-self-weight 1.0 --dba-eval-weight 0.5 --dba-geo-radius-km 2.0 --min-error-km 2.0 --positive-radius-km 0.35 --negative-pred-radius-km 2.0 --negative-min-gt-distance-km 2.0 --negative-max-gt-distance-km 25.0 --max-positives 3 --max-negatives 12 --max-failures-per-query 1 --difficulty-mode error_km_predmix --difficulty-reference-km 10.0 --difficulty-max-weight 3.0 --train-scope vision_encoder --epochs 4 --batch-size 8 --learning-rate 1e-5 --weight-decay 1e-4 --margin 0.08 --temperature 0.07 --ce-weight 0.2 --sample-weight-mode triplet_weight --sample-weight-max 3.0 --device auto --output-dir runs/retrieval_finetune_loop
```

## Index and Data Operations

### Merge multiple Paris retrieval indices

```powershell
.\.venv\Scripts\python -m src.tools.merge_geo_indices --inputs data/geo_index/spacenet_paris_clip.npz data/geo_index/spacenet_paris_test_clip.npz --output data/geo_index/merged_paris_clip.npz --dedupe-radius-m 75
```

### Geo-aware DBA index augmentation

```powershell
.\.venv\Scripts\python -m src.tools.augment_geo_index_embeddings --index data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild.npz --output data/geo_index/spacenet_paris_chips_openai_clip_vit_large_patch14_proj_trainref_v2_mild_dba_geo2_k5.npz --neighbors 5 --self-weight 1.0 --min-similarity 0.0 --temperature 0.07 --max-geo-distance-km 2.0
```

### Dual-index close-range stack

```powershell
.\.venv\Scripts\python -m src.tools.run_geo_eval --retrieval-only --config src/config/paris_close_range_dual_rrf.json --images-dir data/spacenet_paris_test/chips --metadata data/spacenet_paris_test/metadata.csv --limit 180 --seed 42 --diag-samples 180 --output runs/geo_eval_projection_trainref_v2_mild_dual_dba_w100_100_rrf_180.json
```

Related profiles:

- `src/config/paris_close_range_dual_rrf.json`
- `src/config/paris_balanced_dual_rrf.json`
- `src/config/paris_close_range_dual_rrf_graph_kde.json`

## Demo Asset Maintenance

### Regenerate README demo video and screenshot

```powershell
.\.venv\Scripts\python -m pip install playwright
.\.venv\Scripts\python -m playwright install chromium
.\.venv\Scripts\python -m src.tools.record_demo_video --with-analyze
```

Default analyzed sample:

- `data/analysis_tests/paris_street/images/mapillary__1021055432583866.jpg`
- profile: `paris_test`

Optional sample override:

```powershell
.\.venv\Scripts\python -m src.tools.record_demo_video --with-analyze --sample-image data/analysis_tests/paris_street/images/panoramax_osm-fr__29fc50e6-c5ce-4952-b75b-c3d509ea57be.jpg
```

Recommended demo interactions:

1. Launch the app with `.\.venv\Scripts\python -m src.tools.dev_app`.
1. Open the printed `/analysis/` URL.
1. Keep the recording clean with no error banners.
1. Upload an image and run `Analyze Image`.
1. Rotate and pan the 3D globe by dragging.
1. Zoom in and out with the mouse wheel.
1. Use the `Zoom In`, `Zoom Out`, `Paris`, and `Globe` controls.
1. Keep final files at `docs/images/analysis-demo.webm` and `docs/images/analysis-desktop.png`.

## Troubleshooting

If startup fails with `WinError 1392` under `.venv\Lib\site-packages\torch\...`, rebuild the venv:

```powershell
deactivate
rmdir /s /q .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then reinstall your matching `torch`, `torchvision`, and `torchaudio` build and rerun:

```powershell
.\.venv\Scripts\python -m src.tools.dev_app
```

For deeper runtime, profile, and UI notes, see:

- [../src/dashboard/README.md](../src/dashboard/README.md)
- [../src/docs/GEO_TECH.md](../src/docs/GEO_TECH.md)
- [DATA_LAYOUT.md](DATA_LAYOUT.md)
