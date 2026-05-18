# Project Heimdall

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#requirements)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-0A0A0A)](#requirements)
[![License](https://img.shields.io/badge/License-Non--Commercial-orange.svg)](LICENSE)

Project Heimdall is a geospatial perception and analysis platform for image-based localization. It combines object detection, geo-candidate retrieval, probabilistic fusion, and an operator-facing analysis UI for investigation, benchmarking, and tuning workflows.

## What It Does

- Runs image and video analysis workflows.
- Produces ranked geolocation candidates.
- Computes fused location estimates with uncertainty outputs.
- Exposes a local operator UI for review, map interaction, and diagnostics.
- Supports benchmarking, evaluation, and retrieval-tuning workflows.

## Scope

Current scope:

- Detection and oriented bounding-box localization
- Multi-provider geolocation candidate generation
- Probabilistic fusion and uncertainty estimation
- Explainable structured outputs for analysis and audits
- Local dashboard and API workflows for evaluation

Current non-goals:

- Production deployment and operational decision automation
- Scraping pipelines and broad ingestion connectors
- Text or LLM reasoning as a dependency in core inference

## Demo

### Analysis App Video

[![Watch demo video](docs/images/analysis-desktop.png)](docs/images/analysis-demo.webm)

Direct link: [docs/images/analysis-demo.webm](docs/images/analysis-demo.webm)

To regenerate demo assets, see:

- [docs/WORKFLOWS.md](docs/WORKFLOWS.md)
- [src/dashboard/README.md](src/dashboard/README.md)

## Quick Start

### Requirements

- Python 3.10+
- Git
- Optional GPU acceleration: install the PyTorch build that matches your CUDA target

### Install

Create a venv and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For reproducible installs:

```powershell
pip install -r requirements.lock.txt
```

Environment verification:

```powershell
.\.venv\Scripts\python -m src.tools.doctor
```

### Start the app

PowerShell:

```powershell
.\.venv\Scripts\python -m src.tools.dev_app
```

Windows CMD launcher:

```cmd
run_heimdall.cmd
```

Open the printed `/analysis/` URL in your browser. The lab UI is available at `/analysis/lab/`.

## Common Usage

### Run the operator UI

```powershell
.\.venv\Scripts\python -m src.tools.dev_app
```

### Run single-image inference

```powershell
.\.venv\Scripts\python -m src.cli data/analysis_tests/paris_street/images/mapillary__1021055432583866.jpg --json
```

### Run tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

More command recipes, benchmark flows, tuning commands, and demo-maintenance steps are in:

- [docs/WORKFLOWS.md](docs/WORKFLOWS.md)

## App Usage

Basic operator flow:

1. Start the server.
1. Open the `/analysis/` URL printed in terminal.
1. Select a profile.
1. Upload an image and click `Analyze Image`.
1. Review detections, geo candidates, fused estimate, and diagnostics.
1. Use the 3D globe controls (`Zoom In`, `Zoom Out`, `Paris`, `Globe`) to inspect the scene.

For UI structure, endpoints, and demo asset notes, see [src/dashboard/README.md](src/dashboard/README.md).

## Documentation

Start here:

- [docs/DOCS_MAP.md](docs/DOCS_MAP.md): documentation index

Core docs:

- [docs/WORKFLOWS.md](docs/WORKFLOWS.md): command recipes, benchmark flow, training/tuning entry points
- [src/docs/GEO_TECH.md](src/docs/GEO_TECH.md): geolocation architecture, retrieval, fusion, profiles, runtime knobs
- [src/docs/REPRODUCIBILITY.md](src/docs/REPRODUCIBILITY.md): reproducibility and evaluation procedure
- [src/dashboard/README.md](src/dashboard/README.md): analysis UI structure and backend endpoints
- [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md): datasets, model artifacts, and local data layout

Project history and research:

- [docs/engineering/PROGRESS.md](docs/engineering/PROGRESS.md): append-only engineering log
- [docs/research/research.md](docs/research/research.md): chronological evidence ledger with before/after metrics
- [src/docs/RESEARCH_PAPER.md](src/docs/RESEARCH_PAPER.md): full research-style write-up
- [src/docs/MARKET_RESEARCH.md](src/docs/MARKET_RESEARCH.md): market and SOTA review

## Repository Layout

- `src/core/`: detection, geo, fusion, scoring, serialization logic
- `src/tools/`: app server, evaluation, indexing, calibration, training, utility scripts
- `src/dashboard/`: operator UI
- `src/config/`: runtime profiles
- `src/tests/`: test suite
- `docs/`: docs, governance, and benchmark outputs
- `data/`: local datasets and model artifacts
- `runs/`: generated evaluation and experiment outputs

## Troubleshooting

If the app or model stack is unhealthy, run:

```powershell
.\.venv\Scripts\python -m src.tools.doctor
```

If file-watcher reload causes issues, launch without reload:

```powershell
.\.venv\Scripts\python -m src.tools.dev_app --no-reload
```

If your venv is corrupted, rebuild it and reinstall dependencies. Additional runtime and UI notes live in:

- [src/dashboard/README.md](src/dashboard/README.md)
- [src/docs/GEO_TECH.md](src/docs/GEO_TECH.md)

## Contributing and Support

- [docs/governance/CONTRIBUTING.md](docs/governance/CONTRIBUTING.md)
- [docs/governance/SECURITY.md](docs/governance/SECURITY.md)
- [docs/governance/SUPPORT.md](docs/governance/SUPPORT.md)
- [docs/governance/CODE_OF_CONDUCT.md](docs/governance/CODE_OF_CONDUCT.md)

## License

See [LICENSE](LICENSE).
