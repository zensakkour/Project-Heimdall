#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./scripts/run_pipeline.sh /path/to/image.jpg [--weights /path/to/weights] [--geo-model /path/to/model]"
  exit 1
fi

python3 -m src.cli "$@"
