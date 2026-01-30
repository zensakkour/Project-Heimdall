#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./scripts/run_batch.sh /path/to/images [--weights /path/to/weights] [--geo-model /path/to/model] [--output outputs.jsonl]"
  exit 1
fi

python3 batch_run.py "$@"
