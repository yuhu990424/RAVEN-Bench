#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${VIDEO_DATA_ROOT:-}" ]]; then
  echo "Set VIDEO_DATA_ROOT to the directory containing the raw .ts videos." >&2
  exit 1
fi

python3 run_eo_ir_benchmark.py \
  --config configs/paper_smoke.json \
  --steps prepare_dataset export_requests export_model_requests

python3 runners/api/run_gpt54_api.py \
  --model-requests-dir eval_outputs/paper_smoke/model_requests \
  --predictions-dir eval_outputs/paper_smoke/predictions \
  --settings eo_only \
  --limit 1 \
  --dry-run
