#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 run_eo_ir_benchmark.py \
  --config configs/paper_repro.json \
  --steps export_requests export_model_requests
