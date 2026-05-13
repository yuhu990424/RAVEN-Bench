#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 runners/experimental/run_qwen35_9b_text_only.py \
  --questions eval_outputs/paper_repro/questions.jsonl \
  --predictions-dir eval_outputs/paper_repro/predictions_text_only \
  "$@"
