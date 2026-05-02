#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[check] Python syntax"
python3 -m py_compile ./*.py

echo "[check] JSON configs"
python3 -m json.tool configs/paper_repro.json >/dev/null
python3 -m json.tool configs/paper_smoke.json >/dev/null
python3 -m json.tool eo_ir_benchmark_config.example.json >/dev/null

echo "[check] annotation schema"
python3 - <<'PY'
from pathlib import Path
import json
import eo_ir_benchmark as benchmark

annotation_paths = sorted(Path("annotations").glob("*.json"))
if len(annotation_paths) != 48:
    raise SystemExit(f"expected 48 annotation files, found {len(annotation_paths)}")
for path in annotation_paths:
    annotation = json.loads(path.read_text(encoding="utf-8"))
    benchmark.validate_annotation_schema(path, annotation)
    benchmark.validate_group_structure(path, annotation["qa"])
print({"annotation_count": len(annotation_paths)})
PY

echo "[check] config load"
python3 - <<'PY'
from pathlib import Path
import run_eo_ir_benchmark as runner

for config_path in ("configs/paper_repro.json", "configs/paper_smoke.json", "eo_ir_benchmark_config.example.json"):
    config = runner.load_config(Path(config_path).resolve())
    print({"config": config_path, "models": len(config["models"])})
PY

echo "[check] private-path and key scan"
OPENAI_KEY_PATTERN="sk""-proj"
GEMINI_KEY_PATTERN="AI""za"
PRIVATE_URL_PATTERN="git"".terrasense"".research"
PRIVATE_PATH_PATTERN="/media""/storage"
if rg -n "${OPENAI_KEY_PATTERN}|${GEMINI_KEY_PATTERN}|${PRIVATE_URL_PATTERN}|${PRIVATE_PATH_PATTERN}" \
  --glob '!eval_outputs/**' \
  --glob '!__pycache__/**' \
  --glob '!frame_summaries*/**' \
  --glob '!*/scripts/check_artifact.sh' \
  --glob '!scripts/check_artifact.sh' \
  .; then
  echo "Found a private path, private URL, or likely API key pattern." >&2
  exit 1
fi

if [[ -n "${VIDEO_DATA_ROOT:-}" ]]; then
  echo "[check] smoke request export with VIDEO_DATA_ROOT=$VIDEO_DATA_ROOT"
  python3 run_eo_ir_benchmark.py \
    --config configs/paper_smoke.json \
    --steps prepare_dataset export_requests export_model_requests
  python3 audit_model_requests.py \
    --model-requests-dir eval_outputs/paper_smoke/model_requests \
    --model-id gpt_5_4 \
    --output eval_outputs/paper_smoke/request_leakage_audit.json
else
  echo "[check] VIDEO_DATA_ROOT is unset; skipping video-path-dependent request export."
fi

echo "[check] ok"
