#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[check] Python syntax"
python3 -m py_compile ./*.py raven_benchmark/*.py runners/api/*.py runners/local/*.py runners/experimental/*.py tools/*.py tools/maintenance/*.py scripts/smoke_test_first5.py

echo "[check] JSON configs"
python3 -m json.tool configs/paper_repro.json >/dev/null
python3 -m json.tool configs/paper_smoke.json >/dev/null
python3 -m json.tool configs/example.json >/dev/null
python3 -m json.tool metadata/croissant.json >/dev/null
python3 -m json.tool metadata/metadata.json >/dev/null
python3 -m json.tool metadata/mlcroissant.json >/dev/null

echo "[check] required artifact files"
for required_file in \
  LICENSE \
  LICENSE.md \
  data/DATA_LICENSE.md \
  data/DATASET_CARD.md \
  data/datasheet.md \
  CITATION.cff \
  README.md \
  docs/reproducibility.md \
  docs/model_runners.md \
  docs/video_input_protocol.md \
  requirements.txt \
  requirements-api.txt \
  requirements-local.txt \
  metadata/croissant.json \
  metadata/metadata.json \
  metadata/mlcroissant.json
do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required artifact file: $required_file" >&2
    exit 1
  fi
done

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  for tracked_file in requirements.txt requirements-api.txt requirements-local.txt; do
    if ! git ls-files --error-unmatch "$tracked_file" >/dev/null 2>&1; then
      echo "Required dependency file is present but not tracked by git: $tracked_file" >&2
      exit 1
    fi
  done
fi

echo "[check] Croissant NeurIPS metadata"
python3 - <<'PY'
from pathlib import Path
import json

payload = json.loads(Path("metadata/croissant.json").read_text(encoding="utf-8"))

required_fields = [
    "@context",
    "@type",
    "name",
    "url",
    "license",
    "conformsTo",
    "distribution",
    "recordSet",
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
    "rai:dataUseCases",
    "rai:dataSocialImpact",
    "rai:hasSyntheticData",
    "prov:wasDerivedFrom",
    "prov:wasGeneratedBy",
]


def missing_value(value):
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (dict, list)) and not value:
        return True
    return False


missing = [field for field in required_fields if field not in payload or missing_value(payload[field])]
if missing:
    raise SystemExit(f"Croissant metadata is missing required fields: {missing}")

if payload["@type"] != "sc:Dataset":
    raise SystemExit(f"Croissant @type must be sc:Dataset, found {payload['@type']!r}")

url = payload["url"]
if "example.org/to-be-replaced" in url:
    raise SystemExit("Croissant url still uses the placeholder example.org/to-be-replaced URL")
if not url.startswith("https://dataverse.harvard.edu/"):
    raise SystemExit(f"Croissant url must point to the hosted Harvard Dataverse location, found {url!r}")

conforms_to = set(payload["conformsTo"])
required_conforms_to = {
    "http://mlcommons.org/croissant/1.0",
    "http://mlcommons.org/croissant/RAI/1.0",
}
missing_conforms_to = required_conforms_to - conforms_to
if missing_conforms_to:
    raise SystemExit(f"Croissant conformsTo is missing: {sorted(missing_conforms_to)}")

if payload["license"] != "https://www.apache.org/licenses/LICENSE-2.0":
    raise SystemExit(f"Unexpected Croissant license: {payload['license']!r}")

print({"croissant_url": url, "required_fields": len(required_fields)})
PY

echo "[check] annotation schema"
python3 - <<'PY'
from pathlib import Path
import json
from raven_benchmark import benchmark

annotation_paths = sorted(Path("data/annotations").glob("*.json"))
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

for config_path in ("configs/paper_repro.json", "configs/paper_smoke.json", "configs/example.json"):
    config = runner.load_config(Path(config_path).resolve())
    print({"config": config_path, "models": len(config["models"])})
PY

echo "[check] private-path and key scan"
OPENAI_KEY_PATTERN="sk""-[A-Za-z0-9_-]{20,}"
OPENAI_PROJECT_KEY_PATTERN="sk""-proj-[A-Za-z0-9_-]{20,}"
GEMINI_KEY_PATTERN="AI""za[0-9A-Za-z_-]"
PRIVATE_URL_PATTERN="git"".terrasense"".research"
PRIVATE_PATH_PATTERN="/media""/storage"
PRIVATE_HOME_PATTERN="/home""/"
WANDB_PATTERN="wand""b"
PLACEHOLDER_URL_PATTERN="example""\\.org/to-be-replaced"
if rg -n "${OPENAI_KEY_PATTERN}|${OPENAI_PROJECT_KEY_PATTERN}|${GEMINI_KEY_PATTERN}|${PRIVATE_URL_PATTERN}|${PRIVATE_PATH_PATTERN}|${PRIVATE_HOME_PATTERN}|${WANDB_PATTERN}|${PLACEHOLDER_URL_PATTERN}" \
  --glob '!eval_outputs/**' \
  --glob '!__pycache__/**' \
  --glob '!frame_summaries*/**' \
  --glob '!*.pyc' \
  --glob '!*/scripts/check_artifact.sh' \
  --glob '!scripts/check_artifact.sh' \
  .; then
  echo "Found a private path, private URL, placeholder URL, wandb artifact reference, or likely API key pattern." >&2
  exit 1
fi

if [[ -n "${VIDEO_DATA_ROOT:-}" ]]; then
  echo "[check] smoke request export with VIDEO_DATA_ROOT=$VIDEO_DATA_ROOT"
  python3 run_eo_ir_benchmark.py \
    --config configs/paper_smoke.json \
    --steps prepare_dataset export_requests export_model_requests
  python3 tools/audit_model_requests.py \
    --model-requests-dir eval_outputs/paper_smoke/model_requests \
    --model-id gpt_5_4 \
    --output eval_outputs/paper_smoke/request_leakage_audit.json
else
  echo "[check] VIDEO_DATA_ROOT is unset; skipping video-path-dependent request export."
fi

echo "[check] ok"
