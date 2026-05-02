# Reproducibility

This benchmark is script-based. It does not require installing a Python package.
Run commands from the repository root or from `video_understanding_benchmark/`.

## Environment

Recommended base environment:

```bash
python3 -m pip install -r requirements.txt
```

For API models:

```bash
python3 -m pip install -r requirements-api.txt
```

For local open-weight models:

```bash
python3 -m pip install -r requirements-local.txt
```

The exact local-model stack may need model-specific package versions, especially
for remote-code Hugging Face checkpoints.

## Data Root

Raw videos are not distributed with this artifact.

```bash
export VIDEO_DATA_ROOT=/path/to/raw/maritime/videos
```

The metadata file uses relative video filenames and resolves them under
`VIDEO_DATA_ROOT`.

## Prepare Benchmark Requests

Smoke configuration:

```bash
python3 run_eo_ir_benchmark.py \
  --config configs/paper_smoke.json \
  --steps prepare_dataset export_requests export_model_requests
```

Full paper configuration:

```bash
python3 run_eo_ir_benchmark.py \
  --config configs/paper_repro.json \
  --steps prepare_dataset export_requests export_model_requests
```

The full configuration disables all model targets by default. Enable the desired
model in `configs/paper_repro.json` before exporting model-specific requests.

## Run Models

Examples:

```bash
python3 run_gpt54_api.py \
  --model-requests-dir eval_outputs/paper_smoke/model_requests \
  --predictions-dir eval_outputs/paper_smoke/predictions \
  --settings eo_only \
  --limit 1 \
  --dry-run
```

```bash
python3 run_hf_vlm_frames_local.py \
  --model-requests-dir eval_outputs/paper_smoke/model_requests \
  --predictions-dir eval_outputs/paper_smoke/predictions \
  --model-id smolvlm2_2_2b_instruct \
  --model-name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --settings eo_only \
  --limit 1 \
  --dry-run
```

## Score Predictions

```bash
python3 eo_ir_benchmark.py score-predictions \
  --questions eval_outputs/paper_smoke/questions.jsonl \
  --predictions-dir eval_outputs/paper_smoke/predictions \
  --results-dir eval_outputs/paper_smoke/results \
  --allow-partial
```

Prediction and result directories are intentionally ignored by git.
