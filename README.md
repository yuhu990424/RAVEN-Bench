# EO/IR Video Understanding Benchmark

This repository contains the code and annotations used to evaluate multimodal
video understanding over paired electro-optical (EO) and infrared (IR) videos.
It is organized as an anonymous paper artifact: the raw videos, model weights,
prediction files, and large experiment outputs are not included.

## What Is Included

- `annotations/`: 48 benchmark annotation files.
- `metadata/`: example video metadata with relative filenames.
- `configs/`: reproducibility and smoke-test configurations.
- `eo_ir_benchmark.py`: dataset preparation, request export, baselines, and scoring.
- `model_registry.py`: model request registry for local and API models.
- `run_*.py`: model-specific runner entrypoints.
- `scripts/`: lightweight artifact checks and reproducibility helpers.

## What Is Not Included

- Raw `.ts` videos.
- Model checkpoints and downloaded Hugging Face cache files.
- API keys.
- Full predictions, results, frame sheets, and `eval_outputs/` artifacts.

## Data Layout

Set `VIDEO_DATA_ROOT` to the directory containing the raw videos referenced by
`metadata/video_metadata.example.csv`.

```bash
export VIDEO_DATA_ROOT=/path/to/raw/maritime/videos
```

Each metadata row stores relative filenames such as `5th.Wheel_EO.ts`. During
dataset preparation, relative paths are resolved against `VIDEO_DATA_ROOT`.

## Basic Workflow

```bash
python3 run_eo_ir_benchmark.py \
  --config configs/paper_smoke.json \
  --steps prepare_dataset export_requests export_model_requests
```

This writes artifacts under `eval_outputs/paper_smoke/`. The directory is
ignored by git.

To score predictions produced by a runner:

```bash
python3 eo_ir_benchmark.py score-predictions \
  --questions eval_outputs/paper_smoke/questions.jsonl \
  --predictions-dir eval_outputs/paper_smoke/predictions \
  --results-dir eval_outputs/paper_smoke/results \
  --allow-partial
```

## Artifact Check

Run the lightweight repository check before sharing the artifact:

```bash
bash scripts/check_artifact.sh
```

The check validates Python syntax, JSON configs, annotation schema, request
export, and common leakage patterns. It does not download model weights or call
external APIs.
