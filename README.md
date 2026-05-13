# EO/IR Video Understanding Benchmark

This repository contains the code and annotations used to evaluate multimodal
video understanding over paired electro-optical (EO) and infrared (IR) videos.
It is organized as an anonymous paper artifact: the raw videos, model weights,
prediction files, and large experiment outputs are not included.

## What Is Included

- `data/annotations/`: 48 benchmark annotation files.
- `data/metadata/`: video-pair metadata with relative filenames.
- `configs/`: reproducibility and smoke-test configurations.
- `raven_benchmark/`: dataset preparation, request export, baselines, scoring,
  and model request registry.
- `runners/`: model-specific API, local, and experimental runner entrypoints.
- `scripts/`: lightweight artifact checks and reproducibility helpers.
- `metadata/croissant.json`, `metadata/metadata.json`,
  `metadata/mlcroissant.json`: Croissant and
  Responsible AI metadata for the benchmark artifact.
- `data/DATASET_CARD.md`, `data/datasheet.md`: dataset-card and datasheet
  summaries for reviewer-facing documentation.
- `docs/`: reproducibility, model-runner, and video-input protocol notes.
- `LICENSE`, `LICENSE.md`, `data/DATA_LICENSE.md`: repository and data-access
  licensing notes.

## What Is Not Included

- Raw `.ts` videos.
- Model checkpoints and downloaded Hugging Face cache files.
- API keys.
- Full predictions, results, generated model media, and `eval_outputs/`
  artifacts.

Small frame-summary preview sheets may be present for visual inspection. They
are not required for scoring and are not a substitute for the raw video access
described below.

## Data Layout

The reviewer-facing anonymous Dataverse preview URL is:

```text
https://dataverse.harvard.edu/previewurl.xhtml?token=73dc5b15-a0ce-4c20-be84-f8fde196f69b
```

Set `VIDEO_DATA_ROOT` to the directory containing the raw videos referenced by
`data/metadata/video_metadata.example.csv`.

```bash
export VIDEO_DATA_ROOT=/path/to/raw/maritime/videos
```

Each metadata row stores relative filenames such as `5th.Wheel_EO.ts`. During
dataset preparation, relative paths are resolved against `VIDEO_DATA_ROOT`.
The hosted Dataverse files should preserve this relative layout, or reviewers
should receive a private preview/download URL that maps to the same filenames.

## License

Code, documentation, Croissant metadata, and released benchmark annotation files
are distributed under the Apache License 2.0. Raw EO/IR videos are not included
in this anonymous artifact and may be provided separately for review under
separate access terms through the Dataverse preview URL above. See `LICENSE` and
`data/DATA_LICENSE.md`.

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
export, Croissant metadata syntax, required dependency files, and common leakage
patterns. It does not download model weights or call external APIs.

## Additional Documentation

- Reproducibility: `docs/reproducibility.md`
- Model runners: `docs/model_runners.md`
- Video input protocol: `docs/video_input_protocol.md`
- Dataset card: `data/DATASET_CARD.md`
