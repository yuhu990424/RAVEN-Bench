# Data Notes

The raw EO/IR videos are not included in this anonymous paper artifact.

## Expected Layout

Set:

```bash
export VIDEO_DATA_ROOT=/path/to/raw/maritime/videos
```

The directory should contain files named like:

```text
5th.Wheel_EO.ts
5th.Wheel_IR.ts
Airplane_EO.ts
Airplane_IR.ts
```

The full expected file list is given by `data/metadata/video_metadata.example.csv`.

## Metadata

The paper artifact includes anonymized metadata:

- `data/metadata/video_metadata.example.csv`
- `data/metadata/video_metadata.example.json`

Both use relative video filenames, not machine-specific absolute paths.

## Generated Artifacts

Dataset preparation writes generated files under `eval_outputs/`, including:

- `manifest.jsonl`
- `questions.jsonl`
- `model_safe_questions.jsonl`
- `requests/`
- `model_requests/`

These generated files are not committed.
