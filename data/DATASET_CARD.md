# Dataset Card

## Dataset Summary

- Dataset name: RAVEN-Bench Anonymous
- Task: paired EO-IR aerial video question answering
- Data units: EO/IR video pairs plus multiple-choice QA items
- Modalities: EO, IR, EO+IR
- Evaluation settings: EO-only, IR-only, EO+IR
- Splits: dev/test release; no training split is provided in this anonymous artifact
- Labels: model-facing request exports hide gold answers; annotation JSON files in
  this reviewer artifact include gold answers and internal label metadata for
  reproducibility and scoring

RAVEN-Bench Anonymous is an evaluation artifact for multimodal video question
answering over paired electro-optical (EO) and infrared (IR) aerial videos. The
benchmark is designed to test whether models use visible-light and thermal
evidence appropriately across single-modality and paired-modality settings.

## Dataset Structure

Each annotated sample corresponds to a paired EO/IR video instance and a set of
multiple-choice QA items. The artifact includes annotation JSON files, relative
metadata for expected raw video filenames, request-export scripts, scoring code,
and Croissant machine-readable metadata.

Raw videos are not included in this repository snapshot. They must be provided
separately using the documented relative layout.

Reviewer-facing anonymous dataset preview URL:

```text
https://dataverse.harvard.edu/previewurl.xhtml?token=73dc5b15-a0ce-4c20-be84-f8fde196f69b
```

For OpenReview submission, this dataset URL should remain reviewer-accessible
through the full review period. If the Dataverse preview URL changes, update the
Croissant `url` field and regenerate the metadata files before submission.

## Evaluation

The official evaluation settings are:

- EO-only: models receive only the electro-optical video.
- IR-only: models receive only the infrared video.
- EO+IR: models receive both videos in a fixed EO then IR order.

The scoring scripts report question-level accuracy, capability-level scores,
group consistency/coherence metrics, and modality-ablation summaries.

## Intended Use

This dataset is intended for research evaluation of:

- video question answering
- cross-spectral reasoning
- EO/IR evidence use
- temporal reasoning over aerial videos
- robustness to same-scene distractors

## Prohibited Use

This dataset is not intended for:

- surveillance deployment
- identifying real persons or locations
- tracking private individuals
- safety-critical operational decisions
- training or deploying operational monitoring systems without independent
  safety, legal, and ethical review

## Known Limitations

- Small sample size.
- Aerial-domain specificity.
- Limited sensor, geography, weather, and capture-condition coverage.
- No pixel-level registration supervision.
- No claim of broad representativeness across all EO/IR sensing settings.
- Multiple-choice answer formats may introduce wording or distractor artifacts;
  text-only and request-leakage checks are recommended.

## Licensing and Access

Code, documentation, Croissant metadata, and released benchmark annotation files
are distributed under the Apache License 2.0. Raw videos are not included in the
anonymous repository artifact and may be subject to separate reviewer-only access
terms through the Dataverse preview dataset.
