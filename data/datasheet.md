# Datasheet

## Motivation

RAVEN-Bench Anonymous was created to evaluate video-language models on paired
electro-optical and infrared aerial video question answering. The benchmark
focuses on cross-spectral evidence use, temporal reasoning, target grounding,
and distractor rejection under EO-only, IR-only, and EO+IR settings.

## Composition

The artifact contains benchmark annotations, relative video metadata, request
export code, scoring code, model-runner entrypoints, documentation, and
Croissant metadata. Each data unit consists of an EO/IR video pair and
multiple-choice QA items.

Raw EO/IR videos, model weights, API keys, generated predictions, and large
experiment outputs are not included in this repository snapshot.

## Collection Process

The raw data consist of paired EO and IR videos. The released metadata uses
relative filenames and resolves raw videos under `VIDEO_DATA_ROOT`. The
annotations were manually curated as multiple-choice QA items with modality
requirements, capability levels, group metadata, answer labels, and internal
evidence notes.

## Preprocessing and Evaluation Protocol

The evaluation protocol uses full videos, uniformly sampled at 1 fps, with
aspect-ratio-preserving frame resizing under the documented pixel budget. Audio
is disabled. Request export hides gold answers, option roles, evidence notes,
rationales, event descriptions, and annotation time-reference metadata from
model prompts.

## Recommended Uses

The dataset is intended for research evaluation of video QA, cross-spectral
reasoning, EO/IR evidence use, and modality ablation studies.

## Non-Recommended Uses

The dataset should not be used for surveillance deployment, identifying real
persons or locations, tracking private individuals, safety-critical operational
decisions, or claims of broad EO/IR domain representativeness.

## Distribution

The anonymous repository includes code, annotations, metadata, documentation,
and Croissant files. Raw videos must be provided separately for review or final
release according to the applicable access policy.

Reviewer-facing anonymous Dataverse preview URL:

```text
https://dataverse.harvard.edu/previewurl.xhtml?token=73dc5b15-a0ce-4c20-be84-f8fde196f69b
```

This preview URL should remain active through double-blind review. If the hosted
dataset URL changes, update the Croissant `url` field and regenerate the
metadata files before submission.

## Maintenance

This is a static anonymous submission artifact. Any annotation, metadata, or
data-access changes should update the dataset version and regenerate
`metadata/metadata.json`, `metadata/croissant.json`, and
`metadata/mlcroissant.json`.
