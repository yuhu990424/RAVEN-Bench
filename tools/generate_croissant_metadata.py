#!/usr/bin/env python3
"""Generate Croissant + Responsible AI metadata for the EO/IR benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_NAMES = ("metadata/metadata.json", "metadata/croissant.json", "metadata/mlcroissant.json")

DATASET_NAME = "EO/IR Video Understanding Benchmark"
DATASET_URL = "https://dataverse.harvard.edu/previewurl.xhtml?token=73dc5b15-a0ce-4c20-be84-f8fde196f69b"
DATASET_LICENSE = "https://www.apache.org/licenses/LICENSE-2.0"
DATASET_VERSION = "1.0.0-submission-draft"
DATASET_DATE = "2026-05-03"
DATASET_CREATOR = "Anonymous authors"

CORE_CONFORMS_TO = "http://mlcommons.org/croissant/1.0"
RAI_CONFORMS_TO = "http://mlcommons.org/croissant/RAI/1.0"


CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "sc": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "prov": "http://www.w3.org/ns/prov#",
    "dct": "http://purl.org/dc/terms/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileObject": "cr:fileObject",
    "fileProperty": "cr:fileProperty",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


CSV_FIELD_TYPES = {
    "pair_key": "sc:Text",
    "pair_instance_index": "sc:Integer",
    "theme_guess": "sc:Text",
    "pairing_method": "sc:Text",
    "eo_filename": "sc:Text",
    "ir_filename": "sc:Text",
    "eo_duration_seconds": "sc:Float",
    "ir_duration_seconds": "sc:Float",
    "duration_delta_seconds": "sc:Float",
    "eo_fps": "sc:Float",
    "ir_fps": "sc:Float",
    "eo_frame_count": "sc:Integer",
    "ir_frame_count": "sc:Integer",
    "eo_resolution": "sc:Text",
    "ir_resolution": "sc:Text",
    "eo_path": "sc:Text",
    "ir_path": "sc:Text",
    "eo_file_size_bytes": "sc:Integer",
    "ir_file_size_bytes": "sc:Integer",
    "eo_bitrate_mbps": "sc:Float",
    "ir_bitrate_mbps": "sc:Float",
    "eo_duration_source": "sc:Text",
    "ir_duration_source": "sc:Text",
    "eo_metadata_error": "sc:Text",
    "ir_metadata_error": "sc:Text",
}


ANNOTATION_SUMMARY_FIELDS = {
    "sample_id": "sc:Text",
    "type": "sc:Text",
    "main_event": "sc:Text",
    "event_description": "sc:Text",
    "scenario_type": "sc:Text",
}


QUESTION_FIELDS = {
    "uid": "sc:Text",
    "group_id": "sc:Text",
    "group_family": "sc:Text",
    "group_focus": "sc:Text",
    "group_step": "sc:Integer",
    "capability_level": "sc:Text",
    "question": "sc:Text",
    "options": "sc:Text",
    "answer": ["sc:Text", "cr:Label"],
    "option_roles": "sc:Text",
    "question_type": "sc:Text",
    "modality_requirement": "sc:Text",
    "time_reference_eo": "sc:Text",
    "time_reference_ir": "sc:Text",
    "evidence_note": "sc:Text",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_object(path: Path, object_id: str, encoding_format: str, description: str) -> dict:
    relative_path = path.relative_to(ROOT).as_posix()
    return {
        "@type": "cr:FileObject",
        "@id": object_id,
        "name": relative_path,
        "description": description,
        "contentUrl": relative_path,
        "contentSize": f"{path.stat().st_size} B",
        "encodingFormat": encoding_format,
        "sha256": sha256(path),
    }


def csv_field(column_name: str) -> dict:
    return {
        "@type": "cr:Field",
        "@id": f"video_pairs/{column_name}",
        "dataType": CSV_FIELD_TYPES[column_name],
        "source": {
            "fileObject": {"@id": "video-metadata-csv"},
            "extract": {"column": column_name},
        },
    }


def annotation_summary_field(field_name: str) -> dict:
    return {
        "@type": "cr:Field",
        "@id": f"annotation_samples/{field_name}",
        "dataType": ANNOTATION_SUMMARY_FIELDS[field_name],
        "source": {
            "fileSet": {"@id": "annotation-json-files"},
            "extract": {"jsonPath": f"$.{field_name}"},
        },
    }


def question_field(field_name: str) -> dict:
    field = {
        "@type": "cr:Field",
        "@id": f"annotation_samples/questions/{field_name}",
        "dataType": QUESTION_FIELDS[field_name],
        "source": {
            "fileSet": {"@id": "annotation-json-files"},
            "extract": {"jsonPath": f"$.qa[*].{field_name}"},
        },
    }
    if field_name in {"question_type"}:
        field["repeated"] = True
    return field


def load_metadata_rows() -> list[dict[str, str]]:
    with (ROOT / "data/metadata/video_metadata.example.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_annotation_rows() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "data/annotations").glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            rows.append(json.load(handle))
    return rows


def build_distribution() -> list[dict]:
    distribution = [
        {
            "@type": "cr:FileSet",
            "@id": "raw-eo-ir-video-files",
            "name": "Raw paired EO/IR video files",
            "description": (
                "Paired electro-optical and infrared MPEG transport stream videos. "
                "The anonymous repository stores only relative filenames; the hosted "
                "dataset must provide these raw files under the documented layout."
            ),
            "encodingFormat": "video/MP2T",
            "includes": ["*_EO.ts", "*_IR.ts"],
        },
        {
            "@type": "cr:FileSet",
            "@id": "annotation-json-files",
            "name": "Annotation JSON files",
            "description": "One JSON annotation file per annotated EO/IR sample.",
            "encodingFormat": "application/json",
            "includes": "data/annotations/*.json",
        },
    ]

    distribution.append(
        file_object(
            ROOT / "data/metadata/video_metadata.example.csv",
            "video-metadata-csv",
            "text/csv",
            "Anonymized video-pair metadata with relative EO/IR filenames.",
        )
    )
    distribution.append(
        file_object(
            ROOT / "data/metadata/video_metadata.example.json",
            "video-metadata-json",
            "application/json",
            "JSON serialization of the anonymized video-pair metadata.",
        )
    )

    for path in sorted((ROOT / "data/annotations").glob("*.json")):
        object_id = "annotation-" + path.stem.lower().replace(" ", "-").replace("_", "-").replace(".", "-")
        distribution.append(
            file_object(
                path,
                object_id,
                "application/json",
                "Benchmark annotation file containing sample metadata and multiple-choice QA labels.",
            )
        )

    return distribution


def build_video_pairs_record_set(csv_columns: list[str], rows: list[dict[str, str]]) -> dict:
    return {
        "@type": "cr:RecordSet",
        "@id": "video_pairs",
        "name": "Video pair metadata",
        "description": (
            "One record per EO/IR video pair in the anonymized metadata. The paper "
            "statistics use the 48 annotated pairs and exclude the duplicated "
            "Parked.car pair from the 49 metadata rows."
        ),
        "key": [
            {"@id": "video_pairs/pair_key"},
            {"@id": "video_pairs/pair_instance_index"},
        ],
        "field": [csv_field(column_name) for column_name in csv_columns],
        "examples": [
            {
                f"video_pairs/{key}": value
                for key, value in rows[0].items()
                if key in csv_columns
            }
        ],
    }


def build_annotation_record_set(annotation_rows: list[dict]) -> dict:
    question_subfields = [question_field(field_name) for field_name in QUESTION_FIELDS]
    return {
        "@type": "cr:RecordSet",
        "@id": "annotation_samples",
        "name": "Annotation samples",
        "description": (
            "One record per annotated EO/IR sample. Each annotation file contains "
            "scenario metadata and 12 multiple-choice questions with gold labels, "
            "grouping metadata, modality requirements, and internal evidence notes."
        ),
        "key": {"@id": "annotation_samples/sample_id"},
        "field": [
            {
                "@type": "cr:Field",
                "@id": "annotation_samples/annotation_file",
                "dataType": "sc:Text",
                "source": {
                    "fileSet": {"@id": "annotation-json-files"},
                    "extract": {"fileProperty": "filename"},
                },
            },
            *[annotation_summary_field(field_name) for field_name in ANNOTATION_SUMMARY_FIELDS],
            {
                "@type": "cr:Field",
                "@id": "annotation_samples/questions",
                "description": "Nested multiple-choice QA labels for the sample.",
                "dataType": ["cr:RecordSet", "cr:Label"],
                "repeated": True,
                "subField": question_subfields,
            },
        ],
        "examples": [
            {
                "annotation_samples/sample_id": annotation_rows[0]["sample_id"],
                "annotation_samples/type": annotation_rows[0]["type"],
                "annotation_samples/scenario_type": annotation_rows[0]["scenario_type"],
                "annotation_samples/questions/count": len(annotation_rows[0]["qa"]),
            }
        ],
    }


def build_metadata() -> dict:
    metadata_rows = load_metadata_rows()
    annotation_rows = load_annotation_rows()
    csv_columns = list(metadata_rows[0].keys())

    return {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": [CORE_CONFORMS_TO, RAI_CONFORMS_TO],
        "name": DATASET_NAME,
        "description": (
            "A paired electro-optical and infrared video understanding benchmark "
            "for evaluating multimodal models on cross-modal evidence use, temporal "
            "reasoning, target grounding, and distractor rejection. The submission "
            "snapshot contains 48 annotated EO/IR samples, 576 multiple-choice "
            "questions, anonymized video metadata, and scripts for request export "
            "and scoring."
        ),
        "url": DATASET_URL,
        "license": DATASET_LICENSE,
        "creator": [{"@type": "sc:Organization", "name": DATASET_CREATOR}],
        "publisher": [{"@type": "sc:Organization", "name": DATASET_CREATOR}],
        "datePublished": DATASET_DATE,
        "dateCreated": DATASET_DATE,
        "dateModified": DATASET_DATE,
        "version": DATASET_VERSION,
        "keywords": [
            "video understanding",
            "electro-optical video",
            "infrared video",
            "EO/IR",
            "multimodal evaluation",
            "vision-language models",
            "benchmark",
        ],
        "inLanguage": "en",
        "isLiveDataset": False,
        "citeAs": (
            "Anonymous authors. EO/IR Video Understanding Benchmark. "
            "NeurIPS 2026 Evaluations and Datasets submission artifact, 2026."
        ),
        "variableMeasured": [
            {"@type": "sc:PropertyValue", "name": "annotated_samples", "value": 48},
            {"@type": "sc:PropertyValue", "name": "metadata_rows", "value": len(metadata_rows)},
            {"@type": "sc:PropertyValue", "name": "total_questions", "value": 576},
            {"@type": "sc:PropertyValue", "name": "questions_per_sample", "value": 12},
            {"@type": "sc:PropertyValue", "name": "eo_total_duration_seconds", "value": 6498.802},
            {"@type": "sc:PropertyValue", "name": "ir_total_duration_seconds", "value": 6266.08},
            {"@type": "sc:PropertyValue", "name": "eo_ir_total_duration_seconds", "value": 12764.882},
        ],
        "distribution": build_distribution(),
        "recordSet": [
            build_video_pairs_record_set(csv_columns, metadata_rows),
            build_annotation_record_set(annotation_rows),
        ],
        "rai:hasSyntheticData": False,
        "rai:dataCollection": (
            "The dataset consists of paired electro-optical and infrared videos "
            "with manually curated multiple-choice annotations for model evaluation. "
            "The paper artifact stores anonymized relative filenames and benchmark "
            "annotations; raw videos must be provided by the hosted dataset using "
            "the same relative layout."
        ),
        "rai:dataCollectionType": [
            "Physical data collection",
            "Secondary Data analysis",
            "Manual Human Curator",
            "Software Collection",
        ],
        "prov:wasDerivedFrom": [
            {
                "@id": f"{DATASET_URL}#raw-eo-ir-video-files",
                "@type": "sc:Dataset",
                "name": "Hosted raw EO/IR video files",
                "description": (
                    "Primary paired EO/IR video files hosted with the Dataverse "
                    "dataset. No separate external source-dataset URI is asserted "
                    "for this anonymous submission artifact."
                ),
            }
        ],
        "prov:wasGeneratedBy": [
            {
                "@type": "prov:Activity",
                "name": "EO/IR video collection and pairing",
                "description": (
                    "Electro-optical and infrared video streams were organized as "
                    "paired samples through filename metadata and relative EO/IR "
                    "paths."
                ),
            },
            {
                "@type": "prov:Activity",
                "name": "Manual multiple-choice QA annotation",
                "description": (
                    "Each annotated sample was curated into multiple-choice video "
                    "QA items with modality requirements, capability metadata, gold "
                    "answers, and internal evidence notes."
                ),
            },
            {
                "@type": "prov:Activity",
                "name": "Benchmark export and scoring metadata generation",
                "description": (
                    "Repository scripts validate annotations, export model-safe "
                    "requests without label leakage, score predictions, and "
                    "regenerate Croissant metadata and checksums."
                ),
            },
        ],
        "rai:dataCollectionRawData": (
            "Raw data are paired .ts EO and IR videos named with *_EO.ts and *_IR.ts "
            "suffixes. Metadata records video duration, frame count, frame rate, "
            "resolution, bitrate, pairing method, and relative file paths."
        ),
        "rai:dataCollectionMissingData": (
            "The 48 annotated samples have zero missing EO/IR video pairs according "
            "to the current statistics. The anonymized metadata has null file-size "
            "fields because machine-specific raw file sizes were not included in "
            "the paper artifact. The duplicated Parked.car video pair is excluded "
            "from the annotation-counting policy."
        ),
        "rai:dataPreprocessingProtocol": [
            "EO and IR streams are paired through filename metadata and resolved under VIDEO_DATA_ROOT.",
            "The official evaluation protocol uses full videos sampled uniformly at 1.0 fps, preserves aspect ratio, caps each frame at a 512x512 area budget, and disables audio.",
            "Request export hides gold answers, option roles, evidence notes, rationales, event descriptions, and annotation time-reference metadata from model prompts.",
        ],
        "rai:dataAnnotationProtocol": (
            "Each annotated sample has 12 multiple-choice questions with A-D options, "
            "a gold answer, modality requirements, question types, capability level, "
            "group metadata, and internal evidence notes. Questions are designed to "
            "test EO-only, IR-only, and EO+IR reasoning while controlling for same-scene "
            "distractors and wording leakage."
        ),
        "rai:dataAnnotationAnalysis": [
            "The repository provides schema validation for annotation files and group structure.",
            "The artifact check validates JSON syntax, annotation schema, request export, and common leakage patterns.",
            "The annotation policy recommends text-only/no-video baseline checks to detect wording shortcuts before video inference.",
        ],
        "rai:dataReleaseMaintenancePlan": [
            "This is a static submission snapshot, not a live dataset.",
            "Future data or annotation changes should increment the dataset version and regenerate all Croissant files and checksums.",
            "For double-blind review, creator and hosting fields are anonymized and must be updated for the camera-ready public release when applicable.",
        ],
        "rai:personalSensitiveInformation": [
            "No personal or sensitive human attributes are intentionally annotated or used as labels.",
            "The videos focus on vehicles, vessels, aircraft, infrastructure, motion, and cross-modal visual evidence rather than identifying people.",
            "Before public release, the hosted raw videos should be reviewed for incidental identifiers such as readable markings, plates, faces, or location-sensitive details."
        ],
        "rai:dataSocialImpact": (
            "The benchmark supports transparent evaluation of multimodal video models "
            "in EO/IR settings, including whether models use thermal and visible-light "
            "evidence appropriately. Potential risks include dual-use surveillance or "
            "operational monitoring applications; the dataset is intended for research "
            "evaluation and should not be used for identifying people, tracking private "
            "individuals, or safety-critical operational decisions."
        ),
        "rai:dataBiases": [
            "The benchmark is small and intentionally scenario-focused: 48 annotated samples and 576 questions.",
            "Maritime vessels are overrepresented relative to ground vehicles and aircraft, and the sensing contexts are limited to maritime and airborne viewpoints.",
            "Sensor platform, geography, weather, time-of-day, and capture-condition coverage may not represent the broader EO/IR video domain.",
            "Multiple-choice answer distributions and distractor templates may introduce artifacts; text-only leakage checks are part of the recommended validation process.",
        ],
        "rai:dataLimitations": [
            "The dataset is for evaluation and benchmarking, not for training general-purpose perception or surveillance systems.",
            "It does not claim demographic, geographic, sensor, or operational representativeness.",
            "The anonymous paper artifact does not include raw videos; reviewers need the hosted raw data or private preview URL for full inspection.",
            "Gold answers, evidence notes, and time references are labels/internal audit metadata and must not be exposed to evaluated models.",
            "Reviewer access through the hosted dataset URL must remain enabled through review, and public access must be finalized by the camera-ready deadline if accepted.",
        ],
        "rai:dataUseCases": [
            "Evaluate multimodal video-language models on EO-only, IR-only, and EO+IR full-video multiple-choice reasoning.",
            "Study cross-modal evidence use, temporal reasoning, thermal interpretation, target grounding, and distractor rejection.",
            "Run controlled benchmark ablations across modality settings and model families.",
            "Not recommended for model training, person identification, surveillance deployment, safety-critical decision-making, or claims about broad EO/IR domain coverage.",
        ],
        "rai:annotationsPerItem": (
            "Each annotated EO/IR sample has 12 multiple-choice QA items. The current "
            "dataset contains 48 annotated samples and 576 total QA items."
        ),
        "rai:annotatorDemographics": (
            "Annotator demographic information is not collected in this anonymous "
            "artifact and should not be inferred from the annotations."
        ),
        "rai:machineAnnotationTools": [
            "Repository scripts extract video metadata, validate annotation schema, export model-safe requests, and score predictions.",
            "No machine-generated demographic labels or personal-attribute labels are provided.",
        ],
        "rai:dataManipulationProtocol": (
            "The benchmark does not require pixel-level manipulation of the raw videos. "
            "Evaluation-time preprocessing samples frames, resizes frames according to "
            "the documented policy, and packages model requests without label leakage. "
            "The metadata files anonymize machine-specific paths by using relative filenames."
        ),
        "rai:dataImputationProtocol": (
            "No label imputation is used. Missing file sizes remain null in the "
            "anonymized metadata, and metadata extraction issues are represented by "
            "eo_metadata_error and ir_metadata_error fields when applicable."
        ),
    }


def main() -> None:
    metadata = build_metadata()
    for output_name in OUTPUT_NAMES:
        output_path = ROOT / output_name
        output_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
