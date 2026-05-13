#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS_DIR = ROOT_DIR / "data" / "annotations"
DEFAULT_METADATA_CSV = ROOT_DIR / "data" / "metadata" / "video_metadata.example.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval_outputs"
DEFAULT_DATA_ROOT_ENV = "VIDEO_DATA_ROOT"
DEFAULT_OPTION_SHUFFLE_SALT = "eo_ir_option_shuffle_v1"
DEFAULT_MEDIA_ALIAS_SALT = "eo_ir_media_alias_v1"
DEFAULT_VIDEO_INPUT_PROTOCOL = {
    "protocol_id": "eo_ir_full_video_1fps_512px_v1",
    "input_format_version": "1.0",
    "clip_policy": {
        "type": "full_video",
        "start_seconds": 0.0,
        "end_seconds": None,
    },
    "sampling": {
        "method": "uniform_fps",
        "fps": 1.0,
        "max_frames_per_video": None,
        "timestamp_unit": "seconds",
    },
    "frame_resolution": {
        "method": "resize_area_cap",
        "max_pixels": 262144,
        "target_longest_side": 512,
        "preserve_aspect_ratio": True,
    },
    "audio": {
        "enabled": False,
    },
    "multi_video_policy": {
        "eo_ir_order": ["EO", "IR"],
        "alignment": "use the complete decoded EO and IR timelines at the same sampling policy",
    },
    "leakage_policy": {
        "include_gold_answer": False,
        "include_rationale": False,
        "include_time_reference_metadata": False,
    },
}

SETTING_EO_ONLY = "eo_only"
SETTING_IR_ONLY = "ir_only"
SETTING_EO_IR = "eo_ir"
VALID_SETTINGS = (SETTING_EO_ONLY, SETTING_IR_ONLY, SETTING_EO_IR)

VALID_ANSWERS = {"A", "B", "C", "D"}
VALID_CAPABILITY_LEVELS = {"L1", "L2", "L3"}
VALID_GROUP_FAMILIES = {"consistency", "coherence"}
VALID_MODALITY_REQUIREMENTS = {"EO", "IR", "EO+IR"}
VALID_SCENARIO_TYPES = {"EO-favorable", "IR-favorable", "EO+IR-required"}
LEGACY_SCENARIO_TYPE_MAP = {"EO_IR_required": "EO+IR-required"}

STRICT_QUESTION_TYPES = {
    "entity_grounding",
    "scene_grounding",
    "spatial_reasoning",
    "temporal_reasoning",
    "camera_motion_vs_target_motion",
    "trajectory_reasoning",
    "evidence_retrieval",
    "evidence_sufficiency",
    "thermal_evidence_interpretation",
    "cross_modal_phenomenon_explanation",
    "causal_reasoning",
    "group_verdict",
}

QUESTION_TYPE_TO_LEVEL = {
    # release taxonomy
    "entity_grounding": "L1",
    "scene_grounding": "L1",
    "spatial_reasoning": "L1",
    "evidence_retrieval": "L1",
    "temporal_reasoning": "L2",
    "camera_motion_vs_target_motion": "L2",
    "trajectory_reasoning": "L2",
    "evidence_sufficiency": "L2",
    "thermal_evidence_interpretation": "L2",
    "cross_modal_phenomenon_explanation": "L3",
    "causal_reasoning": "L3",
    "group_verdict": "L3",
    # legacy aliases
    "entity_recognition": "L1",
    "key_information_retrieval": "L1",
    "occlusion_detection": "L1",
    "event_understanding": "L2",
    "kinematic_analysis": "L2",
    "anomaly_detection": "L2",
    "reasoning": "L3",
    "cross_modal_comparison": "L3",
    "cross_modal_reasoning": "L3",
    "cross_modal_identity_reasoning": "L3",
    "cross_scale_identity_reasoning": "L3",
    "cross_modal_occlusion_reasoning": "L3",
    "temporal_persistence_reasoning": "L3",
    "occlusion_persistence_reasoning": "L3",
    "cross_modal_chain_reasoning": "L3",
    "entity_integrity_reasoning": "L3",
    "modality_limit_reasoning": "L3",
    "sensor_analysis": "L3",
    "traffic_interaction_reasoning": "L3",
}

REQUIRE_CAPABILITY_LEVEL = True
STRICT_QUESTION_TYPE_VALIDATION = False


@dataclass
class MetadataRecord:
    pair_key: str
    theme_guess: str
    eo_filename: Optional[str]
    ir_filename: Optional[str]
    eo_path: Optional[str]
    ir_path: Optional[str]
    eo_duration_seconds: Optional[float]
    ir_duration_seconds: Optional[float]
    duration_delta_seconds: Optional[float]
    pairing_method: Optional[str]


class BenchmarkError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, request, baseline, and score the EO/IR benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_dataset = subparsers.add_parser(
        "prepare-dataset",
        help="Build manifest.jsonl and questions.jsonl from annotations and metadata.",
    )
    prepare_dataset.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help=f"Directory containing annotation JSON files. Default: {DEFAULT_ANNOTATIONS_DIR}",
    )
    prepare_dataset.add_argument(
        "--metadata-csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help=f"Metadata CSV describing EO/IR pairs. Default: {DEFAULT_METADATA_CSV}",
    )
    prepare_dataset.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Root directory for relative video paths in metadata. "
            f"If omitted, ${DEFAULT_DATA_ROOT_ENV} is used when set; otherwise paths are resolved relative to the metadata CSV."
        ),
    )
    prepare_dataset.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    prepare_dataset.add_argument(
        "--disable-option-shuffle",
        action="store_true",
        help="Preserve original A-D option labels instead of exporting a deterministic per-question permutation.",
    )
    prepare_dataset.add_argument(
        "--option-shuffle-salt",
        default=DEFAULT_OPTION_SHUFFLE_SALT,
        help=(
            "Salt used for deterministic per-question option shuffling. "
            f"Default: {DEFAULT_OPTION_SHUFFLE_SALT}"
        ),
    )
    prepare_dataset.add_argument(
        "--disable-media-aliases",
        action="store_true",
        help="Keep original EO/IR file paths in exported model inputs instead of anonymized aliases.",
    )
    prepare_dataset.add_argument(
        "--media-alias-salt",
        default=DEFAULT_MEDIA_ALIAS_SALT,
        help=(
            "Salt used for deterministic anonymized media alias paths. "
            f"Default: {DEFAULT_MEDIA_ALIAS_SALT}"
        ),
    )

    prepare_clips = subparsers.add_parser(
        "prepare-clips",
        help="Optional debug utility: materialize EO/IR question clips into clips/ using OpenCV.",
    )
    prepare_clips.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "questions.jsonl",
        help="questions.jsonl generated by prepare-dataset.",
    )
    prepare_clips.add_argument(
        "--force",
        action="store_true",
        help="Recreate clips even if they already exist.",
    )
    prepare_clips.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N questions. Useful for smoke tests.",
    )

    export_requests = subparsers.add_parser(
        "export-requests",
        help="Write model-agnostic request JSONL files for eo_only / ir_only / eo_ir using full videos.",
    )
    export_requests.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "questions.jsonl",
        help="questions.jsonl generated by prepare-dataset.",
    )
    export_requests.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact directory. Requests will be placed under requests/.",
    )
    export_requests.add_argument(
        "--settings",
        nargs="+",
        default=list(VALID_SETTINGS),
        choices=VALID_SETTINGS,
        help="Request files to export.",
    )

    random_baseline = subparsers.add_parser(
        "run-random-baseline",
        help="Write random-choice predictions under predictions/{model_id}/.",
    )
    random_baseline.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "questions.jsonl",
        help="questions.jsonl generated by prepare-dataset.",
    )
    random_baseline.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact directory. Predictions will be placed under predictions/.",
    )
    random_baseline.add_argument(
        "--model-id",
        default="random_mcq",
        help="Model identifier used in predictions/{model_id}/.",
    )
    random_baseline.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible baseline predictions.",
    )
    random_baseline.add_argument(
        "--settings",
        nargs="+",
        default=list(VALID_SETTINGS),
        choices=VALID_SETTINGS,
        help="Prediction settings to emit.",
    )

    score_predictions = subparsers.add_parser(
        "score-predictions",
        help="Score predictions/{model}/{setting}.jsonl and write result tables.",
    )
    score_predictions.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "questions.jsonl",
        help="questions.jsonl generated by prepare-dataset.",
    )
    score_predictions.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "predictions",
        help="Directory containing predictions/{model}/{setting}.jsonl.",
    )
    score_predictions.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "results",
        help="Directory where CSV reports will be written.",
    )
    score_predictions.add_argument(
        "--allow-partial",
        action="store_true",
        help="Score partial model coverage instead of requiring all settings and all questions.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-dataset":
        prepare_dataset_command(args)
    elif args.command == "prepare-clips":
        prepare_clips_command(args)
    elif args.command == "export-requests":
        export_requests_command(args)
    elif args.command == "run-random-baseline":
        random_baseline_command(args)
    elif args.command == "score-predictions":
        score_predictions_command(args)
    else:
        raise BenchmarkError(f"Unsupported command: {args.command}")


def prepare_dataset_command(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    clips_dir = output_dir / "clips"
    media_aliases_dir = output_dir / "model_media"
    ensure_dir(output_dir)
    ensure_dir(clips_dir / "eo")
    ensure_dir(clips_dir / "ir")
    ensure_dir(media_aliases_dir / "eo")
    ensure_dir(media_aliases_dir / "ir")

    metadata_rows = load_metadata(
        args.metadata_csv.resolve(),
        data_root=resolve_data_root(getattr(args, "data_root", None)),
    )
    manifest_rows, question_rows, source_answer_labels = build_manifest_and_questions(
        annotations_dir=args.annotations_dir.resolve(),
        metadata_rows=metadata_rows,
        output_dir=output_dir,
        enable_option_shuffle=not getattr(args, "disable_option_shuffle", False),
        option_shuffle_salt=str(getattr(args, "option_shuffle_salt", DEFAULT_OPTION_SHUFFLE_SALT)),
        media_aliases_dir=None if getattr(args, "disable_media_aliases", False) else media_aliases_dir,
        media_alias_salt=str(getattr(args, "media_alias_salt", DEFAULT_MEDIA_ALIAS_SALT)),
    )
    model_safe_question_rows = build_model_safe_questions(question_rows)

    manifest_path = output_dir / "manifest.jsonl"
    questions_path = output_dir / "questions.jsonl"
    model_safe_questions_path = output_dir / "model_safe_questions.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(questions_path, question_rows)
    write_jsonl(model_safe_questions_path, model_safe_question_rows)

    summary = {
        "artifact_directory": str(output_dir),
        "manifest_path": str(manifest_path),
        "questions_path": str(questions_path),
        "model_safe_questions_path": str(model_safe_questions_path),
        "sample_count": len(manifest_rows),
        "question_count": len(question_rows),
        "group_count": len({(row["sample_key"], row["group_id"], row["group_focus"]) for row in question_rows}),
        "source_answer_distribution_audit": audit_answer_labels(source_answer_labels),
        "exported_answer_distribution_audit": audit_answer_distribution(question_rows),
        "option_shuffle": {
            "enabled": not getattr(args, "disable_option_shuffle", False),
            "salt": str(getattr(args, "option_shuffle_salt", DEFAULT_OPTION_SHUFFLE_SALT)),
        },
        "media_aliases": {
            "enabled": not getattr(args, "disable_media_aliases", False),
            "salt": str(getattr(args, "media_alias_salt", DEFAULT_MEDIA_ALIAS_SALT)),
            "directory": str(media_aliases_dir),
        },
        "settings": list(VALID_SETTINGS),
        "video_input": build_video_input_protocol(),
        "protocol": DEFAULT_VIDEO_INPUT_PROTOCOL["protocol_id"],
    }
    write_json(output_dir / "dataset_summary.json", summary)
    print(json.dumps({
        "status": "ok",
        "sample_count": summary["sample_count"],
        "question_count": summary["question_count"],
        "group_count": summary["group_count"],
        "output_dir": str(output_dir),
    }, indent=2))


def prepare_clips_command(args: argparse.Namespace) -> None:
    question_rows = load_jsonl(args.questions.resolve())
    if args.limit is not None:
        question_rows = question_rows[: args.limit]

    prepared = 0
    skipped = 0
    for question in question_rows:
        for modality in required_modalities_for_question(question):
            clip_field = f"{modality.lower()}_clip_path"
            source_field = f"{modality.lower()}_path"
            clip_path = Path(question[clip_field])
            source_path = Path(question[source_field])
            if clip_path.exists() and not args.force:
                skipped += 1
                continue
            ensure_dir(clip_path.parent)
            extract_clip(
                source_path=source_path,
                clip_path=clip_path,
                time_reference=resolve_time_reference(question, modality),
            )
            prepared += 1

    print(json.dumps({
        "status": "ok",
        "clips_prepared": prepared,
        "clips_skipped": skipped,
        "question_count": len(question_rows),
        "note": "prepare-clips is optional and not required for the main full-video benchmark.",
    }, indent=2))


def export_requests_command(args: argparse.Namespace) -> None:
    questions = load_jsonl(args.questions.resolve())
    requests_dir = args.output_dir.resolve() / "requests"
    ensure_dir(requests_dir)

    for setting in args.settings:
        request_rows = [build_request_row(question, setting) for question in questions]
        output_path = requests_dir / f"{setting}.jsonl"
        write_jsonl(output_path, request_rows)

    print(json.dumps({
        "status": "ok",
        "request_dir": str(requests_dir),
        "settings": args.settings,
        "question_count": len(questions),
        "protocol": DEFAULT_VIDEO_INPUT_PROTOCOL["protocol_id"],
    }, indent=2))


def random_baseline_command(args: argparse.Namespace) -> None:
    questions = load_jsonl(args.questions.resolve())
    predictions_root = args.output_dir.resolve() / "predictions" / args.model_id
    ensure_dir(predictions_root)
    rng = random.Random(args.seed)

    for setting in args.settings:
        rows = []
        for question in questions:
            rows.append({
                "model_id": args.model_id,
                "setting": setting,
                "question_id": question["question_id"],
                "sample_id": question["sample_id"],
                "sample_key": question["sample_key"],
                "uid": question["uid"],
                "predicted_answer": rng.choice(sorted(VALID_ANSWERS)),
                "raw_response": None,
            })
        write_jsonl(predictions_root / f"{setting}.jsonl", rows)

    print(json.dumps({
        "status": "ok",
        "model_id": args.model_id,
        "predictions_dir": str(predictions_root),
        "settings": args.settings,
        "question_count": len(questions),
    }, indent=2))


def score_predictions_command(args: argparse.Namespace) -> None:
    questions = load_jsonl(args.questions.resolve())
    question_map = {row["question_id"]: row for row in questions}
    predictions_dir = args.predictions_dir.resolve()
    if not predictions_dir.exists():
        raise BenchmarkError(f"Predictions directory does not exist: {predictions_dir}")

    results_dir = args.results_dir.resolve()
    ensure_dir(results_dir)

    model_metrics: Dict[str, Dict[str, object]] = {}
    ablation_rows: List[Dict[str, object]] = []
    main_rows: List[Dict[str, object]] = []
    level_rows: List[Dict[str, object]] = []
    slices_rows: List[Dict[str, object]] = []
    errors_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    question_detail_rows: List[Dict[str, object]] = []

    model_dirs = [path for path in predictions_dir.iterdir() if path.is_dir()]
    if not model_dirs:
        raise BenchmarkError(f"No model subdirectories found in {predictions_dir}")

    for model_dir in sorted(model_dirs):
        per_setting_metrics: Dict[str, Dict[str, object]] = {}
        for setting in VALID_SETTINGS:
            prediction_file = model_dir / f"{setting}.jsonl"
            if not prediction_file.exists():
                if args.allow_partial:
                    continue
                raise BenchmarkError(f"Missing prediction file: {prediction_file}")
            metrics = score_prediction_file(
                prediction_file=prediction_file,
                questions=questions,
                question_map=question_map,
            )
            per_setting_metrics[setting] = metrics
            slices_rows.extend(metrics["slice_rows"])
            errors_rows.extend(metrics["error_rows"])
            group_rows.extend(metrics["group_rows"])
            question_detail_rows.extend(metrics["question_detail_rows"])

        if not per_setting_metrics:
            continue

        model_id = next(iter(per_setting_metrics.values()))["model_id"]
        model_metrics[model_id] = per_setting_metrics

        eo_ir_metrics = per_setting_metrics.get(SETTING_EO_IR)
        if eo_ir_metrics:
            main_rows.append({
                "model_id": model_id,
                "S_main": round(eo_ir_metrics["main_score"], 6),
                "S_L1": round(eo_ir_metrics["S_L1"], 6),
                "S_L2": round(eo_ir_metrics["S_L2"], 6),
                "S_L3": round(eo_ir_metrics["S_L3"], 6),
                "S_cons": round(eo_ir_metrics["consistency_score"], 6),
                "S_coh": round(eo_ir_metrics["coherence_score"], 6),
                "overall_acc": round(eo_ir_metrics["overall_acc"], 6),
                "group_composite_score": round(eo_ir_metrics["group_composite_score"], 6),
                "strict_group_pass_rate": round(eo_ir_metrics["strict_group_pass_rate"], 6),
                "question_count": eo_ir_metrics["question_count"],
                "group_count": eo_ir_metrics["group_count"],
                "parse_failure_rate": round(eo_ir_metrics["parse_failure_rate"], 6),
            })
            level_metrics = eo_ir_metrics["level_metrics"]
            level_rows.append({
                "model_id": model_id,
                "S_L1": round(level_metrics.get("L1", {}).get("accuracy", 0.0), 6),
                "S_L2": round(level_metrics.get("L2", {}).get("accuracy", 0.0), 6),
                "S_L3": round(level_metrics.get("L3", {}).get("accuracy", 0.0), 6),
                "L1_count": level_metrics.get("L1", {}).get("count", 0),
                "L2_count": level_metrics.get("L2", {}).get("count", 0),
                "L3_count": level_metrics.get("L3", {}).get("count", 0),
            })

        ablation_rows.append(build_ablation_row(model_id, per_setting_metrics))

    scenario_rows = [row for row in slices_rows if row["slice_type"] == "scenario_type"]

    write_csv(results_dir / "main_table.csv", main_rows)
    write_csv(results_dir / "ablation_table.csv", ablation_rows)
    write_csv(results_dir / "level_table.csv", level_rows)
    write_csv(results_dir / "scenario_table.csv", scenario_rows)
    write_csv(results_dir / "slices.csv", slices_rows)
    write_csv(results_dir / "errors.csv", errors_rows)
    write_csv(results_dir / "group_details.csv", group_rows)
    write_csv(results_dir / "question_details.csv", question_detail_rows)

    summary = {
        "status": "ok",
        "models_scored": sorted(model_metrics.keys()),
        "results_dir": str(results_dir),
        "main_table": str(results_dir / "main_table.csv"),
        "ablation_table": str(results_dir / "ablation_table.csv"),
        "level_table": str(results_dir / "level_table.csv"),
        "scenario_table": str(results_dir / "scenario_table.csv"),
        "question_details": str(results_dir / "question_details.csv"),
    }
    write_json(results_dir / "results_summary.json", summary)
    print(json.dumps(summary, indent=2))


def build_manifest_and_questions(
    annotations_dir: Path,
    metadata_rows: Sequence[MetadataRecord],
    output_dir: Path,
    enable_option_shuffle: bool,
    option_shuffle_salt: str,
    media_aliases_dir: Optional[Path],
    media_alias_salt: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[str]]:
    metadata_by_key = {row.pair_key: row for row in metadata_rows}
    manifest_rows: List[Dict[str, object]] = []
    question_rows: List[Dict[str, object]] = []
    seen_sample_keys = set()
    source_answer_labels: List[str] = []

    for annotation_path in sorted(annotations_dir.glob("*.json")):
        annotation = json.loads(annotation_path.read_text())
        validate_annotation_schema(annotation_path, annotation)

        sample_id = annotation["sample_id"]
        canonical_key = canonical_key_from_annotation_filename(annotation_path.name)
        metadata = metadata_by_key.get(canonical_key)
        if metadata is None:
            raise BenchmarkError(
                f"Could not match annotation {annotation_path.name} to metadata using key '{canonical_key}'"
            )
        sample_key = build_sample_key(metadata, annotation_path)
        if sample_key in seen_sample_keys:
            raise BenchmarkError(f"Duplicate sample_key detected: {sample_key}")
        seen_sample_keys.add(sample_key)

        eo_path = require_existing_path(annotation_path.name, "EO", metadata.eo_path)
        ir_path = require_existing_path(annotation_path.name, "IR", metadata.ir_path)
        eo_model_input_path = build_media_alias_path(
            source_path=eo_path,
            modality="EO",
            media_aliases_dir=media_aliases_dir,
            media_alias_salt=media_alias_salt,
        )
        ir_model_input_path = build_media_alias_path(
            source_path=ir_path,
            modality="IR",
            media_aliases_dir=media_aliases_dir,
            media_alias_salt=media_alias_salt,
        )
        group_rules = derive_group_score_rules(annotation)
        validate_group_structure(annotation_path, annotation["qa"])

        scenario_type = canonicalize_scenario_type(str(annotation["scenario_type"]))

        manifest_rows.append({
            "sample_id": sample_id,
            "sample_key": sample_key,
            "sample_name": metadata.theme_guess,
            "canonical_key": canonical_key,
            "annotation_filename": annotation_path.name,
            "annotation_path": str(annotation_path),
            "eo_filename": metadata.eo_filename,
            "ir_filename": metadata.ir_filename,
            "eo_path": eo_path,
            "ir_path": ir_path,
            "eo_model_input_path": eo_model_input_path,
            "ir_model_input_path": ir_model_input_path,
            "type": annotation["type"],
            "main_event": annotation["main_event"],
            "event_description": annotation["event_description"],
            "scenario_type": scenario_type,
            "notes": annotation.get("notes", {}),
            "eo_duration_seconds": metadata.eo_duration_seconds,
            "ir_duration_seconds": metadata.ir_duration_seconds,
            "pair_duration_delta_seconds": metadata.duration_delta_seconds,
            "pairing_method": metadata.pairing_method,
            "status": "ready",
        })

        for qa in annotation["qa"]:
            primary_question_type = qa["question_type"][0]
            level_tag = resolve_level_tag(qa, annotation_path)
            group_focus = normalize_group_focus(qa)
            group_score_rule = group_rules[(str(qa["group_id"]), group_focus)]
            group_family = normalize_group_family(qa, group_score_rule)
            question_id = build_question_id(sample_key, str(qa["uid"]))
            source_answer = str(qa["answer"])
            source_answer_labels.append(source_answer)
            option_payload = remap_question_options(
                question_id=question_id,
                options=qa["options"],
                option_roles=qa["option_roles"],
                correct_answer=source_answer,
                enable_option_shuffle=enable_option_shuffle,
                option_shuffle_salt=option_shuffle_salt,
            )
            question_rows.append({
                "question_id": question_id,
                "sample_id": sample_id,
                "sample_key": sample_key,
                "uid": str(qa["uid"]),
                "group_id": str(qa["group_id"]),
                "group_family": group_family,
                "group_focus": group_focus,
                "group_type": group_focus,
                "group_step": int(qa["group_step"]),
                "group_score_rule": group_score_rule,
                "question": qa["question"],
                "options": option_payload["options"],
                "correct_answer": option_payload["correct_answer"],
                "source_correct_answer": source_answer,
                "option_roles": option_payload["option_roles"],
                "option_permutation_old_to_new": option_payload["old_to_new"],
                "question_type": qa["question_type"],
                "primary_question_type": primary_question_type,
                "level_tag": level_tag,
                "capability_level": level_tag,
                "time_reference_eo": qa.get("time_reference_eo", qa.get("time_reference", "")),
                "time_reference_ir": qa.get("time_reference_ir", qa.get("time_reference", "")),
                "modality_requirement": qa["modality_requirement"],
                "scenario_type": scenario_type,
                "scene_type": annotation["type"],
                "main_event": annotation["main_event"],
                "event_description": annotation["event_description"],
                "annotation_path": str(annotation_path),
                "eo_path": eo_path,
                "ir_path": ir_path,
                "eo_model_input_path": eo_model_input_path,
                "ir_model_input_path": ir_model_input_path,
                "eo_clip_path": str(output_dir / "clips" / "eo" / f"{question_id}.mp4"),
                "ir_clip_path": str(output_dir / "clips" / "ir" / f"{question_id}.mp4"),
            })

    if not manifest_rows:
        raise BenchmarkError(f"No annotations found in {annotations_dir}")
    return manifest_rows, question_rows, source_answer_labels


def load_metadata(path: Path, data_root: Optional[Path] = None) -> List[MetadataRecord]:
    rows: List[MetadataRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pair_key = normalize_key(row["pair_key"])
            rows.append(
                MetadataRecord(
                    pair_key=pair_key,
                    theme_guess=row["theme_guess"],
                    eo_filename=empty_to_none(row["eo_filename"]),
                    ir_filename=empty_to_none(row["ir_filename"]),
                    eo_path=resolve_metadata_media_path(
                        row.get("eo_path"),
                        row.get("eo_filename"),
                        metadata_dir=path.parent,
                        data_root=data_root,
                    ),
                    ir_path=resolve_metadata_media_path(
                        row.get("ir_path"),
                        row.get("ir_filename"),
                        metadata_dir=path.parent,
                        data_root=data_root,
                    ),
                    eo_duration_seconds=parse_optional_float(row["eo_duration_seconds"]),
                    ir_duration_seconds=parse_optional_float(row["ir_duration_seconds"]),
                    duration_delta_seconds=parse_optional_float(row["duration_delta_seconds"]),
                    pairing_method=empty_to_none(row["pairing_method"]),
                )
            )
    if not rows:
        raise BenchmarkError(f"No rows found in metadata CSV: {path}")
    return rows


def resolve_data_root(value: Optional[Path]) -> Optional[Path]:
    if value is not None:
        return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    env_value = os.environ.get(DEFAULT_DATA_ROOT_ENV)
    if env_value:
        return Path(os.path.expandvars(os.path.expanduser(env_value))).resolve()
    return None


def resolve_metadata_media_path(
    path_text: Optional[str],
    filename_text: Optional[str],
    metadata_dir: Path,
    data_root: Optional[Path],
) -> Optional[str]:
    raw_value = empty_to_none(path_text) or empty_to_none(filename_text)
    if raw_value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(raw_value))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    base_dir = data_root if data_root is not None else metadata_dir
    return str((base_dir / path).resolve())


def validate_annotation_schema(path: Path, annotation: Dict[str, object]) -> None:
    required_fields = {"sample_id", "type", "main_event", "event_description", "scenario_type", "qa"}
    missing = required_fields - annotation.keys()
    if missing:
        raise BenchmarkError(f"{path.name} is missing required fields: {sorted(missing)}")
    canonicalize_scenario_type(str(annotation["scenario_type"]))
    if not isinstance(annotation["qa"], list) or not annotation["qa"]:
        raise BenchmarkError(f"{path.name} must contain a non-empty qa list")

    for qa in annotation["qa"]:
        required_qa_fields = {
            "uid", "group_id", "group_step", "question", "options", "answer",
            "option_roles", "question_type", "modality_requirement",
        }
        missing_qa = required_qa_fields - qa.keys()
        if missing_qa:
            raise BenchmarkError(f"{path.name} QA uid={qa.get('uid')} missing fields: {sorted(missing_qa)}")

        if set(qa["options"].keys()) != VALID_ANSWERS:
            raise BenchmarkError(f"{path.name} QA uid={qa['uid']} must have exactly options A-D")
        if set(qa["option_roles"].keys()) != VALID_ANSWERS:
            raise BenchmarkError(f"{path.name} QA uid={qa['uid']} must have option_roles for A-D")
        if qa["answer"] not in VALID_ANSWERS:
            raise BenchmarkError(f"{path.name} QA uid={qa['uid']} has invalid answer {qa['answer']!r}")

        try:
            normalize_group_focus(qa)
        except BenchmarkError as exc:
            raise BenchmarkError(
                f"{path.name} QA uid={qa['uid']} must define a non-empty group_focus or group_type"
            ) from exc

        group_family = qa.get("group_family")
        if group_family is not None:
            normalized_family = str(group_family).strip().lower()
            if normalized_family not in VALID_GROUP_FAMILIES:
                raise BenchmarkError(
                    f"{path.name} QA uid={qa['uid']} has unsupported group_family {group_family!r}; "
                    f"expected one of {sorted(VALID_GROUP_FAMILIES)}"
                )

        capability_level = qa.get("capability_level")
        if REQUIRE_CAPABILITY_LEVEL and capability_level is None:
            raise BenchmarkError(
                f"{path.name} QA uid={qa['uid']} is missing capability_level; release schema requires one of {sorted(VALID_CAPABILITY_LEVELS)}"
            )
        if capability_level is not None:
            normalized_level = str(capability_level).strip().upper()
            if normalized_level not in VALID_CAPABILITY_LEVELS:
                raise BenchmarkError(
                    f"{path.name} QA uid={qa['uid']} has unsupported capability_level {capability_level!r}; expected one of {sorted(VALID_CAPABILITY_LEVELS)}"
                )

        if qa["modality_requirement"] not in VALID_MODALITY_REQUIREMENTS:
            raise BenchmarkError(
                f"{path.name} QA uid={qa['uid']} has invalid modality_requirement {qa['modality_requirement']!r}"
            )
        if not isinstance(qa["question_type"], list) or not qa["question_type"]:
            raise BenchmarkError(f"{path.name} QA uid={qa['uid']} must have a non-empty question_type list")

        if STRICT_QUESTION_TYPE_VALIDATION:
            for qt in qa["question_type"]:
                if qt not in STRICT_QUESTION_TYPES:
                    raise BenchmarkError(
                        f"{path.name} QA uid={qa['uid']} uses unsupported question_type {qt!r} under strict validation"
                    )

        if not any(isinstance(qa.get(field), str) and qa.get(field).strip() for field in ("time_reference_eo", "time_reference_ir", "time_reference")):
            raise BenchmarkError(f"{path.name} QA uid={qa['uid']} must define at least one time reference field")


def validate_group_structure(path: Path, qa_rows: Sequence[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for qa in qa_rows:
        grouped[(str(qa["group_id"]), normalize_group_focus(qa))].append(int(qa["group_step"]))
    if not grouped:
        raise BenchmarkError(f"{path.name} must contain at least one question group")
    for (group_id, group_focus), steps in grouped.items():
        sorted_steps = sorted(steps)
        expected_steps = list(range(1, len(sorted_steps) + 1))
        if sorted_steps != expected_steps:
            raise BenchmarkError(
                f"{path.name} group ({group_id}, {group_focus}) must contain contiguous steps starting at 1; found {sorted_steps}"
            )


def question_type_to_level(question_type: str, annotation_path: Path) -> str:
    level = QUESTION_TYPE_TO_LEVEL.get(question_type)
    if level is None:
        known = ", ".join(sorted(QUESTION_TYPE_TO_LEVEL))
        raise BenchmarkError(
            f"{annotation_path.name} uses unmapped question_type={question_type!r}. Known mappings: {known}"
        )
    return level


def resolve_level_tag(qa: Dict[str, object], annotation_path: Path) -> str:
    capability_level = qa.get("capability_level")
    if capability_level is not None:
        normalized = str(capability_level).strip().upper()
        if normalized in VALID_CAPABILITY_LEVELS:
            return normalized
    primary_question_type = str(qa["question_type"][0])
    return question_type_to_level(primary_question_type, annotation_path)


def normalize_group_focus(qa: Dict[str, object]) -> str:
    for field in ("group_focus", "group_type"):
        value = qa.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise BenchmarkError("Missing group focus")


def normalize_group_family(qa: Dict[str, object], score_rule: Optional[str] = None) -> str:
    value = qa.get("group_family")
    if isinstance(value, str) and value.strip():
        normalized = value.strip().lower()
        if normalized in VALID_GROUP_FAMILIES:
            return normalized
    legacy_group_type = qa.get("group_type")
    if isinstance(legacy_group_type, str) and legacy_group_type.strip():
        normalized_group_type = legacy_group_type.strip().lower()
        if normalized_group_type in VALID_GROUP_FAMILIES:
            return normalized_group_type
    if score_rule == "prefix_ratio":
        return "coherence"
    return "consistency"


def canonicalize_scenario_type(value: str) -> str:
    stripped = value.strip()
    if stripped in LEGACY_SCENARIO_TYPE_MAP:
        return LEGACY_SCENARIO_TYPE_MAP[stripped]
    normalized = re.sub(r"[^a-z0-9]+", " ", stripped.lower()).strip()
    mapping = {
        "eo ir required": "EO+IR-required",
        "eo favorable": "EO-favorable",
        "ir favorable": "IR-favorable",
    }
    canonical = mapping.get(normalized)
    if canonical is None or canonical not in VALID_SCENARIO_TYPES:
        allowed = ", ".join(sorted(VALID_SCENARIO_TYPES | set(LEGACY_SCENARIO_TYPE_MAP)))
        raise BenchmarkError(f"Unsupported scenario_type {value!r}; expected one of {allowed}")
    return canonical


def required_modalities_for_setting(setting: str) -> List[str]:
    if setting == SETTING_EO_ONLY:
        return ["EO"]
    if setting == SETTING_IR_ONLY:
        return ["IR"]
    if setting == SETTING_EO_IR:
        return ["EO", "IR"]
    raise BenchmarkError(f"Unsupported setting: {setting}")


def required_modalities_for_question(question: Dict[str, object]) -> List[str]:
    requirement = question["modality_requirement"]
    if requirement == "EO":
        return ["EO"]
    if requirement == "IR":
        return ["IR"]
    if requirement == "EO+IR":
        return ["EO", "IR"]
    raise BenchmarkError(f"Unsupported modality requirement: {requirement}")


def build_request_row(question: Dict[str, object], setting: str) -> Dict[str, object]:
    modalities = required_modalities_for_setting(setting)
    inputs = []
    if "EO" in modalities:
        inputs.append(build_video_input_descriptor(question, "EO", len(inputs) + 1))
    if "IR" in modalities:
        inputs.append(build_video_input_descriptor(question, "IR", len(inputs) + 1))

    if setting == SETTING_EO_ONLY:
        video_instructions = "Video 1 is the full Electro-Optical (EO) video."
    elif setting == SETTING_IR_ONLY:
        video_instructions = "Video 1 is the full Infrared (IR) video."
    else:
        video_instructions = "Video 1 is the full Electro-Optical (EO) video. Video 2 is the full Infrared (IR) video."

    prompt = build_prompt(question, setting, video_instructions)
    return {
        "question_id": question["question_id"],
        "setting": setting,
        "video_input": build_video_input_protocol(),
        "inputs": inputs,
        "prompt": prompt,
        "response_format": 'Reply only with JSON: {"answer":"<A|B|C|D>"}.',
    }


def build_video_input_protocol() -> Dict[str, object]:
    return copy.deepcopy(DEFAULT_VIDEO_INPUT_PROTOCOL)


def build_video_input_descriptor(question: Dict[str, object], modality: str, video_index: int) -> Dict[str, object]:
    return {
        "video_index": video_index,
        "modality": modality,
        "media_type": "video",
        "path": model_input_path_for_modality(question, modality),
        "role": f"video_{video_index}_{modality.lower()}",
    }


def build_prompt(question: Dict[str, object], setting: str, video_instructions: str) -> str:
    option_lines = [f"{label}. {text}" for label, text in sorted(question["options"].items())]
    protocol_line = {
        SETTING_EO_ONLY: "You must answer using only the EO video.",
        SETTING_IR_ONLY: "You must answer using only the IR video.",
        SETTING_EO_IR: "You must jointly reason over the EO and IR videos.",
    }[setting]
    return "\n".join([
        "Answer this single-answer multiple-choice video question.",
        video_instructions,
        protocol_line,
        f"Question: {question['question']}",
        "Options:",
        *option_lines,
        'Reply only with JSON: {"answer":"<A|B|C|D>"}.',
    ])


def build_model_safe_questions(question_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    safe_rows: List[Dict[str, object]] = []
    for question in question_rows:
        safe_rows.append({
            "question_id": question["question_id"],
            "question": question["question"],
            "options": question["options"],
            "modality_requirement": question["modality_requirement"],
            "eo_path": model_input_path_for_modality(question, "EO"),
            "ir_path": model_input_path_for_modality(question, "IR"),
        })
    return safe_rows


def build_media_alias_path(
    source_path: str,
    modality: str,
    media_aliases_dir: Optional[Path],
    media_alias_salt: str,
) -> str:
    source = Path(source_path).resolve()
    if media_aliases_dir is None:
        return str(source)

    suffix = source.suffix or ".bin"
    digest = hashlib.sha256(f"{media_alias_salt}::{modality}::{source}".encode("utf-8")).hexdigest()[:24]
    alias_path = media_aliases_dir / modality.lower() / f"{digest}{suffix}"
    ensure_dir(alias_path.parent)

    if alias_path.exists():
        if alias_path.resolve() != source:
            raise BenchmarkError(f"Media alias path collision for {alias_path}")
    else:
        alias_path.symlink_to(source)
    return str(alias_path)


def model_input_path_for_modality(question: Dict[str, object], modality: str) -> str:
    normalized = modality.strip().upper()
    preferred_field = f"{normalized.lower()}_model_input_path"
    fallback_field = f"{normalized.lower()}_path"
    value = question.get(preferred_field) or question.get(fallback_field)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"Missing model input path for modality {normalized}")
    return value


def remap_question_options(
    question_id: str,
    options: Dict[str, object],
    option_roles: Dict[str, object],
    correct_answer: str,
    enable_option_shuffle: bool,
    option_shuffle_salt: str,
) -> Dict[str, object]:
    source_labels = sorted(VALID_ANSWERS)
    old_to_new = {label: label for label in source_labels}

    if enable_option_shuffle:
        seed_material = f"{option_shuffle_salt}::{question_id}"
        seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)

        target_labels = source_labels[:]
        desired_correct_label = target_labels[seed % len(target_labels)]
        remaining_source_labels = [label for label in source_labels if label != correct_answer]
        remaining_target_labels = [label for label in target_labels if label != desired_correct_label]
        rng.shuffle(remaining_target_labels)

        old_to_new = {correct_answer: desired_correct_label}
        for source_label, target_label in zip(remaining_source_labels, remaining_target_labels):
            old_to_new[source_label] = target_label

    remapped_options = {
        old_to_new[label]: options[label]
        for label in source_labels
    }
    remapped_option_roles = {
        old_to_new[label]: option_roles[label]
        for label in source_labels
    }

    return {
        "options": remapped_options,
        "option_roles": remapped_option_roles,
        "correct_answer": old_to_new[correct_answer],
        "old_to_new": old_to_new,
    }


def audit_answer_distribution(question_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return audit_answer_labels(str(row["correct_answer"]) for row in question_rows)


def audit_answer_labels(answer_labels: Iterable[str]) -> Dict[str, object]:
    counts = Counter(str(answer) for answer in answer_labels)
    total = sum(counts.values())
    dominant_answer = None
    dominant_count = 0
    if counts:
        dominant_answer, dominant_count = counts.most_common(1)[0]

    per_answer_count = {label: counts.get(label, 0) for label in sorted(VALID_ANSWERS)}
    per_answer_fraction = {
        label: round((count / total), 6) if total else 0.0
        for label, count in per_answer_count.items()
    }

    warnings: List[str] = []
    if dominant_answer is not None and total and dominant_count / total >= 0.9:
        warnings.append(
            "Severe answer-label imbalance detected; benchmark scores may be inflated by answer-prior bias."
        )

    missing_answers = [label for label, count in per_answer_count.items() if count == 0]
    if missing_answers:
        warnings.append(f"Missing answer labels in dataset: {', '.join(missing_answers)}")

    return {
        "total_questions": total,
        "per_answer_count": per_answer_count,
        "per_answer_fraction": per_answer_fraction,
        "dominant_answer": dominant_answer,
        "dominant_answer_fraction": round((dominant_count / total), 6) if total else 0.0,
        "warnings": warnings,
    }


def extract_clip(source_path: Path, clip_path: Path, time_reference: str) -> None:
    with suppress_stderr():
        capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise BenchmarkError(f"Failed to open video for clip extraction: {source_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        if width <= 0 or height <= 0:
            raise BenchmarkError(f"Could not determine frame size for {source_path}")

        frame_ranges = resolve_frame_ranges(time_reference, fps, frame_count)
        if not frame_ranges:
            raise BenchmarkError(f"Invalid frame window for {source_path.name}: {time_reference!r}")

        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise BenchmarkError(f"Failed to open writer for clip: {clip_path}")

        try:
            for start_frame, end_frame in frame_ranges:
                if end_frame <= start_frame:
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current = start_frame
                while current < end_frame:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    writer.write(frame)
                    current += 1
        finally:
            writer.release()
    finally:
        capture.release()

    if not clip_path.exists() or clip_path.stat().st_size == 0:
        raise BenchmarkError(f"Clip extraction produced an empty file: {clip_path}")


def resolve_frame_ranges(time_reference: str, fps: float, frame_count: int) -> List[Tuple[int, int]]:
    windows = extract_time_windows(time_reference)
    if not windows:
        return [(0, frame_count)]

    frame_ranges: List[Tuple[int, int]] = []
    for start_seconds, end_seconds in windows:
        start_frame = max(0, int(start_seconds * fps))
        end_frame = frame_count if end_seconds is None else min(frame_count, int(end_seconds * fps))
        frame_ranges.append((start_frame, end_frame))
    return frame_ranges


def extract_time_windows(time_reference: str) -> List[Tuple[int, Optional[int]]]:
    time_reference = (time_reference or "").strip()
    if not time_reference:
        return []

    windows: List[Tuple[int, Optional[int]]] = []
    parts = [part.strip() for part in time_reference.split(";") if part.strip()]
    for part in parts:
        range_match = re.fullmatch(r"(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2})", part)
        if range_match:
            start_seconds = int(range_match.group(1)) * 60 + int(range_match.group(2))
            end_seconds = int(range_match.group(3)) * 60 + int(range_match.group(4))
            windows.append((start_seconds, end_seconds))
            continue
        post_match = re.fullmatch(r"post[-\s]*(\d{2}):(\d{2})", part, flags=re.IGNORECASE)
        if post_match:
            start_seconds = int(post_match.group(1)) * 60 + int(post_match.group(2))
            windows.append((start_seconds, None))
            continue
        raise BenchmarkError(f"Unsupported time_reference format: {part!r}")
    return windows


def score_prediction_file(prediction_file: Path, questions: Sequence[Dict[str, object]], question_map: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    prediction_rows = load_jsonl(prediction_file)
    if not prediction_rows:
        raise BenchmarkError(f"Prediction file is empty: {prediction_file}")

    by_question_id: Dict[str, Dict[str, object]] = {}
    for row in prediction_rows:
        question_id = row.get("question_id")
        if question_id is None and row.get("sample_key") is not None:
            question_id = build_question_id(str(row["sample_key"]), str(row["uid"]))
        if question_id is None and row.get("sample_id") is not None:
            question_id = build_question_id(str(row["sample_id"]), str(row["uid"]))
        if question_id in by_question_id:
            raise BenchmarkError(f"Duplicate prediction for question_id={question_id} in {prediction_file}")
        if question_id not in question_map:
            raise BenchmarkError(f"Unknown question_id={question_id} in {prediction_file}")
        by_question_id[question_id] = row

    if len(by_question_id) != len(questions):
        raise BenchmarkError(f"{prediction_file} contains {len(by_question_id)} predictions, expected {len(questions)}")

    setting = infer_single_value(prediction_rows, "setting", prediction_file.name.replace(".jsonl", ""))
    model_id = infer_single_value(prediction_rows, "model_id", prediction_file.parent.name)

    scored_rows: List[Dict[str, object]] = []
    for question in questions:
        raw = by_question_id[question["question_id"]]
        predicted_answer, parse_status = normalize_predicted_answer(raw)
        is_correct = predicted_answer == question["correct_answer"] and parse_status == "ok"
        scored_row = {
            "model_id": model_id,
            "setting": setting,
            "question_id": question["question_id"],
            "sample_id": question["sample_id"],
            "sample_key": question["sample_key"],
            "uid": question["uid"],
            "group_id": question["group_id"],
            "group_family": question["group_family"],
            "group_focus": question["group_focus"],
            "group_type": question["group_type"],
            "group_step": question["group_step"],
            "group_score_rule": question["group_score_rule"],
            "question_type": question["question_type"],
            "primary_question_type": question["primary_question_type"],
            "level_tag": question["level_tag"],
            "modality_requirement": question["modality_requirement"],
            "scenario_type": question["scenario_type"],
            "scene_type": question["scene_type"],
            "correct_answer": question["correct_answer"],
            "predicted_answer": predicted_answer,
            "parse_status": parse_status,
            "is_correct": is_correct,
            "raw_response": raw.get("raw_response"),
            "distractor_role": question["option_roles"].get(predicted_answer) if predicted_answer in VALID_ANSWERS and not is_correct else None,
        }
        scored_rows.append(scored_row)

    overall_acc = sum(1 for row in scored_rows if row["is_correct"]) / len(scored_rows)
    parse_failure_rate = sum(1 for row in scored_rows if row["parse_status"] != "ok") / len(scored_rows)

    group_buckets: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in scored_rows:
        group_buckets[(row["sample_key"], row["group_id"], row["group_focus"])].append(row)

    group_rows = []
    consistency_scores = []
    coherence_scores = []
    strict_group_passes = []
    for (sample_key, group_id, group_focus), rows in sorted(group_buckets.items()):
        ordered = sorted(rows, key=lambda item: item["group_step"])
        correct_count = sum(1 for row in ordered if row["is_correct"])
        prefix_correct = 0
        for row in ordered:
            if row["is_correct"]:
                prefix_correct += 1
            else:
                break
        k = len(ordered)
        score_rule = str(ordered[0]["group_score_rule"])
        group_score = compute_group_score(ordered, score_rule)
        if score_rule == "prefix_ratio":
            coherence_scores.append(group_score)
        else:
            consistency_scores.append(group_score)
        strict_group_pass = 1 if correct_count == k else 0
        strict_group_passes.append(strict_group_pass)
        group_rows.append({
            "model_id": model_id,
            "setting": setting,
            "sample_key": sample_key,
            "group_id": group_id,
            "group_family": ordered[0]["group_family"],
            "group_focus": ordered[0]["group_focus"],
            "group_type": group_focus,
            "score_rule": score_rule,
            "question_count": k,
            "correct_count": correct_count,
            "prefix_correct_count": prefix_correct,
            "group_score": round(group_score, 6),
            "strict_group_pass": strict_group_pass,
        })

    consistency_score = average(consistency_scores)
    coherence_score = average(coherence_scores)
    group_composite_score = average(consistency_scores + coherence_scores)
    strict_group_pass_rate = average(strict_group_passes)

    level_metrics = aggregate_accuracy(scored_rows, "level_tag")
    s_l1 = level_metrics.get("L1", {}).get("accuracy", 0.0)
    s_l2 = level_metrics.get("L2", {}).get("accuracy", 0.0)
    s_l3 = level_metrics.get("L3", {}).get("accuracy", 0.0)
    main_score = (s_l1 + s_l2 + s_l3) / 3.0

    slice_rows = []
    for field in ("modality_requirement", "primary_question_type", "level_tag", "group_family", "group_focus", "group_step", "scenario_type", "scene_type"):
        for value, payload in aggregate_accuracy(scored_rows, field).items():
            slice_rows.append({
                "model_id": model_id,
                "setting": setting,
                "slice_type": field,
                "slice_value": value,
                "accuracy": round(payload["accuracy"], 6),
                "count": payload["count"],
            })

    error_rows = []
    question_detail_rows = []
    for row in scored_rows:
        question_detail_rows.append({
            "model_id": row["model_id"],
            "setting": row["setting"],
            "question_id": row["question_id"],
            "sample_id": row["sample_id"],
            "sample_key": row["sample_key"],
            "uid": row["uid"],
            "group_id": row["group_id"],
            "group_family": row["group_family"],
            "group_focus": row["group_focus"],
            "group_step": row["group_step"],
            "group_score_rule": row["group_score_rule"],
            "primary_question_type": row["primary_question_type"],
            "level_tag": row["level_tag"],
            "modality_requirement": row["modality_requirement"],
            "scenario_type": row["scenario_type"],
            "scene_type": row["scene_type"],
            "correct_answer": row["correct_answer"],
            "predicted_answer": row["predicted_answer"],
            "parse_status": row["parse_status"],
            "is_correct": row["is_correct"],
            "distractor_role": row["distractor_role"],
        })
        if row["is_correct"]:
            continue
        error_rows.append({
            "model_id": row["model_id"],
            "setting": row["setting"],
            "question_id": row["question_id"],
            "sample_id": row["sample_id"],
            "sample_key": row["sample_key"],
            "uid": row["uid"],
            "group_id": row["group_id"],
            "group_family": row["group_family"],
            "group_focus": row["group_focus"],
            "group_type": row["group_type"],
            "group_step": row["group_step"],
            "group_score_rule": row["group_score_rule"],
            "primary_question_type": row["primary_question_type"],
            "level_tag": row["level_tag"],
            "modality_requirement": row["modality_requirement"],
            "scenario_type": row["scenario_type"],
            "scene_type": row["scene_type"],
            "correct_answer": row["correct_answer"],
            "predicted_answer": row["predicted_answer"],
            "parse_status": row["parse_status"],
            "distractor_role": row["distractor_role"],
            "raw_response": row["raw_response"],
        })

    return {
        "model_id": model_id,
        "setting": setting,
        "question_count": len(scored_rows),
        "group_count": len(group_rows),
        "overall_acc": overall_acc,
        "parse_failure_rate": parse_failure_rate,
        "consistency_score": consistency_score,
        "coherence_score": coherence_score,
        "group_composite_score": group_composite_score,
        "S_L1": s_l1,
        "S_L2": s_l2,
        "S_L3": s_l3,
        "main_score": main_score,
        "strict_group_pass_rate": strict_group_pass_rate,
        "level_metrics": level_metrics,
        "slice_rows": slice_rows,
        "error_rows": error_rows,
        "group_rows": group_rows,
        "question_detail_rows": question_detail_rows,
    }


def build_ablation_row(model_id: str, per_setting_metrics: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    row: Dict[str, object] = {"model_id": model_id}
    eo_metrics = per_setting_metrics.get(SETTING_EO_ONLY)
    ir_metrics = per_setting_metrics.get(SETTING_IR_ONLY)
    eo_ir_metrics = per_setting_metrics.get(SETTING_EO_IR)
    row["S_eo"] = round(eo_metrics["main_score"], 6) if eo_metrics else None
    row["S_ir"] = round(ir_metrics["main_score"], 6) if ir_metrics else None
    row["S_eo_ir"] = round(eo_ir_metrics["main_score"], 6) if eo_ir_metrics else None
    if eo_metrics and ir_metrics and eo_ir_metrics:
        row["G_cm"] = round(eo_ir_metrics["main_score"] - max(eo_metrics["main_score"], ir_metrics["main_score"]), 6)
    else:
        row["G_cm"] = None
    return row


def aggregate_accuracy(rows: Sequence[Dict[str, object]], field: str) -> Dict[object, Dict[str, object]]:
    buckets: Dict[object, List[bool]] = defaultdict(list)
    for row in rows:
        buckets[row[field]].append(bool(row["is_correct"]))
    return {
        key: {"accuracy": sum(values) / len(values), "count": len(values)}
        for key, values in sorted(buckets.items(), key=lambda item: str(item[0]))
    }


def normalize_predicted_answer(row: Dict[str, object]) -> Tuple[Optional[str], str]:
    for key in ("predicted_answer", "answer"):
        value = row.get(key)
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in VALID_ANSWERS:
                return normalized, "ok"

    raw_response = row.get("raw_response")
    if isinstance(raw_response, str) and raw_response.strip():
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            answer = parsed.get("answer")
            if isinstance(answer, str) and answer.strip().upper() in VALID_ANSWERS:
                return answer.strip().upper(), "ok"
        match = re.search(r"\b([ABCD])\b", raw_response.upper())
        if match:
            return match.group(1), "ok"

    return None, "parse_failure"


def infer_single_value(rows: Sequence[Dict[str, object]], field: str, fallback: str) -> str:
    values = {str(row[field]) for row in rows if row.get(field) is not None}
    if not values:
        return fallback
    if len(values) != 1:
        raise BenchmarkError(f"Inconsistent {field} values in prediction file: {sorted(values)}")
    return next(iter(values))


def build_question_id(sample_id: str, uid: str) -> str:
    return f"{sample_id}__q{uid}"


def resolve_time_reference(question: Dict[str, object], modality: str) -> str:
    eo_reference = question.get("time_reference_eo") or ""
    ir_reference = question.get("time_reference_ir") or ""
    if modality == "EO":
        return select_parseable_time_reference(eo_reference, "")
    if modality == "IR":
        return select_parseable_time_reference(ir_reference, eo_reference)
    raise BenchmarkError(f"Unsupported modality for time reference resolution: {modality}")


def select_parseable_time_reference(primary: str, fallback: str) -> str:
    primary = (primary or "").strip()
    fallback = (fallback or "").strip()
    if primary:
        extract_time_windows(primary)
        return primary
    if fallback:
        extract_time_windows(fallback)
        return fallback
    return ""


def derive_group_score_rules(annotation: Dict[str, object]) -> Dict[Tuple[str, str], str]:
    grouped_types: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for qa in annotation["qa"]:
        grouped_types[(str(qa["group_id"]), normalize_group_focus(qa))].append(qa)

    recommendations = annotation.get("notes", {}).get("group_scoring_recommendation", {})
    rules: Dict[Tuple[str, str], str] = {}
    for key, rows in grouped_types.items():
        group_id, _group_focus = key
        recommendation = str(recommendations.get(group_id, "")).lower()
        group_family = normalize_group_family(rows[0])
        if group_family == "coherence":
            rules[key] = "prefix_ratio"
        elif group_family == "consistency":
            rules[key] = "squared_ratio"
        elif "strict evidence chain" in recommendation or "prefer prefix scoring" in recommendation:
            rules[key] = "prefix_ratio"
        elif "consistency scoring" in recommendation:
            rules[key] = "squared_ratio"
        else:
            rules[key] = "squared_ratio"
    return rules


def compute_group_score(rows: Sequence[Dict[str, object]], score_rule: str) -> float:
    ordered = sorted(rows, key=lambda item: int(item["group_step"]))
    k = len(ordered)
    correct_count = sum(1 for row in ordered if row["is_correct"])
    if score_rule == "squared_ratio":
        return (correct_count / k) ** 2
    if score_rule == "prefix_ratio":
        prefix_correct = 0
        for row in ordered:
            if row["is_correct"]:
                prefix_correct += 1
            else:
                break
        return prefix_correct / k
    raise BenchmarkError(f"Unsupported group score rule: {score_rule}")


@contextmanager
def suppress_stderr() -> Iterator[None]:
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)


def build_sample_key(metadata: MetadataRecord, annotation_path: Path) -> str:
    if metadata.eo_filename:
        stem = Path(metadata.eo_filename).stem
    else:
        stem = annotation_path.stem
    return slugify(stem)


def canonical_key_from_annotation_filename(filename: str) -> str:
    stem = Path(filename).stem.replace("*", " ")
    tokens = [token for token in re.split(r"[ ._\-]+", stem) if token]
    modality_positions = [index for index, token in enumerate(tokens) if token.upper() in {"EO", "IR"}]
    if modality_positions:
        modality_index = modality_positions[-1]
        tokens = tokens[:modality_index] + tokens[modality_index + 1 :]
    return normalize_key(" ".join(tokens))


def normalize_key(text: str) -> str:
    cleaned = re.sub(r"[._\-]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    if cleaned == "tub w barges":
        return "tug w barges"
    return cleaned


def slugify(text: str) -> str:
    text = re.sub(r"[^\w]+", "_", text, flags=re.ASCII)
    text = text.strip("_").lower()
    if not text:
        raise BenchmarkError("Cannot build a stable slug from empty text")
    return text


def require_existing_path(annotation_name: str, modality: str, path_text: Optional[str]) -> str:
    if not path_text:
        raise BenchmarkError(f"{annotation_name} is missing {modality} path in metadata")
    path = Path(path_text)
    if not path.exists():
        raise BenchmarkError(f"{annotation_name} points to missing {modality} path: {path}")
    return str(path.resolve())


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def empty_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def parse_optional_float(value: Optional[str]) -> Optional[float]:
    value = empty_to_none(value)
    if value is None:
        return None
    return float(value)


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
