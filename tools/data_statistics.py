#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import contextlib
import csv
import os
import json
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_VIDEO_ROOT = Path(os.environ.get("VIDEO_DATA_ROOT", ROOT_DIR / "videos"))


@dataclass(frozen=True)
class VideoInfo:
    path: str
    exists: bool
    duration_seconds: float | None
    fps: float | None
    frame_count: int | None
    resolution: str | None
    bitrate_mbps: float | None
    source: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate annotation-driven data statistics for the 48-sample EO/IR benchmark."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT_DIR / "data" / "annotations",
        help="Directory containing annotation JSON files.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help="Directory containing raw EO/IR .ts videos.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT_DIR / "data" / "metadata" / "video_metadata.example.csv",
        help="Metadata CSV used only for filename mapping exceptions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "eval_outputs" / "data_statistics",
        help="Directory where summary.json, paper_tables.md, paper_tables.csv, and figures are written.",
    )
    return parser.parse_args()


def load_annotations(annotation_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(annotation_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["_annotation_file"] = str(path)
        records.append(data)
    return records


def normalize_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in value).split()


def normalize_key(value: str) -> str:
    return " ".join(normalize_name(value))


def load_metadata_by_theme(metadata_path: Path) -> dict[str, dict[str, str]]:
    if not metadata_path.exists():
        return {}

    by_theme: dict[str, dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            theme = row.get("theme_guess", "")
            if not theme:
                continue
            key = normalize_key(theme)
            # Keep instance 1 for duplicated themes, especially Parked Car.
            if key not in by_theme or row.get("pair_instance_index") == "1":
                by_theme[key] = row
    return by_theme


def resolve_video_pair(
    sample_id: str,
    video_root: Path,
    metadata_by_theme: dict[str, dict[str, str]],
) -> dict[str, Any]:
    eo_direct = video_root / f"{sample_id}_EO.ts"
    ir_direct = video_root / f"{sample_id}_IR.ts"
    if eo_direct.exists() and ir_direct.exists():
        return {
            "mapping_source": "direct",
            "eo_path": eo_direct,
            "ir_path": ir_direct,
            "eo_filename": eo_direct.name,
            "ir_filename": ir_direct.name,
            "metadata_row": metadata_by_theme.get(normalize_key(sample_id)),
            "missing": [],
        }

    row = metadata_by_theme.get(normalize_key(sample_id))
    if row:
        eo_path = video_root / row["eo_filename"]
        ir_path = video_root / row["ir_filename"]
        missing = [p.name for p in (eo_path, ir_path) if not p.exists()]
        return {
            "mapping_source": "metadata",
            "eo_path": eo_path,
            "ir_path": ir_path,
            "eo_filename": row["eo_filename"],
            "ir_filename": row["ir_filename"],
            "metadata_row": row,
            "missing": missing,
        }

    return {
        "mapping_source": "unresolved",
        "eo_path": eo_direct,
        "ir_path": ir_direct,
        "eo_filename": eo_direct.name,
        "ir_filename": ir_direct.name,
        "metadata_row": None,
        "missing": [p.name for p in (eo_direct, ir_direct) if not p.exists()],
    }


def run_ffprobe(path: Path) -> VideoInfo | None:
    if shutil.which("ffprobe") is None:
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,bit_rate,duration:format=duration,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [{}])[0]
        fmt = payload.get("format") or {}
        width = stream.get("width")
        height = stream.get("height")
        duration = parse_float(stream.get("duration")) or parse_float(fmt.get("duration"))
        fps = parse_fps(stream.get("avg_frame_rate")) or parse_fps(stream.get("r_frame_rate"))
        frame_count = parse_int(stream.get("nb_frames"))
        bit_rate = parse_float(stream.get("bit_rate")) or parse_float(fmt.get("bit_rate"))
        return VideoInfo(
            path=str(path),
            exists=True,
            duration_seconds=duration,
            fps=fps,
            frame_count=frame_count,
            resolution=f"{width}x{height}" if width and height else None,
            bitrate_mbps=round(bit_rate / 1_000_000, 6) if bit_rate else None,
            source="ffprobe",
        )
    except Exception as exc:  # pragma: no cover - defensive path for local tools.
        return VideoInfo(
            path=str(path),
            exists=True,
            duration_seconds=None,
            fps=None,
            frame_count=None,
            resolution=None,
            bitrate_mbps=None,
            source="ffprobe_error",
            error=str(exc),
        )


def run_opencv(path: Path) -> VideoInfo:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return VideoInfo(
            path=str(path),
            exists=path.exists(),
            duration_seconds=None,
            fps=None,
            frame_count=None,
            resolution=None,
            bitrate_mbps=None,
            source="opencv_unavailable",
            error=str(exc),
        )

    with suppress_native_stderr():
        capture = cv2.VideoCapture(str(path))
        opened = capture.isOpened()
        if opened:
            fps = capture.get(cv2.CAP_PROP_FPS) or None
            frame_count_raw = capture.get(cv2.CAP_PROP_FRAME_COUNT) or None
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) or None
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or None
        capture.release()

    if not opened:
        return VideoInfo(
            path=str(path),
            exists=path.exists(),
            duration_seconds=None,
            fps=None,
            frame_count=None,
            resolution=None,
            bitrate_mbps=None,
            source="opencv_error",
            error="cv2.VideoCapture could not open video",
        )

    frame_count = int(round(frame_count_raw)) if frame_count_raw is not None else None
    duration = frame_count / fps if frame_count and fps else None
    file_size = path.stat().st_size if path.exists() else None
    bitrate = (file_size * 8 / duration / 1_000_000) if file_size and duration else None
    return VideoInfo(
        path=str(path),
        exists=path.exists(),
        duration_seconds=duration,
        fps=fps,
        frame_count=frame_count,
        resolution=f"{int(width)}x{int(height)}" if width and height else None,
        bitrate_mbps=bitrate,
        source="opencv",
    )


def run_metadata_csv(path: Path, metadata_row: dict[str, str] | None, modality: str) -> VideoInfo | None:
    if metadata_row is None:
        return None

    duration = parse_float(metadata_row.get(f"{modality}_duration_seconds"))
    fps = parse_float(metadata_row.get(f"{modality}_fps"))
    frame_count = parse_int(metadata_row.get(f"{modality}_frame_count"))
    resolution = metadata_row.get(f"{modality}_resolution") or None
    bitrate = parse_float(metadata_row.get(f"{modality}_bitrate_mbps"))
    if duration is None and fps is None and frame_count is None and resolution is None and bitrate is None:
        return None

    return VideoInfo(
        path=str(path),
        exists=path.exists(),
        duration_seconds=duration,
        fps=fps,
        frame_count=frame_count,
        resolution=resolution,
        bitrate_mbps=bitrate,
        source="metadata_csv",
    )


@contextlib.contextmanager
def suppress_native_stderr() -> Any:
    """Silence FFmpeg/OpenCV C-level warnings while reading transport streams."""
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def inspect_video(path: Path, metadata_row: dict[str, str] | None, modality: str) -> VideoInfo:
    if not path.exists():
        return VideoInfo(
            path=str(path),
            exists=False,
            duration_seconds=None,
            fps=None,
            frame_count=None,
            resolution=None,
            bitrate_mbps=None,
            source="missing",
            error="file does not exist",
        )
    return run_ffprobe(path) or run_metadata_csv(path, metadata_row, modality) or run_opencv(path)


def parse_float(value: Any) -> float | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_fps(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    if isinstance(value, str) and "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_float = parse_float(denominator)
        if not denominator_float:
            return None
        numerator_float = parse_float(numerator)
        return numerator_float / denominator_float if numerator_float is not None else None
    return parse_float(value)


def summarize_numbers(values: list[float]) -> dict[str, float | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"sum": None, "mean": None, "median": None, "min": None, "max": None}
    return {
        "sum": round(sum(clean), 3),
        "mean": round(statistics.mean(clean), 3),
        "median": round(statistics.median(clean), 3),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
    }


def counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def infer_target_category(sample_id: str, sample_type: str) -> str:
    maritime_terms = (
        "boat",
        "barg",
        "barge",
        "ferry",
        "freighter",
        "houseboat",
        "maritime",
        "naval",
        "powerboat",
        "sailboat",
        "ship",
        "tanker",
        "tug",
        "yacht",
    )
    aircraft_terms = (
        "air canada",
        "aircraft",
        "airliner",
        "airplane",
        "helicopter",
        "plane",
        "seaplane",
    )
    text = normalize_key(f"{sample_id} {sample_type}")
    if any(term in text for term in maritime_terms):
        return "maritime_vessel"
    if any(term in text for term in aircraft_terms):
        return "aircraft"
    return "ground_vehicle"


def percent(count: int, total: int) -> float:
    return round(count * 100 / total, 2) if total else 0.0


def build_statistics(
    annotations: list[dict[str, Any]],
    video_root: Path,
    metadata_by_theme: dict[str, dict[str, str]],
) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    excluded_video_pairs: list[dict[str, str]] = []

    type_counter: Counter[str] = Counter()
    target_category_counter: Counter[str] = Counter()
    sensing_context_counter: Counter[str] = Counter()
    scenario_counter: Counter[str] = Counter()
    modality_counter: Counter[str] = Counter()
    capability_counter: Counter[str] = Counter()
    capability_pattern_counter: Counter[str] = Counter()
    capability_pattern_examples: dict[str, str] = {}
    group_family_counter: Counter[str] = Counter()
    group_focus_counter: Counter[str] = Counter()
    question_type_counter: Counter[str] = Counter()
    qa_per_sample_counter: Counter[str] = Counter()
    time_coverage_counter: Counter[str] = Counter()

    eo_durations: list[float] = []
    ir_durations: list[float] = []
    duration_deltas: list[float] = []
    eo_fps_values: list[float] = []
    ir_fps_values: list[float] = []
    eo_frame_counts: list[float] = []
    ir_frame_counts: list[float] = []
    eo_bitrates: list[float] = []
    ir_bitrates: list[float] = []
    resolution_counter: Counter[str] = Counter()
    duration_source_counter: Counter[str] = Counter()

    total_questions = 0
    missing_video_pairs = 0
    mapped_samples = 0

    for annotation in annotations:
        sample_id = annotation.get("sample_id") or Path(annotation["_annotation_file"]).stem
        qa_items = annotation.get("qa") or []
        sample_type = annotation.get("type") or "<missing>"
        type_counter[sample_type] += 1
        target_category = infer_target_category(sample_id, sample_type)
        target_category_counter[target_category] += 1
        sensing_context_counter[sample_type.split("_", 1)[0]] += 1
        scenario_counter[annotation.get("scenario_type") or "<missing>"] += 1
        qa_per_sample_counter[str(len(qa_items))] += 1
        total_questions += len(qa_items)

        mapping = resolve_video_pair(sample_id, video_root, metadata_by_theme)
        eo_info = inspect_video(mapping["eo_path"], mapping["metadata_row"], "eo")
        ir_info = inspect_video(mapping["ir_path"], mapping["metadata_row"], "ir")
        if eo_info.exists and ir_info.exists:
            mapped_samples += 1
        else:
            missing_video_pairs += 1
            validation_errors.append(f"{sample_id}: missing videos {mapping['missing']}")

        for label, info in (("eo", eo_info), ("ir", ir_info)):
            duration_source_counter[f"{label}:{info.source}"] += 1
            if info.resolution:
                resolution_counter[f"{label}:{info.resolution}"] += 1

        if eo_info.duration_seconds is not None:
            eo_durations.append(eo_info.duration_seconds)
        if ir_info.duration_seconds is not None:
            ir_durations.append(ir_info.duration_seconds)
        if eo_info.duration_seconds is not None and ir_info.duration_seconds is not None:
            duration_deltas.append(abs(eo_info.duration_seconds - ir_info.duration_seconds))
        if eo_info.fps is not None:
            eo_fps_values.append(eo_info.fps)
        if ir_info.fps is not None:
            ir_fps_values.append(ir_info.fps)
        if eo_info.frame_count is not None:
            eo_frame_counts.append(float(eo_info.frame_count))
        if ir_info.frame_count is not None:
            ir_frame_counts.append(float(ir_info.frame_count))
        if eo_info.bitrate_mbps is not None:
            eo_bitrates.append(eo_info.bitrate_mbps)
        if ir_info.bitrate_mbps is not None:
            ir_bitrates.append(ir_info.bitrate_mbps)

        for question in qa_items:
            uid = question.get("uid", "<missing>")
            answer = question.get("answer")
            options = question.get("options") or {}
            if answer not in options:
                validation_errors.append(f"{sample_id} uid={uid}: answer {answer!r} not in options")
            if not question.get("question"):
                validation_errors.append(f"{sample_id} uid={uid}: empty question")
            if len(options) != 4 or any(not value for value in options.values()):
                validation_errors.append(f"{sample_id} uid={uid}: empty or incomplete options")

            modality = question.get("modality_requirement") or "<missing>"
            modality_counter[modality] += 1
            capability_counter[question.get("capability_level") or "<missing>"] += 1
            group_family_counter[question.get("group_family") or "<missing>"] += 1
            group_focus_counter[question.get("group_focus") or "<missing>"] += 1
            for question_type in question.get("question_type") or ["<missing>"]:
                question_type_counter[question_type] += 1

            has_eo_time = bool(question.get("time_reference_eo"))
            has_ir_time = bool(question.get("time_reference_ir"))
            if has_eo_time and has_ir_time:
                time_coverage_counter["both_eo_ir"] += 1
            elif has_eo_time:
                time_coverage_counter["eo_only"] += 1
            elif has_ir_time:
                time_coverage_counter["ir_only"] += 1
            else:
                time_coverage_counter["missing_both"] += 1

        sample_capability_counter = Counter(q.get("capability_level") or "<missing>" for q in qa_items)
        pattern = (
            f"L1={sample_capability_counter.get('L1', 0)}, "
            f"L2={sample_capability_counter.get('L2', 0)}, "
            f"L3={sample_capability_counter.get('L3', 0)}"
        )
        capability_pattern_counter[pattern] += 1
        capability_pattern_examples.setdefault(pattern, sample_id)

        sample_rows.append(
            {
                "sample_id": sample_id,
                "annotation_file": annotation["_annotation_file"],
                "type": sample_type,
                "target_category": target_category,
                "sensing_context": sample_type.split("_", 1)[0],
                "scenario_type": annotation.get("scenario_type"),
                "qa_count": len(qa_items),
                "mapping_source": mapping["mapping_source"],
                "eo_filename": mapping["eo_filename"],
                "ir_filename": mapping["ir_filename"],
                "eo_video": eo_info.__dict__,
                "ir_video": ir_info.__dict__,
            }
        )

    parked_lower = video_root / "Parked.car_EO.ts"
    parked_lower_ir = video_root / "Parked.car_IR.ts"
    if parked_lower.exists() or parked_lower_ir.exists():
        excluded_video_pairs.append(
            {
                "reason": "duplicate Parked Car metadata instance excluded by 48-annotation-sample policy",
                "eo_filename": parked_lower.name,
                "ir_filename": parked_lower_ir.name,
            }
        )

    if any(row["qa_count"] != 12 for row in sample_rows):
        bad = [(row["sample_id"], row["qa_count"]) for row in sample_rows if row["qa_count"] != 12]
        validation_errors.append(f"samples with qa_count != 12: {bad}")

    return {
        "dataset_policy": {
            "counting_unit": "annotation_sample",
            "expected_samples": 48,
            "metadata_duplicate_video_pairs_excluded": True,
            "video_root": str(video_root),
        },
        "overview": {
            "annotated_samples": len(annotations),
            "mapped_samples": mapped_samples,
            "missing_video_pairs": missing_video_pairs,
            "total_questions": total_questions,
            "qa_per_sample_distribution": counter_to_dict(qa_per_sample_counter),
            "eo_duration_seconds": summarize_numbers(eo_durations),
            "ir_duration_seconds": summarize_numbers(ir_durations),
            "eo_plus_ir_duration_seconds": round(sum(eo_durations) + sum(ir_durations), 3),
            "duration_delta_seconds": summarize_numbers(duration_deltas),
        },
        "video_properties": {
            "eo_fps": summarize_numbers(eo_fps_values),
            "ir_fps": summarize_numbers(ir_fps_values),
            "eo_frame_count": summarize_numbers(eo_frame_counts),
            "ir_frame_count": summarize_numbers(ir_frame_counts),
            "eo_bitrate_mbps": summarize_numbers(eo_bitrates),
            "ir_bitrate_mbps": summarize_numbers(ir_bitrates),
            "resolution_distribution": counter_to_dict(resolution_counter),
            "metadata_source_distribution": counter_to_dict(duration_source_counter),
        },
        "category_statistics": {
            "target_category": counter_to_dict(target_category_counter),
            "sensing_context": counter_to_dict(sensing_context_counter),
            "type": counter_to_dict(type_counter),
        },
        "annotation_statistics": {
            "scenario_type": counter_to_dict(scenario_counter),
            "modality_requirement": counter_to_dict(modality_counter),
            "capability_level": counter_to_dict(capability_counter),
            "capability_level_per_sample_pattern": counter_to_dict(capability_pattern_counter),
            "capability_level_pattern_examples": capability_pattern_examples,
            "group_family": counter_to_dict(group_family_counter),
            "group_focus": counter_to_dict(group_focus_counter),
        },
        "qa_reasoning_statistics": {
            "question_type": counter_to_dict(question_type_counter),
        },
        "temporal_annotation_coverage": {
            "counts": counter_to_dict(time_coverage_counter),
            "percentages": {
                key: percent(value, total_questions)
                for key, value in counter_to_dict(time_coverage_counter).items()
            },
        },
        "consistency_checks": {
            "passed": not validation_errors
            and len(annotations) == 48
            and mapped_samples == 48
            and missing_video_pairs == 0
            and excluded_video_pairs
            == [
                {
                    "reason": "duplicate Parked Car metadata instance excluded by 48-annotation-sample policy",
                    "eo_filename": "Parked.car_EO.ts",
                    "ir_filename": "Parked.car_IR.ts",
                }
            ],
            "validation_errors": validation_errors,
            "excluded_video_pairs": excluded_video_pairs,
        },
        "samples": sample_rows,
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def write_markdown(summary: dict[str, Any], output_dir: Path) -> None:
    overview = summary["overview"]
    lines = [
        "# Data Statistics",
        "",
        "Counting unit: 48 annotated EO/IR samples. The duplicated `Parked.car_*` video pair is excluded.",
        "",
        "## Dataset Overview",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Annotated samples", overview["annotated_samples"]],
                ["Mapped EO/IR sample pairs", overview["mapped_samples"]],
                ["Missing video pairs", overview["missing_video_pairs"]],
                ["Total questions", overview["total_questions"]],
                ["EO total duration (s)", overview["eo_duration_seconds"]["sum"]],
                ["IR total duration (s)", overview["ir_duration_seconds"]["sum"]],
                ["EO+IR total duration (s)", overview["eo_plus_ir_duration_seconds"]],
                ["EO mean duration (s)", overview["eo_duration_seconds"]["mean"]],
                ["IR mean duration (s)", overview["ir_duration_seconds"]["mean"]],
                ["Median duration delta (s)", overview["duration_delta_seconds"]["median"]],
            ],
        ),
        "",
        "## High-Level Target Category Distribution",
        counter_markdown(summary["category_statistics"]["target_category"], overview["annotated_samples"]),
        "",
        "## Sensing Context Distribution",
        counter_markdown(summary["category_statistics"]["sensing_context"], overview["annotated_samples"]),
        "",
        "## Fine-Grained Type Distribution",
        counter_markdown(summary["category_statistics"]["type"], overview["annotated_samples"]),
        "",
        "## Modality Requirement",
        counter_markdown(summary["annotation_statistics"]["modality_requirement"], overview["total_questions"]),
        "",
        "## Capability Level",
        counter_markdown(summary["annotation_statistics"]["capability_level"], overview["total_questions"]),
        "",
        "## Capability Level Per-Sample Pattern",
        markdown_table(
            ["Pattern", "Samples", "Example sample"],
            [
                [
                    pattern,
                    count,
                    summary["annotation_statistics"]["capability_level_pattern_examples"].get(pattern, ""),
                ]
                for pattern, count in summary["annotation_statistics"][
                    "capability_level_per_sample_pattern"
                ].items()
            ],
        ),
        "",
        "## Question Type",
        counter_markdown(summary["qa_reasoning_statistics"]["question_type"], overview["total_questions"]),
        "",
        "## Temporal Annotation Coverage",
        markdown_table(
            ["Coverage", "Count", "Percent"],
            [
                [key, value, summary["temporal_annotation_coverage"]["percentages"][key]]
                for key, value in summary["temporal_annotation_coverage"]["counts"].items()
            ],
        ),
        "",
        "## Consistency Checks",
        markdown_table(
            ["Check", "Value"],
            [
                ["Passed", summary["consistency_checks"]["passed"]],
                ["Validation errors", len(summary["consistency_checks"]["validation_errors"])],
                [
                    "Excluded video pair",
                    ", ".join(
                        f"{item['eo_filename']} / {item['ir_filename']}"
                        for item in summary["consistency_checks"]["excluded_video_pairs"]
                    ),
                ],
            ],
        ),
        "",
    ]
    (output_dir / "paper_tables.md").write_text("\n".join(lines), encoding="utf-8")


def counter_markdown(counter: dict[str, int], total: int) -> str:
    return markdown_table(
        ["Item", "Count", "Percent"],
        [[key, value, percent(value, total)] for key, value in counter.items()],
    )


def write_csv_tables(summary: dict[str, Any], output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []

    def add_counter(table: str, counter: dict[str, int], total: int) -> None:
        for key, value in counter.items():
            rows.append(
                {
                    "table": table,
                    "item": key,
                    "count": value,
                    "percent": percent(value, total),
                    "value": "",
                }
            )

    overview = summary["overview"]
    for key, value in flatten_dict("overview", overview).items():
        rows.append({"table": "overview", "item": key, "count": "", "percent": "", "value": value})

    add_counter("target_category", summary["category_statistics"]["target_category"], overview["annotated_samples"])
    add_counter("sensing_context", summary["category_statistics"]["sensing_context"], overview["annotated_samples"])
    add_counter("type", summary["category_statistics"]["type"], overview["annotated_samples"])
    add_counter("modality_requirement", summary["annotation_statistics"]["modality_requirement"], overview["total_questions"])
    add_counter("capability_level", summary["annotation_statistics"]["capability_level"], overview["total_questions"])
    add_counter(
        "capability_level_per_sample_pattern",
        summary["annotation_statistics"]["capability_level_per_sample_pattern"],
        overview["annotated_samples"],
    )
    add_counter("question_type", summary["qa_reasoning_statistics"]["question_type"], overview["total_questions"])
    add_counter("temporal_coverage", summary["temporal_annotation_coverage"]["counts"], overview["total_questions"])

    with (output_dir / "paper_tables.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["table", "item", "count", "percent", "value"])
        writer.writeheader()
        writer.writerows(rows)


def flatten_dict(prefix: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        flattened.update(flatten_dict(f"{prefix}.{key}", child))
    return flattened


def write_figures(summary: dict[str, Any], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    plots = [
        ("category_distribution.png", summary["category_statistics"]["target_category"], "Target Category Distribution"),
        (
            "modality_requirement_distribution.png",
            summary["annotation_statistics"]["modality_requirement"],
            "Modality Requirement",
        ),
        ("capability_level_distribution.png", summary["annotation_statistics"]["capability_level"], "Capability Level"),
        ("question_type_distribution.png", summary["qa_reasoning_statistics"]["question_type"], "Question Type"),
    ]
    for filename, counter, title in plots:
        path = figures_dir / filename
        plot_bar(counter, title, path, plt)
        written.append(str(path))
    return written


def plot_bar(counter: dict[str, int], title: str, path: Path, plt: Any) -> None:
    labels = list(counter.keys())
    values = list(counter.values())
    width = max(8, min(18, len(labels) * 0.5))
    plt.figure(figsize=(width, 5))
    plt.bar(range(len(labels)), values, color="#4c78a8")
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    metadata_by_theme = load_metadata_by_theme(args.metadata)
    summary = build_statistics(annotations, args.video_root, metadata_by_theme)
    write_summary(summary, args.output_dir)
    write_markdown(summary, args.output_dir)
    write_csv_tables(summary, args.output_dir)
    figures = write_figures(summary, args.output_dir)
    summary["generated_files"] = {
        "summary_json": str(args.output_dir / "summary.json"),
        "paper_tables_md": str(args.output_dir / "paper_tables.md"),
        "paper_tables_csv": str(args.output_dir / "paper_tables.csv"),
        "figures": figures,
    }
    write_summary(summary, args.output_dir)

    checks = summary["consistency_checks"]
    print(f"annotated_samples={summary['overview']['annotated_samples']}")
    print(f"mapped_samples={summary['overview']['mapped_samples']}")
    print(f"missing_video_pairs={summary['overview']['missing_video_pairs']}")
    print(f"total_questions={summary['overview']['total_questions']}")
    print(f"consistency_passed={checks['passed']}")
    if checks["validation_errors"]:
        print("validation_errors:")
        for error in checks["validation_errors"]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
