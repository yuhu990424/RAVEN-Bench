#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import re
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "OpenCV (cv2) is required to read video metadata in this environment."
    ) from exc


VIDEO_EXTENSIONS = {".ts", ".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MODALITIES = {"EO", "IR"}
CANONICAL_KEY_ALIASES = {
    # Known filename typo in the source set.
    "tub w barges": "tug w barges",
}


@dataclass
class VideoRecord:
    path: Path
    filename: str
    raw_stem: str
    modality: str
    canonical_key: str
    theme_guess: str
    pairing_method: str = "unpaired"
    file_size_bytes: Optional[int] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None
    bitrate_mbps: Optional[float] = None
    duration_source: str = "unknown"
    metadata_error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input_dir = Path(os.environ.get("VIDEO_DATA_ROOT", script_dir / "data" / "raw_videos"))
    parser = argparse.ArgumentParser(
        description="Extract metadata and EO/IR pairing information for maritime videos."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help=f"Directory containing the source videos. Default: {default_input_dir}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help=f"Directory to write CSV/JSON results. Default: {script_dir}",
    )
    parser.add_argument(
        "--prefix",
        default="maritime_video_metadata",
        help="Filename prefix for generated artifacts.",
    )
    parser.add_argument(
        "--min-fuzzy-ratio",
        type=float,
        default=0.90,
        help="Minimum similarity for fuzzy EO/IR pairing when exact pairing fails.",
    )
    parser.add_argument(
        "--disable-fuzzy-pairing",
        action="store_true",
        help="Disable fuzzy pairing for near-matching EO/IR filenames.",
    )
    parser.add_argument(
        "--duration-mode",
        choices=["auto", "header", "grab"],
        default="auto",
        help=(
            "How to compute duration. "
            "'header' uses container frame_count/fps, "
            "'grab' counts decodable frames, "
            "'auto' falls back to frame-grab when header values look suspicious."
        ),
    )
    parser.add_argument(
        "--max-header-duration-seconds",
        type=float,
        default=1800.0,
        help="In auto mode, header durations above this are treated as suspicious.",
    )
    parser.add_argument(
        "--min-acceptable-bitrate-mbps",
        type=float,
        default=0.5,
        help="In auto mode, header-derived bitrate below this triggers frame-grab fallback.",
    )
    return parser.parse_args()


def title_from_key(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    return " ".join(part.capitalize() for part in cleaned.split())


def normalize_key(text: str) -> str:
    cleaned = re.sub(r"[._-]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return CANONICAL_KEY_ALIASES.get(cleaned, cleaned)


def parse_video_name(stem: str) -> Tuple[str, str, str]:
    tokens = [token for token in re.split(r"[ ._-]+", stem) if token]
    modality_positions = [idx for idx, token in enumerate(tokens) if token.upper() in MODALITIES]
    if not modality_positions:
        raise ValueError(f"Could not find EO/IR token in filename: {stem}")

    modality_index = modality_positions[-1]
    modality = tokens[modality_index].upper()
    title_tokens = tokens[:modality_index] + tokens[modality_index + 1 :]
    canonical_key = normalize_key(" ".join(title_tokens))
    if not canonical_key:
        raise ValueError(f"Unable to build canonical key for filename: {stem}")

    return modality, canonical_key, title_from_key(" ".join(title_tokens))


@contextmanager
def suppress_stderr() -> Iterable[None]:
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)


def should_fallback_to_grab_count(
    duration_mode: str,
    header_duration: Optional[float],
    bitrate_mbps: Optional[float],
    max_header_duration_seconds: float,
    min_acceptable_bitrate_mbps: float,
) -> bool:
    if duration_mode == "grab":
        return True
    if duration_mode == "header":
        return False
    if header_duration is None:
        return True
    if header_duration > max_header_duration_seconds:
        return True
    if bitrate_mbps is not None and bitrate_mbps < min_acceptable_bitrate_mbps:
        return True
    return False


def read_video_metadata(path: Path, args: argparse.Namespace) -> Dict[str, Optional[float]]:
    file_size_bytes = path.stat().st_size
    with suppress_stderr():
        capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        return {
            "file_size_bytes": file_size_bytes,
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
            "duration_seconds": None,
            "bitrate_mbps": None,
            "duration_source": "failed_to_open",
            "metadata_error": "failed_to_open",
        }

    try:
        fps_raw = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count_raw = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width_raw = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        height_raw = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    finally:
        capture.release()

    fps = fps_raw if fps_raw > 0 else None
    header_frame_count = int(round(frame_count_raw)) if frame_count_raw > 0 else None
    width = int(round(width_raw)) if width_raw > 0 else None
    height = int(round(height_raw)) if height_raw > 0 else None

    header_duration = None
    if fps and header_frame_count and fps > 0:
        header_duration = header_frame_count / fps

    bitrate_mbps = None
    if header_duration and header_duration > 0:
        bitrate_mbps = (file_size_bytes * 8) / header_duration / 1_000_000

    use_grab_count = should_fallback_to_grab_count(
        duration_mode=args.duration_mode,
        header_duration=header_duration,
        bitrate_mbps=bitrate_mbps,
        max_header_duration_seconds=args.max_header_duration_seconds,
        min_acceptable_bitrate_mbps=args.min_acceptable_bitrate_mbps,
    )

    frame_count = header_frame_count
    duration_seconds = header_duration
    duration_source = "header"

    if use_grab_count and fps:
        with suppress_stderr():
            capture = cv2.VideoCapture(str(path))
            counted_frames = 0
            while capture.grab():
                counted_frames += 1
            capture.release()
        if counted_frames > 0:
            frame_count = counted_frames
            duration_seconds = counted_frames / fps
            bitrate_mbps = (file_size_bytes * 8) / duration_seconds / 1_000_000
            duration_source = "grab_count"
        else:
            duration_source = "header_after_empty_grab"

    return {
        "file_size_bytes": file_size_bytes,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "bitrate_mbps": bitrate_mbps,
        "duration_source": duration_source,
        "metadata_error": None,
    }


def scan_records(input_dir: Path, args: argparse.Namespace) -> List[VideoRecord]:
    records: List[VideoRecord] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        modality, canonical_key, theme_guess = parse_video_name(path.stem)
        record = VideoRecord(
            path=path,
            filename=path.name,
            raw_stem=path.stem,
            modality=modality,
            canonical_key=canonical_key,
            theme_guess=theme_guess,
        )
        record.__dict__.update(read_video_metadata(path, args))
        records.append(record)
    return records


def group_by_key(records: List[VideoRecord]) -> Dict[str, Dict[str, List[VideoRecord]]]:
    grouped: Dict[str, Dict[str, List[VideoRecord]]] = {}
    for record in records:
        grouped.setdefault(record.canonical_key, {}).setdefault(record.modality, []).append(record)
    for pair in grouped.values():
        for modality_records in pair.values():
            modality_records.sort(key=lambda item: item.filename)
    return grouped


def fuzzy_pair(
    grouped: Dict[str, Dict[str, List[VideoRecord]]],
    min_ratio: float,
) -> List[Dict[str, object]]:
    unmatched_eo = [
        (key, value["EO"][0])
        for key, value in grouped.items()
        if len(value.get("EO", [])) == 1 and len(value.get("IR", [])) == 0
    ]
    unmatched_ir = [
        (key, value["IR"][0])
        for key, value in grouped.items()
        if len(value.get("IR", [])) == 1 and len(value.get("EO", [])) == 0
    ]
    suggestions: List[Tuple[float, str, str]] = []

    for eo_key, _ in unmatched_eo:
        for ir_key, _ in unmatched_ir:
            ratio = SequenceMatcher(None, eo_key, ir_key).ratio()
            if ratio >= min_ratio:
                suggestions.append((ratio, eo_key, ir_key))

    suggestions.sort(reverse=True)
    used_eo = set()
    used_ir = set()
    fuzzy_pairs: List[Dict[str, object]] = []

    for ratio, eo_key, ir_key in suggestions:
        if eo_key in used_eo or ir_key in used_ir:
            continue
        grouped[eo_key]["EO"][0].pairing_method = "fuzzy"
        grouped[ir_key]["IR"][0].pairing_method = "fuzzy"
        fuzzy_pairs.append(
            {
                "pair_key": eo_key,
                "paired_with_key": ir_key,
                "similarity": round(ratio, 6),
                "eo_filename": grouped[eo_key]["EO"][0].filename,
                "ir_filename": grouped[ir_key]["IR"][0].filename,
                "theme_guess": grouped[eo_key]["EO"][0].theme_guess,
            }
        )
        used_eo.add(eo_key)
        used_ir.add(ir_key)

    return fuzzy_pairs


def mark_exact_pairs(grouped: Dict[str, Dict[str, List[VideoRecord]]]) -> None:
    for pair in grouped.values():
        eo_records = pair.get("EO", [])
        ir_records = pair.get("IR", [])
        exact_count = min(len(eo_records), len(ir_records))
        for index in range(exact_count):
            eo_records[index].pairing_method = "exact"
            ir_records[index].pairing_method = "exact"


def safe_round(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None or math.isnan(value):
        return None
    return round(value, digits)


def stat_block(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "total_seconds": None,
            "mean_seconds": None,
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }

    return {
        "count": len(values),
        "total_seconds": safe_round(sum(values)),
        "mean_seconds": safe_round(statistics.mean(values)),
        "median_seconds": safe_round(statistics.median(values)),
        "min_seconds": safe_round(min(values)),
        "max_seconds": safe_round(max(values)),
    }


def build_row(
    pair_key: str,
    theme_guess: str,
    pairing_method: str,
    eo: Optional[VideoRecord],
    ir: Optional[VideoRecord],
    pair_instance_index: int = 1,
) -> Dict[str, object]:
    duration_delta = None
    if eo and ir and eo.duration_seconds is not None and ir.duration_seconds is not None:
        duration_delta = abs(eo.duration_seconds - ir.duration_seconds)

    return {
        "pair_key": pair_key,
        "pair_instance_index": pair_instance_index,
        "theme_guess": theme_guess,
        "pairing_method": pairing_method,
        "eo_filename": eo.filename if eo else None,
        "ir_filename": ir.filename if ir else None,
        "eo_duration_seconds": safe_round(eo.duration_seconds if eo else None),
        "ir_duration_seconds": safe_round(ir.duration_seconds if ir else None),
        "duration_delta_seconds": safe_round(duration_delta),
        "eo_fps": safe_round(eo.fps if eo else None),
        "ir_fps": safe_round(ir.fps if ir else None),
        "eo_frame_count": eo.frame_count if eo else None,
        "ir_frame_count": ir.frame_count if ir else None,
        "eo_resolution": f"{eo.width}x{eo.height}" if eo and eo.width and eo.height else None,
        "ir_resolution": f"{ir.width}x{ir.height}" if ir and ir.width and ir.height else None,
        "eo_path": str(eo.path) if eo else None,
        "ir_path": str(ir.path) if ir else None,
        "eo_file_size_bytes": eo.file_size_bytes if eo else None,
        "ir_file_size_bytes": ir.file_size_bytes if ir else None,
        "eo_bitrate_mbps": safe_round(eo.bitrate_mbps if eo else None),
        "ir_bitrate_mbps": safe_round(ir.bitrate_mbps if ir else None),
        "eo_duration_source": eo.duration_source if eo else None,
        "ir_duration_source": ir.duration_source if ir else None,
        "eo_metadata_error": eo.metadata_error if eo else None,
        "ir_metadata_error": ir.metadata_error if ir else None,
    }


def build_pair_rows(
    grouped: Dict[str, Dict[str, List[VideoRecord]]],
    fuzzy_pairs: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    fuzzy_map = {item["pair_key"]: item for item in fuzzy_pairs}
    inverse_fuzzy_map = {item["paired_with_key"]: item for item in fuzzy_pairs}
    rows: List[Dict[str, object]] = []

    handled_keys = set()

    for key in sorted(grouped):
        if key in handled_keys:
            continue

        pair = grouped[key]
        eo_records = pair.get("EO", [])
        ir_records = pair.get("IR", [])
        pair_key = key
        theme_guess = title_from_key(key)
        exact_count = min(len(eo_records), len(ir_records))
        for index in range(exact_count):
            rows.append(
                build_row(
                    pair_key=pair_key,
                    theme_guess=theme_guess,
                    pairing_method=eo_records[index].pairing_method,
                    eo=eo_records[index],
                    ir=ir_records[index],
                    pair_instance_index=index + 1,
                )
            )

        eo_leftovers = eo_records[exact_count:]
        ir_leftovers = ir_records[exact_count:]

        if eo_leftovers or ir_leftovers:
            if key in fuzzy_map:
                eo = eo_leftovers[0] if eo_leftovers else None
                info = fuzzy_map[key]
                ir_candidates = grouped[info["paired_with_key"]]["IR"]
                ir = ir_candidates[0] if ir_candidates else None
                rows.append(
                    build_row(
                        pair_key=pair_key,
                        theme_guess=info["theme_guess"],
                        pairing_method="fuzzy",
                        eo=eo,
                        ir=ir,
                        pair_instance_index=exact_count + 1,
                    )
                )
                handled_keys.add(info["paired_with_key"])
            elif key in inverse_fuzzy_map:
                continue
            else:
                for offset, eo in enumerate(eo_leftovers, start=exact_count + 1):
                    rows.append(
                        build_row(
                            pair_key=pair_key,
                            theme_guess=theme_guess,
                            pairing_method="unpaired",
                            eo=eo,
                            ir=None,
                            pair_instance_index=offset,
                        )
                    )
                for offset, ir in enumerate(ir_leftovers, start=exact_count + 1):
                    rows.append(
                        build_row(
                            pair_key=pair_key,
                            theme_guess=theme_guess,
                            pairing_method="unpaired",
                            eo=None,
                            ir=ir,
                            pair_instance_index=offset,
                        )
                    )
        handled_keys.add(key)

    return rows


def build_summary(
    records: List[VideoRecord],
    pair_rows: List[Dict[str, object]],
    fuzzy_pairs: List[Dict[str, object]],
    input_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    eo_durations = [record.duration_seconds for record in records if record.modality == "EO" and record.duration_seconds is not None]
    ir_durations = [record.duration_seconds for record in records if record.modality == "IR" and record.duration_seconds is not None]
    paired_deltas = [
        row["duration_delta_seconds"]
        for row in pair_rows
        if row["pairing_method"] in {"exact", "fuzzy"} and row["duration_delta_seconds"] is not None
    ]
    unresolved = [
        row
        for row in pair_rows
        if row["pairing_method"] == "unpaired"
    ]
    duration_source_counts: Dict[str, int] = {}
    for record in records:
        duration_source_counts[record.duration_source] = duration_source_counts.get(record.duration_source, 0) + 1

    return {
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "total_video_files": len(records),
        "eo_files": sum(1 for record in records if record.modality == "EO"),
        "ir_files": sum(1 for record in records if record.modality == "IR"),
        "paired_groups": sum(1 for row in pair_rows if row["pairing_method"] in {"exact", "fuzzy"}),
        "exact_pairs": sum(1 for row in pair_rows if row["pairing_method"] == "exact"),
        "fuzzy_pairs": len(fuzzy_pairs),
        "unpaired_groups": len(unresolved),
        "eo_duration_stats": stat_block(eo_durations),
        "ir_duration_stats": stat_block(ir_durations),
        "pair_duration_delta_stats": stat_block([float(value) for value in paired_deltas]),
        "duration_source_counts": duration_source_counts,
        "unpaired_items": unresolved,
        "fuzzy_pair_details": fuzzy_pairs,
    }


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "pair_key",
        "pair_instance_index",
        "theme_guess",
        "pairing_method",
        "eo_filename",
        "ir_filename",
        "eo_duration_seconds",
        "ir_duration_seconds",
        "duration_delta_seconds",
        "eo_fps",
        "ir_fps",
        "eo_frame_count",
        "ir_frame_count",
        "eo_resolution",
        "ir_resolution",
        "eo_path",
        "ir_path",
        "eo_file_size_bytes",
        "ir_file_size_bytes",
        "eo_bitrate_mbps",
        "ir_bitrate_mbps",
        "eo_duration_source",
        "ir_duration_source",
        "eo_metadata_error",
        "ir_metadata_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = scan_records(args.input_dir, args)
    grouped = group_by_key(records)
    mark_exact_pairs(grouped)

    fuzzy_pairs: List[Dict[str, object]] = []
    if not args.disable_fuzzy_pairing:
        fuzzy_pairs = fuzzy_pair(grouped, min_ratio=args.min_fuzzy_ratio)

    pair_rows = build_pair_rows(grouped, fuzzy_pairs)
    summary = build_summary(records, pair_rows, fuzzy_pairs, args.input_dir, args.output_dir)

    csv_path = args.output_dir / f"{args.prefix}.csv"
    json_path = args.output_dir / f"{args.prefix}.json"
    summary_path = args.output_dir / f"{args.prefix}_summary.json"

    write_csv(pair_rows, csv_path)
    write_json(pair_rows, json_path)
    write_json(summary, summary_path)

    print(f"Wrote pair metadata: {csv_path}")
    print(f"Wrote pair metadata JSON: {json_path}")
    print(f"Wrote summary JSON: {summary_path}")
    print(
        "Summary:",
        json.dumps(
            {
                "total_video_files": summary["total_video_files"],
                "paired_groups": summary["paired_groups"],
                "exact_pairs": summary["exact_pairs"],
                "fuzzy_pairs": summary["fuzzy_pairs"],
                "unpaired_groups": summary["unpaired_groups"],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
