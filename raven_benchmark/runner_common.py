#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from raven_benchmark import benchmark


def as_dict(value: object, label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise benchmark.BenchmarkError(f"Expected {label} to be a dict, got {type(value).__name__}")
    return value


def optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    benchmark.ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def infer_single_provider(rows: Iterable[Dict[str, object]]) -> str:
    providers = {str(row["provider"]) for row in rows}
    if len(providers) != 1:
        raise benchmark.BenchmarkError(f"Expected a single provider per request file, found: {sorted(providers)}")
    return next(iter(providers))


def extract_video_input_policy(row: Dict[str, object]) -> Dict[str, object]:
    provider_payload = as_dict(row.get("provider_payload", {}), label="provider_payload")
    payload_policy = provider_payload.get("video_input")
    if isinstance(payload_policy, dict):
        return payload_policy
    request = as_dict(row.get("request", {}), label="request")
    request_policy = request.get("video_input")
    if isinstance(request_policy, dict):
        return request_policy
    return {}


def resolve_video_runtime_config(
    args: object,
    row: Dict[str, object],
    frame_transport: str,
    default_fps: float = 1.0,
    default_max_pixels: int = 262144,
) -> Dict[str, object]:
    video_input = extract_video_input_policy(row)
    sampling = as_dict(video_input.get("sampling", {}), label="video_input.sampling") if video_input else {}
    frame_resolution = (
        as_dict(video_input.get("frame_resolution", {}), label="video_input.frame_resolution")
        if video_input
        else {}
    )

    policy_fps = optional_float(sampling.get("fps"))
    policy_max_pixels = optional_int(frame_resolution.get("max_pixels"))
    policy_max_frames = optional_int(sampling.get("max_frames_per_video"))

    cli_fps = getattr(args, "fps", None)
    cli_max_pixels = getattr(args, "max_pixels", None)
    cli_max_frames = getattr(args, "max_frames_per_video", None)

    fps = cli_fps if cli_fps is not None else (policy_fps if policy_fps is not None else default_fps)
    max_pixels = (
        cli_max_pixels
        if cli_max_pixels is not None
        else (policy_max_pixels if policy_max_pixels is not None else default_max_pixels)
    )
    max_frames = cli_max_frames if cli_max_frames is not None else policy_max_frames

    return {
        "source": "cli_override"
        if any(value is not None for value in (cli_fps, cli_max_pixels, cli_max_frames))
        else "request_video_input",
        "protocol_id": str(video_input.get("protocol_id", "legacy_cli_defaults")) if video_input else "legacy_cli_defaults",
        "fps": float(fps),
        "max_pixels": int(max_pixels),
        "max_frames_per_video": max_frames,
        "frame_transport": frame_transport,
    }


def extract_request_video_inputs(request: Dict[str, object]) -> List[Dict[str, object]]:
    raw_inputs = request.get("inputs")
    if not isinstance(raw_inputs, list):
        raise benchmark.BenchmarkError("request.inputs must be a list")
    video_inputs = []
    for raw_input in raw_inputs:
        input_dict = as_dict(raw_input, label="request.inputs[]")
        path = input_dict.get("path")
        if not path:
            raise benchmark.BenchmarkError("request input is missing path")
        video_inputs.append(
            {
                "video_index": int(input_dict.get("video_index", len(video_inputs) + 1)),
                "modality": str(input_dict.get("modality", "video")),
                "path": str(path),
            }
        )
    return video_inputs


def normalize_generation_config(
    generation_config: Dict[str, object],
    max_new_tokens_override: int | None = None,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    max_new_tokens = generation_config.get("max_new_tokens") or generation_config.get("max_output_tokens")
    if max_new_tokens_override is not None:
        if max_new_tokens is None:
            max_new_tokens = max_new_tokens_override
        else:
            max_new_tokens = min(int(max_new_tokens), int(max_new_tokens_override))
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = int(max_new_tokens)

    temperature = generation_config.get("temperature")
    if temperature is None or float(temperature) <= 0.0:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = float(temperature)
    return kwargs


def canonical_json_answer(answer: str) -> str:
    return json.dumps({"answer": answer}, separators=(",", ":"))
