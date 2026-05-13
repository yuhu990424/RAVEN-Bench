#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from raven_benchmark import benchmark


DEFAULT_FORBIDDEN_KEY_TERMS = (
    "correct_answer",
    "source_correct_answer",
    "rationale",
    "event_description",
    "time_reference",
    "evidence_note",
    "option_roles",
)
DEFAULT_FORBIDDEN_PROMPT_TERMS = (
    "correct answer",
    "gold answer",
    "correct_answer",
    "source_correct",
    "rationale",
    "event description",
    "time reference",
    "evidence note",
    "shortcut",
)
DEFAULT_PROTECTED_PREFIXES = (
    "5th_wheel",
    "airplane",
    "air_canada_airliner",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exported model_requests for answer leakage.")
    parser.add_argument("--model-requests-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--protected-prefixes",
        nargs="*",
        default=list(DEFAULT_PROTECTED_PREFIXES),
        help="Question-id prefixes allowed to contain known frozen wording such as 'shortcut'.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_requests_dir.resolve() / args.model_id
    rows = load_model_request_rows(model_dir)
    key_hits: List[Dict[str, object]] = []
    prompt_hits: List[Dict[str, object]] = []
    protocol_mismatches: List[Dict[str, object]] = []

    for setting, row in rows:
        question_id = str(as_dict(row["request"], "request")["question_id"])
        for path, value in iter_key_paths(row):
            path_text = ".".join(path)
            if is_allowed_leakage_policy_path(path):
                continue
            lowered_path = path_text.lower()
            for term in DEFAULT_FORBIDDEN_KEY_TERMS:
                if term in lowered_path:
                    key_hits.append(
                        {
                            "setting": setting,
                            "question_id": question_id,
                            "term": term,
                            "path": path_text,
                            "value_preview": preview(value),
                        }
                    )

        for prompt_name, prompt in iter_prompts(row):
            lowered_prompt = prompt.lower()
            for term in DEFAULT_FORBIDDEN_PROMPT_TERMS:
                if term in lowered_prompt:
                    prompt_hits.append(
                        {
                            "setting": setting,
                            "question_id": question_id,
                            "prompt": prompt_name,
                            "term": term,
                            "protected": is_protected_question(question_id, args.protected_prefixes),
                        }
                    )

        protocol = extract_video_input(row)
        sampling = protocol.get("sampling", {}) if isinstance(protocol, dict) else {}
        resolution = protocol.get("frame_resolution", {}) if isinstance(protocol, dict) else {}
        if sampling.get("fps") != 1.0 or resolution.get("max_pixels") != 262144:
            protocol_mismatches.append(
                {
                    "setting": setting,
                    "question_id": question_id,
                    "fps": sampling.get("fps"),
                    "max_pixels": resolution.get("max_pixels"),
                }
            )

    nonprotected_prompt_hits = [hit for hit in prompt_hits if not hit["protected"]]
    summary = {
        "status": "ok",
        "model_id": args.model_id,
        "request_rows": len(rows),
        "bad_key_path_count": len(key_hits),
        "prompt_leakage_term_hits": len(prompt_hits),
        "prompt_leakage_nonprotected_hits": len(nonprotected_prompt_hits),
        "video_protocol_mismatch_count": len(protocol_mismatches),
        "bad_key_path_examples": key_hits[:10],
        "prompt_leakage_examples": prompt_hits[:10],
        "video_protocol_mismatch_examples": protocol_mismatches[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output is not None:
        benchmark.ensure_dir(args.output.parent)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_model_request_rows(model_dir: Path) -> List[Tuple[str, Dict[str, object]]]:
    rows: List[Tuple[str, Dict[str, object]]] = []
    for setting in benchmark.VALID_SETTINGS:
        path = model_dir / f"{setting}.jsonl"
        if not path.exists():
            continue
        rows.extend((setting, row) for row in benchmark.load_jsonl(path))
    return rows


def iter_key_paths(value: object, prefix: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], object]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*prefix, str(key))
            yield child_path, child
            yield from iter_key_paths(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_key_paths(child, (*prefix, str(index)))


def is_allowed_leakage_policy_path(path: Tuple[str, ...]) -> bool:
    path_text = ".".join(path)
    return ".video_input.leakage_policy." in path_text or path_text.endswith("video_input.leakage_policy")


def iter_prompts(row: Dict[str, object]) -> Iterator[Tuple[str, str]]:
    request = as_dict(row.get("request", {}), "request")
    provider_payload = as_dict(row.get("provider_payload", {}), "provider_payload")
    for name, value in (("request.prompt", request.get("prompt")), ("provider_payload.prompt", provider_payload.get("prompt"))):
        if isinstance(value, str):
            yield name, value


def extract_video_input(row: Dict[str, object]) -> Dict[str, object]:
    provider_payload = as_dict(row.get("provider_payload", {}), "provider_payload")
    payload_policy = provider_payload.get("video_input")
    if isinstance(payload_policy, dict):
        return payload_policy
    request = as_dict(row.get("request", {}), "request")
    request_policy = request.get("video_input")
    return request_policy if isinstance(request_policy, dict) else {}


def as_dict(value: object, label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise benchmark.BenchmarkError(f"Expected {label} to be a dict, got {type(value).__name__}")
    return value


def is_protected_question(question_id: str, protected_prefixes: Iterable[str]) -> bool:
    return any(question_id.startswith(prefix) for prefix in protected_prefixes)


def preview(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text[:160]


if __name__ == "__main__":
    main()
