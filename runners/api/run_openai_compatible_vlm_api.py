#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import gc
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from raven_benchmark import benchmark
from runners.api.run_gpt54_api import extract_video_frames


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval_outputs"
DEFAULT_MODEL_REQUESTS_DIR = DEFAULT_OUTPUT_DIR / "model_requests"
DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_DIR / "predictions"
DEFAULT_MODEL_ID = "hosted_vlm"
DEFAULT_MODEL_NAME = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a hosted VLM through an OpenAI-compatible chat-completions API. "
            "Videos are sampled into chronological JPEG data-url frames."
        )
    )
    parser.add_argument("--model-requests-dir", type=Path, default=DEFAULT_MODEL_REQUESTS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api-key-env", default="OPENAI_COMPATIBLE_API_KEY")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=list(benchmark.VALID_SETTINGS),
        choices=benchmark.VALID_SETTINGS,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--question-id",
        dest="question_ids",
        action="append",
        default=None,
        help="Run only the requested question_id. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="auto")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--frame-cache-size", type=int, default=4)
    parser.add_argument("--disable-json-mode", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate request loading and frame extraction without calling the API or writing predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_name:
        raise benchmark.BenchmarkError("Missing --model-name for OpenAI-compatible hosted VLM runner.")

    model_request_dir = args.model_requests_dir.resolve() / args.model_id
    if not model_request_dir.exists():
        raise benchmark.BenchmarkError(
            f"Model request directory does not exist: {model_request_dir}. "
            f"Export model_requests for model_id={args.model_id} first."
        )

    predictions_root = args.predictions_dir.resolve() / args.model_id
    if not args.dry_run:
        benchmark.ensure_dir(predictions_root)

    runner: OpenAICompatibleVLMRunner | None = None
    completed_settings: List[str] = []
    total_requests = 0

    try:
        for setting in args.settings:
            request_path = model_request_dir / f"{setting}.jsonl"
            if not request_path.exists():
                raise benchmark.BenchmarkError(f"Missing model request file: {request_path}")

            output_path = predictions_root / f"{setting}.jsonl"
            request_rows = benchmark.load_jsonl(request_path)
            if args.question_ids:
                wanted_question_ids = set(args.question_ids)
                request_rows = [
                    row
                    for row in request_rows
                    if str(as_dict(row.get("request"), label="request")["question_id"]) in wanted_question_ids
                ]
            if args.limit is not None:
                request_rows = request_rows[: args.limit]
            if not request_rows:
                raise benchmark.BenchmarkError(f"No requests found in {request_path}")

            provider = infer_single_provider(request_rows)
            if provider != "hosted_vlm":
                raise benchmark.BenchmarkError(
                    f"Provider {provider!r} is not supported by this runner. Expected provider='hosted_vlm'."
                )

            if runner is None:
                video_runtime_config = resolve_video_runtime_config(args, request_rows[0])
                print(
                    json.dumps(
                        {"status": "video_runtime_config", "model_id": args.model_id, **video_runtime_config},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                runner = OpenAICompatibleVLMRunner(
                    model_name=args.model_name,
                    api_key_env=args.api_key_env,
                    base_url=args.base_url,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    image_detail=args.image_detail,
                    jpeg_quality=args.jpeg_quality,
                    frame_cache_size=args.frame_cache_size,
                    disable_json_mode=args.disable_json_mode,
                    dry_run=args.dry_run,
                    video_runtime_config=video_runtime_config,
                )

            if args.overwrite and output_path.exists() and not args.dry_run:
                output_path.unlink()

            completed_predictions: List[Dict[str, object]] = []
            if output_path.exists() and not args.dry_run:
                completed_predictions = benchmark.load_jsonl(output_path)
            completed_ids = {str(row["question_id"]) for row in completed_predictions}

            total_for_setting = len(request_rows)
            completed_for_setting = len(completed_predictions)
            for row in request_rows:
                request = as_dict(row.get("request"), label="request")
                question_id = str(request["question_id"])
                if question_id in completed_ids:
                    continue
                prediction = runner.run_request(row)
                if not args.dry_run:
                    append_jsonl(output_path, prediction)
                    completed_predictions.append(prediction)
                    completed_ids.add(question_id)
                    completed_for_setting = len(completed_predictions)
                else:
                    completed_for_setting += 1
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "model_id": args.model_id,
                            "setting": setting,
                            "completed": completed_for_setting,
                            "total": total_for_setting,
                            "question_id": question_id,
                        }
                    ),
                    flush=True,
                )
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)

            completed_settings.append(setting)
            total_requests += completed_for_setting
    finally:
        if runner is not None:
            runner.close()

    print(
        json.dumps(
            {
                "status": "ok",
                "model_id": args.model_id,
                "provider": "hosted_vlm",
                "dry_run": bool(args.dry_run),
                "settings": completed_settings,
                "request_count": total_requests,
                "predictions_dir": str(predictions_root),
            },
            indent=2,
        )
    )


def resolve_video_runtime_config(args: argparse.Namespace, row: Dict[str, object]) -> Dict[str, object]:
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

    fps = args.fps if args.fps is not None else (policy_fps if policy_fps is not None else 1.0)
    max_pixels = (
        args.max_pixels
        if args.max_pixels is not None
        else (policy_max_pixels if policy_max_pixels is not None else 262144)
    )
    max_frames = args.max_frames_per_video if args.max_frames_per_video is not None else policy_max_frames

    return {
        "source": "cli_override"
        if any(value is not None for value in (args.fps, args.max_pixels, args.max_frames_per_video))
        else "request_video_input",
        "protocol_id": str(video_input.get("protocol_id", "legacy_cli_defaults")) if video_input else "legacy_cli_defaults",
        "fps": float(fps),
        "max_pixels": int(max_pixels),
        "max_frames_per_video": max_frames,
        "frame_transport": "openai_compatible_chat_image_url",
    }


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


class OpenAICompatibleVLMRunner:
    def __init__(
        self,
        model_name: str,
        api_key_env: str,
        base_url: str,
        timeout: float,
        max_retries: int,
        image_detail: str,
        jpeg_quality: int,
        frame_cache_size: int,
        disable_json_mode: bool,
        dry_run: bool,
        video_runtime_config: Dict[str, object],
    ) -> None:
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.image_detail = image_detail
        self.jpeg_quality = max(1, min(int(jpeg_quality), 100))
        self.frame_cache_size = max(int(frame_cache_size), 0)
        self.disable_json_mode = bool(disable_json_mode)
        self.dry_run = bool(dry_run)
        self.video_runtime_config = dict(video_runtime_config)
        self.frame_cache: OrderedDict[str, List[Dict[str, object]]] = OrderedDict()

        self.api_key = None
        if not self.dry_run:
            self.api_key = os.environ.get(api_key_env)
            if not self.api_key:
                raise benchmark.BenchmarkError(f"Missing API key in environment variable {api_key_env}")

    def run_request(self, row: Dict[str, object]) -> Dict[str, object]:
        request = as_dict(row.get("request"), label="request")
        generation_config = as_dict(row.get("generation_config", {}), label="generation_config")
        prompt = str(request["prompt"])
        video_inputs = self.extract_request_video_inputs(request)
        if not video_inputs:
            raise benchmark.BenchmarkError("Hosted VLM request is missing request.inputs video paths")

        content, frame_summary = self.build_content(prompt=prompt, video_inputs=video_inputs)
        api_runtime_config = {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "image_detail": self.image_detail,
            "jpeg_quality": self.jpeg_quality,
            "json_mode": not self.disable_json_mode,
        }

        if self.dry_run:
            raw_response = json.dumps({"answer": "A"}, separators=(",", ":"))
            predicted_answer = "A"
            response_id = None
            usage = None
        else:
            response = self.create_chat_completion(content=content, generation_config=generation_config)
            raw_response = extract_response_text(response)
            predicted_answer, _ = benchmark.normalize_predicted_answer({"raw_response": raw_response})
            response_id = str(response.get("id")) if response.get("id") is not None else None
            usage = response.get("usage")

        return {
            "model_id": str(row["model_id"]),
            "setting": str(request["setting"]),
            "question_id": str(request["question_id"]),
            "predicted_answer": predicted_answer,
            "raw_response": raw_response,
            "response_id": response_id,
            "usage": usage,
            "video_runtime_config": self.video_runtime_config,
            "api_runtime_config": api_runtime_config,
            "frame_summary": frame_summary,
        }

    def extract_request_video_inputs(self, request: Dict[str, object]) -> List[Dict[str, object]]:
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

    def build_content(
        self,
        prompt: str,
        video_inputs: List[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        content: List[Dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    f"{prompt}\n\n"
                    "The videos are provided below as chronological frame samples. "
                    "Use the frame order and timestamp labels when reasoning about temporal events. "
                    "Return JSON only, exactly in the form {\"answer\":\"A\"}, {\"answer\":\"B\"}, "
                    "{\"answer\":\"C\"}, or {\"answer\":\"D\"}."
                ),
            }
        ]
        frame_summary = []

        for video_input in video_inputs:
            video_index = int(video_input["video_index"])
            modality = str(video_input["modality"])
            frames = self.get_video_frames(str(video_input["path"]))
            frame_summary.append(
                {
                    "video_index": video_index,
                    "modality": modality,
                    "path": str(video_input["path"]),
                    "frame_count": len(frames),
                    "first_timestamp": frames[0]["timestamp_sec"] if frames else None,
                    "last_timestamp": frames[-1]["timestamp_sec"] if frames else None,
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Video {video_index} ({modality}) frame samples, ordered chronologically. "
                        f"Sampling fps={self.video_runtime_config['fps']}."
                    ),
                }
            )
            for frame in frames:
                content.append({"type": "text", "text": f"Video {video_index} ({modality}) timestamp {frame['timestamp_sec']:.2f}s"})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(frame["data_url"]),
                            "detail": self.image_detail,
                        },
                    }
                )

        return content, frame_summary

    def get_video_frames(self, video_path: str) -> List[Dict[str, object]]:
        cache_key = json.dumps(
            {
                "path": str(Path(video_path).resolve()),
                "fps": self.video_runtime_config["fps"],
                "max_pixels": self.video_runtime_config["max_pixels"],
                "max_frames_per_video": self.video_runtime_config["max_frames_per_video"],
                "jpeg_quality": self.jpeg_quality,
            },
            sort_keys=True,
        )
        cached = self.frame_cache.get(cache_key)
        if cached is not None:
            self.frame_cache.move_to_end(cache_key)
            return cached

        frames = extract_video_frames(
            Path(video_path),
            fps=float(self.video_runtime_config["fps"]),
            max_pixels=int(self.video_runtime_config["max_pixels"]),
            max_frames_per_video=optional_int(self.video_runtime_config.get("max_frames_per_video")),
            jpeg_quality=self.jpeg_quality,
        )
        if self.frame_cache_size > 0:
            self.frame_cache[cache_key] = frames
            while len(self.frame_cache) > self.frame_cache_size:
                self.frame_cache.popitem(last=False)
        return frames

    def create_chat_completion(self, content: List[Dict[str, object]], generation_config: Dict[str, object]) -> Dict[str, object]:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, object] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
        }
        max_tokens = (
            generation_config.get("max_tokens")
            or generation_config.get("max_output_tokens")
            or generation_config.get("max_new_tokens")
        )
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        temperature = generation_config.get("temperature")
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if not self.disable_json_mode:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_body = response.read().decode("utf-8")
                return json.loads(response_body)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2.0 ** attempt, 8.0))
        raise benchmark.BenchmarkError(f"OpenAI-compatible API request failed: {last_error}")

    def close(self) -> None:
        self.frame_cache.clear()
        gc.collect()


def extract_response_text(response: Dict[str, object]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    pieces = []
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            pieces.append(part["text"])
                    if pieces:
                        return "\n".join(pieces).strip()
            text = first.get("text")
            if isinstance(text, str):
                return text.strip()
    return json.dumps(response, ensure_ascii=False, sort_keys=True)


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


def as_dict(value: object, label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise benchmark.BenchmarkError(f"Expected {label} to be a dict, got {type(value).__name__}")
    return value


if __name__ == "__main__":
    try:
        main()
    except benchmark.BenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
