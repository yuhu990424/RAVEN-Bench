#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from raven_benchmark import benchmark
from runners.api.run_gpt54_api import extract_video_frames


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval_outputs"
DEFAULT_MODEL_REQUESTS_DIR = DEFAULT_OUTPUT_DIR / "model_requests"
DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_DIR / "predictions"
DEFAULT_MODEL_ID = "gemini_3_flash_preview"
DEFAULT_MODEL_NAME = "gemini-3-flash-preview"
DEFAULT_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {
            "type": "STRING",
            "enum": sorted(benchmark.VALID_ANSWERS),
        }
    },
    "required": ["answer"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gemini REST API against exported benchmark model_requests. "
            "Video files are sampled into ordered JPEG frames according to request.video_input."
        )
    )
    parser.add_argument("--model-requests-dir", type=Path, default=DEFAULT_MODEL_REQUESTS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
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
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override request.video_input.sampling.fps. Official policy defaults to 1.0 fps.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Override request.video_input.frame_resolution.max_pixels. Official policy defaults to 262144.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Optional uniform frame cap per video. Omit for the official no-cap policy.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--frame-cache-size",
        type=int,
        default=4,
        help="Number of decoded frame sequences to cache in memory across questions.",
    )
    parser.add_argument(
        "--disable-response-schema",
        action="store_true",
        help="Use prompt-only JSON instruction instead of Gemini responseSchema.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay after each API request for rate limiting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate request loading and frame extraction without calling Gemini or writing predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_request_dir = args.model_requests_dir.resolve() / args.model_id
    if not model_request_dir.exists():
        raise benchmark.BenchmarkError(
            f"Model request directory does not exist: {model_request_dir}. "
            f"Export model_requests for model_id={args.model_id} first."
        )

    predictions_root = args.predictions_dir.resolve() / args.model_id
    if not args.dry_run:
        benchmark.ensure_dir(predictions_root)

    runner: GeminiRunner | None = None
    completed_settings: List[str] = []
    total_requests = 0

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
        if provider != "gemini":
            raise benchmark.BenchmarkError(
                f"Provider {provider!r} is not supported by this runner. Expected provider='gemini'."
            )

        if runner is None:
            video_runtime_config = resolve_video_runtime_config(args, request_rows[0])
            print(
                json.dumps(
                    {
                        "status": "video_runtime_config",
                        "model_id": args.model_id,
                        **video_runtime_config,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            runner = GeminiRunner(
                model_name=args.model_name,
                api_base_url=args.api_base_url,
                api_key_env=args.api_key_env,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
                jpeg_quality=args.jpeg_quality,
                frame_cache_size=args.frame_cache_size,
                disable_response_schema=args.disable_response_schema,
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
            print(
                json.dumps(
                    {
                        "status": "progress",
                        "model_id": args.model_id,
                        "setting": setting,
                        "completed": len(completed_predictions) if not args.dry_run else "dry_run",
                        "total": total_for_setting,
                        "question_id": question_id,
                    }
                ),
                flush=True,
            )
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        completed_settings.append(setting)
        total_requests += len(completed_predictions) if not args.dry_run else len(request_rows)

    print(
        json.dumps(
            {
                "status": "ok",
                "model_id": args.model_id,
                "provider": "gemini",
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
        "frame_transport": "gemini_generate_content_inline_data",
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


class GeminiRunner:
    def __init__(
        self,
        model_name: str,
        api_base_url: str,
        api_key_env: str,
        timeout: float,
        max_retries: int,
        retry_sleep_seconds: float,
        jpeg_quality: int,
        frame_cache_size: int,
        disable_response_schema: bool,
        dry_run: bool,
        video_runtime_config: Dict[str, object],
    ) -> None:
        self.model_name = model_name
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(int(max_retries), 0)
        self.retry_sleep_seconds = max(float(retry_sleep_seconds), 0.0)
        self.jpeg_quality = max(1, min(int(jpeg_quality), 100))
        self.frame_cache_size = max(int(frame_cache_size), 0)
        self.disable_response_schema = disable_response_schema
        self.dry_run = dry_run
        self.video_runtime_config = dict(video_runtime_config)
        self.frame_cache: Dict[str, List[Dict[str, object]]] = {}
        self.frame_cache_order: List[str] = []

        self.api_key = None
        if not dry_run:
            self.api_key = os.environ.get(api_key_env)
            if not self.api_key:
                raise benchmark.BenchmarkError(f"Missing Gemini API key in environment variable {api_key_env}")

    def run_request(self, row: Dict[str, object]) -> Dict[str, object]:
        request = as_dict(row.get("request"), label="request")
        generation_config = as_dict(row.get("generation_config", {}), label="generation_config")
        prompt = str(request["prompt"])
        video_inputs = self.extract_request_video_inputs(request)
        if not video_inputs:
            raise benchmark.BenchmarkError("Gemini request is missing request.inputs video paths")

        parts, frame_summary = self.build_parts(prompt=prompt, video_inputs=video_inputs)
        api_runtime_config = {
            "model_name": self.model_name,
            "jpeg_quality": self.jpeg_quality,
            "response_schema": not self.disable_response_schema,
        }

        if self.dry_run:
            raw_response = json.dumps({"answer": "A"}, separators=(",", ":"))
            predicted_answer = "A"
            response_payload = None
            usage = None
        else:
            response_payload = self.generate_content(parts=parts, generation_config=generation_config)
            raw_response = extract_gemini_text(response_payload)
            predicted_answer, _ = benchmark.normalize_predicted_answer({"raw_response": raw_response})
            usage = response_payload.get("usageMetadata") if isinstance(response_payload, dict) else None

        return {
            "model_id": str(row["model_id"]),
            "setting": str(request["setting"]),
            "question_id": str(request["question_id"]),
            "predicted_answer": predicted_answer,
            "raw_response": raw_response,
            "usage": usage,
            "gemini_response_metadata": compact_gemini_metadata(response_payload),
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

    def build_parts(
        self,
        prompt: str,
        video_inputs: List[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        parts: List[Dict[str, object]] = [
            {
                "text": (
                    f"{prompt}\n\n"
                    "The videos are provided below as chronological frame samples. "
                    "Use the frame order and timestamp labels when reasoning about temporal events."
                )
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
            parts.append(
                {
                    "text": (
                        f"Video {video_index} ({modality}) frame samples, ordered chronologically. "
                        f"Sampling fps={self.video_runtime_config['fps']}."
                    )
                }
            )
            for frame in frames:
                parts.append({"text": f"Video {video_index} ({modality}) timestamp {frame['timestamp_sec']:.2f}s"})
                data_url = str(frame["data_url"])
                prefix = "data:image/jpeg;base64,"
                if not data_url.startswith(prefix):
                    raise benchmark.BenchmarkError("Expected JPEG data URL from frame extractor")
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": data_url[len(prefix):],
                        }
                    }
                )
        return parts, frame_summary

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
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key]

        frames = extract_video_frames(
            Path(video_path),
            fps=float(self.video_runtime_config["fps"]),
            max_pixels=int(self.video_runtime_config["max_pixels"]),
            max_frames_per_video=optional_int(self.video_runtime_config.get("max_frames_per_video")),
            jpeg_quality=self.jpeg_quality,
        )
        if self.frame_cache_size > 0:
            self.frame_cache[cache_key] = frames
            self.frame_cache_order.append(cache_key)
            while len(self.frame_cache_order) > self.frame_cache_size:
                old_key = self.frame_cache_order.pop(0)
                self.frame_cache.pop(old_key, None)
        return frames

    def generate_content(self, parts: List[Dict[str, object]], generation_config: Dict[str, object]) -> Dict[str, object]:
        if not self.api_key:
            raise benchmark.BenchmarkError("Gemini API key is not initialized")

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": normalize_generation_config(generation_config, self.disable_response_schema),
        }
        url = f"{self.api_base_url}/models/{self.model_name}:generateContent"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key,
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                text = exc.read().decode("utf-8", errors="replace")
                last_error = GeminiHTTPError(exc.code, redact_api_key(text, self.api_key))
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    break
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
            if self.retry_sleep_seconds > 0:
                time.sleep(self.retry_sleep_seconds * (attempt + 1))
        raise benchmark.BenchmarkError(f"Gemini API request failed: {last_error}") from last_error


def normalize_generation_config(generation_config: Dict[str, object], disable_response_schema: bool) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    temperature = generation_config.get("temperature")
    if temperature is not None:
        normalized["temperature"] = float(temperature)
    max_output_tokens = generation_config.get("max_output_tokens") or generation_config.get("max_new_tokens")
    if max_output_tokens is not None:
        normalized["maxOutputTokens"] = int(max_output_tokens)
    thinking_config: Dict[str, object] = {}
    thinking_level = generation_config.get("thinking_level") or generation_config.get("thinkingLevel")
    if thinking_level is not None:
        thinking_config["thinkingLevel"] = str(thinking_level)
    thinking_budget = generation_config.get("thinking_budget") or generation_config.get("thinkingBudget")
    if thinking_budget is not None:
        thinking_config["thinkingBudget"] = int(thinking_budget)
    include_thoughts = generation_config.get("include_thoughts")
    if include_thoughts is None:
        include_thoughts = generation_config.get("includeThoughts")
    if include_thoughts is not None:
        thinking_config["includeThoughts"] = bool(include_thoughts)
    if thinking_config:
        normalized["thinkingConfig"] = thinking_config
    if not disable_response_schema:
        normalized["responseMimeType"] = "application/json"
        normalized["responseSchema"] = ANSWER_SCHEMA
    return normalized


class GeminiHTTPError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def extract_gemini_text(response_payload: Dict[str, object]) -> str:
    candidates = response_payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return json.dumps(response_payload, ensure_ascii=False, sort_keys=True)
    first = candidates[0]
    if not isinstance(first, dict):
        return json.dumps(response_payload, ensure_ascii=False, sort_keys=True)
    content = first.get("content")
    if not isinstance(content, dict):
        return json.dumps(response_payload, ensure_ascii=False, sort_keys=True)
    parts = content.get("parts")
    if not isinstance(parts, list):
        return json.dumps(response_payload, ensure_ascii=False, sort_keys=True)
    texts = [part.get("text") for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
    return "\n".join(texts).strip() if texts else json.dumps(response_payload, ensure_ascii=False, sort_keys=True)


def compact_gemini_metadata(response_payload: object) -> object:
    if not isinstance(response_payload, dict):
        return None
    metadata: Dict[str, object] = {}
    if "promptFeedback" in response_payload:
        metadata["promptFeedback"] = response_payload["promptFeedback"]
    candidates = response_payload.get("candidates")
    if isinstance(candidates, list):
        metadata["candidates"] = [
            {
                key: value
                for key, value in candidate.items()
                if key in {"finishReason", "safetyRatings", "citationMetadata"}
            }
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
    return metadata


def redact_api_key(text: str, api_key: str) -> str:
    return text.replace(api_key, "<GEMINI_API_KEY>")


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
