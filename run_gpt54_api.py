#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")
import cv2
import eo_ir_benchmark as benchmark

try:
    cv2.setLogLevel(0)
except AttributeError:
    pass


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval_outputs"
DEFAULT_MODEL_REQUESTS_DIR = DEFAULT_OUTPUT_DIR / "model_requests"
DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_DIR / "predictions"
DEFAULT_MODEL_ID = "gpt_5_4"
DEFAULT_MODEL_NAME = "gpt-5.4"

ANSWER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {
            "type": "string",
            "enum": sorted(benchmark.VALID_ANSWERS),
        }
    },
    "required": ["answer"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GPT-5.4/OpenAI Responses API against exported benchmark model_requests. "
            "Video files are sampled into ordered image frames according to request.video_input."
        )
    )
    parser.add_argument("--model-requests-dir", type=Path, default=DEFAULT_MODEL_REQUESTS_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
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
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--organization", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=2)
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
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="auto")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--frame-cache-size",
        type=int,
        default=4,
        help="Number of decoded frame sequences to cache in memory across questions.",
    )
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help="Do not send temperature even if generation_config contains it.",
    )
    parser.add_argument(
        "--disable-json-schema",
        action="store_true",
        help="Use prompt-only JSON instruction instead of Responses structured output.",
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
        help="Validate request loading and frame extraction without calling the API or writing predictions.",
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

    runner: OpenAIGPTRunner | None = None
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
        if provider != "gpt":
            raise benchmark.BenchmarkError(
                f"Provider {provider!r} is not supported by this runner. Expected provider='gpt'."
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
            runner = OpenAIGPTRunner(
                model_name=args.model_name,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                organization=args.organization,
                project=args.project,
                timeout=args.timeout,
                max_retries=args.max_retries,
                image_detail=args.image_detail,
                jpeg_quality=args.jpeg_quality,
                frame_cache_size=args.frame_cache_size,
                omit_temperature=args.omit_temperature,
                disable_json_schema=args.disable_json_schema,
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

    if runner is not None:
        runner.close()

    print(
        json.dumps(
            {
                "status": "ok",
                "model_id": args.model_id,
                "provider": "gpt",
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
        "frame_transport": "openai_responses_input_image",
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


class OpenAIGPTRunner:
    def __init__(
        self,
        model_name: str,
        api_key_env: str,
        base_url: Optional[str],
        organization: Optional[str],
        project: Optional[str],
        timeout: float,
        max_retries: int,
        image_detail: str,
        jpeg_quality: int,
        frame_cache_size: int,
        omit_temperature: bool,
        disable_json_schema: bool,
        dry_run: bool,
        video_runtime_config: Dict[str, object],
    ) -> None:
        self.model_name = model_name
        self.image_detail = image_detail
        self.jpeg_quality = max(1, min(int(jpeg_quality), 100))
        self.frame_cache_size = max(int(frame_cache_size), 0)
        self.omit_temperature = omit_temperature
        self.disable_json_schema = disable_json_schema
        self.dry_run = dry_run
        self.video_runtime_config = dict(video_runtime_config)
        self.frame_cache: OrderedDict[str, List[Dict[str, object]]] = OrderedDict()

        self.client = None
        if not dry_run:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise benchmark.BenchmarkError(f"Missing OpenAI API key in environment variable {api_key_env}")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise benchmark.BenchmarkError(
                    "The OpenAI Python SDK is required. Install/activate an environment with `openai` available."
                ) from exc
            client_kwargs: Dict[str, object] = {
                "api_key": api_key,
                "timeout": timeout,
                "max_retries": max_retries,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            if organization:
                client_kwargs["organization"] = organization
            if project:
                client_kwargs["project"] = project
            self.client = OpenAI(**client_kwargs)

    def run_request(self, row: Dict[str, object]) -> Dict[str, object]:
        request = as_dict(row.get("request"), label="request")
        generation_config = as_dict(row.get("generation_config", {}), label="generation_config")
        prompt = str(request["prompt"])
        video_inputs = self.extract_request_video_inputs(request)
        if not video_inputs:
            raise benchmark.BenchmarkError("GPT request is missing request.inputs video paths")

        content, frame_summary = self.build_content(prompt=prompt, video_inputs=video_inputs)
        api_runtime_config = {
            "model_name": self.model_name,
            "image_detail": self.image_detail,
            "jpeg_quality": self.jpeg_quality,
            "structured_output": not self.disable_json_schema,
        }

        if self.dry_run:
            raw_response = json.dumps({"answer": "A"}, separators=(",", ":"))
            predicted_answer = "A"
            response_id = None
            usage = None
        else:
            response = self.create_response(content=content, generation_config=generation_config)
            raw_response = extract_response_text(response)
            predicted_answer, _ = benchmark.normalize_predicted_answer({"raw_response": raw_response})
            response_id = str(getattr(response, "id", "")) or None
            usage = to_plain_json(getattr(response, "usage", None))

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
                "type": "input_text",
                "text": (
                    f"{prompt}\n\n"
                    "The videos are provided below as chronological frame samples. "
                    "Use the frame order and timestamp labels when reasoning about temporal events."
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
                    "type": "input_text",
                    "text": (
                        f"Video {video_index} ({modality}) frame samples, ordered chronologically. "
                        f"Sampling fps={self.video_runtime_config['fps']}."
                    ),
                }
            )
            for frame in frames:
                content.append(
                    {
                        "type": "input_text",
                        "text": f"Video {video_index} ({modality}) timestamp {frame['timestamp_sec']:.2f}s",
                    }
                )
                content.append(
                    {
                        "type": "input_image",
                        "image_url": str(frame["data_url"]),
                        "detail": self.image_detail,
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

    def create_response(self, content: List[Dict[str, object]], generation_config: Dict[str, object]) -> object:
        if self.client is None:
            raise benchmark.BenchmarkError("OpenAI client is not initialized")
        kwargs: Dict[str, object] = {
            "model": self.model_name,
            "input": [{"role": "user", "content": content}],
            "store": False,
        }
        max_output_tokens = generation_config.get("max_output_tokens") or generation_config.get("max_new_tokens")
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = int(max_output_tokens)
        temperature = generation_config.get("temperature")
        if temperature is not None and not self.omit_temperature:
            kwargs["temperature"] = float(temperature)
        if not self.disable_json_schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "multiple_choice_answer",
                    "strict": True,
                    "schema": ANSWER_JSON_SCHEMA,
                }
            }
        return self.client.responses.create(**kwargs)

    def close(self) -> None:
        self.frame_cache.clear()
        gc.collect()


def extract_video_frames(
    video_path: Path,
    fps: float,
    max_pixels: int,
    max_frames_per_video: Optional[int],
    jpeg_quality: int,
) -> List[Dict[str, object]]:
    if fps <= 0:
        raise benchmark.BenchmarkError(f"fps must be positive, got {fps}")
    if max_pixels <= 0:
        raise benchmark.BenchmarkError(f"max_pixels must be positive, got {max_pixels}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise benchmark.BenchmarkError(f"Failed to open video for OpenAI runner: {video_path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = 30.0
        frames = extract_frames_sequential(capture, source_fps, fps, max_pixels, jpeg_quality)
    finally:
        capture.release()

    if max_frames_per_video is not None and max_frames_per_video > 0 and len(frames) > max_frames_per_video:
        frames = uniform_subsample(frames, max_frames_per_video)
    if not frames:
        raise benchmark.BenchmarkError(f"No frames extracted from {video_path}")
    return frames


def extract_frames_by_timestamp(
    capture: cv2.VideoCapture,
    source_fps: float,
    frame_count: int,
    target_fps: float,
    max_pixels: int,
    jpeg_quality: int,
) -> List[Dict[str, object]]:
    duration = frame_count / source_fps
    timestamps = []
    timestamp = 0.0
    while timestamp < duration:
        timestamps.append(timestamp)
        timestamp += 1.0 / target_fps
    if not timestamps:
        timestamps = [0.0]
    frames = []
    last_frame_index = -1
    for timestamp in timestamps:
        frame_index = min(frame_count - 1, int(round(timestamp * source_fps)))
        if frame_index == last_frame_index:
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        frames.append(encode_frame(frame, timestamp_sec=frame_index / source_fps, max_pixels=max_pixels, jpeg_quality=jpeg_quality))
        last_frame_index = frame_index
    return frames


def extract_frames_sequential(
    capture: cv2.VideoCapture,
    source_fps: float,
    target_fps: float,
    max_pixels: int,
    jpeg_quality: int,
) -> List[Dict[str, object]]:
    frames = []
    frame_index = 0
    next_sample_time = 0.0
    sample_interval = 1.0 / target_fps
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp_sec = frame_index / source_fps
        if timestamp_sec + (0.5 / source_fps) >= next_sample_time:
            frames.append(
                encode_frame(
                    frame,
                    timestamp_sec=timestamp_sec,
                    max_pixels=max_pixels,
                    jpeg_quality=jpeg_quality,
                )
            )
            next_sample_time += sample_interval
        frame_index += 1
    return frames


def encode_frame(frame: object, timestamp_sec: float, max_pixels: int, jpeg_quality: int) -> Dict[str, object]:
    resized = resize_to_pixel_budget(frame, max_pixels=max_pixels)
    ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise benchmark.BenchmarkError("Failed to encode video frame as JPEG")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    height, width = resized.shape[:2]
    return {
        "timestamp_sec": float(timestamp_sec),
        "width": int(width),
        "height": int(height),
        "data_url": f"data:image/jpeg;base64,{payload}",
    }


def resize_to_pixel_budget(frame: object, max_pixels: int) -> object:
    height, width = frame.shape[:2]
    pixels = width * height
    if pixels <= max_pixels:
        return frame
    scale = (max_pixels / float(pixels)) ** 0.5
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def uniform_subsample(frames: List[Dict[str, object]], max_count: int) -> List[Dict[str, object]]:
    if max_count <= 0 or len(frames) <= max_count:
        return frames
    if max_count == 1:
        return [frames[0]]
    last = len(frames) - 1
    indexes = sorted({int(round(i * last / (max_count - 1))) for i in range(max_count)})
    return [frames[index] for index in indexes]


def extract_response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    pieces: List[str] = []
    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        pieces.append(text)
                    refusal = getattr(part, "refusal", None)
                    if isinstance(refusal, str):
                        pieces.append(refusal)
    if pieces:
        return "\n".join(pieces).strip()
    return json.dumps(to_plain_json(response), ensure_ascii=False, sort_keys=True)


def to_plain_json(value: object) -> object:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_plain_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain_json(item) for key, item in value.items()}
    return str(value)


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
