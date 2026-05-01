#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
DEFAULT_MODEL_ID = "internvl3_5_8b"
DEFAULT_MODEL_NAME = "OpenGVLab/InternVL3_5-8B-HF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local InternVL3.5-8B against exported benchmark model_requests and "
            "write predictions/{model_id}/{setting}.jsonl."
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
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video sampling FPS. Defaults to request.video_input.sampling.fps.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Override processor min_pixels. Defaults to max_pixels.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Override processor max_pixels. Defaults to request.video_input.frame_resolution.max_pixels.",
    )
    parser.add_argument("--frame-sample-count", type=int, default=None)
    parser.add_argument("--frame-resize", type=int, default=None)
    parser.add_argument(
        "--processor-video-size",
        type=int,
        default=None,
        help="Square native video size for InternVL frames. Defaults to the model vision_config image_size.",
    )
    parser.add_argument(
        "--processor-video-decoding",
        action="store_true",
        help=(
            "Pass video files directly to the processor. By default, this runner decodes "
            "frames with OpenCV at the request video_input FPS/resolution policy."
        ),
    )
    parser.add_argument(
        "--frame-cache-size",
        type=int,
        default=6,
        help="Number of decoded/resized videos to keep in memory for repeated questions. Use 0 to disable.",
    )
    parser.add_argument(
        "--visual-feature-cache-size",
        type=int,
        default=2,
        help="Number of per-video visual feature tensors to cache on CPU for repeated questions. Use 0 to disable.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--video-cache-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "local_runner_cache" / "videos_internvl35",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_request_dir = args.model_requests_dir.resolve() / args.model_id
    if not model_request_dir.exists():
        raise benchmark.BenchmarkError(f"Model request directory does not exist: {model_request_dir}")

    predictions_root = args.predictions_dir.resolve() / args.model_id
    benchmark.ensure_dir(predictions_root)

    runner: InternVLRunner | None = None
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
        if provider != "internvl":
            raise benchmark.BenchmarkError(
                f"Provider {provider!r} is not supported by this runner. Expected provider='internvl'."
            )

        if runner is None:
            video_runtime_config = resolve_video_runtime_config(args, request_rows[0])
            print(
                json.dumps(
                    {
                        "status": "video_runtime_config",
                        "model_id": args.model_id,
                        "source": video_runtime_config["source"],
                        "protocol_id": video_runtime_config["protocol_id"],
                        "fps": video_runtime_config["fps"],
                        "min_pixels": video_runtime_config["min_pixels"],
                        "max_pixels": video_runtime_config["max_pixels"],
                        "target_longest_side": video_runtime_config["target_longest_side"],
                        "processor_video_size": args.processor_video_size,
                        "max_frames_per_video": video_runtime_config["max_frames_per_video"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            runner = InternVLRunner.from_request_row(
                request_rows[0],
                model_name=args.model_name,
                fps=float(video_runtime_config["fps"]),
                min_pixels=int(video_runtime_config["min_pixels"]),
                max_pixels=int(video_runtime_config["max_pixels"]),
                target_longest_side=optional_int(video_runtime_config["target_longest_side"]),
                frame_sample_count=args.frame_sample_count,
                frame_resize=args.frame_resize,
                processor_video_size=args.processor_video_size,
                decode_with_cv2=not args.processor_video_decoding,
                frame_cache_size=args.frame_cache_size,
                visual_feature_cache_size=args.visual_feature_cache_size,
                device=args.device,
                load_in_8bit=args.load_in_8bit,
                video_cache_dir=args.video_cache_dir.resolve(),
            )

        if args.overwrite and output_path.exists():
            output_path.unlink()

        completed_predictions: List[Dict[str, object]] = []
        if output_path.exists():
            completed_predictions = benchmark.load_jsonl(output_path)
        completed_ids = {str(row["question_id"]) for row in completed_predictions}

        total_for_setting = len(request_rows)
        for row in request_rows:
            request = as_dict(row.get("request"), label="request")
            question_id = str(request["question_id"])
            if question_id in completed_ids:
                continue
            prediction = runner.run_request(row)
            append_jsonl(output_path, prediction)
            completed_predictions.append(prediction)
            completed_ids.add(question_id)
            print(
                json.dumps(
                    {
                        "status": "progress",
                        "model_id": args.model_id,
                        "setting": setting,
                        "completed": len(completed_predictions),
                        "total": total_for_setting,
                        "question_id": question_id,
                    }
                ),
                flush=True,
            )

        completed_settings.append(setting)
        total_requests += len(completed_predictions)

    if runner is not None:
        runner.close()

    print(
        json.dumps(
            {
                "status": "ok",
                "model_id": args.model_id,
                "provider": "internvl",
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

    fps = args.fps if args.fps is not None else (policy_fps if policy_fps is not None else 0.25)
    max_pixels = (
        args.max_pixels
        if args.max_pixels is not None
        else (policy_max_pixels if policy_max_pixels is not None else 100352)
    )
    min_pixels = args.min_pixels if args.min_pixels is not None else max_pixels

    return {
        "source": (
            "cli_override"
            if any(value is not None for value in (args.fps, args.min_pixels, args.max_pixels))
            else "request_video_input"
        ),
        "protocol_id": str(video_input.get("protocol_id", "legacy_cli_defaults")) if video_input else "legacy_cli_defaults",
        "fps": float(fps),
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "target_longest_side": optional_int(frame_resolution.get("target_longest_side")),
        "max_frames_per_video": sampling.get("max_frames_per_video"),
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


def optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    return int(value)


class CachedImageFeatureOutput:
    def __init__(self, pooler_output: object) -> None:
        self.pooler_output = pooler_output


class InternVLRunner:
    def __init__(
        self,
        model_name: str,
        generation_config: Dict[str, object],
        fps: float,
        min_pixels: int,
        max_pixels: int,
        target_longest_side: int | None,
        frame_sample_count: int | None,
        frame_resize: int | None,
        processor_video_size: int | None,
        decode_with_cv2: bool,
        frame_cache_size: int,
        visual_feature_cache_size: int,
        device: str,
        load_in_8bit: bool,
        video_cache_dir: Path,
    ) -> None:
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        import torch

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise benchmark.BenchmarkError(
                "This runner requires a recent Transformers build with AutoModelForImageTextToText support."
            ) from exc
        if load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise benchmark.BenchmarkError(
                    "8-bit loading requires a Transformers build with BitsAndBytesConfig support."
                ) from exc

        self.torch = torch
        self.model_name = model_name
        self.base_generation_config = dict(generation_config)
        self.fps = fps
        self.max_pixels = max_pixels
        self.target_longest_side = target_longest_side
        self.frame_sample_count = frame_sample_count
        self.frame_resize = frame_resize
        self.processor_video_size = processor_video_size
        self.decode_with_cv2 = decode_with_cv2
        self.frame_cache_size = max(0, frame_cache_size)
        self.frame_cache: OrderedDict[tuple[object, ...], List[object]] = OrderedDict()
        self.visual_feature_cache_size = max(0, visual_feature_cache_size)
        self.visual_feature_cache: OrderedDict[tuple[object, ...], CachedImageFeatureOutput] = OrderedDict()
        self.active_visual_cache_key: tuple[object, ...] | None = None
        self.video_cache_dir = video_cache_dir
        benchmark.ensure_dir(self.video_cache_dir)

        dtype = torch.bfloat16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        device_map = "auto" if device == "auto" else None
        load_kwargs: Dict[str, object] = {
            "dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }
        if load_in_8bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        if device_map == "auto":
            offload_dir = self.video_cache_dir / "model_offload"
            benchmark.ensure_dir(offload_dir)
            load_kwargs["offload_folder"] = str(offload_dir)
            load_kwargs["offload_state_dict"] = True
            if torch.cuda.is_available():
                gpu_total_gib = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                load_kwargs["max_memory"] = {
                    0: os.environ.get("INTERNVL_GPU_MAX_MEMORY", f"{max(gpu_total_gib - 3, 1)}GiB"),
                    "cpu": os.environ.get("INTERNVL_CPU_MAX_MEMORY", "8GiB"),
                }
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **load_kwargs,
        )
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        elif device == "cpu":
            self.model = self.model.to("cpu")
        if self.processor_video_size is None:
            self.processor_video_size = infer_native_vision_size(self.model)
        self.install_visual_feature_cache()
        print(
            json.dumps(
                {
                    "status": "internvl_processor_config",
                    "processor_video_size": self.processor_video_size,
                    "decode_with_cv2": self.decode_with_cv2,
                    "preserve_aspect_with_padding": self.decode_with_cv2 and self.processor_video_size is not None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if self.processor_video_size is not None:
            self.processor.image_seq_length = image_seq_length_for_video_size(self.processor_video_size)

    @classmethod
    def from_request_row(
        cls,
        row: Dict[str, object],
        model_name: str,
        fps: float,
        min_pixels: int,
        max_pixels: int,
        target_longest_side: int | None,
        frame_sample_count: int | None,
        frame_resize: int | None,
        processor_video_size: int | None,
        decode_with_cv2: bool,
        frame_cache_size: int,
        visual_feature_cache_size: int,
        device: str,
        load_in_8bit: bool,
        video_cache_dir: Path,
    ) -> "InternVLRunner":
        return cls(
            model_name=model_name,
            generation_config=as_dict(row.get("generation_config"), label="generation_config"),
            fps=fps,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            target_longest_side=target_longest_side,
            frame_sample_count=frame_sample_count,
            frame_resize=frame_resize,
            processor_video_size=processor_video_size,
            decode_with_cv2=decode_with_cv2,
            frame_cache_size=frame_cache_size,
            visual_feature_cache_size=visual_feature_cache_size,
            device=device,
            load_in_8bit=load_in_8bit,
            video_cache_dir=video_cache_dir,
        )

    def run_request(self, row: Dict[str, object]) -> Dict[str, object]:
        provider_payload = as_dict(row.get("provider_payload"), label="provider_payload")
        request = as_dict(row.get("request"), label="request")
        prompt = str(provider_payload.get("prompt") or request["prompt"])
        source_video_paths = [str(path) for path in provider_payload.get("video_paths", [])]
        video_paths = [self.prepare_video_path(path) for path in source_video_paths]
        if not video_paths:
            raise benchmark.BenchmarkError("InternVL request is missing video_paths")

        raw_response = self.generate_text(prompt=prompt, video_paths=video_paths, generation_config=provider_payload)
        predicted_answer, _ = benchmark.normalize_predicted_answer({"raw_response": raw_response})
        self.release_runtime_memory()
        return {
            "model_id": str(row["model_id"]),
            "setting": str(request["setting"]),
            "question_id": str(request["question_id"]),
            "predicted_answer": predicted_answer,
            "raw_response": raw_response,
        }

    def generate_text(
        self,
        prompt: str,
        video_paths: List[str],
        generation_config: Dict[str, object],
    ) -> str:
        try:
            return self._generate_text_once(prompt=prompt, video_paths=video_paths, generation_config=generation_config)
        except IndexError as exc:
            if "Invalid frame index" not in str(exc):
                raise
            safe_paths = [self.build_safe_tail_trim_copy(path, drop_last_frames=3) for path in video_paths]
            return self._generate_text_once(prompt=prompt, video_paths=safe_paths, generation_config=generation_config)

    def _generate_text_once(
        self,
        prompt: str,
        video_paths: List[str],
        generation_config: Dict[str, object],
    ) -> str:
        chat_text = self.build_chat_text(prompt=prompt, video_count=len(video_paths))
        inputs = self.build_inputs(chat_text=chat_text, video_paths=video_paths)
        inputs = move_batch_to_model_device(inputs, resolve_model_device(self.model))
        generate_kwargs = normalize_generation_config(
            generation_config=as_dict(generation_config.get("generation_config", {}), label="generation_config")
        )
        output_ids = None
        generated_ids = None
        output_text = None
        try:
            self.active_visual_cache_key = self.visual_cache_key_for_paths(video_paths)
            with self.torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generate_kwargs)
            generated_ids = [
                output_ids_row[len(input_ids_row):]
                for input_ids_row, output_ids_row in zip(inputs.input_ids, output_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            return str(output_text[0]).strip() if output_text else ""
        finally:
            self.active_visual_cache_key = None
            del inputs
            if output_ids is not None:
                del output_ids
            if generated_ids is not None:
                del generated_ids
            if output_text is not None:
                del output_text
            self.release_runtime_memory()

    def install_visual_feature_cache(self) -> None:
        if self.visual_feature_cache_size <= 0:
            return
        internvl_model = getattr(self.model, "model", None)
        if internvl_model is None or not hasattr(internvl_model, "get_image_features"):
            return
        original_get_image_features = internvl_model.get_image_features

        def cached_get_image_features(*args: object, **kwargs: object) -> CachedImageFeatureOutput:
            cache_key = self.active_visual_cache_key
            if cache_key is not None and cache_key in self.visual_feature_cache:
                cached_output = self.visual_feature_cache.pop(cache_key)
                self.visual_feature_cache[cache_key] = cached_output
                return cached_output

            output = original_get_image_features(*args, **kwargs)
            pooler_output = getattr(output, "pooler_output")
            cached_output = CachedImageFeatureOutput(pooler_output.detach().cpu())
            if cache_key is not None:
                self.visual_feature_cache[cache_key] = cached_output
                while len(self.visual_feature_cache) > self.visual_feature_cache_size:
                    self.visual_feature_cache.popitem(last=False)
            return cached_output

        internvl_model.get_image_features = cached_get_image_features

    def visual_cache_key_for_paths(self, video_paths: List[str]) -> tuple[object, ...]:
        return (
            tuple(str(Path(path).resolve()) for path in video_paths),
            self.fps,
            self.processor_video_size,
            self.frame_sample_count,
            self.frame_resize,
        )

    def build_chat_text(self, prompt: str, video_count: int) -> str:
        conversation = [
            {
                "role": "user",
                "content": [*({"type": "video"} for _ in range(video_count)), {"type": "text", "text": prompt}],
            }
        ]
        return self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

    def build_inputs(self, chat_text: str, video_paths: List[str]) -> object:
        videos: object = video_paths
        if self.frame_sample_count is not None:
            videos = [
                sample_video_frames_with_cv2(Path(path), self.frame_sample_count, self.frame_resize)
                for path in video_paths
            ]
        elif self.decode_with_cv2:
            videos = [
                self.sample_frames_for_path(path)
                for path in video_paths
            ]
            videos = pad_video_frame_lists_to_equal_length(videos)
        processor_video_kwargs: Dict[str, object] = {}
        if self.frame_resize is not None:
            processor_video_kwargs["size"] = {"height": self.frame_resize, "width": self.frame_resize}
        elif self.decode_with_cv2 and self.processor_video_size is not None:
            processor_video_kwargs["do_resize"] = False
        elif self.processor_video_size is not None:
            processor_video_kwargs["size"] = {
                "height": self.processor_video_size,
                "width": self.processor_video_size,
            }

        last_error: Exception | None = None
        base_attempts = (
            {"text": [chat_text], "videos": videos, "return_tensors": "pt", **processor_video_kwargs},
            {"text": [chat_text], "videos": [videos], "return_tensors": "pt", **processor_video_kwargs},
            {"text": chat_text, "videos": videos, "return_tensors": "pt", **processor_video_kwargs},
        )
        for base_kwargs in base_attempts:
            for kwargs in ({**base_kwargs, "fps": self.fps}, base_kwargs):
                try:
                    return self.processor(**kwargs)
                except Exception as exc:
                    last_error = exc
                    continue
        raise benchmark.BenchmarkError(
            f"InternVL failed to build inputs for local video inference: {last_error}"
        ) from last_error

    def sample_frames_for_path(self, source_path: str) -> List[object]:
        path = Path(source_path)
        cache_key = (
            str(path.resolve()),
            self.fps,
            min_optional_int(self.target_longest_side, self.processor_video_size),
            self.max_pixels,
            self.processor_video_size,
        )
        if self.frame_cache_size > 0 and cache_key in self.frame_cache:
            frames = self.frame_cache.pop(cache_key)
            self.frame_cache[cache_key] = frames
            return frames

        frames = sample_video_frames_with_cv2_at_fps(
            path,
            fps=self.fps,
            target_longest_side=min_optional_int(self.target_longest_side, self.processor_video_size),
            max_pixels=self.max_pixels,
            pad_to_square_size=self.processor_video_size,
        )
        if self.frame_cache_size > 0:
            self.frame_cache[cache_key] = frames
            while len(self.frame_cache) > self.frame_cache_size:
                self.frame_cache.popitem(last=False)
        return frames

    def prepare_video_path(self, source_path: str) -> str:
        source = Path(source_path)
        if source.suffix.lower() == ".mp4":
            return str(source)
        cache_key = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:16]
        cached_path = self.video_cache_dir / f"{source.stem}_{cache_key}.mp4"
        if cached_path.exists() and cached_path.stat().st_size > 0:
            return str(cached_path)
        transcode_video_with_cv2(source, cached_path)
        return str(cached_path)

    def build_safe_tail_trim_copy(self, source_path: str, drop_last_frames: int) -> str:
        source = Path(source_path)
        cache_key = hashlib.sha1(f"{source.resolve()}::{drop_last_frames}".encode("utf-8")).hexdigest()[:16]
        safe_path = self.video_cache_dir / f"{source.stem}_{cache_key}_trim.mp4"
        if safe_path.exists() and safe_path.stat().st_size > 0:
            return str(safe_path)
        transcode_video_with_cv2(source, safe_path, drop_last_frames=drop_last_frames)
        return str(safe_path)

    def release_runtime_memory(self) -> None:
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        self.release_runtime_memory()


def move_batch_to_model_device(batch: object, device: object) -> object:
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def resolve_model_device(model: object) -> object:
    if hasattr(model, "device"):
        return getattr(model, "device")
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return "cpu"


def infer_native_vision_size(model: object) -> int | None:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    image_size = getattr(vision_config, "image_size", None)
    if isinstance(image_size, int):
        return image_size
    if isinstance(image_size, (list, tuple)) and image_size:
        return int(min(image_size))
    return None


def min_optional_int(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def image_seq_length_for_video_size(video_size: int) -> int:
    # InternVL3.5 uses a 14px ViT patch and 0.5 pixel-shuffle downsampling.
    return (video_size // 28) ** 2


def normalize_generation_config(generation_config: Dict[str, object]) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    max_new_tokens = generation_config.get("max_new_tokens") or generation_config.get("max_output_tokens")
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = int(max_new_tokens)
    temperature = generation_config.get("temperature")
    if temperature is None or float(temperature) <= 0.0:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = float(temperature)
    return kwargs


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    benchmark.ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def transcode_video_with_cv2(source_path: Path, output_path: Path, drop_last_frames: int = 0) -> None:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise benchmark.BenchmarkError(f"Failed to open source video for local runner: {source_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        if width <= 0 or height <= 0:
            raise benchmark.BenchmarkError(f"Could not determine frame size for {source_path}")

        benchmark.ensure_dir(output_path.parent)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise benchmark.BenchmarkError(f"Failed to create cached mp4 for {source_path}")

        try:
            wrote_frames = 0
            tail_buffer: List[object] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if drop_last_frames > 0:
                    tail_buffer.append(frame)
                    if len(tail_buffer) <= drop_last_frames:
                        continue
                    frame = tail_buffer.pop(0)
                writer.write(frame)
                wrote_frames += 1
        finally:
            writer.release()

        if wrote_frames == 0 or not output_path.exists() or output_path.stat().st_size == 0:
            raise benchmark.BenchmarkError(f"Cached mp4 conversion produced an empty file for {source_path}")
    finally:
        capture.release()


def sample_video_frames_with_cv2(source_path: Path, frame_count: int, resize: int | None) -> List[object]:
    if frame_count <= 0:
        raise benchmark.BenchmarkError("--frame-sample-count must be positive when provided")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise benchmark.BenchmarkError(f"Failed to open video for frame sampling: {source_path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            raise benchmark.BenchmarkError(f"Could not determine frame count for {source_path}")

        if frame_count == 1:
            indices = [total_frames // 2]
        else:
            indices = [round(i * (total_frames - 1) / (frame_count - 1)) for i in range(frame_count)]

        frames: List[object] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                if resize is not None:
                    frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not frames:
            raise benchmark.BenchmarkError(f"Frame sampling produced no frames for {source_path}")
        return frames
    finally:
        capture.release()


def sample_video_frames_with_cv2_at_fps(
    source_path: Path,
    fps: float,
    target_longest_side: int | None,
    max_pixels: int | None,
    pad_to_square_size: int | None,
) -> List[object]:
    if fps <= 0:
        raise benchmark.BenchmarkError("Video sampling fps must be positive")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise benchmark.BenchmarkError(f"Failed to open video for fps sampling: {source_path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if source_fps <= 0 or total_frames <= 0:
            raise benchmark.BenchmarkError(f"Could not determine fps/frame count for {source_path}")

        duration_seconds = total_frames / source_fps
        frame_count = max(1, int(math.ceil(duration_seconds * fps)))
        indices = [
            min(total_frames - 1, int(round(index * source_fps / fps)))
            for index in range(frame_count)
        ]

        frames: List[object] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            frame = resize_frame_for_policy(
                frame,
                target_longest_side=target_longest_side,
                max_pixels=max_pixels,
            )
            if pad_to_square_size is not None:
                frame = pad_frame_to_square(frame, pad_to_square_size)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not frames:
            raise benchmark.BenchmarkError(f"FPS sampling produced no frames for {source_path}")
        return frames
    finally:
        capture.release()


def resize_frame_for_policy(frame: object, target_longest_side: int | None, max_pixels: int | None) -> object:
    height, width = frame.shape[:2]
    scale = 1.0
    if target_longest_side is not None and target_longest_side > 0:
        scale = min(scale, target_longest_side / max(height, width))
    if max_pixels is not None and max_pixels > 0 and height * width > max_pixels:
        scale = min(scale, (max_pixels / float(height * width)) ** 0.5)
    if scale >= 1.0:
        return frame
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def pad_frame_to_square(frame: object, square_size: int) -> object:
    if square_size <= 0:
        return frame
    height, width = frame.shape[:2]
    if height > square_size or width > square_size:
        scale = min(square_size / height, square_size / width)
        frame = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        height, width = frame.shape[:2]
    top = (square_size - height) // 2
    bottom = square_size - height - top
    left = (square_size - width) // 2
    right = square_size - width - left
    return cv2.copyMakeBorder(
        frame,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def pad_video_frame_lists_to_equal_length(videos: List[List[object]]) -> List[List[object]]:
    if len(videos) <= 1:
        return videos
    max_length = max(len(frames) for frames in videos)
    padded_videos: List[List[object]] = []
    for frames in videos:
        if not frames:
            raise benchmark.BenchmarkError("Cannot pad an empty decoded video")
        if len(frames) < max_length:
            frames = [*frames, *([frames[-1]] * (max_length - len(frames)))]
        padded_videos.append(frames)
    return padded_videos


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
