#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import eo_ir_benchmark as benchmark


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval_outputs"
DEFAULT_MODEL_REQUESTS_DIR = DEFAULT_OUTPUT_DIR / "model_requests"
DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_DIR / "predictions"
DEFAULT_MODEL_ID = "qwen3_5_9b"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B"
ANSWER_JSON_PREFIX = '{"answer":"'
ANSWER_JSON_SUFFIX = '"}'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local Qwen3.5-9B against exported benchmark model_requests and write "
            "predictions/{model_id}/{setting}.jsonl."
        )
    )
    parser.add_argument(
        "--model-requests-dir",
        type=Path,
        default=DEFAULT_MODEL_REQUESTS_DIR,
        help="Directory containing model_requests/{model_id}/{setting}.jsonl.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help="Directory where predictions/{model_id}/{setting}.jsonl will be written.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Model identifier matching a subdirectory under model_requests/. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Transformers checkpoint name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=list(benchmark.VALID_SETTINGS),
        choices=benchmark.VALID_SETTINGS,
        help="Settings to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N requests in each setting.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video sampling rate. By default this is read from request.video_input.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Override minimum pixels passed to AutoProcessor. By default this is read from request.video_input.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Override maximum pixels passed to AutoProcessor. By default this is read from request.video_input.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Execution device preference.",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=("none", "8bit", "4bit"),
        help=(
            "Quantize model weights to reduce memory without changing video input. "
            "Use with official video_input to keep frame rate/resolution unchanged."
        ),
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Compute dtype for bitsandbytes 4-bit quantization.",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        default="nf4",
        choices=("nf4", "fp4"),
        help="Quantization type for bitsandbytes 4-bit quantization.",
    )
    parser.add_argument(
        "--bnb-4bit-double-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable nested/double quantization for bitsandbytes 4-bit loading.",
    )
    parser.add_argument(
        "--llm-int8-cpu-offload",
        action="store_true",
        help="Enable fp32 CPU offload for bitsandbytes 8-bit loading.",
    )
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "local_runner_cache" / "offload_qwen35",
        help="Folder used by Transformers for optional CPU/disk offload.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction files.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Keep Qwen3.5 thinking mode enabled. By default this runner requests direct non-thinking answers.",
    )
    parser.add_argument(
        "--video-cache-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "local_runner_cache" / "videos_qwen35",
        help="Directory for local transcoded video copies used by the runner.",
    )
    parser.add_argument(
        "--video-feature-cache-size",
        type=int,
        default=12,
        help="Max number of unique video tuples to keep in the in-memory preprocessed feature cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_request_dir = args.model_requests_dir.resolve() / args.model_id
    if not model_request_dir.exists():
        raise benchmark.BenchmarkError(
            f"Model request directory does not exist: {model_request_dir}. "
            "Export model_requests for model_id=qwen3_5_9b first."
        )

    predictions_root = args.predictions_dir.resolve() / args.model_id
    benchmark.ensure_dir(predictions_root)

    runner: Qwen35Runner | None = None
    completed_settings: List[str] = []
    total_requests = 0

    for setting in args.settings:
        request_path = model_request_dir / f"{setting}.jsonl"
        if not request_path.exists():
            raise benchmark.BenchmarkError(f"Missing model request file: {request_path}")

        output_path = predictions_root / f"{setting}.jsonl"
        request_rows = benchmark.load_jsonl(request_path)
        if args.limit is not None:
            request_rows = request_rows[: args.limit]
        if not request_rows:
            raise benchmark.BenchmarkError(f"No requests found in {request_path}")

        provider = infer_single_provider(request_rows)
        if provider != "qwen":
            raise benchmark.BenchmarkError(
                f"Provider {provider!r} is not supported by this runner. Expected provider='qwen'."
            )

        if runner is None:
            video_runtime_config = resolve_video_runtime_config(args, request_rows[0])
            runner = Qwen35Runner(
                model_name=args.model_name,
                generation_config=as_dict(request_rows[0].get("generation_config"), label="generation_config"),
                fps=video_runtime_config["fps"],
                min_pixels=video_runtime_config["min_pixels"],
                max_pixels=video_runtime_config["max_pixels"],
                device=args.device,
                video_cache_dir=args.video_cache_dir.resolve(),
                enable_thinking=args.enable_thinking,
                video_feature_cache_size=args.video_feature_cache_size,
                video_runtime_config=video_runtime_config,
                quantization=args.quantization,
                bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_double_quant=args.bnb_4bit_double_quant,
                llm_int8_cpu_offload=args.llm_int8_cpu_offload,
                offload_folder=args.offload_folder.resolve(),
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
            runner = None

    print(
        json.dumps(
            {
                "status": "ok",
                "model_id": args.model_id,
                "provider": "qwen",
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
        "source": "cli_override" if any(value is not None for value in (args.fps, args.min_pixels, args.max_pixels)) else "request_video_input",
        "protocol_id": str(video_input.get("protocol_id", "legacy_cli_defaults")) if video_input else "legacy_cli_defaults",
        "fps": float(fps),
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
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


class Qwen35Runner:
    def __init__(
        self,
        model_name: str,
        generation_config: Dict[str, object],
        fps: float,
        min_pixels: int,
        max_pixels: int,
        device: str,
        video_cache_dir: Path,
        enable_thinking: bool,
        video_feature_cache_size: int,
        video_runtime_config: Dict[str, object],
        quantization: str,
        bnb_4bit_compute_dtype: str,
        bnb_4bit_quant_type: str,
        bnb_4bit_double_quant: bool,
        llm_int8_cpu_offload: bool,
        offload_folder: Path,
    ) -> None:
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        import torch

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise benchmark.BenchmarkError(
                "This runner requires a recent Transformers build with "
                "AutoModelForImageTextToText support."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.base_generation_config = dict(generation_config)
        self.fps = fps
        self.device_preference = device
        self.video_cache_dir = video_cache_dir
        self.enable_thinking = enable_thinking
        self.video_runtime_config = dict(video_runtime_config)
        self.model_runtime_config = {
            "quantization": quantization,
            "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype if quantization == "4bit" else None,
            "bnb_4bit_quant_type": bnb_4bit_quant_type if quantization == "4bit" else None,
            "bnb_4bit_double_quant": bool(bnb_4bit_double_quant) if quantization == "4bit" else None,
            "llm_int8_cpu_offload": bool(llm_int8_cpu_offload) if quantization == "8bit" else None,
        }
        self.video_feature_cache_size = max(int(video_feature_cache_size), 1)
        self.video_feature_cache: OrderedDict[Tuple[str, ...], Dict[str, object]] = OrderedDict()
        benchmark.ensure_dir(self.video_cache_dir)
        benchmark.ensure_dir(offload_folder)

        dtype = torch.bfloat16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        device_map = "auto" if device == "auto" else None
        model_kwargs: Dict[str, object] = {
            "dtype": dtype,
            "attn_implementation": "sdpa",
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        if quantization != "none":
            if device == "cpu" or not torch.cuda.is_available():
                raise benchmark.BenchmarkError("bitsandbytes quantization requires CUDA; use --device auto or --device cuda.")
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise benchmark.BenchmarkError(
                    "Transformers BitsAndBytesConfig is required for --quantization."
                ) from exc
            if quantization == "4bit":
                compute_dtype = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                }[bnb_4bit_compute_dtype]
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_double_quant,
                )
            elif quantization == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=llm_int8_cpu_offload,
                )
            model_kwargs["device_map"] = "auto"
            model_kwargs["offload_folder"] = str(offload_folder)

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **model_kwargs,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        except ValueError as exc:
            if "qwen3_5" not in str(exc):
                raise
            raise benchmark.BenchmarkError(
                "The installed Transformers build does not recognize model_type='qwen3_5'. "
                "Upgrade to a newer Transformers build before running Qwen/Qwen3.5-9B."
            ) from exc

        if quantization == "none" and device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        elif quantization == "none" and device == "cpu":
            self.model = self.model.to("cpu")

    def run_request(self, row: Dict[str, object]) -> Dict[str, object]:
        provider_payload = as_dict(row.get("provider_payload"), label="provider_payload")
        request = as_dict(row.get("request"), label="request")
        prompt = str(request["prompt"])
        video_paths = [self.prepare_video_path(str(path)) for path in provider_payload.get("video_paths", [])]
        if not video_paths:
            raise benchmark.BenchmarkError("Qwen request is missing video_paths")

        raw_response, predicted_answer, answer_source, initial_response = self.generate_answer(
            prompt=prompt,
            video_paths=video_paths,
            generation_config=provider_payload,
        )
        self.release_runtime_memory()
        prediction = {
            "model_id": str(row["model_id"]),
            "setting": str(request["setting"]),
            "question_id": str(request["question_id"]),
            "predicted_answer": predicted_answer,
            "raw_response": raw_response,
            "answer_source": answer_source,
            "video_runtime_config": self.video_runtime_config,
            "model_runtime_config": self.model_runtime_config,
        }
        if initial_response is not None and initial_response != raw_response:
            prediction["raw_response_initial"] = initial_response
        return prediction

    def generate_answer(
        self,
        prompt: str,
        video_paths: List[str],
        generation_config: Dict[str, object],
    ) -> Tuple[str, str | None, str, str | None]:
        try:
            return self._generate_answer_once(prompt=prompt, video_paths=video_paths, generation_config=generation_config)
        except IndexError as exc:
            if "Invalid frame index" not in str(exc):
                raise
            safe_paths = [self.build_safe_tail_trim_copy(path, drop_last_frames=3) for path in video_paths]
            return self._generate_answer_once(prompt=prompt, video_paths=safe_paths, generation_config=generation_config)

    def _generate_answer_once(
        self,
        prompt: str,
        video_paths: List[str],
        generation_config: Dict[str, object],
    ) -> Tuple[str, str | None, str, str | None]:
        prompt_text = self.build_prompt_text(prompt=prompt, video_paths=video_paths)
        initial_response = self.generate_prefilled_json(prompt_text=prompt_text, video_paths=video_paths, generation_config=generation_config)
        predicted_answer, _ = benchmark.normalize_predicted_answer({"raw_response": initial_response})
        if predicted_answer is not None:
            canonical = canonical_json_answer(predicted_answer)
            return canonical, predicted_answer, "json_prefill", initial_response

        fallback_answer = self.score_answer_options(prompt_text=prompt_text, video_paths=video_paths)
        canonical = canonical_json_answer(fallback_answer)
        return canonical, fallback_answer, "option_scoring_fallback", initial_response

    def build_prompt_text(self, prompt: str, video_paths: List[str]) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    *({"type": "video", "path": path} for path in video_paths),
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        template_kwargs: Dict[str, object] = {
            "add_generation_prompt": True,
        }
        if not self.enable_thinking:
            template_kwargs["enable_thinking"] = False

        try:
            return self.processor.apply_chat_template(conversation, tokenize=False, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            return self.processor.apply_chat_template(conversation, tokenize=False, **template_kwargs)

    def generate_prefilled_json(
        self,
        prompt_text: str,
        video_paths: List[str],
        generation_config: Dict[str, object],
    ) -> str:
        inputs = self.build_model_inputs(prompt_text=f"{prompt_text}{ANSWER_JSON_PREFIX}", video_paths=video_paths)
        inputs = move_batch_to_model_device(inputs, resolve_model_device(self.model))
        generate_kwargs = normalize_generation_config(
            generation_config=as_dict(generation_config.get("generation_config", {}), label="generation_config"),
            max_new_tokens_override=8,
        )
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
        completion = str(output_text[0]).strip() if output_text else ""
        return f"{ANSWER_JSON_PREFIX}{completion}"

    def score_answer_options(self, prompt_text: str, video_paths: List[str]) -> str:
        prefix_text = f"{prompt_text}{ANSWER_JSON_PREFIX}"
        prefix_inputs = self.build_model_inputs(prompt_text=prefix_text, video_paths=video_paths)
        prefix_len = int(prefix_inputs["input_ids"].shape[1])

        best_answer = "A"
        best_score = float("-inf")
        for answer in benchmark.VALID_ANSWERS:
            candidate_text = f"{prefix_text}{answer}{ANSWER_JSON_SUFFIX}"
            candidate_inputs = self.build_model_inputs(prompt_text=candidate_text, video_paths=video_paths)
            labels = candidate_inputs["input_ids"].clone()
            labels[:, :prefix_len] = -100
            model_inputs = move_batch_to_model_device(candidate_inputs, resolve_model_device(self.model))
            labels = labels.to(model_inputs["input_ids"].device)
            with self.torch.inference_mode():
                outputs = self.model(**model_inputs, labels=labels)
            token_count = int((labels != -100).sum().item())
            score = float(-(outputs.loss.item() * max(token_count, 1)))
            if score > best_score:
                best_score = score
                best_answer = answer

        return best_answer

    def build_model_inputs(self, prompt_text: str, video_paths: List[str]) -> object:
        text = self.expand_video_placeholders(prompt_text, video_paths)
        text_inputs = self.processor.tokenizer(
            [text],
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        self.processor._check_special_mm_tokens([text], text_inputs, modalities=["video"])
        mm_token_type_ids = self.processor.create_mm_token_type_ids(text_inputs["input_ids"])
        text_inputs["mm_token_type_ids"] = self.torch.tensor(mm_token_type_ids, dtype=self.torch.long)

        cached_video_inputs = self.get_cached_video_inputs(video_paths)
        text_inputs["pixel_values_videos"] = cached_video_inputs["pixel_values_videos"].clone()
        text_inputs["video_grid_thw"] = cached_video_inputs["video_grid_thw"].clone()
        return text_inputs

    def get_cached_video_inputs(self, video_paths: List[str]) -> Dict[str, object]:
        cache_key = tuple(video_paths)
        cached = self.video_feature_cache.get(cache_key)
        if cached is not None:
            self.video_feature_cache.move_to_end(cache_key)
            return cached

        processed = self.processor.video_processor(
            videos=video_paths,
            fps=self.fps,
            return_metadata=True,
        )
        cached = {
            "pixel_values_videos": processed["pixel_values_videos"].cpu(),
            "video_grid_thw": processed["video_grid_thw"].cpu(),
            "video_metadata": processed["video_metadata"],
        }
        self.video_feature_cache[cache_key] = cached
        while len(self.video_feature_cache) > self.video_feature_cache_size:
            self.video_feature_cache.popitem(last=False)
        return cached

    def expand_video_placeholders(self, prompt_text: str, video_paths: List[str]) -> str:
        cached_video_inputs = self.get_cached_video_inputs(video_paths)
        text = prompt_text
        merge_length = self.processor.video_processor.merge_size**2
        video_grid_thw = cached_video_inputs["video_grid_thw"]
        video_metadata = cached_video_inputs["video_metadata"]

        for index in range(len(video_paths)):
            metadata = video_metadata[index]
            if metadata.fps is None:
                metadata.fps = 24
            curr_timestamp = self.processor._calculate_timestamps(
                metadata.frames_indices,
                metadata.fps,
                self.processor.video_processor.temporal_patch_size,
            )
            video_placeholder = ""
            frame_count = int(video_grid_thw[index][0].item())
            frame_seqlen = int(video_grid_thw[index][1:].prod().item()) // merge_length
            for frame_idx in range(frame_count):
                curr_time = curr_timestamp[frame_idx]
                video_placeholder += f"<{curr_time:.1f} seconds>"
                video_placeholder += (
                    self.processor.vision_start_token
                    + self.processor.video_token * frame_seqlen
                    + self.processor.vision_end_token
                )

            wrapped_token = (
                f"{self.processor.vision_start_token}{self.processor.video_token}{self.processor.vision_end_token}"
            )
            if wrapped_token in text:
                text = text.replace(wrapped_token, video_placeholder, 1)
            else:
                text = text.replace(self.processor.video_token, video_placeholder, 1)

        return text

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
        self.video_feature_cache.clear()
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
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
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
