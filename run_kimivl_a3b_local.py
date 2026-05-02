#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
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
DEFAULT_MODEL_ID = "kimivl_a3b_instruct"
DEFAULT_MODEL_NAME = "moonshotai/Kimi-VL-A3B-Instruct"
ANSWER_JSON_PREFIX = '{"answer":"'
ANSWER_JSON_SUFFIX = '"}'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local Kimi-VL A3B Instruct against exported benchmark model_requests and "
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
    parser.add_argument("--overwrite", action="store_true")
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
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Execution device preference.",
    )
    parser.add_argument(
        "--quantization",
        default="4bit",
        choices=("none", "8bit", "4bit"),
        help="Quantize model weights to reduce memory without changing video input.",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        default="nf4",
        choices=("nf4", "fp4"),
    )
    parser.add_argument(
        "--bnb-4bit-double-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--llm-int8-cpu-offload", action="store_true")
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "local_runner_cache" / "offload_kimivl_a3b",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "local_runner_cache" / "hf",
        help="Hugging Face cache directory for model weights and remote code.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate request loading and frame extraction without loading Kimi-VL or writing predictions.",
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

    runner: KimiVLA3BRunner | None = None
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
            if provider != "kimi":
                raise benchmark.BenchmarkError(
                    f"Provider {provider!r} is not supported by this runner. Expected provider='kimi'."
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
                runner = KimiVLA3BRunner(
                    model_name=args.model_name,
                    generation_config=as_dict(request_rows[0].get("generation_config", {}), label="generation_config"),
                    video_runtime_config=video_runtime_config,
                    device=args.device,
                    quantization=args.quantization,
                    bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
                    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                    bnb_4bit_double_quant=args.bnb_4bit_double_quant,
                    llm_int8_cpu_offload=args.llm_int8_cpu_offload,
                    offload_folder=args.offload_folder.resolve(),
                    hf_cache_dir=args.hf_cache_dir.resolve(),
                    jpeg_quality=args.jpeg_quality,
                    frame_cache_size=args.frame_cache_size,
                    dry_run=args.dry_run,
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
                "provider": "kimi",
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
        "frame_transport": "kimivl_transformers_pil_images",
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


class KimiVLA3BRunner:
    def __init__(
        self,
        model_name: str,
        generation_config: Dict[str, object],
        video_runtime_config: Dict[str, object],
        device: str,
        quantization: str,
        bnb_4bit_compute_dtype: str,
        bnb_4bit_quant_type: str,
        bnb_4bit_double_quant: bool,
        llm_int8_cpu_offload: bool,
        offload_folder: Path,
        hf_cache_dir: Path,
        jpeg_quality: int,
        frame_cache_size: int,
        dry_run: bool,
    ) -> None:
        self.model_name = model_name
        self.base_generation_config = dict(generation_config)
        self.video_runtime_config = dict(video_runtime_config)
        self.jpeg_quality = max(1, min(int(jpeg_quality), 100))
        self.frame_cache_size = max(int(frame_cache_size), 0)
        self.frame_cache: OrderedDict[str, List[Dict[str, object]]] = OrderedDict()
        self.dry_run = bool(dry_run)
        self.model_runtime_config = {
            "model_name": model_name,
            "quantization": quantization,
            "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype if quantization == "4bit" else None,
            "bnb_4bit_quant_type": bnb_4bit_quant_type if quantization == "4bit" else None,
            "bnb_4bit_double_quant": bool(bnb_4bit_double_quant) if quantization == "4bit" else None,
            "llm_int8_cpu_offload": bool(llm_int8_cpu_offload) if quantization == "8bit" else None,
            "device": device,
            "hf_cache_dir": str(hf_cache_dir),
        }

        if self.dry_run:
            self.torch = None
            self.model = None
            self.processor = None
            return

        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        import torch

        try:
            from transformers import AutoConfig, AutoProcessor
        except ImportError as exc:
            raise benchmark.BenchmarkError("This runner requires Transformers with AutoModelForCausalLM.") from exc
        ensure_transformers_remote_code_compat()

        self.torch = torch
        benchmark.ensure_dir(offload_folder)
        benchmark.ensure_dir(hf_cache_dir)
        model_kwargs: Dict[str, object] = {
            "trust_remote_code": True,
            "torch_dtype": "auto",
            "cache_dir": str(hf_cache_dir),
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"

        if quantization != "none":
            if device == "cpu" or not torch.cuda.is_available():
                raise benchmark.BenchmarkError("bitsandbytes quantization requires CUDA; use --device auto or --device cuda.")
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise benchmark.BenchmarkError("Transformers BitsAndBytesConfig is required for --quantization.") from exc
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

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, cache_dir=str(hf_cache_dir))
        normalize_kimivl_config(config)
        model_class = load_patched_kimivl_model_class(model_name=model_name, hf_cache_dir=hf_cache_dir)
        self.model = model_class.from_pretrained(model_name, config=config, **model_kwargs)
        patch_kimivl_language_prepare(self.model)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=str(hf_cache_dir))
        self.model.eval()
        if quantization == "none" and device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        elif quantization == "none" and device == "cpu":
            self.model = self.model.to("cpu")

    def run_request(self, row: Dict[str, object]) -> Dict[str, object]:
        request = as_dict(row.get("request"), label="request")
        prompt = str(request["prompt"])
        video_inputs = self.extract_request_video_inputs(request)
        if not video_inputs:
            raise benchmark.BenchmarkError("KimiVL request is missing request.inputs video paths")

        images, frame_summary = self.build_images_and_summary(video_inputs)
        if self.dry_run:
            raw_response = json.dumps({"answer": "A"}, separators=(",", ":"))
            predicted_answer = "A"
            answer_source = "dry_run"
            initial_response = None
        else:
            raw_response, predicted_answer, answer_source, initial_response = self.generate_answer(
                prompt=prompt,
                video_inputs=video_inputs,
                images=images,
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
            "frame_summary": frame_summary,
        }
        if initial_response is not None and initial_response != raw_response:
            prediction["raw_response_initial"] = initial_response
        return prediction

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

    def build_images_and_summary(
        self,
        video_inputs: List[Dict[str, object]],
    ) -> Tuple[List[object], List[Dict[str, object]]]:
        images = []
        frame_summary = []
        for video_input in video_inputs:
            frames = self.get_video_frames(str(video_input["path"]))
            images.extend(frame["image"] for frame in frames)
            frame_summary.append(
                {
                    "video_index": int(video_input["video_index"]),
                    "modality": str(video_input["modality"]),
                    "path": str(video_input["path"]),
                    "frame_count": len(frames),
                    "first_timestamp": frames[0]["timestamp_sec"] if frames else None,
                    "last_timestamp": frames[-1]["timestamp_sec"] if frames else None,
                    "first_frame_size": [frames[0]["width"], frames[0]["height"]] if frames else None,
                    "last_frame_size": [frames[-1]["width"], frames[-1]["height"]] if frames else None,
                }
            )
        return images, frame_summary

    def generate_answer(
        self,
        prompt: str,
        video_inputs: List[Dict[str, object]],
        images: List[object],
    ) -> Tuple[str, str | None, str, str | None]:
        prompt_text = self.build_prompt_text(prompt=prompt, video_inputs=video_inputs)
        initial_response = self.generate_prefilled_json(prompt_text=prompt_text, images=images)
        predicted_answer, _ = benchmark.normalize_predicted_answer({"raw_response": initial_response})
        if predicted_answer is not None:
            canonical = canonical_json_answer(predicted_answer)
            return canonical, predicted_answer, "json_prefill", initial_response

        fallback_answer = self.score_answer_options(prompt_text=prompt_text, images=images)
        canonical = canonical_json_answer(fallback_answer)
        return canonical, fallback_answer, "option_scoring_fallback", initial_response

    def generate_prefilled_json(self, prompt_text: str, images: List[object]) -> str:
        if self.model is None or self.processor is None or self.torch is None:
            raise benchmark.BenchmarkError("KimiVL model is not initialized")
        inputs = self.build_model_inputs(prompt_text=f"{prompt_text}{ANSWER_JSON_PREFIX}", images=images)
        inputs = move_batch_to_model_device(inputs, resolve_model_device(self.model))
        generate_kwargs = normalize_generation_config(self.base_generation_config, max_new_tokens_override=8)
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = [
            output_ids_row[len(input_ids_row):]
            for input_ids_row, output_ids_row in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        completion = str(output_text[0]).strip() if output_text else ""
        return f"{ANSWER_JSON_PREFIX}{completion}"

    def score_answer_options(self, prompt_text: str, images: List[object]) -> str:
        if self.model is None or self.processor is None or self.torch is None:
            raise benchmark.BenchmarkError("KimiVL model is not initialized")
        prefix_text = f"{prompt_text}{ANSWER_JSON_PREFIX}"
        prefix_inputs = self.build_model_inputs(prompt_text=prefix_text, images=images)
        prefix_len = int(prefix_inputs["input_ids"].shape[1])

        best_answer = "A"
        best_score = float("-inf")
        for answer in benchmark.VALID_ANSWERS:
            candidate_text = f"{prefix_text}{answer}{ANSWER_JSON_SUFFIX}"
            candidate_inputs = self.build_model_inputs(prompt_text=candidate_text, images=images)
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

    def build_model_inputs(self, prompt_text: str, images: List[object]) -> object:
        if self.processor is None:
            raise benchmark.BenchmarkError("KimiVL processor is not initialized")
        return self.processor(
            images=images,
            text=prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    def build_prompt_text(self, prompt: str, video_inputs: List[Dict[str, object]]) -> str:
        messages = self.build_messages(prompt=prompt, video_inputs=video_inputs)
        return self.build_chat_template(messages)

    def build_messages(self, prompt: str, video_inputs: List[Dict[str, object]]) -> List[Dict[str, object]]:
        content: List[Dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    f"{prompt}\n\n"
                    "The videos are provided below as chronological frame samples. "
                    "Use the frame order and timestamp labels when reasoning about temporal events."
                ),
            }
        ]
        for video_input in video_inputs:
            video_index = int(video_input["video_index"])
            modality = str(video_input["modality"])
            frames = self.get_video_frames(str(video_input["path"]))
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
                content.append({"type": "image", "image": ""})
        return [{"role": "user", "content": content}]

    def build_chat_template(self, messages: List[Dict[str, object]]) -> object:
        try:
            return self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        except TypeError:
            return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

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
        )
        if self.frame_cache_size > 0:
            self.frame_cache[cache_key] = frames
            while len(self.frame_cache) > self.frame_cache_size:
                self.frame_cache.popitem(last=False)
        return frames

    def release_runtime_memory(self) -> None:
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def close(self) -> None:
        self.frame_cache.clear()
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        self.release_runtime_memory()


def extract_video_frames(
    video_path: Path,
    fps: float,
    max_pixels: int,
    max_frames_per_video: Optional[int],
) -> List[Dict[str, object]]:
    if fps <= 0:
        raise benchmark.BenchmarkError(f"fps must be positive, got {fps}")
    if max_pixels <= 0:
        raise benchmark.BenchmarkError(f"max_pixels must be positive, got {max_pixels}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise benchmark.BenchmarkError(f"Failed to open video for KimiVL runner: {video_path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = 30.0
        frames = extract_frames_sequential(capture, source_fps, fps, max_pixels)
    finally:
        capture.release()

    if max_frames_per_video is not None and max_frames_per_video > 0 and len(frames) > max_frames_per_video:
        frames = uniform_subsample(frames, max_frames_per_video)
    if not frames:
        raise benchmark.BenchmarkError(f"No frames extracted from {video_path}")
    return frames


def extract_frames_sequential(
    capture: cv2.VideoCapture,
    source_fps: float,
    target_fps: float,
    max_pixels: int,
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
            frames.append(encode_frame(frame, timestamp_sec=timestamp_sec, max_pixels=max_pixels))
            next_sample_time += sample_interval
        frame_index += 1
    return frames


def encode_frame(frame: object, timestamp_sec: float, max_pixels: int) -> Dict[str, object]:
    from PIL import Image

    resized = resize_to_pixel_budget(frame, max_pixels=max_pixels)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGB")
    height, width = resized.shape[:2]
    return {
        "timestamp_sec": float(timestamp_sec),
        "width": int(width),
        "height": int(height),
        "image": image,
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


def ensure_transformers_remote_code_compat() -> None:
    try:
        import transformers.utils.import_utils as import_utils
    except ImportError:
        import_utils = None
    if import_utils is not None and not hasattr(import_utils, "is_torch_fx_available"):

        def is_torch_fx_available() -> bool:
            try:
                import torch.fx  # noqa: F401
            except Exception:
                return False
            return True

        import_utils.is_torch_fx_available = is_torch_fx_available
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return
    if not hasattr(DynamicCache, "seen_tokens"):

        @property
        def seen_tokens(self: object) -> int:
            get_seq_length = getattr(self, "get_seq_length", None)
            if callable(get_seq_length):
                return int(get_seq_length())
            return 0

        DynamicCache.seen_tokens = seen_tokens
    if not hasattr(DynamicCache, "get_usable_length"):

        def get_usable_length(self: object, new_seq_length: object = None, layer_idx: int = 0) -> int:
            get_seq_length = getattr(self, "get_seq_length", None)
            if callable(get_seq_length):
                try:
                    return int(get_seq_length(layer_idx))
                except TypeError:
                    return int(get_seq_length())
            return 0

        DynamicCache.get_usable_length = get_usable_length
    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls: type, past_key_values: object = None) -> object:
            if past_key_values is None:
                return cls()
            if isinstance(past_key_values, cls):
                return past_key_values
            return cls(ddp_cache_data=past_key_values)

        DynamicCache.from_legacy_cache = from_legacy_cache
    if not hasattr(DynamicCache, "to_legacy_cache"):

        def to_legacy_cache(self: object) -> Tuple[object, ...]:
            legacy_layers = []
            for layer in getattr(self, "layers", []):
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is None:
                    keys = getattr(layer, "key_cache", None)
                if values is None:
                    values = getattr(layer, "value_cache", None)
                if keys is not None and values is not None:
                    legacy_layers.append((keys, values))
            return tuple(legacy_layers)

        DynamicCache.to_legacy_cache = to_legacy_cache
    if not hasattr(DynamicCache, "_kimivl_truthy_cache_patch"):

        def __bool__(self: object) -> bool:
            return True

        DynamicCache.__bool__ = __bool__
        DynamicCache._kimivl_truthy_cache_patch = True


def normalize_kimivl_config(config: object) -> None:
    text_config = getattr(config, "text_config", None)
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "default" and "type" not in rope_scaling:
        text_config.rope_scaling = None


def load_patched_kimivl_model_class(model_name: str, hf_cache_dir: Path) -> type:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_class = get_class_from_dynamic_module(
        "modeling_kimi_vl.KimiVLForConditionalGeneration",
        model_name,
        cache_dir=str(hf_cache_dir),
    )
    original_tie_weights = model_class.tie_weights
    if not getattr(original_tie_weights, "_kimivl_accepts_extra_args", False):

        def tie_weights(self: object, *args: object, **kwargs: object) -> object:
            return original_tie_weights(self)

        tie_weights._kimivl_accepts_extra_args = True
        model_class.tie_weights = tie_weights

    original_prepare_inputs = model_class.prepare_inputs_for_generation
    if not getattr(original_prepare_inputs, "_kimivl_cache_position_patch", False):

        def prepare_inputs_for_generation(
            self: object,
            input_ids: object,
            past_key_values: object = None,
            inputs_embeds: object = None,
            pixel_values: object = None,
            attention_mask: object = None,
            image_grid_hws: object = None,
            cache_position: object = None,
            **kwargs: object,
        ) -> object:
            if cache_position is None:
                import torch

                if input_ids is not None:
                    seq_len = int(input_ids.shape[-1])
                    device = input_ids.device
                elif inputs_embeds is not None:
                    seq_len = int(inputs_embeds.shape[-2])
                    device = inputs_embeds.device
                else:
                    seq_len = 1
                    device = "cpu"

                cache_start = 0
                get_seq_length = getattr(past_key_values, "get_seq_length", None)
                if callable(get_seq_length):
                    cache_start = int(get_seq_length())
                else:
                    seen_tokens = getattr(past_key_values, "seen_tokens", None)
                    if seen_tokens is not None:
                        cache_start = int(seen_tokens)
                cache_position = torch.arange(cache_start, cache_start + seq_len, device=device)

            return original_prepare_inputs(
                self,
                input_ids=input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_hws=image_grid_hws,
                cache_position=cache_position,
                **kwargs,
            )

        prepare_inputs_for_generation._kimivl_cache_position_patch = True
        model_class.prepare_inputs_for_generation = prepare_inputs_for_generation
    return model_class


def patch_kimivl_language_prepare(model: object) -> None:
    import types

    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return
    original_prepare = getattr(language_model, "prepare_inputs_for_generation", None)
    if original_prepare is None or getattr(original_prepare, "_kimivl_position_ids_patch", False):
        return

    def prepare_inputs_for_generation(
        self: object,
        input_ids: object,
        past_key_values: object = None,
        attention_mask: object = None,
        inputs_embeds: object = None,
        **kwargs: object,
    ) -> Dict[str, object]:
        model_inputs = original_prepare(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        position_ids = model_inputs.get("position_ids")
        target_len = None
        prepared_input_ids = model_inputs.get("input_ids")
        prepared_inputs_embeds = model_inputs.get("inputs_embeds")
        if prepared_input_ids is not None:
            target_len = int(prepared_input_ids.shape[1])
        elif prepared_inputs_embeds is not None:
            target_len = int(prepared_inputs_embeds.shape[1])
        if position_ids is not None and target_len is not None and int(position_ids.shape[-1]) != target_len:
            model_inputs["position_ids"] = position_ids[:, -target_len:]
        prepared_attention_mask = model_inputs.get("attention_mask")
        if (
            attention_mask is not None
            and prepared_attention_mask is not None
            and int(prepared_attention_mask.shape[-1]) < int(attention_mask.shape[-1])
        ):
            model_inputs["attention_mask"] = attention_mask
        return model_inputs

    prepare_inputs_for_generation._kimivl_position_ids_patch = True
    language_model.prepare_inputs_for_generation = types.MethodType(prepare_inputs_for_generation, language_model)


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
