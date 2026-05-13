#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
from raven_benchmark import benchmark


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval_outputs"
DEFAULT_MODEL_REQUESTS_DIR = DEFAULT_OUTPUT_DIR / "model_requests"
DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_DIR / "predictions"
DEFAULT_MODEL_ID = "videollama3_7b"
DEFAULT_MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local VideoLLaMA3-7B against exported benchmark model_requests and "
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
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--video-size", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--video-cache-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "local_runner_cache" / "videos_videollama3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_request_dir = args.model_requests_dir.resolve() / args.model_id
    if not model_request_dir.exists():
        raise benchmark.BenchmarkError(f"Model request directory does not exist: {model_request_dir}")

    predictions_root = args.predictions_dir.resolve() / args.model_id
    benchmark.ensure_dir(predictions_root)

    runner: VideoLLaMA3Runner | None = None
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
        if provider != "videollama":
            raise benchmark.BenchmarkError(
                f"Provider {provider!r} is not supported by this runner. Expected provider='videollama'."
            )

        if runner is None:
            runner = VideoLLaMA3Runner.from_request_row(
                request_rows[0],
                model_name=args.model_name,
                fps=args.fps,
                max_frames=args.max_frames,
                video_size=args.video_size,
                device=args.device,
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
                "provider": "videollama",
                "settings": completed_settings,
                "request_count": total_requests,
                "predictions_dir": str(predictions_root),
            },
            indent=2,
        )
    )


class VideoLLaMA3Runner:
    def __init__(
        self,
        model_name: str,
        generation_config: Dict[str, object],
        fps: float,
        max_frames: int,
        video_size: int | None,
        device: str,
        video_cache_dir: Path,
    ) -> None:
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        python_bin_dir = Path(sys.executable).resolve().parent
        os.environ["PATH"] = f"{python_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        import torch

        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise benchmark.BenchmarkError(
                "This runner requires a recent Transformers build with AutoModelForCausalLM support."
            ) from exc
        try:
            from transformers import VideoLlama3ForConditionalGeneration, VideoLlama3Processor
        except ImportError:
            VideoLlama3ForConditionalGeneration = None
            VideoLlama3Processor = None

        self.torch = torch
        self.model_name = model_name
        self.base_generation_config = dict(generation_config)
        self.fps = fps
        self.max_frames = max_frames
        self.video_size = video_size
        self.video_cache_dir = video_cache_dir
        self.processor_style = "remote"
        benchmark.ensure_dir(self.video_cache_dir)

        dtype = torch.bfloat16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        device_map = "auto" if device == "auto" else None
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        native_supported = getattr(model_config, "model_type", None) == "video_llama_3"
        if native_supported and VideoLlama3ForConditionalGeneration is not None and VideoLlama3Processor is not None:
            try:
                self.model = VideoLlama3ForConditionalGeneration.from_pretrained(
                    model_name,
                    config=model_config,
                    dtype=dtype,
                    device_map=device_map,
                    attn_implementation="sdpa",
                )
                self.processor = VideoLlama3Processor.from_pretrained(model_name)
                self.processor_style = "native"
            except RuntimeError as exc:
                if "ignore_mismatched_sizes" not in str(exc):
                    raise
                self.release_runtime_memory()
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=model_config,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
                ensure_videollama3_processor_compat()
                self.processor = load_videollama3_remote_processor(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=model_config,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            ensure_videollama3_processor_compat()
            self.processor = load_videollama3_remote_processor(model_name)
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        elif device == "cpu":
            self.model = self.model.to("cpu")

    @classmethod
    def from_request_row(
        cls,
        row: Dict[str, object],
        model_name: str,
        fps: float,
        max_frames: int,
        video_size: int | None,
        device: str,
        video_cache_dir: Path,
    ) -> "VideoLLaMA3Runner":
        return cls(
            model_name=model_name,
            generation_config=as_dict(row.get("generation_config"), label="generation_config"),
            fps=fps,
            max_frames=max_frames,
            video_size=video_size,
            device=device,
            video_cache_dir=video_cache_dir,
        )

    def run_request(self, row: Dict[str, object]) -> Dict[str, object]:
        provider_payload = as_dict(row.get("provider_payload"), label="provider_payload")
        request = as_dict(row.get("request"), label="request")
        prompt = str(provider_payload.get("prompt") or request["prompt"])
        video_paths = [self.prepare_video_path(str(path)) for path in provider_payload.get("video_paths", [])]
        if not video_paths:
            raise benchmark.BenchmarkError("VideoLLaMA request is missing video_paths")

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
        inputs = self.build_inputs(prompt=prompt, video_paths=video_paths)
        inputs = move_batch_to_model_device(inputs, resolve_model_device(self.model))
        inputs = align_multimodal_tensor_dtypes(inputs, self.model)
        generate_kwargs = normalize_generation_config(
            generation_config=as_dict(generation_config.get("generation_config", {}), label="generation_config")
        )
        if "pad_token_id" not in generate_kwargs and getattr(self.model, "config", None) is not None:
            eos = getattr(self.model.config, "eos_token_id", None)
            if eos is not None:
                generate_kwargs["pad_token_id"] = eos
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        prompt_len = int(inputs["input_ids"].shape[1])
        output_text = self.processor.batch_decode(
            output_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        decoded = str(output_text[0]).strip() if output_text else ""
        if decoded:
            return decoded

        # Some remote-code generation paths return only newly generated ids,
        # unlike the standard generate() full-sequence convention.
        output_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return str(output_text[0]).strip() if output_text else ""

    def build_inputs(self, prompt: str, video_paths: List[str]) -> object:
        if self.processor_style == "remote":
            content = [*({"type": "video", "video": self.build_remote_video_payload(path)} for path in video_paths)]
            content.append({"type": "text", "text": prompt})
            conversation = [{"role": "user", "content": content}]
            try:
                return self.processor(conversation=conversation, return_tensors="pt", add_generation_prompt=True)
            except TypeError:
                return self.processor(conversation, return_tensors="pt")

        conversation = [
            {
                "role": "user",
                "content": [*({"type": "video"} for _ in video_paths), {"type": "text", "text": prompt}],
            }
        ]
        chat_text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.processor(
            text=[chat_text],
            videos=video_paths,
            return_tensors="pt",
            fps=self.fps,
            max_frames=self.max_frames,
        )

    def build_remote_video_payload(self, path: str) -> Dict[str, object]:
        payload: Dict[str, object] = {"video_path": path, "fps": self.fps, "max_frames": self.max_frames}
        if self.video_size is not None:
            payload["size"] = self.video_size
            payload["size_divisible"] = 14
        return payload

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


def resolve_model_dtype(model: object) -> object:
    if hasattr(model, "dtype"):
        return getattr(model, "dtype")
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).dtype
        except StopIteration:
            pass
    return None


def align_multimodal_tensor_dtypes(batch: object, model: object) -> object:
    dtype = resolve_model_dtype(model)
    if dtype is None:
        return batch
    for key in ("pixel_values", "pixel_values_videos", "image_values", "video_values"):
        try:
            value = batch.get(key)
        except AttributeError:
            continue
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            batch[key] = value.to(dtype=dtype)
    return batch


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


def ensure_videollama3_processor_compat() -> None:
    try:
        import transformers.image_utils as image_utils
    except ImportError:
        return
    if not hasattr(image_utils, "VideoInput"):
        from typing import Any

        image_utils.VideoInput = Any


def load_videollama3_remote_processor(model_name: str) -> object:
    import types

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    processor_class = get_class_from_dynamic_module(
        "processing_videollama3.Videollama3Qwen2Processor",
        model_name,
        trust_remote_code=True,
    )
    args = processor_class._get_arguments_from_pretrained(model_name, trust_remote_code=True)
    processor = processor_class(*args)

    def _merge_kwargs(self, _model_processor_kwargs, tokenizer_init_kwargs=None, **kwargs):
        return_tensors = kwargs.get("return_tensors")
        text_kwargs = {
            "padding": kwargs.get("padding", False),
            "padding_side": kwargs.get("padding_side", getattr(self.tokenizer, "padding_side", "right")),
        }
        image_kwargs: Dict[str, object] = {}
        if return_tensors is not None:
            text_kwargs["return_tensors"] = return_tensors
            image_kwargs["return_tensors"] = return_tensors
        return {
            "text_kwargs": text_kwargs,
            "images_kwargs": image_kwargs,
            "audio_kwargs": {},
            "videos_kwargs": {},
            "chat_template_kwargs": {
                "chat_template": kwargs.get("chat_template"),
                "add_system_prompt": kwargs.get("add_system_prompt", False),
                "add_generation_prompt": kwargs.get("add_generation_prompt", False),
            },
            "common_kwargs": {},
        }

    processor._merge_kwargs = types.MethodType(_merge_kwargs, processor)
    return processor


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
