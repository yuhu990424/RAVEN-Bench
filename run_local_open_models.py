#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import eo_ir_benchmark as benchmark


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "eval_outputs"
DEFAULT_MODEL_REQUESTS_DIR = DEFAULT_OUTPUT_DIR / "model_requests"
DEFAULT_PREDICTIONS_DIR = DEFAULT_OUTPUT_DIR / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local open-source multimodal models against exported benchmark "
            "model_requests and write predictions/{model_id}/{setting}.jsonl."
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
        required=True,
        help="Model identifier matching a subdirectory under model_requests/.",
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
        default=1.0,
        help="Video sampling rate passed to the Qwen processor.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum pixels passed to AutoProcessor to control video resolution.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=768 * 28 * 28,
        help="Maximum pixels passed to AutoProcessor to control video resolution.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Execution device preference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction files.",
    )
    parser.add_argument(
        "--video-cache-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "local_runner_cache" / "videos",
        help="Directory for local transcoded video copies used by the runner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_request_dir = args.model_requests_dir.resolve() / args.model_id
    if not model_request_dir.exists():
        raise benchmark.BenchmarkError(f"Model request directory does not exist: {model_request_dir}")

    predictions_root = args.predictions_dir.resolve() / args.model_id
    benchmark.ensure_dir(predictions_root)

    runner: Optional[QwenRunner] = None
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
                f"Provider {provider!r} is not implemented yet in this runner. "
                "Current runner supports only qwen."
            )
        if runner is None:
            runner = QwenRunner.from_request_row(
                request_rows[0],
                fps=args.fps,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
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
        for index, row in enumerate(request_rows, start=1):
            question_id = str(as_dict(row.get("request"), label="request")["question_id"])
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


class QwenRunner:
    def __init__(
        self,
        model_name: str,
        generation_config: Dict[str, object],
        fps: float,
        min_pixels: int,
        max_pixels: int,
        device: str,
        video_cache_dir: Path,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.model_name = model_name
        self.base_generation_config = dict(generation_config)
        self.fps = fps
        self.device_preference = device
        self.video_cache_dir = video_cache_dir
        benchmark.ensure_dir(self.video_cache_dir)

        torch_dtype = torch.bfloat16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        device_map = "auto" if device == "auto" else None
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        )
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        elif device == "cpu":
            self.model = self.model.to("cpu")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    @classmethod
    def from_request_row(
        cls,
        row: Dict[str, object],
        fps: float,
        min_pixels: int,
        max_pixels: int,
        device: str,
        video_cache_dir: Path,
    ) -> "QwenRunner":
        provider_payload = as_dict(row.get("provider_payload"), label="provider_payload")
        return cls(
            model_name=str(provider_payload["model"]),
            generation_config=as_dict(row.get("generation_config"), label="generation_config"),
            fps=fps,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            device=device,
            video_cache_dir=video_cache_dir,
        )

    def run_request(self, row: Dict[str, object]) -> Dict[str, object]:
        provider_payload = as_dict(row.get("provider_payload"), label="provider_payload")
        request = as_dict(row.get("request"), label="request")
        prompt = str(request["prompt"])
        video_paths = [self.prepare_video_path(str(path)) for path in provider_payload.get("video_paths", [])]
        if not video_paths:
            raise benchmark.BenchmarkError("Qwen request is missing video_paths")

        raw_response = self.generate_text(prompt=prompt, video_paths=video_paths, generation_config=provider_payload)
        predicted_answer, _parse_status = benchmark.normalize_predicted_answer({"raw_response": raw_response})
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
        conversation = [
            {
                "role": "user",
                "content": [
                    *({"type": "video", "path": path} for path in video_paths),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            fps=self.fps,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = move_batch_to_model_device(inputs, self.model.device)

        generate_kwargs = normalize_generation_config(
            generation_config=as_dict(generation_config.get("generation_config", {}), label="generation_config"),
        )
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


def move_batch_to_model_device(batch: object, device: object) -> object:
    if hasattr(batch, "to"):
        return batch.to(device)
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
