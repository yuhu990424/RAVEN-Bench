#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import eo_ir_benchmark as benchmark
from run_qwen35_9b_local import (
    ANSWER_JSON_PREFIX,
    ANSWER_JSON_SUFFIX,
    append_jsonl,
    canonical_json_answer,
    move_batch_to_model_device,
    normalize_generation_config,
    resolve_model_device,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = ROOT_DIR / "eval_outputs" / "v14_deep_annotations_validation" / "questions.jsonl"
DEFAULT_PREDICTIONS_DIR = ROOT_DIR / "eval_outputs" / "v14_deep_annotations_validation" / "predictions_text_only"
DEFAULT_MODEL_ID = "qwen3_5_9b_text_only"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5-9B on MCQ text only, without EO/IR videos."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--quantization", default="none", choices=("none", "8bit", "4bit"))
    parser.add_argument("--llm-int8-cpu-offload", action="store_true")
    parser.add_argument("--offload-folder", type=Path, default=ROOT_DIR / "eval_outputs" / "local_runner_cache" / "offload_qwen35_text_only")
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = benchmark.load_jsonl(args.questions.resolve())
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        raise benchmark.BenchmarkError(f"No questions found: {args.questions}")

    output_path = args.predictions_dir.resolve() / args.model_id / "eo_ir.jsonl"
    benchmark.ensure_dir(output_path.parent)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    completed = benchmark.load_jsonl(output_path) if output_path.exists() else []
    completed_ids = {str(row["question_id"]) for row in completed}

    runner = Qwen35TextOnlyRunner(
        model_name=args.model_name,
        generation_config={"temperature": 0.0, "max_new_tokens": 32},
        device=args.device,
        quantization=args.quantization,
        llm_int8_cpu_offload=args.llm_int8_cpu_offload,
        offload_folder=args.offload_folder.resolve(),
        enable_thinking=args.enable_thinking,
    )
    try:
        for question in questions:
            question_id = str(question["question_id"])
            if question_id in completed_ids:
                continue
            prediction = runner.run_question(question=question, model_id=args.model_id)
            append_jsonl(output_path, prediction)
            completed.append(prediction)
            completed_ids.add(question_id)
            print(
                json.dumps(
                    {
                        "status": "progress",
                        "model_id": args.model_id,
                        "completed": len(completed),
                        "total": len(questions),
                        "question_id": question_id,
                    }
                ),
                flush=True,
            )
    finally:
        runner.close()

    print(
        json.dumps(
            {
                "status": "ok",
                "model_id": args.model_id,
                "setting": "text_only",
                "request_count": len(completed),
                "predictions_file": str(output_path),
            },
            indent=2,
        )
    )


class Qwen35TextOnlyRunner:
    def __init__(
        self,
        model_name: str,
        generation_config: Dict[str, object],
        device: str,
        quantization: str,
        llm_int8_cpu_offload: bool,
        offload_folder: Path,
        enable_thinking: bool,
    ) -> None:
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        import torch

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise benchmark.BenchmarkError("Text-only Qwen runner requires a recent Transformers build.") from exc

        self.torch = torch
        self.base_generation_config = dict(generation_config)
        self.enable_thinking = enable_thinking
        self.model_runtime_config = {
            "mode": "text_only_no_video",
            "quantization": quantization,
            "llm_int8_cpu_offload": bool(llm_int8_cpu_offload) if quantization == "8bit" else None,
        }
        benchmark.ensure_dir(offload_folder)

        dtype = torch.bfloat16 if (device != "cpu" and torch.cuda.is_available()) else torch.float32
        model_kwargs: Dict[str, object] = {"dtype": dtype, "attn_implementation": "sdpa"}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        if quantization != "none":
            if device == "cpu" or not torch.cuda.is_available():
                raise benchmark.BenchmarkError("bitsandbytes quantization requires CUDA.")
            if quantization == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=llm_int8_cpu_offload,
                )
            elif quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            model_kwargs["device_map"] = "auto"
            model_kwargs["offload_folder"] = str(offload_folder)

        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name)
        if quantization == "none" and device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        elif quantization == "none" and device == "cpu":
            self.model = self.model.to("cpu")

    def run_question(self, question: Dict[str, object], model_id: str) -> Dict[str, object]:
        prompt = build_text_only_prompt(question)
        raw_response, predicted_answer, answer_source, initial_response = self.generate_answer(prompt)
        self.release_runtime_memory()
        prediction = {
            "model_id": model_id,
            "setting": "text_only",
            "question_id": str(question["question_id"]),
            "predicted_answer": predicted_answer,
            "raw_response": raw_response,
            "answer_source": answer_source,
            "model_runtime_config": self.model_runtime_config,
        }
        if initial_response is not None and initial_response != raw_response:
            prediction["raw_response_initial"] = initial_response
        return prediction

    def generate_answer(self, prompt: str) -> Tuple[str, str | None, str, str | None]:
        prompt_text = self.build_prompt_text(prompt)
        initial_response = self.generate_prefilled_json(prompt_text)
        predicted_answer, _ = benchmark.normalize_predicted_answer({"raw_response": initial_response})
        if predicted_answer is not None:
            return canonical_json_answer(predicted_answer), predicted_answer, "json_prefill", initial_response
        fallback_answer = self.score_answer_options(prompt_text)
        return canonical_json_answer(fallback_answer), fallback_answer, "option_scoring_fallback", initial_response

    def build_prompt_text(self, prompt: str) -> str:
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        kwargs: Dict[str, object] = {"add_generation_prompt": True}
        if not self.enable_thinking:
            kwargs["enable_thinking"] = False
        try:
            return self.processor.apply_chat_template(conversation, tokenize=False, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.processor.apply_chat_template(conversation, tokenize=False, **kwargs)

    def build_model_inputs(self, prompt_text: str) -> object:
        return self.processor.tokenizer(
            [prompt_text],
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )

    def generate_prefilled_json(self, prompt_text: str) -> str:
        inputs = self.build_model_inputs(f"{prompt_text}{ANSWER_JSON_PREFIX}")
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
            clean_up_tokenization_spaces=True,
        )
        completion = str(output_text[0]).strip() if output_text else ""
        return f"{ANSWER_JSON_PREFIX}{completion}"

    def score_answer_options(self, prompt_text: str) -> str:
        prefix_text = f"{prompt_text}{ANSWER_JSON_PREFIX}"
        prefix_inputs = self.build_model_inputs(prefix_text)
        prefix_len = int(prefix_inputs["input_ids"].shape[1])
        best_answer = "A"
        best_score = float("-inf")
        for answer in benchmark.VALID_ANSWERS:
            candidate_text = f"{prefix_text}{answer}{ANSWER_JSON_SUFFIX}"
            candidate_inputs = self.build_model_inputs(candidate_text)
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


def build_text_only_prompt(question: Dict[str, object]) -> str:
    option_lines = [f"{label}. {text}" for label, text in sorted(question["options"].items())]
    return "\n".join(
        [
            "Answer this single-answer multiple-choice question using only the text below.",
            "No video, image, filename, timestamp, answer key, rationale, or metadata is provided.",
            f"Question: {question['question']}",
            "Options:",
            *option_lines,
            'Reply only with JSON: {"answer":"<A|B|C|D>"}.',
        ]
    )


if __name__ == "__main__":
    main()
