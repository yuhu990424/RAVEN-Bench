#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import eo_ir_benchmark as benchmark


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    provider: str
    model_name: str
    enabled: bool
    generation_config: Dict[str, object]
    notes: Optional[str] = None


SUPPORTED_PROVIDERS = {
    "gemini",
    "gpt",
    "kimi",
    "qwen",
    "llava",
    "internvl",
    "videollama",
    "hf_vlm",
    "hosted_vlm",
    "unirs",
}

DEFAULT_MODEL_CONFIGS: List[Dict[str, object]] = [
    {
        "model_id": "gemini_2_5_flash",
        "provider": "gemini",
        "model_name": "gemini-2.5-flash",
        "enabled": True,
        "generation_config": {"temperature": 0.0, "max_output_tokens": 32},
        "notes": "Fast multimodal API baseline.",
    },
    {
        "model_id": "gemini_3_1_flash",
        "provider": "gemini",
        "model_name": "gemini-3.1-flash",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_output_tokens": 32},
        "notes": "Gemini 3.1 Flash REST API benchmark target.",
    },
    {
        "model_id": "gemini_3_1_pro_preview",
        "provider": "gemini",
        "model_name": "gemini-3.1-pro-preview",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_output_tokens": 512, "thinking_level": "low"},
        "notes": "Gemini 3.1 Pro Preview REST API benchmark target.",
    },
    {
        "model_id": "gemini_3_flash_preview",
        "provider": "gemini",
        "model_name": "gemini-3-flash-preview",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_output_tokens": 512, "thinking_level": "minimal"},
        "notes": "Gemini 3 Flash Preview REST API benchmark target.",
    },
    {
        "model_id": "gpt_4_1_mini",
        "provider": "gpt",
        "model_name": "gpt-4.1-mini",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_output_tokens": 32},
        "notes": "GPT-style OpenAI Responses payload.",
    },
    {
        "model_id": "gpt_5_4",
        "provider": "gpt",
        "model_name": "gpt-5.4",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_output_tokens": 32},
        "notes": "GPT-5.4 OpenAI Responses API benchmark target.",
    },
    {
        "model_id": "kimi_k2_vision",
        "provider": "kimi",
        "model_name": "kimi-k2-vision",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_output_tokens": 32},
        "notes": "Kimi-style OpenAI-compatible multimodal payload.",
    },
    {
        "model_id": "kimivl_a3b_instruct",
        "provider": "kimi",
        "model_name": "moonshotai/Kimi-VL-A3B-Instruct",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local Kimi-VL A3B Instruct benchmark target using Hugging Face Transformers.",
    },
    {
        "model_id": "smolvlm2_2_2b_instruct",
        "provider": "hf_vlm",
        "model_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local generic Hugging Face frame-sequence runner target.",
    },
    {
        "model_id": "qwen2_5_vl_3b",
        "provider": "qwen",
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local Qwen smoke-test default. Smaller than the 7B release model and suitable for local call validation.",
    },
    {
        "model_id": "qwen3_vl_2b_instruct",
        "provider": "qwen",
        "model_name": "Qwen/Qwen3-VL-2B-Instruct",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local Qwen3-VL 2B benchmark target using the Qwen local runner.",
    },
    {
        "model_id": "qwen3_5_9b",
        "provider": "qwen",
        "model_name": "Qwen/Qwen3.5-9B",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local Qwen3.5 benchmark target. Requires a recent Transformers build with Qwen3.5 multimodal support.",
    },
    {
        "model_id": "qwen3_5_9b_instruct",
        "provider": "qwen",
        "model_name": "Qwen/Qwen3.5-9B",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Alias entry for the Qwen3.5 9B Instruct row in the benchmark table.",
    },
    {
        "model_id": "qwen3_vl_8b_instruct",
        "provider": "qwen",
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local Qwen3-VL 8B benchmark target using the Qwen local runner.",
    },
    {
        "model_id": "qwen3_vl_30b_a3b_instruct",
        "provider": "qwen",
        "model_name": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local Qwen3-VL 30B-A3B benchmark target. Usually requires multi-GPU or aggressive quantization.",
    },
    {
        "model_id": "llava_next_video_7b",
        "provider": "llava",
        "model_name": "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local LLaVA video smoke-test default. Kept at 7B because the public Transformers video checkpoint is 7B.",
    },
    {
        "model_id": "llava_video_7b_qwen2",
        "provider": "hf_vlm",
        "model_name": "lmms-lab/LLaVA-Video-7B-Qwen2",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local generic Hugging Face frame-sequence runner target for LLaVA-Video-7B-Qwen2.",
    },
    {
        "model_id": "internvl3_2b",
        "provider": "internvl",
        "model_name": "OpenGVLab/InternVL3-2B-hf",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local InternVL smoke-test default using the native Transformers 2B checkpoint.",
    },
    {
        "model_id": "internvl3_5_8b",
        "provider": "internvl",
        "model_name": "OpenGVLab/InternVL3_5-8B-HF",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local InternVL3.5 benchmark target. Uses the official 8B Hugging Face checkpoint.",
    },
    {
        "model_id": "internvl3_5_30b_a3b",
        "provider": "internvl",
        "model_name": "OpenGVLab/InternVL3_5-30B-A3B-HF",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local InternVL3.5 30B-A3B benchmark target. Checkpoint name can be overridden by CLI.",
    },
    {
        "model_id": "minicpm_v45",
        "provider": "hf_vlm",
        "model_name": "openbmb/MiniCPM-V-4_5",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local generic Hugging Face frame-sequence runner target.",
    },
    {
        "model_id": "molmo2_8b",
        "provider": "hf_vlm",
        "model_name": "allenai/Molmo2-8B",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local generic Hugging Face frame-sequence runner target. Override model_name if using a different Molmo2 release.",
    },
    {
        "model_id": "keye_vl15_8b",
        "provider": "hf_vlm",
        "model_name": "Kwai-Keye/Keye-VL-1.5-8B",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local generic Hugging Face frame-sequence runner target. Override model_name if the release path differs.",
    },
    {
        "model_id": "mimo_vl_7b_rl_2508",
        "provider": "hf_vlm",
        "model_name": "XiaomiMiMo/MiMo-VL-7B-RL-2508",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local generic Hugging Face frame-sequence runner target. Override model_name if the release path differs.",
    },
    {
        "model_id": "videollama3_7b",
        "provider": "videollama",
        "model_name": "DAMO-NLP-SG/VideoLLaMA3-7B",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local VideoLLaMA3 benchmark target using the official 7B video-chat checkpoint.",
    },
    {
        "model_id": "videollama3_2b",
        "provider": "videollama",
        "model_name": "DAMO-NLP-SG/VideoLLaMA3-2B",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local VideoLLaMA3 2B benchmark target using the VideoLLaMA3 runner.",
    },
    {
        "model_id": "videoxl_pro_3b",
        "provider": "hf_vlm",
        "model_name": "BAAI/Video-XL-Pro-3B",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local generic Hugging Face frame-sequence runner target. Override model_name if the release path differs.",
    },
    {
        "model_id": "internvideo25_chat_8b",
        "provider": "hf_vlm",
        "model_name": "OpenGVLab/InternVideo2_5_Chat_8B",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local generic Hugging Face frame-sequence runner target. May require project-specific remote code.",
    },
    {
        "model_id": "mplug_owl3_7b",
        "provider": "hf_vlm",
        "model_name": "mPLUG/mPLUG-Owl3-7B",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Local generic Hugging Face frame-sequence runner target. Override model_name if the release path differs.",
    },
    {
        "model_id": "qwen3_vl_235b_a22b_instruct",
        "provider": "hosted_vlm",
        "model_name": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_tokens": 32},
        "notes": "Hosted open-weight model via OpenAI-compatible chat-completions API.",
    },
    {
        "model_id": "internvl3_5_241b_a28b",
        "provider": "hosted_vlm",
        "model_name": "OpenGVLab/InternVL3_5-241B-A28B",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_tokens": 32},
        "notes": "Hosted open-weight model via OpenAI-compatible chat-completions API. Override provider model name as needed.",
    },
    {
        "model_id": "unirs",
        "provider": "unirs",
        "model_name": "IDEA-Research/UniRS",
        "enabled": False,
        "generation_config": {"temperature": 0.0, "max_new_tokens": 32},
        "notes": "Remote-sensing-specific baseline entry. Uses the generic HF frame runner unless replaced with a UniRS-specific adapter.",
    },
]


def default_model_configs() -> List[Dict[str, object]]:
    return [
        {
            "model_id": str(config["model_id"]),
            "provider": str(config["provider"]),
            "model_name": str(config["model_name"]),
            "enabled": bool(config["enabled"]),
            "generation_config": dict(config["generation_config"]),
            "notes": str(config["notes"]) if config.get("notes") is not None else None,
        }
        for config in DEFAULT_MODEL_CONFIGS
    ]


def validate_model_specs(model_specs: List[ModelSpec]) -> None:
    seen = set()
    for spec in model_specs:
        if spec.provider not in SUPPORTED_PROVIDERS:
            raise benchmark.BenchmarkError(
                f"Unsupported provider {spec.provider!r} for model_id={spec.model_id!r}. "
                f"Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
            )
        if spec.model_id in seen:
            raise benchmark.BenchmarkError(f"Duplicate model_id in config: {spec.model_id}")
        seen.add(spec.model_id)


def parse_model_specs(raw_models: List[Dict[str, object]]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for raw in raw_models:
        specs.append(
            ModelSpec(
                model_id=str(raw["model_id"]),
                provider=str(raw["provider"]),
                model_name=str(raw["model_name"]),
                enabled=bool(raw.get("enabled", True)),
                generation_config=dict(raw.get("generation_config", {})),
                notes=str(raw["notes"]) if raw.get("notes") is not None else None,
            )
        )
    validate_model_specs(specs)
    return specs


def enabled_model_specs(model_specs: List[ModelSpec]) -> List[ModelSpec]:
    return [spec for spec in model_specs if spec.enabled]


def build_model_request(question: Dict[str, object], setting: str, model: ModelSpec) -> Dict[str, object]:
    generic = benchmark.build_request_row(question, setting)
    provider_payload = build_provider_payload(question, setting, model)
    return {
        "model_id": model.model_id,
        "provider": model.provider,
        "model_name": model.model_name,
        "generation_config": model.generation_config,
        "notes": model.notes,
        "request": generic,
        "provider_payload": provider_payload,
    }


def build_provider_payload(
    question: Dict[str, object],
    setting: str,
    model: ModelSpec,
) -> Dict[str, object]:
    if model.provider == "gemini":
        return build_gemini_payload(question, setting, model)
    if model.provider == "gpt":
        return build_gpt_payload(question, setting, model)
    if model.provider == "kimi":
        return build_kimi_payload(question, setting, model)
    if model.provider == "qwen":
        return build_qwen_payload(question, setting, model)
    if model.provider == "llava":
        return build_llava_payload(question, setting, model)
    if model.provider == "internvl":
        return build_internvl_payload(question, setting, model)
    if model.provider == "videollama":
        return build_videollama_payload(question, setting, model)
    if model.provider == "hf_vlm":
        return build_hf_vlm_payload(question, setting, model)
    if model.provider == "hosted_vlm":
        return build_hosted_vlm_payload(question, setting, model)
    if model.provider == "unirs":
        return build_unirs_payload(question, setting, model)
    raise benchmark.BenchmarkError(f"Unsupported provider: {model.provider}")


def build_gemini_payload(question: Dict[str, object], setting: str, model: ModelSpec) -> Dict[str, object]:
    video_paths = select_video_paths(question, setting)
    video_input = benchmark.build_video_input_protocol()
    parts = [{"type": "text", "text": benchmark.build_prompt(question, setting, build_video_instruction(setting))}]
    parts.extend({"type": "video_file", "path": path} for path in video_paths)
    return {
        "provider": "gemini",
        "model": model.model_name,
        "contents": [{"role": "user", "parts": parts}],
        "generation_config": model.generation_config,
        "video_input": video_input,
        "video_metadata": {
            "fps": video_input["sampling"]["fps"],
        },
        "media_resolution": "default",
    }


def build_gpt_payload(question: Dict[str, object], setting: str, model: ModelSpec) -> Dict[str, object]:
    video_paths = select_video_paths(question, setting)
    video_input = benchmark.build_video_input_protocol()
    content = [{"type": "input_text", "text": benchmark.build_prompt(question, setting, build_video_instruction(setting))}]
    content.extend({"type": "input_file", "path": path} for path in video_paths)
    return {
        "provider": "gpt",
        "model": model.model_name,
        "api_style": "openai_responses",
        "input": [{"role": "user", "content": content}],
        "generation_config": model.generation_config,
        "video_input": video_input,
        "frame_extraction": {
            "fps": video_input["sampling"]["fps"],
            "max_frames_per_video": video_input["sampling"]["max_frames_per_video"],
            "frame_resolution": video_input["frame_resolution"],
        },
    }


def build_kimi_payload(question: Dict[str, object], setting: str, model: ModelSpec) -> Dict[str, object]:
    video_paths = select_video_paths(question, setting)
    video_input = benchmark.build_video_input_protocol()
    content = [{"type": "text", "text": benchmark.build_prompt(question, setting, build_video_instruction(setting))}]
    content.extend({"type": "video", "path": path} for path in video_paths)
    return {
        "provider": "kimi",
        "model": model.model_name,
        "api_style": "openai_compatible_chat",
        "messages": [{"role": "user", "content": content}],
        "generation_config": model.generation_config,
        "video_input": video_input,
    }


def build_qwen_payload(question: Dict[str, object], setting: str, model: ModelSpec) -> Dict[str, object]:
    video_input = benchmark.build_video_input_protocol()
    return {
        "provider": "qwen",
        "model": model.model_name,
        "runtime": "local_transformers",
        "messages": [
            {
                "role": "user",
                "content": benchmark.build_prompt(question, setting, build_video_instruction(setting)),
            }
        ],
        "video_paths": select_video_paths(question, setting),
        "generation_config": model.generation_config,
        "video_input": video_input,
        "processor_hint": "qwen2_5_vl",
    }


def build_llava_payload(question: Dict[str, object], setting: str, model: ModelSpec) -> Dict[str, object]:
    video_input = benchmark.build_video_input_protocol()
    return {
        "provider": "llava",
        "model": model.model_name,
        "runtime": "local_transformers",
        "conversation": [
            {
                "role": "user",
                "content": benchmark.build_prompt(question, setting, build_video_instruction(setting)),
            }
        ],
        "video_paths": select_video_paths(question, setting),
        "generation_config": model.generation_config,
        "video_input": video_input,
        "processor_hint": "llava_video",
    }


def build_internvl_payload(question: Dict[str, object], setting: str, model: ModelSpec) -> Dict[str, object]:
    video_input = benchmark.build_video_input_protocol()
    return {
        "provider": "internvl",
        "model": model.model_name,
        "runtime": "local_python",
        "prompt": benchmark.build_prompt(question, setting, build_video_instruction(setting)),
        "video_paths": select_video_paths(question, setting),
        "generation_config": model.generation_config,
        "video_input": video_input,
        "processor_hint": "internvl_chat",
    }


def build_videollama_payload(question: Dict[str, object], setting: str, model: ModelSpec) -> Dict[str, object]:
    video_input = benchmark.build_video_input_protocol()
    return {
        "provider": "videollama",
        "model": model.model_name,
        "runtime": "local_transformers",
        "prompt": benchmark.build_prompt(question, setting, build_video_instruction(setting)),
        "video_paths": select_video_paths(question, setting),
        "generation_config": model.generation_config,
        "video_input": video_input,
        "processor_hint": "videollama3_chat",
    }


def build_hf_vlm_payload(question: Dict[str, object], setting: str, model: ModelSpec) -> Dict[str, object]:
    video_input = benchmark.build_video_input_protocol()
    return {
        "provider": "hf_vlm",
        "model": model.model_name,
        "runtime": "local_transformers_frame_sequence",
        "prompt": benchmark.build_prompt(question, setting, build_video_instruction(setting)),
        "video_paths": select_video_paths(question, setting),
        "generation_config": model.generation_config,
        "video_input": video_input,
        "processor_hint": "auto_chat_template_images",
    }


def build_hosted_vlm_payload(question: Dict[str, object], setting: str, model: ModelSpec) -> Dict[str, object]:
    video_paths = select_video_paths(question, setting)
    video_input = benchmark.build_video_input_protocol()
    return {
        "provider": "hosted_vlm",
        "model": model.model_name,
        "api_style": "openai_compatible_chat_completions",
        "messages": [
            {
                "role": "user",
                "content": benchmark.build_prompt(question, setting, build_video_instruction(setting)),
            }
        ],
        "video_paths": video_paths,
        "generation_config": model.generation_config,
        "video_input": video_input,
        "frame_extraction": {
            "fps": video_input["sampling"]["fps"],
            "max_frames_per_video": video_input["sampling"]["max_frames_per_video"],
            "frame_resolution": video_input["frame_resolution"],
        },
    }


def build_unirs_payload(question: Dict[str, object], setting: str, model: ModelSpec) -> Dict[str, object]:
    payload = build_hf_vlm_payload(question, setting, model)
    payload["provider"] = "unirs"
    payload["processor_hint"] = "unirs_or_auto_chat_template_images"
    return payload


def select_video_paths(question: Dict[str, object], setting: str) -> List[str]:
    if setting == benchmark.SETTING_EO_ONLY:
        return [benchmark.model_input_path_for_modality(question, "EO")]
    if setting == benchmark.SETTING_IR_ONLY:
        return [benchmark.model_input_path_for_modality(question, "IR")]
    if setting == benchmark.SETTING_EO_IR:
        return [
            benchmark.model_input_path_for_modality(question, "EO"),
            benchmark.model_input_path_for_modality(question, "IR"),
        ]
    raise benchmark.BenchmarkError(f"Unsupported setting: {setting}")


def build_video_instruction(setting: str) -> str:
    if setting == benchmark.SETTING_EO_ONLY:
        return "Video 1 is the full Electro-Optical (EO) video."
    if setting == benchmark.SETTING_IR_ONLY:
        return "Video 1 is the full Infrared (IR) video."
    if setting == benchmark.SETTING_EO_IR:
        return "Video 1 is the full Electro-Optical (EO) video. Video 2 is the full Infrared (IR) video."
    raise benchmark.BenchmarkError(f"Unsupported setting: {setting}")


def export_model_requests(
    questions_path: Path,
    output_dir: Path,
    settings: List[str],
    model_specs: List[ModelSpec],
) -> Dict[str, object]:
    questions = benchmark.load_jsonl(questions_path)
    request_root = output_dir / "model_requests"
    benchmark.ensure_dir(request_root)

    enabled_specs = enabled_model_specs(model_specs)
    for spec in enabled_specs:
        model_dir = request_root / spec.model_id
        benchmark.ensure_dir(model_dir)
        for setting in settings:
            rows = [build_model_request(question, setting, spec) for question in questions]
            benchmark.write_jsonl(model_dir / f"{setting}.jsonl", rows)

    return {
        "status": "ok",
        "request_dir": str(request_root),
        "enabled_models": [spec.model_id for spec in enabled_specs],
        "question_count": len(questions),
        "settings": settings,
        "video_input": benchmark.build_video_input_protocol(),
    }
