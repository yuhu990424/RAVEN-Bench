# Model Runners

All runners consume exported `model_requests/{model_id}/{setting}.jsonl` files
and write predictions under `predictions/{model_id}/{setting}.jsonl`.

## API Models

OpenAI/GPT:

```bash
export OPENAI_API_KEY=...
python3 runners/api/run_gpt54_api.py --model-requests-dir eval_outputs/paper_smoke/model_requests --dry-run --limit 1
```

Gemini:

```bash
export GEMINI_API_KEY=...
python3 runners/api/run_gemini3_flash_api.py --model-requests-dir eval_outputs/paper_smoke/model_requests --dry-run --limit 1
python3 runners/api/run_gemini31_pro_api.py --model-requests-dir eval_outputs/paper_smoke/model_requests --dry-run --limit 1
```

Hosted OpenAI-compatible models:

```bash
export OPENAI_COMPATIBLE_API_KEY=...
python3 runners/api/run_openai_compatible_vlm_api.py \
  --base-url https://provider.example/v1 \
  --model-id qwen3_vl_235b_a22b_instruct \
  --model-name provider/model-name \
  --model-requests-dir eval_outputs/paper_smoke/model_requests \
  --dry-run \
  --limit 1
```

## Local Models

Qwen-family local runners:

```bash
python3 runners/local/run_qwen35_9b_local.py --help
python3 runners/local/run_qwen3vl_2b_local.py --help
python3 runners/local/run_qwen3vl_8b_local.py --help
```

InternVL-family local runners:

```bash
python3 runners/local/run_internvl35_8b_local.py --help
python3 runners/local/run_internvl35_30b_a3b_local.py --help
```

KimiVL:

```bash
python3 runners/local/run_kimivl_a3b_local.py --help
```

Generic Hugging Face frame-sequence runner:

```bash
python3 runners/local/run_hf_vlm_frames_local.py \
  --model-id smolvlm2_2_2b_instruct \
  --model-name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --model-requests-dir eval_outputs/paper_smoke/model_requests \
  --settings eo_only \
  --limit 1 \
  --dry-run
```

Local-model smoke tests may still require downloading model weights unless
`--dry-run` is used.
