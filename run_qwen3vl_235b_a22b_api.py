#!/usr/bin/env python3

from __future__ import annotations

import run_openai_compatible_vlm_api as runner


runner.DEFAULT_MODEL_ID = "qwen3_vl_235b_a22b_instruct"
runner.DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Instruct"


if __name__ == "__main__":
    runner.main()
