#!/usr/bin/env python3

from __future__ import annotations

import run_qwen35_9b_local as runner


runner.DEFAULT_MODEL_ID = "qwen3_vl_8b_instruct"
runner.DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


if __name__ == "__main__":
    runner.main()
