#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runners.local import run_qwen35_9b_local as runner


runner.DEFAULT_MODEL_ID = "qwen3_vl_30b_a3b_instruct"
runner.DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"


if __name__ == "__main__":
    runner.main()
