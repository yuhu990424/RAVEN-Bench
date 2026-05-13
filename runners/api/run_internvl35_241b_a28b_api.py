#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runners.api import run_openai_compatible_vlm_api as runner


runner.DEFAULT_MODEL_ID = "internvl3_5_241b_a28b"
runner.DEFAULT_MODEL_NAME = "OpenGVLab/InternVL3_5-241B-A28B"


if __name__ == "__main__":
    runner.main()
