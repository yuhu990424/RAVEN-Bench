#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runners.local import run_internvl35_8b_local as runner


runner.DEFAULT_MODEL_ID = "internvl3_5_30b_a3b"
runner.DEFAULT_MODEL_NAME = "OpenGVLab/InternVL3_5-30B-A3B-HF"


if __name__ == "__main__":
    runner.main()
