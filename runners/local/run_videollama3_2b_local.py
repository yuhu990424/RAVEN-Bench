#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runners.local import run_videollama3_7b_local as runner


runner.DEFAULT_MODEL_ID = "videollama3_2b"
runner.DEFAULT_MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-2B"


if __name__ == "__main__":
    runner.main()
