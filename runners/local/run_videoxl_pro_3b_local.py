#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runners.local import run_hf_vlm_frames_local as runner


runner.DEFAULT_MODEL_ID = "videoxl_pro_3b"
runner.DEFAULT_MODEL_NAME = "BAAI/Video-XL-Pro-3B"
runner.DEFAULT_ADAPTER = "auto_chat_template"


if __name__ == "__main__":
    runner.main()
