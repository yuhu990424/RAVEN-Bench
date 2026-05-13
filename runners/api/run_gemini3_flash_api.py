#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from runners.api import run_gemini31_flash_api as gemini_api


gemini_api.DEFAULT_MODEL_ID = "gemini_3_flash_preview"
gemini_api.DEFAULT_MODEL_NAME = "gemini-3-flash-preview"


if __name__ == "__main__":
    gemini_api.main()
