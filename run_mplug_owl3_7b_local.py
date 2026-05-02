#!/usr/bin/env python3

from __future__ import annotations

import run_hf_vlm_frames_local as runner


runner.DEFAULT_MODEL_ID = "mplug_owl3_7b"
runner.DEFAULT_MODEL_NAME = "mPLUG/mPLUG-Owl3-7B"
runner.DEFAULT_ADAPTER = "auto_chat_template"


if __name__ == "__main__":
    runner.main()
