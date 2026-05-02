#!/usr/bin/env python3

from __future__ import annotations

import run_hf_vlm_frames_local as runner


runner.DEFAULT_MODEL_ID = "molmo2_8b"
runner.DEFAULT_MODEL_NAME = "allenai/Molmo2-8B"
runner.DEFAULT_ADAPTER = "auto_chat_template"


if __name__ == "__main__":
    runner.main()
