#!/usr/bin/env python3

from __future__ import annotations

import run_hf_vlm_frames_local as runner


runner.DEFAULT_MODEL_ID = "unirs"
runner.DEFAULT_MODEL_NAME = "IDEA-Research/UniRS"
runner.DEFAULT_ADAPTER = "unirs_or_auto_chat_template"


if __name__ == "__main__":
    runner.main()
