#!/usr/bin/env python3

from __future__ import annotations

import run_hf_vlm_frames_local as runner


runner.DEFAULT_MODEL_ID = "smolvlm2_2_2b_instruct"
runner.DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
runner.DEFAULT_ADAPTER = "auto_chat_template"


if __name__ == "__main__":
    runner.main()
