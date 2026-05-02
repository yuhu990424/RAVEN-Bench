#!/usr/bin/env python3

from __future__ import annotations

import run_hf_vlm_frames_local as runner


runner.DEFAULT_MODEL_ID = "llava_video_7b_qwen2"
runner.DEFAULT_MODEL_NAME = "lmms-lab/LLaVA-Video-7B-Qwen2"
runner.DEFAULT_ADAPTER = "auto_chat_template"


if __name__ == "__main__":
    runner.main()
