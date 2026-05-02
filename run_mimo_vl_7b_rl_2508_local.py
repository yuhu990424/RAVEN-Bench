#!/usr/bin/env python3

from __future__ import annotations

import run_hf_vlm_frames_local as runner


runner.DEFAULT_MODEL_ID = "mimo_vl_7b_rl_2508"
runner.DEFAULT_MODEL_NAME = "XiaomiMiMo/MiMo-VL-7B-RL-2508"
runner.DEFAULT_ADAPTER = "auto_chat_template"


if __name__ == "__main__":
    runner.main()
