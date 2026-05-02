#!/usr/bin/env python3

from __future__ import annotations

import run_hf_vlm_frames_local as runner


runner.DEFAULT_MODEL_ID = "keye_vl15_8b"
runner.DEFAULT_MODEL_NAME = "Kwai-Keye/Keye-VL-1.5-8B"
runner.DEFAULT_ADAPTER = "auto_chat_template"


if __name__ == "__main__":
    runner.main()
