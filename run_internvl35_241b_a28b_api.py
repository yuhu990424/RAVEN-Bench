#!/usr/bin/env python3

from __future__ import annotations

import run_openai_compatible_vlm_api as runner


runner.DEFAULT_MODEL_ID = "internvl3_5_241b_a28b"
runner.DEFAULT_MODEL_NAME = "OpenGVLab/InternVL3_5-241B-A28B"


if __name__ == "__main__":
    runner.main()
