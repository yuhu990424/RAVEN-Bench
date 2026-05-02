#!/usr/bin/env python3

from __future__ import annotations

import run_internvl35_8b_local as runner


runner.DEFAULT_MODEL_ID = "internvl3_5_8b"
runner.DEFAULT_MODEL_NAME = "OpenGVLab/InternVL3_5-8B-HF"


if __name__ == "__main__":
    runner.main()
