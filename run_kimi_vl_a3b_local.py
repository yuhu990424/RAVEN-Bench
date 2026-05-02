#!/usr/bin/env python3

from __future__ import annotations

import run_kimivl_a3b_local as runner


runner.DEFAULT_MODEL_ID = "kimivl_a3b_instruct"
runner.DEFAULT_MODEL_NAME = "moonshotai/Kimi-VL-A3B-Instruct"


if __name__ == "__main__":
    runner.main()
