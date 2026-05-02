#!/usr/bin/env python3

from __future__ import annotations

import run_gemini31_flash_api as gemini_api


gemini_api.DEFAULT_MODEL_ID = "gemini_3_1_pro_preview"
gemini_api.DEFAULT_MODEL_NAME = "gemini-3.1-pro-preview"


if __name__ == "__main__":
    gemini_api.main()
