#!/usr/bin/env python3

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

from raven_benchmark import benchmark


def post_json_with_retries(
    url: str,
    payload: Dict[str, object],
    headers: Dict[str, str],
    timeout: float,
    max_retries: int,
) -> Dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
            return json.loads(response_body)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2.0 ** attempt, 8.0))
    raise benchmark.BenchmarkError(f"JSON API request failed: {last_error}")


def extract_openai_compatible_response_text(response: Dict[str, object]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    pieces = []
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            pieces.append(part["text"])
                    if pieces:
                        return "\n".join(pieces).strip()
            text = first.get("text")
            if isinstance(text, str):
                return text.strip()
    return json.dumps(response, ensure_ascii=False, sort_keys=True)


def redact_secret(text: str, secret: Optional[str], placeholder: str = "<redacted>") -> str:
    if not secret:
        return text
    return text.replace(secret, placeholder)
