from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


def get_dashscope_client(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    api_key = api_key or _env("DASHSCOPE_API_KEY") or _env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set DASHSCOPE_API_KEY (recommended) or OPENAI_API_KEY."
        )

    base_url = (
        base_url
        or _env("DASHSCOPE_BASE_URL")
        or _env("OPENAI_BASE_URL")
        or DEFAULT_DASHSCOPE_BASE_URL
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def chat_completions_text(
    *,
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    retries: int = 2,
    retry_sleep_s: float = 1.0,
) -> str:
    """
    Calls OpenAI-compatible chat completions and returns the first choice text.

    `messages` should follow OpenAI-compatible schema. This supports both:
    - text-only: {"role":"user","content":"..."}
    - multimodal: {"role":"user","content":[{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]}
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra:
        payload.update(extra)

    last_err: Optional[BaseException] = None
    for attempt in range(max(0, retries) + 1):
        try:
            completion = client.chat.completions.create(**payload)
            text = completion.choices[0].message.content
            return text or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt >= max(0, retries):
                raise
            time.sleep(retry_sleep_s * (2**attempt))

    raise RuntimeError(f"Chat completion failed: {last_err}")
