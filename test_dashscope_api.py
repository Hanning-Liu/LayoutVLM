"""Smoke test: DashScope (Aliyun Bailian) OpenAI-compatible chat API using DASHSCOPE_API_KEY from .env."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from openai import OpenAI

# Repo root is the directory containing this file (LayoutVLM/).
REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

# 华北2（北京）; override with DASHSCOPE_BASE_URL if you use another region.
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def main() -> int:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key or not api_key.strip():
        print("error: DASHSCOPE_API_KEY is missing or empty in the environment (.env).", file=sys.stderr)
        return 1

    base_url = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL).strip()
    model = os.getenv("DASHSCOPE_TEST_MODEL", "qwen3.6-plus").strip()

    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "你是谁？"}],
    )
    text = completion.choices[0].message.content
    print(text if text is not None else "(empty response)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(2)
