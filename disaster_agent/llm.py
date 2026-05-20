"""
LLM client — thin wrapper around Ollama (OpenAI-compatible API).
Set AGENT_MODEL env var to switch models without touching code:
    AGENT_MODEL=qwen3-coder       python run_agent.py
    AGENT_MODEL=qwen2.5-coder:7b  python run_agent.py
    AGENT_MODEL=phi4:14b          python run_agent.py
"""

import os
import sys
import time
from typing import Optional

from openai import OpenAI, APIConnectionError, APIError

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("AGENT_MODEL", "qwen2.5-coder:14b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
TEMPERATURE     = float(os.environ.get("AGENT_TEMP", "0.3"))
# 8192 gives qwen2.5-coder and similar verbose models headroom to write
# a complete script without truncation.
MAX_TOKENS      = int(os.environ.get("AGENT_MAX_TOKENS", "8192"))
MAX_RETRIES     = 3
RETRY_BACKOFF_S = 2.0


def get_client() -> OpenAI:
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def _unescape_template_braces(s: str) -> str:
    """
    Prompt templates double their braces ({{ / }}) so they survive
    str.format(). For prompts that bypass .format() (e.g. FIRST_EXPERIMENT_PROMPT),
    we unescape here so the LLM sees real Python syntax.
    """
    return s.replace("{{", "{").replace("}}", "}")


def _chat_once(prompt: str, system: Optional[str]) -> str:
    client = get_client()
    prompt = _unescape_template_braces(prompt)
    if system:
        system = _unescape_template_braces(system)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    content = response.choices[0].message.content
    return content or ""


def call_llm(prompt: str, system: Optional[str] = None) -> str:
    """
    Send a prompt and return the raw text response.
    Retries on transient connection / 5xx errors so a flaky local Ollama
    instance doesn't kill the whole agent run.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _chat_once(prompt, system)
        except APIConnectionError as e:
            last_err = e
            print(f"  [llm] Connection to {OLLAMA_BASE_URL} failed "
                  f"(attempt {attempt}/{MAX_RETRIES}): {e}",
                  file=sys.stderr)
        except APIError as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status and 400 <= status < 500:
                raise
            print(f"  [llm] API error (attempt {attempt}/{MAX_RETRIES}): {e}",
                  file=sys.stderr)
        except Exception as e:                      # pylint: disable=broad-except
            last_err = e
            print(f"  [llm] Unexpected error (attempt {attempt}/{MAX_RETRIES}): {e}",
                  file=sys.stderr)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_S * attempt)

    print(f"  [llm] Giving up after {MAX_RETRIES} attempts. "
          f"Last error: {last_err}", file=sys.stderr)
    return ""

