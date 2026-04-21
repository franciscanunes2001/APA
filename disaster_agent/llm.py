"""
LLM client — thin wrapper around Ollama (OpenAI-compatible API).
Swap MODEL_NAME to change the local model without touching anything else.
"""

import os
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("AGENT_MODEL", "gemma4:latest")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
TEMPERATURE = 0.3   # low = more deterministic code output
MAX_TOKENS = 4096


def get_client() -> OpenAI:
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def call_llm(prompt: str, system: str | None = None) -> str:
    """Send a prompt and return the raw text response."""
    client = get_client()

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
    return response.choices[0].message.content
