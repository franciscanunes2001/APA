"""
Extracts structured data from LLM responses (v2).

Robust to formatting differences across Ollama models (gemma, qwen, deepseek-r1,
phi4, etc.). Different models wrap code differently:
  - ```python\n...```          (gemma, most)
  - ```Python\n...```           (qwen sometimes — capital P)
  - ```py\n...```               (some smaller models)
  - ```\n...```                 (deepseek-r1, when no language tag)
  - ~~~python\n...~~~           (rare, but some templates)
  - <think>...</think> ```...```  (deepseek-r1 reasoning models)

v2 additions:
  - extract_architecture_name() returns "unknown" when code is None,
    instead of crashing on the [:3000] slice.
  - extract_learning_curve() pulls EPOCH_METRIC lines printed by the
    Keras callback injected in executor_v2.py.
"""

import re
from typing import Optional


# All accepted opening fences. Order matters: try language-tagged first,
# fall back to plain ``` only if nothing else hits.
_FENCE_PATTERNS = [
    # ```python ... ```  /  ```Python ... ```  /  ```py ... ```
    r"```[ \t]*(?:python|Python|PYTHON|py|python3)[ \t]*\r?\n(.*?)```",
    # ~~~python ... ~~~
    r"~~~[ \t]*(?:python|py)[ \t]*\r?\n(.*?)~~~",
    # bare ``` ... ```  (last resort — only used if nothing above matches)
    r"```[ \t]*\r?\n(.*?)```",
]


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models like
    deepseek-r1. The actual answer comes after."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def extract_code(response: str) -> str | None:
    """
    Pull the first Python code block out of an LLM response.
    Returns the code string, or None if no block found.
    Tolerant of capitalisation, ~~~ fences, and reasoning-model <think> tags.
    """
    if not response:
        return None

    cleaned = _strip_think_blocks(response)

    for pattern in _FENCE_PATTERNS:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Sanity check: must look like Python (avoid grabbing a stray
            # bash/json fence). At least one of these should appear.
            if any(tok in code for tok in ("import ", "def ", "print(", "=")):
                return code
    return None


def extract_f1(output: str) -> Optional[float]:
    """
    Parse the mandatory output line:   RESULT: f1=0.7823
    Tolerant of whitespace and case.
    Returns the float value, or None if not found.
    """
    if not output:
        return None
    match = re.search(r"RESULT\s*:\s*f1\s*=\s*([0-9]*\.?[0-9]+)",
                      output, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_architecture_name(code: str | None) -> str:
    """
    Ask the LLM to summarise the architecture name from the code itself.

    v2 fix: previously crashed when `code` was None (e.g. when the LLM
    response had no code block). Now returns "unknown" gracefully.
    """
    from .llm_v2 import call_llm
    if not code:
        return "unknown"
    prompt = (
        "Look at this Python ML script and reply with ONLY a short name "
        "for the model architecture it implements (e.g. 'TF-IDF + LogisticRegression', "
        "'Keras Embedding + BiLSTM'). No explanation, no markdown, just the name.\n\n"
        f"```python\n{code[:3000]}\n```"
    )
    response = call_llm(prompt)
    if not response:
        return "unknown"
    first_line = response.strip().splitlines()[0].strip() if response.strip() else ""
    return first_line or "unknown"


# ── Learning-curve extraction ────────────────────────────────────────────────
#
# The preamble injected by executor_v2.py monkey-patches Keras so every
# model.fit() call automatically appends a callback that prints lines like:
#
#   EPOCH_METRIC epoch=1 loss=0.5234 val_loss=0.4321
#   EPOCH_METRIC epoch=2 loss=0.4012 val_loss=0.3987
#
# We parse them back into a list[dict] for logging in experiments.json.
_EPOCH_LINE_RE = re.compile(
    r"EPOCH_METRIC\s+epoch=(\d+)\s+loss=([0-9.eE+-]+)\s+val_loss=([0-9.eE+-]+)"
)


def extract_learning_curve(output: str) -> list[dict]:
    """
    Parse all EPOCH_METRIC lines out of stdout into a list of
    {"epoch": int, "loss": float, "val_loss": float}.

    Returns an empty list when the script is sklearn-only (no Keras lines).
    """
    if not output:
        return []
    curve = []
    for m in _EPOCH_LINE_RE.finditer(output):
        try:
            curve.append({
                "epoch":    int(m.group(1)),
                "loss":     float(m.group(2)),
                "val_loss": float(m.group(3)),
            })
        except ValueError:
            continue
    return curve
