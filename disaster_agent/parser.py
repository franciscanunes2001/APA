"""
Extracts structured data from LLM responses.

Robust to formatting differences across Ollama models (gemma, qwen, deepseek-r1,
phi4, etc.). Different models wrap code differently:
  - ```python\n...```          (gemma, most)
  - ```Python\n...```           (qwen sometimes — capital P)
  - ```py\n...```               (some smaller models)
  - ```\n...```                 (deepseek-r1, when no language tag)
  - ~~~python\n...~~~           (rare, but some templates)
  - <think>...</think> ```...```  (deepseek-r1 reasoning models)
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


# Strip leading markdown / labels that small models love to add:
#   "**Architecture name:** TF-IDF + LR"   ->  "TF-IDF + LR"
#   'Architecture name: "TF-IDF + LR"'     ->  "TF-IDF + LR"
#   "### TF-IDF + LR"                      ->  "TF-IDF + LR"
_ARCH_PREFIX_RE = re.compile(
    r"""^\s*
        (?:[*_#>\-]+\s*)*                          # markdown bullets/headers
        (?:\*\*)?                                   # optional bold open
        (?:architecture(?:\s*name)?\s*[:\-]\s*)?    # "Architecture name:" label
        (?:\*\*)?                                   # optional bold close
        \s*[\"']?                                   # optional opening quote
    """,
    re.IGNORECASE | re.VERBOSE,
)
_ARCH_SUFFIX_RE = re.compile(r"[\"'\s*_]+$")


def _clean_arch_line(line: str) -> str:
    line = _ARCH_PREFIX_RE.sub("", line)
    line = _ARCH_SUFFIX_RE.sub("", line)
    return line.strip()


def extract_architecture_name(response: str) -> str:
    """
    The prompts ask the LLM to write the architecture name on the first line
    before the code block. Extract it and strip markdown/label noise.
    """
    if not response:
        return "unknown"

    cleaned = _strip_think_blocks(response).strip()

    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Skip pure markdown headers/horizontal rules
        if line.startswith("```") or line.startswith("~~~"):
            break
        if set(line) <= set("-=_*#"):  # e.g. "----" separators
            continue
        candidate = _clean_arch_line(line)
        # Skip empty results and obvious non-name preludes like "Sure!" or "Here is"
        if not candidate or len(candidate) > 120:
            continue
        if re.match(r"^(sure|okay|ok|here|alright|certainly|of course)\b",
                    candidate, re.IGNORECASE):
            continue
        return candidate

    return "unknown"


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
