"""
Extracts structured data from LLM responses.
"""

import re
from typing import Optional


def extract_code(response: str) -> Optional[str]:
    """
    Pull the first ```python ... ``` block out of an LLM response.
    Returns the code string, or None if no block found.
    """
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_architecture_name(response: str) -> str:
    """
    The prompts ask the LLM to write the architecture name on the first line
    before the code block. Extract it.
    """
    lines = response.strip().splitlines()
    for line in lines:
        line = line.strip()
        # Skip empty lines and markdown headers
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        # Stop before the code block
        if "```python" in line:
            break
        return line
    return "unknown"


def extract_f1(output: str) -> Optional[float]:
    """
    Parse the mandatory output line:   RESULT: f1=0.7823
    Returns the float value, or None if not found.
    """
    match = re.search(r"RESULT:\s*f1=([0-9.]+)", output)
    if match:
        return float(match.group(1))
    return None
