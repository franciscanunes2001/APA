"""
Sandboxed code executor.
Writes generated code to a temp file and runs it in a subprocess
with a timeout. Captures stdout/stderr. Never uses eval/exec.
"""

import os
import sys
import subprocess
import tempfile
import textwrap
from pathlib import Path


TIMEOUT_SECONDS = 240  # 4 minutes hard cap per experiment


def run_code(code: str, data_dir: str) -> dict:
    """
    Execute `code` as a standalone Python script.

    Prepends `DATA_DIR = "<data_dir>"` so the script can use it directly.

    Returns a dict:
      {
        "stdout":    str,
        "stderr":    str,
        "exit_code": int,
        "timed_out": bool,
      }
    """
    # Inject DATA_DIR at the top of the script
    preamble = f'DATA_DIR = {data_dir!r}\n\n'
    full_code = preamble + code

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix="agent_exp_",
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        return {
            "stdout":    result.stdout,
            "stderr":    result.stderr,
            "exit_code": result.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout":    "",
            "stderr":    f"Timed out after {TIMEOUT_SECONDS}s.",
            "exit_code": -1,
            "timed_out": True,
        }
    finally:
        os.unlink(tmp_path)


def execution_succeeded(result: dict) -> bool:
    return result["exit_code"] == 0 and not result["timed_out"]
