"""
Experiment memory — reads and writes experiments.json.

Each entry:
{
  "experiment_id": 1,
  "architecture":  "TF-IDF + LogisticRegression",
  "f1":            0.7823,        # null if failed
  "status":        "success",     # "success" | "failed" | "timeout"
  "error":         null,          # error message if failed
  "code":          "...",         # the generated code
  "llm_rationale": "...",         # second line from LLM response
  "stdout":        "...",
}
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


def _load(log_path: Path) -> list[dict]:
    if log_path.exists():
        return json.loads(log_path.read_text())
    return []


def _save(log_path: Path, experiments: list[dict]) -> None:
    log_path.write_text(json.dumps(experiments, indent=2))


def load_experiments(log_path: Path) -> list[dict]:
    return _load(log_path)


def add_experiment(log_path: Path, entry: dict) -> None:
    experiments = _load(log_path)
    entry["experiment_id"] = len(experiments) + 1
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    experiments.append(entry)
    _save(log_path, experiments)


def get_best(experiments: list[dict]) -> tuple[float, Optional[dict]]:
    """Returns (best_f1, best_experiment). best_f1=0.0 if no successes."""
    successful = [e for e in experiments if e.get("f1") is not None]
    if not successful:
        return 0.0, None
    best = max(successful, key=lambda e: e["f1"])
    return best["f1"], best


def get_tried_architectures(experiments: list[dict]) -> list[str]:
    return [e["architecture"] for e in experiments]


def format_history_for_prompt(experiments: list[dict], n: int = 5) -> str:
    """Format the last n experiments as a compact summary for the LLM."""
    recent = experiments[-n:]
    lines = []
    for e in recent:
        f1_str = f"{e['f1']:.4f}" if e.get("f1") is not None else "FAILED"
        rationale = e.get("llm_rationale", "")
        lines.append(
            f"  Exp #{e['experiment_id']}: {e['architecture']} → F1={f1_str}"
            + (f"  [{rationale}]" if rationale else "")
        )
    return "\n".join(lines) if lines else "  (none yet)"
