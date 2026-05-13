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


LOW_SCORE_THRESHOLD = 0.70


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
    entry.setdefault("assigned_family", "")
    entry.setdefault("prompt", "")
    entry.setdefault("learning_curve", [])
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


def get_failed_architectures(experiments: list[dict]) -> list[str]:
    """
    Architectures that failed and never succeeded later.
    """
    successful = {
        e.get("architecture", "unknown")
        for e in experiments
        if e.get("f1") is not None
    }
    failed = []
    seen = set()
    for e in experiments:
        name = e.get("architecture", "unknown")
        if e.get("f1") is None and name not in successful and name not in seen:
            failed.append(name)
            seen.add(name)
    return failed


def get_low_scoring_architectures(
    experiments: list[dict],
    threshold: float = LOW_SCORE_THRESHOLD,
) -> list[tuple[str, float]]:
    """
    Architectures that ran but whose best observed F1 is below threshold.
    """
    best_per_arch: dict[str, float] = {}
    for e in experiments:
        f1 = e.get("f1")
        if f1 is None:
            continue
        name = e.get("architecture", "unknown")
        if f1 > best_per_arch.get(name, -1.0):
            best_per_arch[name] = f1
    return [(name, f1) for name, f1 in best_per_arch.items() if f1 < threshold]


def format_history_for_prompt(experiments: list[dict], n: int = 5) -> str:
    """Format the last n experiments as a compact summary for the LLM."""
    recent = experiments[-n:]
    lines = []
    for e in recent:
        f1_str = f"{e['f1']:.4f}" if e.get("f1") is not None else "FAILED"
        rationale = e.get("llm_rationale", "")
        family = e.get("assigned_family", "")
        family_part = f" | family={family}" if family else ""
        lines.append(
            f"  Exp #{e['experiment_id']}: {e['architecture']} -> F1={f1_str}{family_part}"
            + (f"  [{rationale}]" if rationale else "")
        )
    return "\n".join(lines) if lines else "  (none yet)"


def format_failed_for_prompt(experiments: list[dict]) -> str:
    failed = get_failed_architectures(experiments)
    if not failed:
        return "  (none)"
    return "\n".join(f"  - {name}" for name in failed)


def format_low_scoring_for_prompt(
    experiments: list[dict],
    threshold: float = LOW_SCORE_THRESHOLD,
) -> str:
    low = get_low_scoring_architectures(experiments, threshold)
    if not low:
        return "  (none)"
    return "\n".join(f"  - {name} (best F1={f1:.4f})" for name, f1 in low)
