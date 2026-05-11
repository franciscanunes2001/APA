"""
Experiment memory (v2) — reads and writes experiments_v2.json.

Each entry now includes:
{
  "experiment_id":  1,
  "architecture":   "TF-IDF + LogisticRegression",
  "f1":             0.7823,        # null if failed
  "status":         "success",     # "success" | "failed" | "timeout"
  "error":          null,          # error message if failed
  "code":           "...",         # the generated code
  "llm_rationale":  "...",         # second line from LLM response
  "stdout":         "...",
  "prompt":         "...",         # NEW in v2 — the full prompt sent to the LLM
  "learning_curve": [              # NEW in v2 — empty for sklearn-only runs
      {"epoch": 1, "loss": 0.52, "val_loss": 0.43},
      ...
  ],
}

Two new helpers split the architecture history into:
  - low-scoring (ran successfully but F1 < threshold)
  - failed     (crashed / timed out / no code block)
so the PROPOSE prompt can show them as separate lists.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


# Experiments below this F1 are surfaced in the PROPOSE prompt as
# "tried-and-weak" so the LLM avoids tweaking dead-end families.
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
    # Make sure the v2-only fields always exist (defensive defaulting).
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
    Architectures the agent attempted but never got a working F1 from.
    Excludes architectures that *also* succeeded somewhere else in
    history — those count as "tried" but not "failed".
    """
    successful = {e["architecture"] for e in experiments
                  if e.get("f1") is not None}
    failed = []
    seen = set()
    for e in experiments:
        if e.get("f1") is None and e.get("architecture") not in successful:
            name = e.get("architecture", "unknown")
            if name not in seen:
                failed.append(name)
                seen.add(name)
    return failed


def get_low_scoring_architectures(
    experiments: list[dict],
    threshold: float = LOW_SCORE_THRESHOLD,
) -> list[tuple[str, float]]:
    """
    Architectures that ran successfully but scored below `threshold`.
    Returns (architecture, best_f1_for_that_arch) tuples.
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
        lines.append(
            f"  Exp #{e['experiment_id']}: {e['architecture']} → F1={f1_str}"
            + (f"  [{rationale}]" if rationale else "")
        )
    return "\n".join(lines) if lines else "  (none yet)"


def format_failed_for_prompt(experiments: list[dict]) -> str:
    """Bulleted list of architectures the agent should NEVER try again."""
    failed = get_failed_architectures(experiments)
    if not failed:
        return "  (none)"
    return "\n".join(f"  - {name}" for name in failed)


def format_low_scoring_for_prompt(
    experiments: list[dict],
    threshold: float = LOW_SCORE_THRESHOLD,
) -> str:
    """Bulleted list of architectures that ran but scored low."""
    low = get_low_scoring_architectures(experiments, threshold)
    if not low:
        return "  (none)"
    return "\n".join(f"  - {name} (best F1={f1:.4f})" for name, f1 in low)
