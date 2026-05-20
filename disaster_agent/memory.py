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
import re
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


LOW_SCORE_THRESHOLD = 0.76


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
    entry.setdefault("variation_axis", "")
    experiments.append(entry)
    _save(log_path, experiments)


def get_best(experiments: list[dict]) -> tuple[float, Optional[dict]]:
    """Returns (best_f1, best_experiment). best_f1=0.0 if no successes."""
    successful = [e for e in experiments if e.get("f1") is not None]
    if not successful:
        return 0.0, None
    best = max(successful, key=lambda e: e["f1"])
    return best["f1"], best


def _is_deep_learning(exp: dict) -> bool:
    name = (exp.get("architecture") or "").lower()
    code = (exp.get("code") or "").lower()
    return "keras embedding" in name or "embedding(" in code


def _is_classical(exp: dict) -> bool:
    if _is_deep_learning(exp):
        return False
    name = (exp.get("architecture") or "").lower()
    code = (exp.get("code") or "").lower()
    return (
        "tf-idf" in name
        or "tfidf" in name
        or "tfidfvectorizer" in code
        or "countvectorizer" in code
    )


def get_best_in_family(
    experiments: list[dict], family_kind: str,
) -> tuple[float, Optional[dict]]:
    """
    family_kind: 'classical' | 'deep_learning' | 'overall'.

    Used by the exploit/refinement curriculum steps to deterministically pick
    the baseline to refine, rather than asking the LLM to guess from a list.
    """
    successful = [e for e in experiments if e.get("f1") is not None]
    if family_kind == "classical":
        filtered = [e for e in successful if _is_classical(e)]
    elif family_kind == "deep_learning":
        filtered = [e for e in successful if _is_deep_learning(e)]
    else:
        filtered = successful
    if not filtered:
        return 0.0, None
    best = max(filtered, key=lambda e: e["f1"])
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
) -> list[tuple[str, float, str]]:
    """
    Architectures that ran but whose best observed F1 is below threshold.
    Returns (architecture_name, best_f1, assigned_family) so callers can
    filter by family.
    """
    best_per_arch: dict[str, tuple[float, str]] = {}
    for e in experiments:
        f1 = e.get("f1")
        if f1 is None:
            continue
        name = e.get("architecture", "unknown")
        family = e.get("assigned_family", "")
        if f1 > best_per_arch.get(name, (-1.0, ""))[0]:
            best_per_arch[name] = (f1, family)
    return [
        (name, f1, family) for name, (f1, family) in best_per_arch.items()
        if f1 < threshold
    ]


_ANALYSIS_PREVIEW_CHARS = 280  # per-experiment cap to keep the history compact

# Lines whose presence in `code` we want to surface in the prompt history.
# This lets the LLM see the actual hyperparameters of past experiments without
# us having to parse them into a structured field. Brittle by design — if a
# pattern is missed, the LLM just doesn't see that line. No prompt-contract
# change required.
_HPARAM_PATTERNS = re.compile(
    r"(TfidfVectorizer|CountVectorizer|LogisticRegression|LinearSVC|"
    r"ComplementNB|MultinomialNB|MLPClassifier|RandomForestClassifier|"
    r"RidgeClassifier|SGDClassifier|GradientBoostingClassifier|"
    r"Embedding\(|LSTM\(|GRU\(|Bidirectional\(|Conv1D\(|MultiHeadAttention\(|"
    r"pad_sequences\(|EarlyStopping\(|model\.fit\(|\.compile\()"
)


def _extract_hparam_lines(code: str, max_chars: int = 500) -> str:
    """Return a compact ' | '-joined summary of constructor/fit lines in code."""
    if not code:
        return ""
    keep = []
    seen_signatures: set[str] = set()
    for raw in code.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "import ", "from ")):
            continue
        if _HPARAM_PATTERNS.search(line):
            # Dedupe lines that differ only in their LHS variable name
            # (e.g. three identical pad_sequences calls for tr/val/test).
            sig = line.split("=", 1)[-1].strip() if "=" in line else line
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            keep.append(line)
    joined = " | ".join(keep)
    if len(joined) > max_chars:
        joined = joined[:max_chars].rstrip() + "…"
    return joined


def format_history_for_prompt(experiments: list[dict], n: int = 5) -> str:
    """
    Format the last n experiments as a compact summary for the LLM.

    Each experiment contributes its arch + F1 + family + one-line LLM
    rationale, followed by a truncated dump of the per-iteration
    `iteration_analysis` from that run. Feeding the prior analysis back
    here is what makes "iterate based on what it learned" actually mean
    something — without it the analysis is generated and never read.
    """
    recent = experiments[-n:]
    lines = []
    for e in recent:
        f1_str = f"{e['f1']:.4f}" if e.get("f1") is not None else "FAILED"
        rationale = e.get("llm_rationale", "")
        family = e.get("assigned_family", "")
        family_part = f" | family={family}" if family else ""
        header = (
            f"  Exp #{e['experiment_id']}: {e['architecture']} -> F1={f1_str}{family_part}"
            + (f"  [{rationale}]" if rationale else "")
        )
        lines.append(header)

        hparams = _extract_hparam_lines(e.get("code") or "")
        if hparams:
            lines.append(f"      hparams: {hparams}")

        analysis = (e.get("iteration_analysis") or "").strip()
        if analysis:
            if len(analysis) > _ANALYSIS_PREVIEW_CHARS:
                analysis = analysis[:_ANALYSIS_PREVIEW_CHARS].rstrip() + "…"
            # Indent so it's visually attached to its experiment line.
            lines.append(f"      analysis: {analysis}")
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
    """
    All architectures whose best observed F1 fell below `threshold`,
    including those in the family the controller is currently assigning.
    Surfacing the in-family weak runs is the signal that pushes the LLM
    away from re-running near-duplicates inside an exploit phase.
    """
    low = get_low_scoring_architectures(experiments, threshold)
    if not low:
        return "  (none)"
    return "\n".join(
        f"  - {name} (best F1={f1:.4f}, family={fam})"
        for name, f1, fam in low
    )


def get_tried_signatures(
    experiments: list[dict], family: Optional[str] = None,
) -> list[tuple[int, str, str, Optional[float], str]]:
    """
    Return (experiment_id, architecture, variation_axis, f1, hparam_sig)
    for prior runs, optionally filtered to a single assigned family.
    The hparam_sig is a compact dump of constructor/fit calls — what the
    LLM should consult to avoid proposing a near-duplicate config.
    """
    items = experiments
    if family:
        items = [e for e in experiments if e.get("assigned_family") == family]
    out = []
    for e in items:
        out.append((
            e.get("experiment_id", -1),
            e.get("architecture", "unknown"),
            (e.get("variation_axis") or "").strip(),
            e.get("f1"),
            _extract_hparam_lines(e.get("code") or "", max_chars=260),
        ))
    return out


def format_tried_in_family_for_prompt(
    experiments: list[dict], family: str,
) -> str:
    sigs = get_tried_signatures(experiments, family=family)
    if not sigs:
        return "  (none — this is the first attempt at this family)"
    lines = []
    for eid, name, axis, f1, sig in sigs:
        f1_str = f"F1={f1:.4f}" if isinstance(f1, (int, float)) else "F1=FAILED"
        axis_str = f"  axis={axis}" if axis else ""
        lines.append(f"  - #{eid} {name}  [{f1_str}]{axis_str}")
        if sig:
            lines.append(f"      hparams: {sig}")
    return "\n".join(lines)


def get_variation_axes_in_family(
    experiments: list[dict], family: str,
) -> Counter:
    """Counter of variation_axis values among prior runs of `family`."""
    c: Counter = Counter()
    for e in experiments:
        if e.get("assigned_family") != family:
            continue
        axis = (e.get("variation_axis") or "").strip().lower()
        c[axis or "(none declared)"] += 1
    return c


def format_variation_axes_in_family_for_prompt(
    experiments: list[dict], family: str,
) -> str:
    c = get_variation_axes_in_family(experiments, family)
    if not c:
        return "  (none — first attempt at this family)"
    return "\n".join(
        f"  - {axis}: {count}" for axis, count in c.most_common()
    )


def format_history_stratified_for_prompt(
    experiments: list[dict], family: str, top_k: int = 5,
) -> str:
    """
    Stratified view of history for the propose step:
      - top-K by F1 across ALL families (so the LLM sees what already worked)
      - every run in `family` (so it sees in-family detail without recency bias)

    Replaces a chronological dump that buries the strongest signals once the
    log grows past a couple of dozen entries.
    """
    successful = [e for e in experiments if isinstance(e.get("f1"), (int, float))]
    top = sorted(successful, key=lambda e: e["f1"], reverse=True)[:top_k]
    in_family = [e for e in experiments if e.get("assigned_family") == family]

    seen_ids: set[int] = set()
    sections: list[str] = []

    def _render(label: str, runs: list[dict]) -> Optional[str]:
        if not runs:
            return None
        lines = [label]
        for e in runs:
            eid = e.get("experiment_id", -1)
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            f1 = e.get("f1")
            f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "FAILED"
            axis = (e.get("variation_axis") or "").strip()
            axis_str = f"  axis={axis}" if axis else ""
            rationale = (e.get("llm_rationale") or "").strip()
            rationale_str = f"  [{rationale}]" if rationale else ""
            fam = e.get("assigned_family", "")
            lines.append(
                f"  Exp #{eid}: {e.get('architecture','?')} -> F1={f1_str}"
                f" | family={fam}{axis_str}{rationale_str}"
            )
            hparams = _extract_hparam_lines(e.get("code") or "")
            if hparams:
                lines.append(f"      hparams: {hparams}")
            analysis = (e.get("iteration_analysis") or "").strip()
            if analysis:
                if len(analysis) > _ANALYSIS_PREVIEW_CHARS:
                    analysis = analysis[:_ANALYSIS_PREVIEW_CHARS].rstrip() + "…"
                lines.append(f"      analysis: {analysis}")
        return "\n".join(lines) if len(lines) > 1 else None

    section_top = _render(f"TOP {top_k} RUNS BY F1 (across all families):", top)
    if section_top:
        sections.append(section_top)
    section_fam = _render(f"ALL RUNS IN ASSIGNED FAMILY ({family}):", in_family)
    if section_fam:
        sections.append(section_fam)

    return "\n\n".join(sections) if sections else "  (none yet)"
