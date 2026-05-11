"""
Autonomous Research Agent (v2) — main loop.

Usage:
    python -m disaster_agent.agent_v2 [--max-iter N] [--target-f1 F]

What changed from agent.py:
  1. Curriculum index is computed proportionally to max_iter so the
     last N iterations don't all collapse onto "Final best-model
     refinement" when max_iter > len(CURRICULUM).
  2. The full prompt sent to the LLM is logged in experiments_v2.json
     under the "prompt" field.
  3. Keras learning curves (loss / val_loss per epoch) are extracted
     from stdout and logged under "learning_curve".
  4. PROPOSE prompt receives two separate lists — failed and
     low-scoring architectures — so the LLM understands the difference
     between "don't repeat that mistake" and "don't waste a slot tweaking".
  5. plot_results() saves an F1-progression chart (results.png) at the
     end of the run, with failed experiments marked clearly.
"""

import argparse
from pathlib import Path

from .llm_v2 import call_llm
from .parser_v2 import (
    extract_code,
    extract_architecture_name,
    extract_f1,
    extract_learning_curve,
)
from .executor_v2 import run_code, execution_succeeded
from .memory_v2 import (
    load_experiments,
    add_experiment,
    get_best,
    get_tried_architectures,
    format_history_for_prompt,
    format_failed_for_prompt,
    format_low_scoring_for_prompt,
)
from .prompts_v2 import (
    SYSTEM_PROMPT,
    FIRST_EXPERIMENT_PROMPT,
    PROPOSE_PROMPT_TEMPLATE,
    FIX_PROMPT_TEMPLATE,
    ANALYZE_PROMPT_TEMPLATE,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = str(PROJECT_ROOT)
LOG_PATH     = PROJECT_ROOT / "experiments_v2.json"
PLOT_PATH    = PROJECT_ROOT / "results.png"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MAX_ITER  = 12
DEFAULT_TARGET_F1 = 0.82
MAX_FIX_RETRIES   = 3   # weaker models occasionally need a third shot

CURRICULUM = [
    "TF-IDF / bag-of-words strong linear baseline",
    "TF-IDF / bag-of-words probabilistic baseline",
    "TF-IDF / bag-of-words Dense MLP",
    "Keras Embedding + 1D CNN",
    "Keras Embedding + GRU or BiLSTM",
    "Keras Embedding + CNN + GRU/BiGRU hybrid",
    "Keras Embedding + small Transformer/self-attention block",
    "Exploit best classical family so far",
    "Exploit best deep-learning family so far",
    "Final best-model refinement",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    line = "─" * 60
    print(f"\n{line}\n  {text}\n{line}")


def _curriculum_family(iteration_idx: int, max_iter: int) -> str:
    """
    Map iteration index -> curriculum entry, proportionally.

    With max_iter=12 and 10 curriculum entries the original code mapped
    iters 9..11 all onto the final entry. v2 spreads them evenly:
    iteration_idx=0 -> entry 0, iteration_idx=max_iter-1 -> last entry.

    We never want to repeat the very first family on a non-first iteration,
    so we bias the index forward by 0.5 of a slot.
    """
    if max_iter <= 1:
        return CURRICULUM[0]
    # Map [0, max_iter-1] -> [0, len(CURRICULUM)-1] with linear interpolation.
    pos = iteration_idx * (len(CURRICULUM) - 1) / (max_iter - 1)
    idx = int(round(pos))
    idx = max(0, min(len(CURRICULUM) - 1, idx))
    return CURRICULUM[idx]


def _run_with_fixes(
    code: str,
    architecture: str,
    data_dir: str,
) -> tuple[dict, str]:
    """
    Execute code; if it fails, ask the LLM to fix it.

    Strategy:
      attempt 1: run the original code
      attempts 2..MAX_FIX_RETRIES-1: send error + code, ask for fix
      final attempt (MAX_FIX_RETRIES): give up on the broken code entirely
                                       and ask for a fresh rewrite from scratch
    """
    current_code = code

    for attempt in range(MAX_FIX_RETRIES + 1):
        result = run_code(current_code, data_dir)

        if execution_succeeded(result):
            return result, current_code

        error_msg = (result["stderr"] or "Unknown error")[:2000]

        if attempt == MAX_FIX_RETRIES:
            print(f"  [executor] Still failing after {MAX_FIX_RETRIES} fix attempts.")
            return result, current_code

        is_last_attempt = (attempt == MAX_FIX_RETRIES - 1)
        if is_last_attempt:
            print(f"  [executor] Attempt {attempt + 1} failed. "
                  f"Asking LLM for a fresh rewrite (final attempt)...")
            fix_prompt = (
                SYSTEM_PROMPT + "\n\n"
                f"You previously tried to write a '{architecture}' model and it failed.\n"
                f"Last error was:\n```\n{error_msg[:500]}\n```\n\n"
                f"Forget the previous attempt entirely. Write a NEW complete script "
                f"for '{architecture}' from scratch. Use the pre-defined helpers "
                f"(clean_text, make_features, DATA_DIR). Be conservative — use only "
                f"sklearn defaults you are confident about, or follow the KERAS "
                f"TEMPLATE exactly if this is a Keras model. "
                f"Return ONE ```python``` block."
            )
        else:
            print(f"  [executor] Attempt {attempt + 1} failed. Asking LLM to fix...")
            fix_prompt = FIX_PROMPT_TEMPLATE.format(
                architecture=architecture,
                error=error_msg,
                code=current_code,
            )

        fix_response = call_llm(fix_prompt)
        fixed_code = extract_code(fix_response)
        if fixed_code:
            current_code = fixed_code
        else:
            print("  [llm] Could not extract fixed code from response.")
            if is_last_attempt:
                return result, current_code

    return result, current_code  # unreachable


def _propose_experiment(
    experiments: list[dict],
    iteration_idx: int,
    max_iter: int,
) -> tuple[str, str, str, str, str]:
    """
    Returns (llm_response, full_prompt_sent, assigned_family,
             architecture_name, code).

    The full prompt is returned so the caller can log it in experiments.json.
    """
    if not experiments:
        assigned_family = CURRICULUM[0]
        prompt_used = FIRST_EXPERIMENT_PROMPT

        print("  [llm] Requesting first experiment (TF-IDF baseline)...")
        response = call_llm(prompt_used)
    else:
        best_f1, _ = get_best(experiments)
        history_str    = format_history_for_prompt(experiments)
        failed_str     = format_failed_for_prompt(experiments)
        low_scoring_str = format_low_scoring_for_prompt(experiments)

        assigned_family = _curriculum_family(iteration_idx, max_iter)

        base_prompt = PROPOSE_PROMPT_TEMPLATE.format(
            n=min(len(experiments), 5),
            history=history_str,
            best_f1=best_f1,
            failed_list=failed_str,
            low_scoring_list=low_scoring_str,
        )
        prompt_used = base_prompt + f"""

        IMPORTANT:
        For this experiment, you MUST implement exactly this architecture family:

        {assigned_family}

        Do NOT choose another family.
        Do NOT propose any architecture from the FAILED list above.
        During exploration:
        - prioritize diversity of architecture families;
        - keep models lightweight and fast.

        During exploitation/refinement:
        - change only ONE meaningful design choice or parameter at a time;
        - explain briefly why the change may improve F1.

        Remember:
        Line 1 of your response must be the REAL architecture implemented,
        not the curriculum family name.
        """

        print(f"  [llm] Requesting experiment: {assigned_family}")
        response = call_llm(prompt_used)

    code = extract_code(response)
    architecture = extract_architecture_name(code)
    return response, prompt_used, assigned_family, architecture, code


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(experiments: list[dict], out_path: Path = PLOT_PATH) -> None:
    """
    Save an F1-progression chart to results.png.

    - Successful experiments: blue dots connected by a line, with the
      running-best F1 drawn as a dashed line on top.
    - Failed / timed-out experiments: red 'x' markers along the bottom
      so they're visible but don't distort the y-axis.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not installed — skipping results.png")
        return

    if not experiments:
        print("  [plot] No experiments to plot.")
        return

    iters, f1s, labels = [], [], []
    fail_iters, fail_labels = [], []
    for e in experiments:
        i = e["experiment_id"]
        if e.get("f1") is not None:
            iters.append(i)
            f1s.append(e["f1"])
            labels.append(e.get("architecture", ""))
        else:
            fail_iters.append(i)
            fail_labels.append(e.get("architecture", "failed"))

    # Running best
    best_so_far = []
    cur = 0.0
    by_iter = {e["experiment_id"]: e.get("f1") for e in experiments}
    for i in sorted(by_iter.keys()):
        f1 = by_iter[i]
        if f1 is not None and f1 > cur:
            cur = f1
        best_so_far.append((i, cur))

    fig, ax = plt.subplots(figsize=(10, 5))
    if iters:
        ax.plot(iters, f1s, marker="o", linewidth=1.5, color="#1f77b4",
                label="experiment F1")
    if best_so_far:
        bx, by = zip(*best_so_far)
        ax.plot(bx, by, linestyle="--", color="#2ca02c", label="best so far")
    if fail_iters:
        # Pin failed runs at y=0 so they don't squash the scale.
        ax.scatter(fail_iters, [0.0] * len(fail_iters),
                   marker="x", s=60, color="#d62728", label="failed / timeout")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("F1 (validation)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Disaster-tweet agent — F1 per iteration")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] Saved {out_path}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_agent(max_iter: int = DEFAULT_MAX_ITER, target_f1: float = DEFAULT_TARGET_F1) -> None:
    _banner(f"Disaster Tweet Agent v2  |  max_iter={max_iter}  target_f1={target_f1}")

    experiments = load_experiments(LOG_PATH)
    print(f"  Loaded {len(experiments)} previous experiments from {LOG_PATH.name}")

    successful_iterations = 0

    while successful_iterations < max_iter:

        _banner(f"Iteration {successful_iterations + 1}/{max_iter}")

        # 1. Propose
        response, prompt_used, assigned_family, architecture, code = (
            _propose_experiment(experiments, successful_iterations, max_iter)
        )
        print(f"  Family:       {assigned_family}")
        print(f"  Architecture: {architecture}")

        # Optional rationale (second non-empty, non-comment, non-fence line)
        rationale_lines = [
            l.strip() for l in response.splitlines()
            if l.strip() and not l.strip().startswith("#") and "```" not in l
        ]
        llm_rationale = rationale_lines[1] if len(rationale_lines) > 1 else ""

        if code is None:
            print("  [parser] No code block found in LLM response. "
                  "Logging as failed and continuing.")
            add_experiment(LOG_PATH, {
                "assigned_family": assigned_family,
                "architecture":   architecture,
                "f1":             None,
                "status":         "failed",
                "error":          "No code block in LLM response",
                "code":           "",
                "llm_rationale":  llm_rationale,
                "stdout":         "",
                "prompt":         prompt_used,
                "learning_curve": [],
            })
            experiments = load_experiments(LOG_PATH)
            successful_iterations += 1
            continue

        successful_iterations += 1

        # 2. Execute (with fix-retries)
        print("  [executor] Running...")
        result, final_code = _run_with_fixes(code, architecture, DATA_DIR)

        # 3. Parse F1 + learning curve
        f1 = None
        status = "failed"
        error_msg = None
        learning_curve = extract_learning_curve(result["stdout"])

        if result["timed_out"]:
            status = "timeout"
            error_msg = "Timed out"
            print("  [executor] TIMED OUT")
        elif execution_succeeded(result):
            f1 = extract_f1(result["stdout"])
            if f1 is not None:
                status = "success"
                print(f"  [result]   F1 = {f1:.4f}")
                if learning_curve:
                    print(f"  [result]   Captured {len(learning_curve)} epoch(s) "
                          f"of learning-curve data")
            else:
                status = "failed"
                error_msg = "Script ran but RESULT line not found in stdout"
                print(f"  [result]   No RESULT line found. "
                      f"stdout:\n{result['stdout'][:500]}")
        else:
            error_msg = (result["stderr"] or "")[:1000]
            print(f"  [executor] FAILED:\n{error_msg[:300]}")

        # 4. Log
        add_experiment(LOG_PATH, {
            "assigned_family": assigned_family,
            "architecture":   architecture,
            "f1":             f1,
            "status":         status,
            "error":          error_msg,
            "code":           final_code,
            "llm_rationale":  llm_rationale,
            "stdout":         result["stdout"][:2000],
            "prompt":         prompt_used,
            "learning_curve": learning_curve,
        })
        experiments = load_experiments(LOG_PATH)

        # 5. Status print
        best_f1, best_exp = get_best(experiments)
        print(f"  [memory]   Best so far: F1={best_f1:.4f} "
              f"({best_exp['architecture'] if best_exp else 'none'})")


    # ── Wrap-up ───────────────────────────────────────────────────────────────
    _banner("Agent loop complete")
    experiments = load_experiments(LOG_PATH)
    best_f1, best_exp = get_best(experiments)

    # Always try to plot, even if no experiment succeeded.
    plot_results(experiments)

    if not best_exp:
        print("No successful experiments. Cannot generate submission.")
        return

    print(f"Best experiment: #{best_exp['experiment_id']} — "
          f"{best_exp['architecture']} — F1={best_f1:.4f}")

    history_str = format_history_for_prompt(experiments, n=len(experiments))
    analysis_prompt = ANALYZE_PROMPT_TEMPLATE.format(
        history=history_str,
        best_f1=best_f1,
    )
    print("\n[llm] Requesting analysis of results...")
    analysis = call_llm(analysis_prompt)
    print(f"\n── LLM Analysis ──\n{analysis}\n")

    (PROJECT_ROOT / "analysis_v2.txt").write_text(analysis)

    _banner("Generating Kaggle submission")
    print("  Re-running best code to generate test predictions...")

    from .submission_v2 import generate_submission_v2
    sub_result = generate_submission_v2(best_exp["code"], DATA_DIR)

    if execution_succeeded(sub_result):
        print(f"  submission.csv written to {PROJECT_ROOT}/submission.csv")
    else:
        print(f"  Submission generation failed:\n{sub_result['stderr'][:500]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Disaster Tweet autonomous agent (v2)")
    parser.add_argument("--max-iter",  type=int,   default=DEFAULT_MAX_ITER)
    parser.add_argument("--target-f1", type=float, default=DEFAULT_TARGET_F1)
    args = parser.parse_args()
    run_agent(max_iter=args.max_iter, target_f1=args.target_f1)


if __name__ == "__main__":
    main()
