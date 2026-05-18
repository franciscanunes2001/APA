"""
Autonomous Research Agent — main loop.

Usage:
    python -m disaster_agent.agent [--max-iter N] [--target-f1 F]
"""

import argparse
from pathlib import Path

from .llm import call_llm
from .parser import (
    extract_code,
    extract_architecture_name,
    extract_f1,
    extract_learning_curve,
)
from .executor import run_code, execution_succeeded
from .memory import (
    load_experiments,
    add_experiment,
    get_best,
    get_tried_architectures,
    format_history_for_prompt,
    format_failed_for_prompt,
    format_low_scoring_for_prompt,
)
from .prompts import (
    FIRST_EXPERIMENT_PROMPT,
    PROPOSE_PROMPT_TEMPLATE,
    FIX_PROMPT_TEMPLATE,
    ANALYZE_PROMPT_TEMPLATE,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = str(PROJECT_ROOT)
LOG_PATH     = PROJECT_ROOT / "experiments.json"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MAX_ITER  = 10
DEFAULT_TARGET_F1 = 0.82
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
MAX_FIX_RETRIES   = 3   # was 2 — bumped to give weaker models a third shot

# ── Model Constants ───────────────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_TRAINING_EXAMPLES = 7600  # Approximate from dataset (~7600 rows)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
STEPS_PER_EPOCH = int(NUM_TRAINING_EXAMPLES * TRAIN_SPLIT) // BATCH_SIZE
EPOCHS = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    line = "─" * 60
    print(f"\n{line}\n  {text}\n{line}")


def _curriculum_family(iteration_idx: int, max_iter: int) -> str:
    """
    Choose the architecture family assigned to this iteration.

    Short runs walk the curriculum in order. Longer runs are spread across the
    full curriculum so the agent explores first, then exploits/refines later.
    """
    if max_iter <= 1:
        return CURRICULUM[0]
    if max_iter <= len(CURRICULUM):
        idx = min(iteration_idx, len(CURRICULUM) - 1)
    else:
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
      attempt 2: send error + code, ask for fix
      attempt 3: same as 2 but with stronger framing
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

        # On the LAST fix attempt, throw away the broken code and ask for
        # a clean rewrite. Iterating on broken code tends to compound errors.
        is_last_attempt = (attempt == MAX_FIX_RETRIES - 1)
        if is_last_attempt:
            print(f"  [executor] Attempt {attempt + 1} failed. "
                  f"Asking LLM for a fresh rewrite (final attempt)...")
            fix_prompt = (
                f"You previously tried to write a '{architecture}' model and it failed.\n"
                f"Last error was:\n```\n{error_msg[:500]}\n```\n\n"
                f"Forget the previous attempt entirely. Write a NEW complete script "
                f"for '{architecture}' from scratch. Use the pre-defined helpers "
                f"(clean_text, make_features, DATA_DIR). Be conservative — use only "
                f"sklearn defaults you are confident about. Return ONE ```python``` block."
            )
            # Re-prepend SYSTEM_PROMPT context manually
            from .prompts import SYSTEM_PROMPT
            fix_prompt = SYSTEM_PROMPT + "\n\n" + fix_prompt
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
            # Don't update current_code — try again with same code or bail next loop
            if is_last_attempt:
                return result, current_code

    return result, current_code  # unreachable


def _propose_experiment(
    experiments: list[dict],
    iteration_idx: int,
    max_iter: int,
) -> tuple[str, str, str, str, str]:
    """Returns (llm_response, prompt_used, assigned_family, architecture_name, code)."""
    assigned_family = _curriculum_family(iteration_idx, max_iter)
    if not experiments:
        assigned_family = CURRICULUM[0]
        print("  [llm] Requesting first experiment (TF-IDF baseline)...")
        prompt_used = FIRST_EXPERIMENT_PROMPT
        response = call_llm(prompt_used)
    else:
        best_f1, best_exp = get_best(experiments)
        history_str = format_history_for_prompt(experiments)
        tried_list  = ", ".join(get_tried_architectures(experiments))
        failed_list = format_failed_for_prompt(experiments)
        low_scoring_list = format_low_scoring_for_prompt(experiments)

        prompt_used = PROPOSE_PROMPT_TEMPLATE.format(
            n=min(len(experiments), 5),
            history=history_str,
            best_f1=best_f1,
            tried_list=tried_list,
            failed_list=failed_list,
            low_scoring_list=low_scoring_list,
        )
        prompt_used += f"""

IMPORTANT - CONTROLLER ASSIGNED FAMILY:
For this experiment, you MUST implement exactly this architecture family:

{assigned_family}

The current best model overall is: {best_exp['architecture'] if best_exp else 'none yet'} with F1={best_f1:.4f}

For "Exploit best classical family": ignore the overall best if it uses Keras Embedding.
Base your implementation on the best TF-IDF based model found so far.

For "Exploit best deep-learning family": ignore the overall best if it uses TF-IDF.
Base your implementation on the best Keras Embedding model found so far.
BEST KERAS EMBEDDING MODEL SO FAR: look in the experiment history above and find
the highest F1 among models using Keras Embedding layers — use THAT as your base.

Do NOT switch to a different architecture family than assigned.

Do NOT choose another family.
Do NOT write the family name as the architecture name. Line 1 must be the
actual concrete architecture implemented, for example "TF-IDF + ComplementNB"
or "Keras Embedding + CNN + GRU".
During exploration, prioritize diversity of architecture families.
During exploitation/refinement, change only ONE meaningful design choice at a time.
EXPLOITATION WARNING: do NOT add extra Dense layers — this causes overfitting on 
small datasets. Instead tune: batch_size, dropout rate, embedding dim, or patience.
"""
        print(f"  [llm] Requesting experiment: {assigned_family}")
        response = call_llm(prompt_used)

    architecture = extract_architecture_name(response)
    code = extract_code(response)
    return response, prompt_used, assigned_family, architecture, code


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_agent(max_iter: int = DEFAULT_MAX_ITER, target_f1: float = DEFAULT_TARGET_F1) -> None:
    _banner(f"Disaster Tweet Agent  |  max_iter={max_iter}  target_f1={target_f1}")

    experiments = load_experiments(LOG_PATH)
    print(f"  Loaded {len(experiments)} previous experiments from {LOG_PATH.name}")

    successful_iterations = 0

    for iteration in range(1, max_iter * 2):
        if successful_iterations >= max_iter:
            break

        _banner(f"Iteration {successful_iterations + 1}/{max_iter}")

        # 1. Propose
        response, prompt_used, assigned_family, architecture, code = _propose_experiment(
            experiments,
            successful_iterations,
            max_iter,
        )
        print(f"  Family:       {assigned_family}")
        print(f"  Architecture: {architecture}")

        # Optional rationale
        rationale_lines = [
            l.strip() for l in response.splitlines()
            if l.strip() and not l.strip().startswith("#") and "```" not in l
        ]
        llm_rationale = rationale_lines[1] if len(rationale_lines) > 1 else ""

        if code is None:
            print("  [parser] No code block found in LLM response. Retrying without counting.")
            add_experiment(LOG_PATH, {
                "architecture":  architecture,
                "f1":            None,
                "status":        "failed",
                "error":         "No code block in LLM response",
                "assigned_family": assigned_family,
                "code":          "",
                "llm_rationale": llm_rationale,
                "stdout":        "",
                "prompt":        prompt_used,
                "learning_curve": [],
            })
            experiments = load_experiments(LOG_PATH)
            continue

        successful_iterations += 1

        # 2. Execute
        print("  [executor] Running...")
        result, final_code = _run_with_fixes(code, architecture, DATA_DIR)

        # 3. Parse F1
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
                    print(f"  [result]   Captured {len(learning_curve)} Keras epoch(s)")
            else:
                status = "failed"
                error_msg = "Script ran but RESULT line not found in stdout"
                print(f"  [result]   No RESULT line found. stdout:\n{result['stdout'][:500]}")
        else:
            error_msg = (result["stderr"] or "")[:1000]
            print(f"  [executor] FAILED:\n{error_msg[:300]}")

        # 4. Log
        add_experiment(LOG_PATH, {
            "architecture":  architecture,
            "f1":            f1,
            "status":        status,
            "error":         error_msg,
            "assigned_family": assigned_family,
            "code":          final_code,
            "llm_rationale": llm_rationale,
            "stdout":        result["stdout"][:2000],
            "prompt":        prompt_used,
            "learning_curve": learning_curve,
        })
        experiments = load_experiments(LOG_PATH)

        # 5. Stopping criterion
        best_f1, best_exp = get_best(experiments)
        print(f"  [memory]   Best so far: F1={best_f1:.4f} "
              f"({best_exp['architecture'] if best_exp else 'none'})")

        if best_f1 >= target_f1:
            _banner(f"Target F1 {target_f1} reached! Stopping early.")
            break

    # ── Wrap-up ───────────────────────────────────────────────────────────────
    _banner("Agent loop complete")
    experiments = load_experiments(LOG_PATH)
    best_f1, best_exp = get_best(experiments)

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

    (PROJECT_ROOT / "analysis.txt").write_text(analysis)

    _banner("Generating Kaggle submission")
    print("  Re-running best code to generate test predictions...")

    from .submission import generate_submission
    sub_result = generate_submission(best_exp["code"], DATA_DIR)

    if execution_succeeded(sub_result):
        print(f"  submission.csv written to {PROJECT_ROOT}/submission.csv")
    else:
        print(f"  Submission generation failed:\n{sub_result['stderr'][:500]}")
        print("\n  Tip: run `python disaster_agent/generate_submission.py` manually.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Disaster Tweet autonomous agent")
    parser.add_argument("--max-iter",  type=int,   default=DEFAULT_MAX_ITER)
    parser.add_argument("--target-f1", type=float, default=DEFAULT_TARGET_F1)
    args = parser.parse_args()
    run_agent(max_iter=args.max_iter, target_f1=args.target_f1)


if __name__ == "__main__":
    main()
