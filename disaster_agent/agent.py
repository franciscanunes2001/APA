"""
Autonomous Research Agent — main loop.

Usage:
    python -m disaster_agent.agent [--max-iter N] [--target-f1 F]

The loop:
  1. If no experiments yet → run FIRST_EXPERIMENT_PROMPT (TF-IDF baseline)
  2. Otherwise → call PROPOSE_PROMPT_TEMPLATE with history
  3. Extract code from LLM response
  4. Execute code; if it crashes → call FIX_PROMPT_TEMPLATE (up to 2 retries)
  5. Parse F1 from stdout
  6. Log result to experiments.json
  7. Repeat until max_iter reached or F1 ≥ target_f1
  8. Generate Kaggle submission from best experiment
"""

import argparse
import textwrap
from pathlib import Path

from .llm import call_llm
from .parser import extract_code, extract_architecture_name, extract_f1
from .executor import run_code, execution_succeeded
from .memory import (
    load_experiments,
    add_experiment,
    get_best,
    get_tried_architectures,
    format_history_for_prompt,
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
DEFAULT_MAX_ITER  = 7
DEFAULT_TARGET_F1 = 0.82
MAX_FIX_RETRIES   = 2


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    line = "─" * 60
    print(f"\n{line}\n  {text}\n{line}")


def _run_with_fixes(
    code: str,
    architecture: str,
    data_dir: str,
) -> tuple[dict, str]:
    """
    Execute code; if it fails, ask the LLM to fix it (up to MAX_FIX_RETRIES).
    Returns (execution_result, final_code).
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
            print("  [llm] Could not extract fixed code.")
            return result, current_code

    return result, current_code  # unreachable but satisfies linter


def _propose_experiment(experiments: list[dict]) -> tuple[str, str, str]:
    """
    Ask the LLM for the next experiment.
    Returns (llm_response, architecture_name, code).
    """
    if not experiments:
        print("  [llm] Requesting first experiment (TF-IDF baseline)...")
        response = call_llm(FIRST_EXPERIMENT_PROMPT)
    else:
        best_f1, _ = get_best(experiments)
        history_str = format_history_for_prompt(experiments)
        tried_list  = ", ".join(get_tried_architectures(experiments))

        prompt = PROPOSE_PROMPT_TEMPLATE.format(
            n=min(len(experiments), 5),
            history=history_str,
            best_f1=best_f1,
            tried_list=tried_list,
        )
        print("  [llm] Requesting next experiment proposal...")
        response = call_llm(prompt)

    architecture = extract_architecture_name(response)
    code = extract_code(response)
    return response, architecture, code


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_agent(max_iter: int = DEFAULT_MAX_ITER, target_f1: float = DEFAULT_TARGET_F1) -> None:
    _banner(f"Disaster Tweet Agent  |  max_iter={max_iter}  target_f1={target_f1}")

    experiments = load_experiments(LOG_PATH)
    print(f"  Loaded {len(experiments)} previous experiments from {LOG_PATH.name}")

    successful_iterations = 0  # only count experiments that ran (not parse failures)

    for iteration in range(1, max_iter * 2):  # allow extra attempts for failed parses
        if successful_iterations >= max_iter:
            break

        _banner(f"Iteration {successful_iterations + 1}/{max_iter}")

        # 1. Propose
        response, architecture, code = _propose_experiment(experiments)
        print(f"  Architecture: {architecture}")

        # Extract optional rationale (second non-empty line before code block)
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
                "code":          "",
                "llm_rationale": llm_rationale,
                "stdout":        "",
            })
            experiments = load_experiments(LOG_PATH)
            continue  # don't count against max_iter

        successful_iterations += 1

        # 2. Execute (with auto-fix retries)
        print("  [executor] Running...")
        result, final_code = _run_with_fixes(code, architecture, DATA_DIR)

        # 3. Parse F1
        f1 = None
        status = "failed"
        error_msg = None

        if result["timed_out"]:
            status = "timeout"
            error_msg = "Timed out"
            print("  [executor] TIMED OUT")
        elif execution_succeeded(result):
            f1 = extract_f1(result["stdout"])
            if f1 is not None:
                status = "success"
                print(f"  [result]   F1 = {f1:.4f}")
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
            "code":          final_code,
            "llm_rationale": llm_rationale,
            "stdout":        result["stdout"][:2000],
        })
        experiments = load_experiments(LOG_PATH)

        # 5. Check stopping criterion
        best_f1, best_exp = get_best(experiments)
        print(f"  [memory]   Best so far: F1={best_f1:.4f} ({best_exp['architecture'] if best_exp else 'none'})")

        if best_f1 >= target_f1:
            _banner(f"Target F1 {target_f1} reached! Stopping early.")
            break

    # ── End of loop: analysis + submission ───────────────────────────────────
    _banner("Agent loop complete")
    experiments = load_experiments(LOG_PATH)
    best_f1, best_exp = get_best(experiments)

    if not best_exp:
        print("No successful experiments. Cannot generate submission.")
        return

    print(f"Best experiment: #{best_exp['experiment_id']} — {best_exp['architecture']} — F1={best_f1:.4f}")

    # Optional LLM analysis
    history_str = format_history_for_prompt(experiments, n=len(experiments))
    analysis_prompt = ANALYZE_PROMPT_TEMPLATE.format(
        history=history_str,
        best_f1=best_f1,
    )
    print("\n[llm] Requesting analysis of results...")
    analysis = call_llm(analysis_prompt)
    print(f"\n── LLM Analysis ──\n{analysis}\n")

    # Save analysis to file
    (PROJECT_ROOT / "analysis.txt").write_text(analysis)

    # Generate Kaggle submission
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
    parser.add_argument("--max-iter",  type=int,   default=DEFAULT_MAX_ITER,  help="Max iterations")
    parser.add_argument("--target-f1", type=float, default=DEFAULT_TARGET_F1, help="Stop when F1 ≥ this")
    args = parser.parse_args()
    run_agent(max_iter=args.max_iter, target_f1=args.target_f1)


if __name__ == "__main__":
    main()





