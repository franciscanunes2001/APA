"""
Kaggle submission generator (v2).

Same logic as submission.py but imports from executor_v2 so the v2
package is fully self-contained.

Primary path: the LLM-generated script already writes submission.csv
(SYSTEM_PROMPT rule 6 instructs it to do so). This module handles
the fallback case where it didn't.
"""

from pathlib import Path
from .executor_v2 import run_code, execution_succeeded


# Appended to the best experiment's code when it didn't produce a submission.csv.
# Uses only variables that sklearn and Keras scripts reliably define.
_SKLEARN_SUFFIX = """
# ── submission fallback (sklearn) ───────────────────────────────────────────
import pandas as _pd
_test = _pd.read_csv(f"{DATA_DIR}/test.csv")
_sub  = _pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

_kw  = _test['keyword'].fillna('').apply(clean_text)
_txt = _test['text'].fillna('').apply(clean_text)
_X_test = vectorizer.transform((_kw + ' ' + _txt).str.strip())
_sub['target'] = model.predict(_X_test)
_sub.to_csv(f"{DATA_DIR}/submission.csv", index=False)
print("Submission saved.")
"""

_KERAS_SUFFIX = """
# ── submission fallback (keras) ─────────────────────────────────────────────
import numpy as _np
import pandas as _pd
_test = _pd.read_csv(f"{DATA_DIR}/test.csv")
_sub  = _pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

_kw  = _test['keyword'].fillna('').apply(clean_text)
_txt = _test['text'].fillna('').apply(clean_text)
_texts = (_kw + ' ' + _txt).str.strip().tolist()

from tensorflow.keras.preprocessing.sequence import pad_sequences as _pad
_seqs  = tokenizer.texts_to_sequences(_texts)
_X     = _pad(_seqs, maxlen=MAX_LEN, padding='post', truncating='post')
_preds = (model.predict(_X, verbose=0).ravel() > 0.5).astype(int)
_sub['target'] = _preds
_sub.to_csv(f"{DATA_DIR}/submission.csv", index=False)
print("Submission saved.")
"""


def submission_exists(data_dir: str) -> bool:
    return (Path(data_dir) / "submission.csv").exists()


def generate_submission_v2(best_code: str, data_dir: str) -> dict:
    """
    Re-run best experiment code. If submission.csv already exists (written
    by the script itself), we're done. Otherwise try sklearn then keras suffix.
    """
    # First: re-run the script as-is — it may already write submission.csv
    result = run_code(best_code, data_dir)
    if execution_succeeded(result) and submission_exists(data_dir):
        return result

    # Fallback: try sklearn variable names
    result = run_code(best_code + "\n\n" + _SKLEARN_SUFFIX, data_dir)
    if execution_succeeded(result) and submission_exists(data_dir):
        return result

    # Fallback: try keras variable names
    result = run_code(best_code + "\n\n" + _KERAS_SUFFIX, data_dir)
    return result
