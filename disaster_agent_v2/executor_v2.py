"""
Sandboxed code executor (v2).

Same idea as executor.py: write the LLM-generated code to a temp file
and run it as a subprocess with a timeout.

v2 additions on top of the original preamble:
  - Keras model.fit() is monkey-patched to ALWAYS attach an
    EpochPrinter callback. The callback prints one
        EPOCH_METRIC epoch=<i> loss=<x> val_loss=<y>
    line per epoch, which parser_v2.extract_learning_curve() picks up.
  - The patch is opt-out safe: if Keras isn't installed (sklearn-only
    runs) the wrapper does nothing.

Nothing else changes — DATA_DIR, clean_text and make_features are still
injected exactly as before so existing prompts keep working.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path


TIMEOUT_SECONDS = 240  # 4 minutes hard cap per experiment


# Injected at the top of every generated script. Saves ~80 lines of LLM
# output budget per experiment AND eliminates `NameError: make_features
# is not defined` and `df['keywor...` truncation bugs.
PREAMBLE_TEMPLATE = '''\
# ── Auto-injected by executor_v2.py ─────────────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATA_DIR = {data_dir!r}

import re as _re
import pandas as pd
import numpy as np

_CONTRACTIONS = {{
    "don't":"do not","doesn't":"does not","didn't":"did not","won't":"will not",
    "can't":"cannot","couldn't":"could not","isn't":"is not","aren't":"are not",
    "wasn't":"was not","weren't":"were not","it's":"it is","i'm":"i am",
    "i've":"i have","i'll":"i will","i'd":"i would","you're":"you are",
    "you've":"you have","you'll":"you will","you'd":"you would",
    "he's":"he is","she's":"she is","we're":"we are","we've":"we have",
    "we'll":"we will","they're":"they are","they've":"they have",
    "they'll":"they will","that's":"that is","what's":"what is",
    "there's":"there is","let's":"let us","who's":"who is",
}}

def clean_text(text):
    text = str(text).lower()
    text = _re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    text = _re.sub(r'<.*?>', '', text)
    text = _re.sub(r'[^\\x00-\\x7F]+', '', text)
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    text = _re.sub(r'[^a-z0-9\s]', ' ', text)
    text = _re.sub(r'\\s+', ' ', text).strip()
    return text

def make_features(df):
    kw  = df['keyword'].fillna('').apply(clean_text)
    txt = df['text'].fillna('').apply(clean_text)
    return (kw + ' ' + txt).str.strip()


# ── Keras epoch-by-epoch logger (v2) ────────────────────────────────────────
# Monkey-patches keras.Model.fit so EVERY fit call automatically attaches
# a callback that prints one parseable line per epoch. parser_v2 reads
# these to build the learning curve. Silently no-ops if Keras isn't around.
try:
    from tensorflow.keras.callbacks import Callback as _KCallback
    from tensorflow.keras.models import Model as _KModel

    class _EpochPrinter(_KCallback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {{}}
            loss     = logs.get('loss', float('nan'))
            val_loss = logs.get('val_loss', float('nan'))
            # epoch is 0-indexed in Keras; print 1-indexed for humans
            print(f"EPOCH_METRIC epoch={{epoch + 1}} "
                  f"loss={{loss:.6f}} val_loss={{val_loss:.6f}}",
                  flush=True)

    _orig_fit = _KModel.fit

    def _patched_fit(self, *args, **kwargs):
        cbs = list(kwargs.get('callbacks') or [])
        if not any(isinstance(c, _EpochPrinter) for c in cbs):
            cbs.append(_EpochPrinter())
        kwargs['callbacks'] = cbs
        return _orig_fit(self, *args, **kwargs)

    _KModel.fit = _patched_fit
except Exception:
    # tensorflow not installed, or import failed — skip silently
    pass
# ── End preamble ────────────────────────────────────────────────────────────

'''


def run_code(code: str, data_dir: str) -> dict:
    """
    Execute `code` as a standalone Python script with the helper preamble
    prepended. Returns a dict with stdout, stderr, exit_code, timed_out.
    """
    full_code = PREAMBLE_TEMPLATE.format(data_dir=data_dir) + code

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix="agent_v2_exp_",
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
