"""
Prompt templates for the Disaster Tweets autonomous research agent.

Helpers (clean_text, make_features) and DATA_DIR are pre-injected into
every script by executor.py. The LLM should USE them, not redefine them.
"""

SYSTEM_PROMPT = """You are an ML engineer building NLP models for binary text classification.

TASK: Classify tweets as disaster-related (target=1) or not (target=0).
METRIC: F1 score (binary, positive class = 1).

ALREADY DEFINED FOR YOU (do NOT redefine these — just use them):
  - DATA_DIR              : str path to the data directory
  - clean_text(text)      : returns a cleaned lowercase string
  - make_features(df)     : returns a Series of "keyword + text" cleaned strings
  Tensorflow logging is already silenced.

DATASET:
  - train.csv: columns [id, keyword, location, text, target]  (~7600 rows)
  - test.csv:  columns [id, keyword, location, text]          (~3200 rows)
  - sample_submission.csv: columns [id, target]

AVAILABLE LIBRARIES (use ONLY these):
  - sklearn  (TfidfVectorizer, CountVectorizer, LogisticRegression, MLPClassifier,
              SVC, LinearSVC, SGDClassifier, RidgeClassifier,
              RandomForestClassifier, GradientBoostingClassifier, etc.)
  - tensorflow / keras (for deep learning models)
  - numpy, pandas, re, string

NOT AVAILABLE (do NOT import — your script will crash):
  - torch, transformers, spacy, nltk, xgboost, lightgbm, catboost, gensim

SKLEARN GOTCHAS (the agent has crashed on these before — pay attention):
  - LinearSVC, RidgeClassifier, SGDClassifier do NOT accept n_jobs.
  - GradientBoostingClassifier does NOT accept class_weight.
  - LinearSVC has no predict_proba; use decision_function for soft voting.
  - For ensembles, use sklearn.ensemble.VotingClassifier.

EVALUATION — read this carefully:
  - val_f1 = f1_score(y_val, y_pred_val)
  - NEVER pass y_train as the first argument. NEVER compare to y_train.

OUTPUT FORMAT (THIS IS THE MOST IMPORTANT RULE):
  Your response MUST be:

  <architecture name on one line, no markdown, no quotes, no labels>
  <one sentence rationale>

  ```python
  <complete Python script>
  ```

  - Use lowercase ```python (not ```Python, not ```py, not bare ```).
  - The opening fence must be on its own line.
  - Do NOT add any prose AFTER the closing ``` fence.
  - Do NOT use **bold** or "Architecture name:" labels on line 1.

SCRIPT REQUIREMENTS:
  1. Load data:
       import pandas as pd
       train = pd.read_csv(f"{{DATA_DIR}}/train.csv")
       test  = pd.read_csv(f"{{DATA_DIR}}/test.csv")
  2. Build features WITH the pre-defined helper:
       X_train_text = make_features(train)
       X_test_text  = make_features(test)
  3. 80/20 stratified split with random_state=42.
  4. For Keras: max 5 epochs, EarlyStopping(patience=2), verbose=0,
     class_weight={{0: 1.0, 1: 1.5}}.
  5. Total runtime under 3 minutes.
  6. Save submission BEFORE the RESULT line:
       _sub = pd.read_csv(f"{{DATA_DIR}}/sample_submission.csv")
       _sub['target'] = test_predictions.astype(int)
       _sub.to_csv(f"{{DATA_DIR}}/submission.csv", index=False)
  7. The VERY LAST line printed by your script must be EXACTLY:
       print(f"RESULT: f1={{val_f1:.4f}}")
"""

FIRST_EXPERIMENT_PROMPT = SYSTEM_PROMPT + """
## YOUR TASK
This is experiment #1. Build a TF-IDF + Logistic Regression baseline.

  - Use the pre-defined make_features (do NOT redefine it)
  - TF-IDF: max_features=10000, ngram_range=(1,2), sublinear_tf=True
  - LogisticRegression: C=1.0, max_iter=1000, class_weight='balanced', solver='saga', n_jobs=-1
  - Evaluate F1 on the validation set
  - Save submission.csv
  - Print the RESULT line LAST

Example of the required first two lines (architecture, rationale, blank, then code):

TF-IDF + LogisticRegression
Standard linear baseline on bag-of-ngrams features for binary text classification.
"""

PROPOSE_PROMPT_TEMPLATE = SYSTEM_PROMPT + """
## EXPERIMENT HISTORY (last {n} runs):
{history}

## CURRENT BEST F1: {best_f1:.4f}
## ALL ARCHITECTURES TRIED: {tried_list}

## YOUR TASK
Propose a NEW experiment that improves on the current best F1.
Do NOT repeat any architecture from the tried list above.

Pick the next UNTRIED option from this progression:
  1. TF-IDF + LogisticRegression              (class_weight='balanced', n_jobs=-1)
  2. TF-IDF + SGDClassifier                   (loss='modified_huber', NO n_jobs)
  3. TF-IDF + RidgeClassifier                 (alpha=1.0, class_weight='balanced', NO n_jobs)
  4. TF-IDF + LinearSVC                       (C=0.5, class_weight='balanced', NO n_jobs)
  5. TF-IDF + ComplementNB                    (alpha=0.5 — strong on imbalanced text)
  6. TF-IDF + RandomForestClassifier          (n_estimators=200, class_weight='balanced', n_jobs=-1)
  7. TF-IDF (char n-grams) + LogisticRegression  (analyzer='char_wb', ngram_range=(3,5))
  8. Keras Embedding + 1D CNN                 (filters=128, kernel_size=3, GlobalMaxPool)
  9. Keras Embedding + BiLSTM                 (units=64, recurrent_dropout=0.2)
 10. Keras Embedding + CNN + BiGRU            (conv then recurrent)

DO NOT propose VotingClassifier or GradientBoostingClassifier — these have
sklearn-API incompatibilities that the agent has repeatedly failed to handle.

REMINDER: import every sklearn class you use. Common imports needed:
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
  from sklearn.svm import LinearSVC
  from sklearn.naive_bayes import ComplementNB
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import f1_score

Keras-specific:
  - Tokenizer fit on training texts only, transform train+val+test
  - Pad sequences to maxlen=100, padding='post', truncating='post'
  - Embedding dim=64, trainable=True
  - Final layer: Dense(1, activation='sigmoid'), binary_crossentropy
  - class_weight={{0: 1.0, 1: 1.5}} in model.fit()

OUTPUT FORMAT (follow EXACTLY — first three lines of your response):
Line 1: architecture name with no markdown, no quotes, no labels
Line 2: one short sentence on why it should beat F1={best_f1:.4f}
Line 3: blank
Line 4+: the ```python ... ``` code block
"""

FIX_PROMPT_TEMPLATE = SYSTEM_PROMPT + """
## PROBLEM
The following Python script for a "{architecture}" model crashed with this error:

```
{error}
```

## ORIGINAL CODE
```python
{code}
```

## YOUR TASK
Return the COMPLETE corrected script inside a single ```python ... ``` block.
Do not explain, do not add prose — just the fixed code in one fenced block.
Remember: clean_text, make_features, and DATA_DIR are already defined.
"""

ANALYZE_PROMPT_TEMPLATE = """You are reviewing ML experiment results for a disaster tweet classifier.

## RESULTS SO FAR:
{history}

Briefly answer (3-5 sentences total):
1. What pattern do you see in the results?
2. Which approach worked best and why?
3. What should be tried next to push F1 above {best_f1:.4f}?
"""