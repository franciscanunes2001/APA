# Disaster Tweets — Autonomous ML Research Agent

University project (Track A — NLP). An autonomous agent that proposes,
generates, executes and iterates on ML experiments for the
[Kaggle "Real or Not? NLP with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-started)
binary-classification task.

The agent runs entirely **locally** against an [Ollama](https://ollama.com)
server — no cloud LLM calls, no API keys.

---

## How it works

For each iteration the agent:

1. Picks an architecture family from a curriculum (TF-IDF baselines →
   Keras CNN/RNN/Transformer → exploitation/refinement).
2. Asks the LLM to write a complete training script for it.
3. Runs the script in a sandboxed subprocess (with a timeout) and
   captures stdout / stderr.
4. If the script crashes, asks the LLM to fix it (up to 3 retries).
5. Parses the validation F1 from a `RESULT: f1=0.XXXX` line and logs
   everything to `experiments.json` — including the full prompt and,
   for Keras runs, the per-epoch learning curve.
6. After all iterations, asks the LLM for a written analysis
   (`analysis.txt`) and regenerates `submission.csv` from the best model.
(Note) The correct folder is disaster_agent (not the disaster_agent_v2, as this was a previous model we used and discarded but kept it in the code so you could see our methedology and process)

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model

Install Ollama from https://ollama.com — it provides a local
OpenAI-compatible API on `http://localhost:11434`.

Then pull at least one code-capable model (any of these will work):

```bash
ollama pull qwen2.5-coder:7b      # recommended — good quality, 7B is fine on most laptops
ollama pull qwen3-coder            # if you have more RAM/VRAM
ollama pull gemma2:9b              # alternative
```

Make sure the Ollama server is running:

```bash
ollama serve
```

### 3. Place the dataset

Download the three CSVs from
[the Kaggle competition](https://www.kaggle.com/c/nlp-getting-started/data)
and put them at the project root:

```
APA/
├── train.csv
├── test.csv
├── sample_submission.csv
└── ...
```


---

## Run the agent

A single command runs the whole pipeline:

```bash
AGENT_MODEL=qwen2.5-coder:7b python run_agent.py --max-iter 12
```

Flags:
- `--max-iter N` — number of experiment iterations (default 10)
- `--target-f1 F` — informational target F1 (default 0.82)

Useful environment variables:
- `AGENT_MODEL` — the Ollama model tag to use
- `AGENT_TEMP`  — sampling temperature (default 0.3)
- `OLLAMA_URL`  — base URL of the Ollama server (default `http://localhost:11434/v1`)

---

## Outputs

After a run you get:

| File               | Contents                                                        |
| ------------------ | --------------------------------------------------------------- |
| `experiments.json` | Full log of every experiment (prompt, code, F1, learning curve) |
| `analysis.txt`     | LLM-written analysis of the run                                 |
| `submission.csv`   | Kaggle submission generated from the best model                 |

---

## Repo layout

```
disaster_agent/
├── agent.py        # main loop, curriculum, propose/fix/log
├── executor.py     # sandboxed subprocess runner + Keras epoch-logger preamble
├── llm.py          # Ollama OpenAI-compatible client
├── memory.py       # experiments.json read/write + history helpers
├── parser.py       # extract code / architecture / F1 / learning curve
├── prompts.py      # prompt templates incl. phase-aware iteration analysis
└── submission.py   # Kaggle submission generation
```
