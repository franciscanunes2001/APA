<<<<<<< HEAD
# Disaster Tweets - Autonomous ML Research Agent

University project for Advanced Predictive Analytics 2025/2026, Track A - NLP with Disaster Tweets. The goal is to build an autonomous machine learning research agent for the Kaggle "Real or Not? NLP with Disaster Tweets" binary classification task.

The agent runs locally through an Ollama OpenAI-compatible API. It does not use cloud LLM calls or external API keys.

## How It Works

For each iteration, the agent:

1. Selects an architecture family from a fixed curriculum: TF-IDF baselines, Keras CNN/RNN/Transformer models, then exploitation/refinement steps.
2. Sends the task, curriculum family, previous experiment history, failed/low-scoring architecture lists, and current best result to the local LLM.
3. Asks the LLM to generate a complete executable Python training script.
4. Runs the generated script as a temporary file in a subprocess sandbox with a four-minute timeout.
5. If the script crashes, asks the LLM to repair it for up to three attempts. On the final repair attempt, the agent requests a fresh rewrite.
6. Parses the validation F1 score and any Keras `EPOCH_METRIC` learning-curve lines from stdout.
7. Asks the LLM for a short analysis of the iteration, then stores that analysis in memory.
8. Logs the prompt, generated code, architecture, assigned family, status, F1, stdout/stderr snippets, learning curve, and iteration analysis to `experiments.json`.
9. After the run, asks the LLM for a final written analysis and generates `submission.csv` from the best experiment.
=======
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
>>>>>>> Tiago

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model

<<<<<<< HEAD
Install Ollama from https://ollama.com. It provides a local OpenAI-compatible API at:

```text
http://localhost:11434/v1
```

Pull at least one code-capable local model, for example:

```bash
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull gemma4:latest
=======
Install Ollama from https://ollama.com — it provides a local
OpenAI-compatible API on `http://localhost:11434`.

Then pull at least one code-capable model (any of these will work):

```bash
ollama pull qwen2.5-coder:7b      # recommended — good quality, 7B is fine on most laptops
ollama pull qwen3-coder            # if you have more RAM/VRAM
ollama pull gemma2:9b              # alternative
>>>>>>> Tiago
```

Make sure the Ollama server is running:

```bash
ollama serve
```

### 3. Place the dataset

<<<<<<< HEAD
Download the competition files from Kaggle and place them in the project root:

```text
APA/
|-- train.csv
|-- test.csv
|-- sample_submission.csv
`-- ...
```

## Run the Agent

The main entry point is `run_agent.py`:

```bash
python run_agent.py --max-iter 10 --target-f1 0.99
```

On macOS/Linux, set the local LLM model like this:

```bash
AGENT_MODEL=qwen2.5-coder:14b python run_agent.py --max-iter 10 --target-f1 0.99
```

On Windows PowerShell:

```powershell
$env:AGENT_MODEL = "qwen2.5-coder:14b"
python run_agent.py --max-iter 10 --target-f1 0.99
```

Useful environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `AGENT_MODEL` | Ollama model tag used by the agent | `gemma4:latest` |
| `AGENT_TEMP` | LLM sampling temperature | `0.3` |
| `AGENT_MAX_TOKENS` | Maximum tokens requested from the LLM | `8192` |
| `OLLAMA_URL` | Ollama OpenAI-compatible base URL | `http://localhost:11434/v1` |

Command-line flags:

| Flag | Purpose | Default |
| --- | --- | --- |
| `--max-iter N` | Number of experiment iterations | `10` |
| `--target-f1 F` | Early stopping target F1 | `0.82` |

## Outputs

After a run, the main files are:

| File | Contents |
| --- | --- |
| `experiments.json` | Full experiment log: prompts, code, architecture, family, status, F1, stdout/stderr snippets, learning curves, and iteration analyses |
| `analysis.txt` | Final LLM-written analysis of the experiment run |
| `submission.csv` | Kaggle submission generated from the best experiment |

## Repo Layout

```text
disaster_agent/
|-- agent.py        # main loop, curriculum, repair loop, logging, final analysis
|-- executor.py     # subprocess sandbox and injected preprocessing/Keras helpers
|-- llm.py          # Ollama OpenAI-compatible client
|-- memory.py       # experiments.json read/write and memory formatting
|-- parser.py       # code, architecture, F1, and learning-curve extraction
|-- prompts.py      # system, propose, fix, iteration-analysis, and final-analysis prompts
`-- submission.py   # Kaggle submission generation from the best model

run_agent.py        # project entry point
requirements.txt    # Python dependencies
```

The `disaster_agent_v2/` folder and `run_agent_v2.py` are retained as earlier development versions. The current final submission path is `run_agent.py`, which uses the modules in `disaster_agent/`.
=======
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
>>>>>>> Tiago
