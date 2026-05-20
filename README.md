# Disaster Tweets - Autonomous ML Research Agent

University project for Advanced Predictive Analytics 2025/2026, Track A - NLP with Disaster Tweets. The goal is to build an autonomous machine learning research agent for the Kaggle "Real or Not? NLP with Disaster Tweets" binary classification task.

The agent runs locally through an Ollama OpenAI-compatible API. It does not use cloud LLM calls or external API keys.

## How It Works

For each iteration, the agent:

1. Selects an architecture family from a fixed curriculum: TF-IDF baselines, Keras CNN/RNN/Transformer models, then exploitation/refinement steps.
2. Sends the task, assigned family, current best result, stratified experiment memory, tried architectures, failed architectures, low-scoring architectures, hyperparameter signatures, and variation axes to the local LLM.
3. Asks the LLM to generate a complete executable Python training script.
4. Runs the generated script as a temporary file in a subprocess sandbox with a timeout.
5. If the script crashes, asks the LLM to repair it for up to three attempts. On the final repair attempt, the agent requests a fresh rewrite.
6. Parses the validation F1 score and any Keras `EPOCH_METRIC` learning-curve lines from stdout.
7. Asks the LLM for a short analysis of the iteration, then stores that analysis in memory.
8. Logs the prompt, generated code, architecture, assigned family, status, F1, stdout/stderr snippets, learning curve, variation axis, and iteration analysis to `experiments.json`.
9. After the run, asks the LLM for a final written analysis and generates `submission.csv` from the best experiment.

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama and pull a model

Install Ollama from https://ollama.com. It provides a local OpenAI-compatible API at:

```text
http://localhost:11434/v1
```

Pull at least one code-capable local model, for example:

```bash
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull gemma4:latest
```

Make sure the Ollama server is running:

```bash
ollama serve
```

### 3. Place the dataset

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

## Useful Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `AGENT_MODEL` | Ollama model tag used by the agent | `gemma4:latest` |
| `AGENT_TEMP` | LLM sampling temperature | `0.3` |
| `AGENT_MAX_TOKENS` | Maximum tokens requested from the LLM | `8192` |
| `OLLAMA_URL` | Ollama OpenAI-compatible base URL | `http://localhost:11434/v1` |

## Command-Line Flags

| Flag | Purpose | Default |
|---|---|---|
| `--max-iter N` | Number of experiment iterations | `10` |
| `--target-f1 F` | Early stopping target F1 | `0.82` |

## Outputs

After a run, the main files are:

| File | Contents |
|---|---|
| `experiments.json` | Full experiment log: prompts, code, architecture, family, status, F1, stdout/stderr snippets, learning curves, variation axes, and iteration analyses |
| `analysis.txt` | Final LLM-written analysis of the experiment run |
| `submission.csv` | Kaggle submission generated from the best experiment |

## Repo Layout

```text
disaster_agent/
|-- agent.py        # main loop, curriculum, repair loop, logging, final analysis
|-- executor.py     # subprocess sandbox and injected preprocessing/Keras helpers
|-- llm.py          # Ollama OpenAI-compatible client
|-- memory.py       # experiments.json read/write and memory formatting
|-- parser.py       # code, architecture, F1, variation-axis, and learning-curve extraction
|-- prompts.py      # system, propose, fix, iteration-analysis, and final-analysis prompts
`-- submission.py   # Kaggle submission generation from the best model

run_agent.py        # project entry point
requirements.txt    # Python dependencies
```

The `disaster_agent_v2/` folder and `run_agent_v2.py` are retained as earlier development versions. The current final submission path is `run_agent.py`, which uses the modules in `disaster_agent/`.
