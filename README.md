# hospital-rag

A RAG over the records of a fictional hospital (the Centre Hospitalier de Valmont: 280 documents, 60 patients, 33 drug notices), built as a Python application and used to **measure** optimisations against a golden dataset of 106 questions. Every experiment is a TOML file; every run is stored in PostgreSQL with its full configuration and its scores, question by question.

All the data is synthetic: patients, doctors and values are invented, and the doses have not been checked by a doctor.

## Requirements

- **Python 3.12** and [uv](https://docs.astral.sh/uv/).
- **Docker** with Compose (PostgreSQL 17 and Qdrant).
- **One NVIDIA GPU with 24 GB** (developed on an RTX PRO 4000 Blackwell). The embedder, the reranker and the generator fit together (≈ 11 GB); the judge (≈ 20 GB) runs on its own.
- **llama.cpp** (`llama-server`) for the LLMs and **vLLM** for the embedder and the reranker.
- The models, in one folder (`MODELS_PATH`):

  ```text
  $MODELS_PATH/
  ├── embedder/bge-m3                      # also qwen3-embedding-0.6b, harrier-oss-v1-0.6b (§5)
  ├── reranker/bge-reranker-v2-m3          # also qwen3-reranker-0.6b + qwen3_reranker.jinja (§5)
  └── docling/                             # Docling's layout and table models (offline conversion)
  ```

  plus the GGUF files of the generator (`Qwen3.5-9B-UD-Q4_K_XL.gguf`) and of the judge (`Qwen3.8-27B-UD-Q4_K_XL.gguf`).

## Installation

```bash
uv sync                      # Python dependencies, including the dev ones (pytest)
cp .env.example .env         # then fill it in, see below
docker compose up -d         # PostgreSQL on port 5433, Qdrant on 6333
```

In `.env`:

| Key | What |
|---|---|
| `DB_USER`, `DB_PASSWORD`, `DB_NAME` | PostgreSQL credentials, read by `docker-compose.yaml` |
| `POSTGRES_URL` | the same, as a URL: `postgresql://user:password@localhost:5433/db_name` |
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant; the key is also given to the container |
| `EMBEDDINGS_URL`, `RERANK_URL`, `LLM_URL`, `JUDGE_URL` | the model servers (ports 8001, 8002, 8000, 8003) |
| `JUDGE_MODEL` | the judge's served name (`qwen3.8-27b`) |
| `MODELS_PATH` | the models folder above |
| `DEEPEVAL_TELEMETRY_OPT_OUT=YES` | keep it: DeepEval sends usage data to an external service otherwise |

The tables are created on the first run. Everything is run **from the repository root**: paths in `config.py` and in the experiment files are relative to it.

## Model servers

One terminal per server. Start the vLLM servers first (vLLM reserves a fixed share of the GPU at start-up, llama-server only what it needs). They take 30 s to 2 min to load; wait until `curl -s localhost:<port>/v1/models` answers.

```bash
# Embedder (8001)
~/vllm_server/.venv/bin/vllm serve $MODELS_PATH/embedder/bge-m3 \
  --runner pooling --served-model-name bge-m3 --port 8001 \
  --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.10 --enforce-eager

# Reranker (8002)
~/vllm_server/.venv/bin/vllm serve $MODELS_PATH/reranker/bge-reranker-v2-m3 \
  --runner pooling --served-model-name bge-reranker --port 8002 \
  --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.10 --enforce-eager

# Generator (8000)
llama-server -m <path>/Qwen3.5-9B-UD-Q4_K_XL.gguf -a qwen3.5-9b --port 8000 \
  -ngl 99 -c 16384 -np 1 -fa on --reasoning off

# Judge (8003): stop the generator first
llama-server -m <path>/Qwen3.8-27B-UD-Q4_K_XL.gguf -a qwen3.8-27b --port 8003 \
  -ngl 99 -c 16384 -np 1 -fa on --reasoning off
```

What each flag does, the §5 models (Qwen3, Harrier) and why these models were chosen: `PLAN.md`, section *Services*.

**An embedder's served name must be its folder name** (`bge-m3`, `qwen3-embedding-0.6b`…): `chunk.py` loads the tokenizer from `$MODELS_PATH/embedder/<served name>` to count the chunks in the embedder's own tokens.

| What you run | Needs |
|---|---|
| `ingest.py`, retrieval-only experiments | embedder, reranker |
| `ask.py`, experiments with generation, LLM filter, decomposition | embedder, reranker, generator |
| `judge_experiment.py` | judge only (it reads the stored answers) |

## Usage

### 1. Convert and index

```bash
uv run python scripts/ingest.py experiments/baseline.toml [experiments/other.toml ...] [--force]
```

Converts the documents of `data/docs/` with Docling once (cached as JSON in `data/conversions/`, 35 s), then chunks and indexes each configuration into its own Qdrant collection (`hopital__<embedder>__<max_tokens>_<overlap>`, ≈ 7 s). A collection that is already complete is skipped; `--force` rebuilds it.

### 2. Ask a question

```bash
uv run python scripts/ask.py "Quelle dose de ramipril prend actuellement M. Bruno Martin ?"
uv run python scripts/ask.py --id Q013 --full                # a golden-dataset question, with its expected answer
uv run python scripts/ask.py --id Q013 --config experiments/decomposition_gen.toml
uv run python scripts/ask.py --id Q019 --retrieve-only       # chunks only, no LLM
```

Shows the answer, then each chunk with its rerank and search scores: the tool for finding whether a failure comes from retrieval, reranking or generation.

### 3. Run an experiment

```bash
uv run python scripts/run_experiment.py experiments/baseline.toml [experiments/other.toml ...]
```

Runs the 106 questions (≈ 20 s retrieval only, ≈ 2 min with generation) and stores the answers, the chunks and the deterministic scores: hit, MRR, recall and nDCG at 3, 5 and 10 (100 questions with ground truth), refusals, citations, filter quality.

### 4. Judge it

```bash
uv run python scripts/judge_experiment.py baseline [other_name or id ...] [--judge-model gemma4-26b-a4b] [--force]
```

DeepEval with the judge LLM on a fixed sample of 26 questions (2 per type): contextual precision and recall, faithfulness, answer relevancy, correctness. ≈ 26 min per experiment. A name means its most recent finished run. An interrupted pass resumes where it stopped.

### 5. Compare

```bash
uv run python scripts/report.py              # → data/reports/metrics.html, open it in a browser
```

A self-contained page: leaderboard (with the config keys that differ from `baseline`), scores by question type, and a comparison of two experiments question by question. Re-run it after each experiment.

### Recompute a metric on stored runs

```bash
uv run python scripts/rescore.py --all       # or: rescore.py baseline 39
```

Computes the redundancy metrics (`redundancy@k`, `near_duplicates@k`, `documents@k`) of stored runs from their stored chunks: PostgreSQL only, no model. New runs get them automatically.

### 6. Explore (dashboard)

```bash
uv run streamlit run ui/app.py               # → http://localhost:8501
```

A live, read-only view of the database (its connection is opened read-only: PostgreSQL refuses any write). Three pages:
**Leaderboard** (choose the metrics, sort, the config keys that differ from `baseline`), **Compare two experiments**
(by question type, then the questions improved and degraded), **Question detail** (the expected answer and the evidence,
then for each chosen experiment its answer, its scores with the judge's explanations, and its chunks with a ✓ on the
relevant ones). "Reload from the database" after a new run.

### Tests

```bash
uv run pytest                # metrics, refusals and citations, patient filters, decomposition
```

## Experiments

An experiment is a copy of `experiments/baseline.toml` with **one** change. Copy it, change `name` and the key you test, and check with `diff experiments/baseline.toml experiments/<new>.toml`.

```toml
name = "baseline"            # runs with the same name are repeats, averaged together

[chunking]                   # changing it means a new collection (ingest.py)
max_tokens = 512
overlap = 0
patient_line = true

[embedder]
model = "bge-m3"
query_instruction = ""       # Qwen3 / Harrier: a task instruction before the question

[search]
method = "hybrid"            # dense, sparse or hybrid
fusion = "rrf"               # rrf or dbsf
rrf_k = 60
candidates = 30              # given to the reranker
top_k = 5                    # given to the LLM

[reranker]
model = "bge-reranker"       # or "none"
# threshold = 0.05           # optional: refuse without calling the LLM below this score

[generation]
enabled = true               # false: retrieval only
model = "qwen3.5-9b"
system_prompt = "prompts/strict.txt"
temperature = 0.0
top_p = 1.0
top_k = 40                   # sampling, not the number of chunks
max_tokens = 512

# Optional sections, off when absent:
# [filters]        patient = "names" | "llm", scope = "patient_or_notice" | "patient_only"
# [decomposition]  enabled = true, max_subquestions = 3, gate = true (split only record + notice questions)
```

**A name always means one configuration.** The runner refuses a run whose name already ran with another config, prompt *text* included (the prompt text is stored with each experiment). So: **never edit a prompt file once it has run**: copy it to a new file (`combined.txt` → `combined_v2.txt`), and give the experiment a new name.

## Layout

```text
├── data/
│   ├── docs/                    # the 280 documents (DOCX, PDF)
│   ├── hopital_documents.json   # metadata of each document (patient, type, date, service)
│   ├── hopital_eval.json        # the golden dataset: 106 questions, expected answers, evidence
│   ├── conversions/             # Docling JSON (generated, not committed)
│   └── reports/                 # metrics.html (generated, not committed)
├── experiments/                 # one TOML file per experiment
├── prompts/                     # system prompts, one text file each, never edited once used
├── scripts/                     # thin entry points: ingest, ask, run_experiment, judge_experiment, report
├── src/hospital_rag/
│   ├── config.py                # settings from .env
│   ├── db.py, sql/              # PostgreSQL: one query per .sql file
│   ├── etl/                     # convert (Docling) → chunk → index (Qdrant)
│   ├── rag/                     # filters, decompose, search, rerank, generate, pipeline
│   ├── evaluation/              # metrics, judge, runner, report
│   └── templates/report.html
├── tests/
├── ui/app.py                    # Streamlit dashboard (read-only)
├── PLAN.md                      # the step-by-step log of the exercise, with every result
└── RAPPORT.md                   # the final report
```

Scripts, notebooks and the future API only call `src/`: `rag/pipeline.py` (`retrieve()`, `answer()`) is the single entry point of the RAG.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `database "…" does not exist` | `POSTGRES_DB` is only read when the volume is first created: create the database by hand (`docker exec … psql -c "CREATE DATABASE …"`) or remove the volume. |
| vLLM: `Repo id must be in the form 'repo_name'…` | the model path doesn't exist (check `MODELS_PATH` and the folder names). |
| A server answers `Unauthorized` on 8001 | another vLLM (another project) holds the port: stop it. |
| vLLM fails at start-up for lack of memory | the GPU is shared: check `nvidia-smi`; the judge doesn't fit next to the generator. |
| `experiment '…' already ran … with another config` | the guard: give the new configuration another name. |
| Two runs of the same config give slightly different answers at temperature 0 | llama-server's state between requests; see `PLAN.md` step 27. Retrieval scores are exactly reproducible; compare answers on more than one run. |
