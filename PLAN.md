# Plan — Exercice 3 : industrialiser et optimiser le RAG

Steps in order, based on `exercice_03_recap_enonce.md`. How to start the model servers: see [Services](#services).

## Status (2026-10-28)

- **Phase 0:** done.
- **Phase 1:** steps 7 (`db.py`), 8 (`etl/convert.py`) and 9 (`etl/chunk.py`) done and tested; `config.py` fixed. Next: step 10 (`etl/index.py`).
- **Phases 2–4:** not started.

## Phase 0 — Fix the setup

- [x] 1. **Dependencies:** `deepeval`, `pydantic-settings` and `streamlit` added.
- [x] 2. **Postgres port:** 5433 everywhere (`docker-compose.yaml`, `.env.example`, `.env`).
- [x] 3. **Broken imports:** removed from `scripts/ingest.py` and `evaluation/runner.py` (both empty now).
- 4. **Experiment files:**
   - [x] `chunk_256_overlap_32.toml` renamed to `chunks_256_overlap_32.toml`.
   - [x] Create `experiments/baseline.toml`: copy of the 256 file with `name = "baseline"`, `max_tokens = 512`, `overlap = 0`.
- [x] 5. **Prompts in French:** `minimal.txt` and `strict.txt`.
- 6. **Services:**
   - [x] `docker compose up -d`: Postgres and Qdrant are up (Qdrant answers 401 without the API key, as expected).
   - [x] Embedder, reranker and LLM servers: see [Services](#services). All three answer; GPU at 10.3 GB / 24 GB.
   - [x] Database `hospital_db` created: the volume had been initialised with an older `DB_NAME` (`hospital`, empty, can be dropped). `POSTGRES_DB` is only read the first time a volume is created.

## Phase 1 — Move the notebook code into modules

Bottom-up. Test each module before starting the next.

- [x] 7. **`db.py`:** connection, `CREATE TABLE IF NOT EXISTS` for `experiments`, `answers` and `scores` (§7), and the insert functions.
   - `connect()`, `create_tables()`, `start_experiment()` → id, `finish_experiment()`, `insert_answers()`, `insert_scores()`.
   - Differences from the §7 schema: `ON DELETE CASCADE` on the foreign keys (deleting a test experiment deletes its rows), `answer` can be NULL (retrieval-only runs), `chunks` also stores the chunk text (step 18b).
   - `insert_scores()` is an upsert: re-running the judge replaces its scores.
   - The queries live in `src/hospital_rag/sql/*.sql` (one per file), read by `load_sql()` with `importlib.resources`, so they work from scripts, notebooks, the UI or the API. `compare_experiments()` runs the §7 comparison query.
   - Tested: tables created twice, insert, upsert, JSONB query, comparison query, cascade delete; SQL files load from outside the repo.
- [x] 8. **`etl/convert.py`:** Docling conversion of DOCX and PDF files, cached as JSON so each document is converted once.
   - `get_converter(ocr=False)`, `convert_documents(force=False, ocr=False)` → list of failures, `load_document(name)` → `DoclingDocument`.
   - The old fix list is covered: syntax error, `PdfFormatOption` import, `ocr_settings.ocr_num_threads`, `artifacts_path` always set, `InputFormat.DOC` dropped, loop over the corpus with a cache.
   - OCR off: the 24 PDFs have a text layer (about 700 characters, no images). Layout and table models still run, from `MODELS_PATH/docling`.
   - Each JSON is written to a `.tmp` file then renamed, so a crash can't leave a half-written file that the cache would skip.
   - A failing document is logged and listed, the others carry on.
   - `data/conversions/` added to `.gitignore`: generated files, rebuilt by `scripts/ingest.py`.
   - Tested: 280/280 converted in 35 s, 0 failures; second run converts nothing; lab report tables come out as tables; PDF text present; conversion works with `HF_HUB_OFFLINE=1`.
   - Seen for step 9: in the PDFs, the header (patient, IPP, date, author) is merged into one paragraph. The patient line must come from `hopital_documents.json`, not from parsing the header.

   And in `config.py` (done with step 7, since `db.py` needs it):
   - [x] `extra="ignore"`: `.env` also holds `DB_USER`, `DB_PASSWORD`, `DB_NAME` for docker-compose, and pydantic-settings rejected them;
   - [x] `Path('../../data/docs')` depended on the current directory: now `data/docs`, `data/conversions`, plus `documents_path` and `eval_path` (scripts run from the repo root);
   - [x] machine paths moved to `.env` and `.env.example`: `MODELS_PATH` (holds `embedder/`, `reranker/`, `docling/`); `settings.docling_models_path` is derived from it;
   - [x] `ip_address` and the unused `import os` removed;
   - [x] `JUDGE_URL` added (see step 18b).
- [x] 9. **`etl/chunk.py`:** HybridChunker with `max_tokens - overlap`, then `add_overlap()` on `chunk.text`, then the patient line and headings, in that order (§2).
   - `chunk_document(meta, config)` and `chunk_corpus(config)` → dicts with the document's metadata (from `hopital_documents.json`), `chunk_index` and `text`.
   - Tokens are counted with the embedder's own tokenizer, loaded from `MODELS_PATH/embedder/{embedder.model}`. **Rule: an embedder's served name = its folder name** (see [Services](#swap-models-for-5)).
   - Patient line: `Patient : M. Bruno MARTIN (IPP 4100002) — Ordonnance du 13/04/2025`, built from the metadata; none for notices (their headings name the drug).
   - The chunker gets `max_tokens - overlap - tokens of the patient line`, so the final text stays under `max_tokens`.
   - Tested (evidence coverage = grade-2 evidence whose snippets are all in one chunk, a ceiling on recall):

     | Config | Chunks | Max tokens | Coverage |
     |---|---|---|---|
     | 512 / 0 (baseline) | 1173 | 378 | 174/174 |
     | 256 / 32 | 1207 | 255 | 174/174 |
     | 128 / 0 | 1456 | 127 | 170/174 (4 `split`) |
     | 128 / 16 | 1608 | 128 | 170/174 (4 `split`) |
     | 64 / 0 | 3357 | 64 | 156/174 |

   - **Seen for step 21:** HybridChunker already cuts at every heading, and the sections are short (85 tokens on average at 512). Sizes of 256 and above barely change the chunks; the size only starts to matter at 128 and below. Overlap 16 doesn't recover the 4 `split` misses at 128: try a larger overlap.
- [ ] 10. **`etl/index.py`:**
    - Dense embeddings plus BM25 sparse vectors.
    - One collection per `{embedder}__{max_tokens}_{overlap}`, skipped if it already exists.
    - The payload carries the metadata from `hopital_documents.json`, with payload indexes on `patient_ipp`, `doc_type` and `date`.
- [ ] 11. **`scripts/ingest.py`:** read the TOML and call the indexing function. Run it for the baseline.
- [ ] 12. **`rag/search.py`:** dense, sparse, and hybrid search with RRF or DBSF and an optional filter (from notebook 05).
- [ ] 13. **`rag/rerank.py`:** call `RERANK_URL` and keep the top_k results.
- [ ] 14. **`rag/generate.py`:** read the prompt file, build the messages with numbered extracts (`[1] ...`, `[2] ...`, then `Question : ...`), and pass temperature, top_p, max_tokens and `extra_body` top_k from the config.
- [ ] 15. **`rag/pipeline.py`:** `answer(question, config) -> (answer, chunks)`. Scripts, evaluation and the future API all call this.
- [ ] 16. **`scripts/ask.py`:** a few lines. Use it as a smoke test.
- [ ] 17. **`evaluation/metrics.py`:** `judge_chunk()`, hit@k, MRR, recall@k and nDCG@k (from notebook 06).
- [ ] 18. **`evaluation/judge.py`:** the DeepEval metrics (faithfulness, answer relevancy, contextual precision and recall, GEval correctness) plus `is_refusal()`. The judge model is served on `JUDGE_URL`.
- [ ] 18b. **Judge in a separate pass:** the judge (about 20 GB) doesn't fit on the GPU next to the generator. So:
    - `run_experiment.py` does retrieval, generation and deterministic metrics, and stores answers and chunks **with their text** (faithfulness needs it);
    - `scripts/judge_experiment.py <experiment_id>` reads the answers back from PostgreSQL and writes the judge scores, once the generator is stopped and the judge is started;
    - bonus: old experiments can be re-judged later (second judge, §8) without generating again.
- [ ] 19. **`evaluation/runner.py`:** `run_experiment(config)` does the following:
    - inserts an `experiments` row with the full config **and the prompt text**;
    - loops over the 106 questions and stores answers and latency (generation can be skipped for retrieval-only experiments);
    - computes the deterministic scores on the 100 questions with ground truth;
    - sets the status to `done` or `failed` and records the duration.

## Phase 2 — The baseline

- [ ] 20. Run `uv run python scripts/run_experiment.py experiments/baseline.toml`, then the judge pass. The numbers should be close to the chapter 04 ones. If they aren't, there's a bug in the refactor: find it before going further.

## Phase 3 — Experiments, one change at a time, one TOML each

- [ ] 21. **§2 Chunking:** 128, 256 and 512 tokens × overlap 0 and about 15%. Needs reindexing. Watch `split`, `buried` and `table`; note the number of chunks.
- [ ] 22. **§3 Retrieval parameters:** top_k 3, 5, 10; candidates 10, 30, 50; RRF `k` vs DBSF. No reindexing.
- [ ] 23. **§3 Generation parameters:** temperature 0, 0.3, 0.7, 1, plus top_p and top_k. Judge only, **3 runs each**; look at the spread, not just the mean, and watch the refusals.
- [ ] 24. **§4 Prompts:** `minimal`, `strict`, `citations`, `patient` (each variant = `strict.txt` + one rule), at temperature 0.
    - Count refusals both ways: questions with an answer that were refused, and `out_of_corpus` questions that were answered.
    - Check citations without an LLM: `cited_passages()` + `judge_chunk()`.
- [ ] 25. **§5 Models:**
    - Embedders: Harrier-OSS-v1-0.6b and Qwen3-Embedding-0.6B, each with its `query_instruction` and its own collection. Read the vector size from the model config.
    - Reranker: Qwen3-Reranker-0.6B with the special vLLM command (see [Services](#swap-models-for-5)).
    - Record indexing time, latency per question and GPU memory (`nvidia-smi`).
- [ ] 26. **§6 Filters (`rag/filters.py`):**
    - The LLM extracts JSON; the code maps the name to an IPP using `hopital_documents.json`. Never trust an ID written by the LLM.
    - If the filtered search returns nothing, fall back to an unfiltered search.
    - No patient filter for questions without a patient (`drug_info`, `cross_notice`, `cross_patient`).
    - Measure extraction accuracy against `patient_ipp` (correct, wrong, missed), the effect per question type, then compare with a plain name match that uses no LLM.
- [ ] 27. **§8:** pick at least two. Cheapest: removing the patient line, reranker threshold. Most instructive: contextual retrieval, parent-child. A second judge from another family (Gemma4-26B-A4B) measures the self-preference bias of the Qwen judge.

## Phase 4 — Deliverables

- [ ] 28. **(Bonus) `ui/app.py`:** a read-only Streamlit app that imports from `src/`. It shows a leaderboard per metric, a two-experiment comparison (per question type, questions improved or degraded) and question details.
- [ ] 29. **`README.md`:** install, `docker compose`, model servers (copy the [Services](#services) section), ingest, run, ask.
- [ ] 30. **`RAPPORT.md`:**
    - a comparison table built from the §7 SQL query;
    - the best config, chosen on quality, cost and latency;
    - what changed nothing or made results worse;
    - three failures of the best config analysed: where it fails (retrieval, rerank or generation) and the fix to try.

**Tip:** commit after each module in phase 1 and after each experiment's TOML in phase 3, so the git history mirrors the experiment log.

---

## Services

### Hardware constraint

One GPU: RTX PRO 4000 Blackwell, **24 GB**. The embedder, the reranker and the generator must fit together; the judge runs on its own afterwards (step 18b). Before starting, stop anything else that uses the GPU (`nvidia-smi`), for example a leftover `llama-server`.

### Model choice

| Role | Model | Server | Port | Approx. VRAM |
|---|---|---|---|---|
| Embedder (baseline) | `~/models/rag/models/embedder/bge-m3` | vLLM | 8001 | ~2.4 GB |
| Reranker (baseline) | `~/models/rag/models/reranker/bge-reranker-v2-m3` | vLLM | 8002 | ~2.4 GB |
| Generator | `~/models/qwen3.5-9b/llama/Qwen3.5-9B-UD-Q4_K_XL.gguf` | llama-server | 8000 | ~7–8 GB |
| Judge | `~/models/qwen3.8-27b/llama/Qwen3.8-27B-UD-Q4_K_XL.gguf` | llama-server, alone | 8003 | ~20 GB |
| Second judge (§8) | `~/models/gemma4-26b-a4b/llama/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` | llama-server, alone | 8003 | ~19 GB |

The VRAM figures are estimates from the file sizes: check them with `nvidia-smi` after each start.

Why these:

- **Embedder and reranker:** `~/models/rag/models/` holds exactly the models the exercise names. bge-m3 and bge-reranker-v2-m3 are the reference: multilingual, good in French, 8k tokens, and bge-m3 needs no query instruction. The Qwen3 and Harrier models are for §5 only.
- **Generator, Qwen3.5-9B Q4:** fits next to the embedder and reranker with room left for context, which a 27B doesn't. Fast enough for 106 questions × dozens of experiments, and a 9B model follows the « Je ne sais pas » rule reliably. The GGUF with llama-server also matches the exercise's `extra_body={"top_k": 40}`, a llama-server parameter.
- **Judge, Qwen3.8-27B:** the strongest model available, and it only scores the 26-question sample, so speed matters less. Same family as the generator, so it may favour its answers: the Gemma second judge (§8) measures that.
- **Not used:** diffusiongemma, muse-glimmer-30b, nemotron3-nano-4b; glm-ocr and paddleocr-vl (Docling's built-in OCR is enough for the 24 imaging PDFs; point it at `~/models/rag/models/docling` so it runs offline); gemma4-12b (vLLM only; a candidate for the "other LLM" test in §8).

### Start order

One terminal per server. **Start the two vLLM servers first, then llama-server**: vLLM reserves a fixed share of the GPU when it starts, while llama-server only takes what it needs. The ports match `.env` (`LLM_URL` 8000, `EMBEDDINGS_URL` 8001, `RERANK_URL` 8002).

### Terminal 1 — embedder (port 8001)

```bash
~/vllm_server/.venv/bin/vllm serve ~/models/rag/models/embedder/bge-m3 \
  --runner pooling --served-model-name bge-m3 --port 8001 \
  --dtype float16 --max-model-len 2048 \
  --gpu-memory-utilization 0.10 --enforce-eager
```

### Terminal 2 — reranker (port 8002)

```bash
~/vllm_server/.venv/bin/vllm serve ~/models/rag/models/reranker/bge-reranker-v2-m3 \
  --runner pooling --served-model-name bge-reranker --port 8002 \
  --dtype float16 --max-model-len 2048 \
  --gpu-memory-utilization 0.10 --enforce-eager
```

### Terminal 3 — generator (port 8000)

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen3.5-9b/llama/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -a qwen3.5-9b --port 8000 \
  -ngl 99 -c 16384 -np 1 -fa on --reasoning off
```

### Flags explained

vLLM (version 0.31.0, in `~/vllm_server/.venv`):

| Flag | Why |
|---|---|
| `--runner pooling` | Serves the model as an embedder / scorer (`/v1/embeddings`, `/rerank`) instead of a text generator. |
| `--served-model-name` | The name to pass as `model=` in the client. Keep it equal to `embedder.model` / `reranker.model` in the TOML files. For embedders, also equal to the model's folder name: `chunk.py` loads the tokenizer from `MODELS_PATH/embedder/{embedder.model}`. |
| `--dtype float16` | bge-m3 and bge-reranker are stored in fp32; fp16 halves the memory with no measurable quality loss. The Qwen3 models are already bf16, so no flag needed. |
| `--max-model-len 2048` | A 512-token chunk plus the question fits with margin. A smaller context means less memory reserved. |
| `--gpu-memory-utilization 0.10` | Each vLLM instance reserves this share of the **whole** GPU (~2.4 GB), enough for a 0.6B model. If startup fails with a memory error, raise it to 0.12. |
| `--enforce-eager` | Skips CUDA graph capture: less memory, faster startup, almost no speed cost for small models. |

llama-server (`~/llama.cpp/build/bin/llama-server`):

| Flag | Why |
|---|---|
| `-a qwen3.5-9b` | Model name (alias) to pass as `model=` in the OpenAI client. |
| `-ngl 99` | Put all layers on the GPU. |
| `-c 16384` | Context size: room for 10 chunks of 512 tokens plus the prompt and the answer. |
| `-np 1` | One slot. Without it, llama-server may split the context across several slots, and each one gets less than 16k. |
| `-fa on` | Flash attention: smaller KV cache, faster. |
| `--reasoning off` | No thinking text in the answers, so `is_refusal()` and the 1–3 sentence rule work. |

### Check that they're up

```bash
curl -s localhost:8001/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":["test"]}' | head -c 200; echo
curl -s localhost:8002/rerank -H 'Content-Type: application/json' \
  -d '{"model":"bge-reranker","query":"dose de ramipril","documents":["Ramipril 5 mg","Metformine 1000 mg"]}'; echo
curl -s localhost:8000/v1/models; echo
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

The reranker should score "Ramipril 5 mg" higher. Total GPU use should be around 13 GB.

### Swap models for §5

Stop the baseline server on that port first.

Embedders (terminal 1). They need the `Instruct: {task}\nQuery: {question}` format on queries (see their model cards), set with `query_instruction` in the TOML:

```bash
~/vllm_server/.venv/bin/vllm serve ~/models/rag/models/embedder/qwen3-embedding-0.6b \
  --runner pooling --served-model-name qwen3-embedding-0.6b --port 8001 \
  --max-model-len 2048 --gpu-memory-utilization 0.10 --enforce-eager

~/vllm_server/.venv/bin/vllm serve ~/models/rag/models/embedder/harrier-oss-v1-0.6b \
  --runner pooling --served-model-name harrier-oss-v1-0.6b --port 8001 \
  --max-model-len 2048 --gpu-memory-utilization 0.10 --enforce-eager
```

Reranker (terminal 2), the command from the exercise with local paths:

```bash
~/vllm_server/.venv/bin/vllm serve ~/models/rag/models/reranker/qwen3-reranker-0.6b \
  --runner pooling --served-model-name qwen3-reranker --port 8002 \
  --max-model-len 2048 --gpu-memory-utilization 0.10 --enforce-eager \
  --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
  --chat-template ~/models/rag/models/reranker/qwen3_reranker.jinja
```

| Flag | Why |
|---|---|
| `--hf-overrides` | Qwen3-Reranker is stored as a text generator (`Qwen3ForCausalLM`). This turns it into a classifier whose score is the probability of "yes" against "no". |
| `--chat-template` | The prompt format the reranker was trained on (`<Instruct>`, `<Query>`, `<Document>`, then yes/no). Without it, the scores are meaningless. |

### Judge (port 8003)

Stop the generator (terminal 3) first. If it still runs out of memory, stop the two vLLM servers too: the judge pass doesn't need them.

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen3.8-27b/llama/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -a qwen3.8-27b --port 8003 \
  -ngl 99 -c 16384 -np 1 -fa on --reasoning off
```

Second judge for §8, same flags:

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/gemma4-26b-a4b/llama/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  -a gemma4-26b-a4b --port 8003 \
  -ngl 99 -c 16384 -np 1 -fa on --reasoning off
```

Add `JUDGE_URL=http://localhost:8003/v1` to `.env` and `.env.example`.
