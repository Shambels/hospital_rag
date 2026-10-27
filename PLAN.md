# Plan — Exercice 3 : industrialiser et optimiser le RAG

Steps in order, based on `exercice_03_recap_enonce.md`.

## Phase 0 — Fix the setup

1. **Add the missing dependencies:** `uv add pydantic-settings deepeval`. `config.py` imports `pydantic_settings`, and the judge needs DeepEval. Add `streamlit` later for the bonus.
2. **Fix the Postgres port:** `docker-compose.yaml` exposes Postgres on **5433**, but `.env.example` points to **5432**. Make them match.
3. **Fix the broken imports:** `scripts/ingest.py` and `evaluation/runner.py` do `import config` and use `'{llm_url}'` (not an f-string). Use `from hospital_rag.config import settings`, then `settings.llm_url`, or delete those lines: clients belong in `rag/` or `etl/`.
4. **Sort out the experiment files:**
   - [ ] Rename `chunk_256_overlap_32.toml` to `chunks_256_overlap_32.toml` so it matches its `name`.
   - [ ] Create `experiments/baseline.toml`: 512 tokens, patient line, bge-m3, hybrid search, bge-reranker, top_k 5.
5. **Write the prompts in French:**
   - [ ] `minimal.txt`: « Réponds à la question à partir des extraits fournis. »
   - [x] `strict.txt`: extracts only, « Je ne sais pas. » otherwise, one to three sentences.
6. **Start the services:** `docker compose up -d`, plus the embedder, reranker and LLM servers. Check that each URL in `.env` answers.

## Phase 1 — Move the notebook code into modules

Bottom-up. Test each module before starting the next.

7. **`db.py`:** connection, `CREATE TABLE IF NOT EXISTS` for `experiments`, `answers` and `scores` (§7), and the insert functions.
8. **`etl/convert.py`:** Docling conversion of DOCX and PDF files, cached as JSON in `data/conversions/` so each document is converted once. The corpus is in `data/docs/`, not `data/hopital/`: put that path in config, not in the code.
9. **`etl/chunk.py`:** HybridChunker with `max_tokens - overlap`, then `add_overlap()` on `chunk.text`, then the patient line and headings, in that order (§2).
10. **`etl/index.py`:**
    - Dense embeddings plus BM25 sparse vectors.
    - One collection per `{embedder}__{max_tokens}_{overlap}`, skipped if it already exists.
    - The payload carries the metadata from `hopital_documents.json`, with payload indexes on `patient_ipp`, `doc_type` and `date`.
11. **`scripts/ingest.py`:** read the TOML and call the indexing function. Run it for the baseline.
12. **`rag/search.py`:** dense, sparse, and hybrid search with RRF or DBSF and an optional filter (from notebook 05).
13. **`rag/rerank.py`:** call `RERANK_URL` and keep the top_k results.
14. **`rag/generate.py`:** read the prompt file, build the messages with numbered extracts (`[1] ...`, `[2] ...`, then `Question : ...`), and pass temperature, top_p, max_tokens and `extra_body` top_k from the config.
15. **`rag/pipeline.py`:** `answer(question, config) -> (answer, chunks)`. Scripts, evaluation and the future API all call this.
16. **`scripts/ask.py`:** a few lines. Use it as a smoke test.
17. **`evaluation/metrics.py`:** `judge_chunk()`, hit@k, MRR, recall@k and nDCG@k (from notebook 06).
18. **`evaluation/judge.py`:** the DeepEval metrics (faithfulness, answer relevancy, contextual precision and recall, GEval correctness) plus `is_refusal()`.
19. **`evaluation/runner.py`:** `run_experiment(config)` does the following:
    - inserts an `experiments` row with the full config **and the prompt text**;
    - loops over the 106 questions and stores answers and latency;
    - computes the deterministic scores on the 100 questions with ground truth;
    - runs the judge on the 26-question sample (2 per type) when the config asks for it;
    - sets the status to `done` or `failed` and records the duration.

## Phase 2 — The baseline

20. Run `uv run python scripts/run_experiment.py experiments/baseline.toml` with the judge on. The numbers should be close to the chapter 04 ones. If they aren't, there's a bug in the refactor: find it before going further.

## Phase 3 — Experiments, one change at a time, one TOML each

21. **§2 Chunking:** 128, 256 and 512 tokens × overlap 0 and about 15%. Needs reindexing. Watch `split`, `buried` and `table`; note the number of chunks.
22. **§3 Retrieval parameters:** top_k 3, 5, 10; candidates 10, 30, 50; RRF `k` vs DBSF. No reindexing.
23. **§3 Generation parameters:** temperature 0, 0.3, 0.7, 1, plus top_p and top_k. Judge only, **3 runs each**; look at the spread, not just the mean, and watch the refusals.
24. **§4 Prompts:** `minimal`, `strict`, `citations`, `patient` (each variant = `strict.txt` + one rule), at temperature 0.
    - Count refusals both ways: questions with an answer that were refused, and `out_of_corpus` questions that were answered.
    - Check citations without an LLM: `cited_passages()` + `judge_chunk()`.
25. **§5 Models:**
    - Embedders: Harrier-OSS-v1-0.6b and Qwen3-Embedding-0.6B, each with its `query_instruction` and its own collection. Read the vector size from the model config.
    - Reranker: Qwen3-Reranker-0.6B with the special vLLM command.
    - Record indexing time, latency per question and GPU memory (`nvidia-smi`).
26. **§6 Filters (`rag/filters.py`):**
    - The LLM extracts JSON; the code maps the name to an IPP using `hopital_documents.json`. Never trust an ID written by the LLM.
    - If the filtered search returns nothing, fall back to an unfiltered search.
    - No patient filter for questions without a patient (`drug_info`, `cross_notice`, `cross_patient`).
    - Measure extraction accuracy against `patient_ipp` (correct, wrong, missed), the effect per question type, then compare with a plain name match that uses no LLM.
27. **§8:** pick at least two. Cheapest: removing the patient line, reranker threshold. Most instructive: contextual retrieval, parent-child.

## Phase 4 — Deliverables

28. **(Bonus) `ui/app.py`:** a read-only Streamlit app that imports from `src/`. It shows a leaderboard per metric, a two-experiment comparison (per question type, questions improved or degraded) and question details.
29. **`README.md`:** install, `docker compose`, model servers, ingest, run, ask.
30. **`RAPPORT.md`:**
    - a comparison table built from the §7 SQL query;
    - the best config, chosen on quality, cost and latency;
    - what changed nothing or made results worse;
    - three failures of the best config analysed: where it fails (retrieval, rerank or generation) and the fix to try.

**Tip:** commit after each module in phase 1 and after each experiment's TOML in phase 3, so the git history mirrors the experiment log.
