# Plan — Exercice 3 : industrialiser et optimiser le RAG

Steps in order, based on `exercice_03_recap_enonce.md`. How to start the model servers: see [Services](#services).

## Status (2026-11-05)

- **Phase 0:** done.
- **Phase 1:** **done.** ETL, RAG, `ask.py`, metrics (unit tests: `uv run pytest`), LLM judge, and the two experiment passes (`run_experiment.py`, `judge_experiment.py`).
- **Phase 2:** done. Baseline = experiment 10 (run + judge), the reference of every experiment.
- **Phase 3:** steps 21–23 done: baseline kept everywhere (chunking, retrieval parameters, temperature 0, `top_k = 5`). Step 23 corrected: its runs used an edited prompt (see step 23). Steps 24 (prompts, judge pass partly skipped), 25 (models), 26 (filters) and 27 (reranker threshold, question decomposition) done. **Phase 3 complete.**
- **Phase 4:** done: README, RAPPORT and the bonus UI. **Every step of the plan is done.**

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

   - **Seen for step 21:** HybridChunker already cuts at every heading, and the sections are short (85 tokens on average at 512). Sizes of 256 and above barely change the chunks; the size only starts to matter at 128 and below. Overlap 16 doesn't recover the 4 `split` misses at 128: try a - [x] 10. **`etl/index.py`:**
    - Dense embeddings plus BM25 sparse vectors.
    - One collection per `{embedder}__{max_tokens}_{overlap}`, skipped if it already exists.
    - The payload carries the metadata from `hopital_documents.json`, with payload indexes on `patient_ipp`, `doc_type` and `date`.
    - `index_corpus(config, force=False)` → `{collection, chunks, created, seconds}`; `embed(texts, model)` and `bm25(text)` are reused by `rag/search.py`, so chunks and questions are always encoded the same way.
    - Name: `hopital__bge-m3__512_0`, plus `__no-patient-line` when `patient_line = false`. Any new `[chunking]` option must be added to `collection_name()`. `query_instruction` isn't in the name: it only changes the questions.
    - BM25 is computed by the Qdrant server (`qdrant/bm25`, French: stemming and stopwords), with the IDF modifier: no extra package.
    - Dense size read from the embedder's answer, cosine distance. Payload indexes: `patient_ipp`, `doc_type`, `care_path`, `document` (keyword) and `date` (datetime, `YYYY-MM-DD` accepted).
    - "Already indexed" = the collection exists **and** has as many points as the chunking produces; otherwise it is rebuilt (crash halfway, or chunking changed under the same name).
    - Tested: 1173 chunks indexed in 7.2 s; second run skips; dense size 1024; Q007 in hybrid search returns the 3 evidence documents in the top 3.
    - **Seen for later:** the top chunk for Q007 says « Ramipril 10 mg » (consultation of 13/04), but the expected answer is 5 mg (the dose was changed). The right documents are found; choosing the most recent one is the `temporal` trap, for the reranker and the prompt.
pe` and `date`.
- [x] 11. **`scripts/ingest.py`:** read the TOML and call the indexing function. Run it for the baseline.
    - `uv run python scripts/ingest.py experiments/baseline.toml [other.toml ...] [--force]`: converts the missing documents, stops if one fails, then indexes each config (skipped if complete).
    - `--force` rebuilds the collections; it never re-runs Docling (use `convert_documents(force=True)` for that).
    - Logs: `httpx`/`httpx2` and `docling` at WARNING, `transformers` at ERROR (harmless "Token indices sequence length" warning), Qdrant's "insecure connection" warning ignored (local, plain http).
    - Done: `hopital__bge-m3__512_0` (1173 chunks) and `hopital__bge-m3__256_32` (1207 chunks, 5.7 s).
- [x] 12. **`rag/search.py`:** dense, sparse, and hybrid search with RRF or DBSF and an optional filter (from notebook 05).
    - `search(question, config, query_filter=None)` → the `[search] candidates` best chunks (payload + `search_score`); `query_text()` adds the §5 instruction.
    - New `[search]` keys in both TOMLs: `fusion = "rrf"` (or `"dbsf"`) and `rrf_k = 60`, so every stored config says which fusion it used.
    - The instruction goes before the question for the **dense** vector only; BM25 gets the bare question.
    - The filter is passed as `query_filter`: Qdrant also applies it to the prefetches.
    - Tested on the 100 questions with ground truth, baseline index, no reranker (quick relevance check: grade-2 evidence document + all its snippets):

      | Method | hit@5 | hit@30 | ms / question |
      |---|---|---|---|
      | dense | 0.87 | 1.00 | 31 |
      | sparse (BM25) | 0.79 | 0.99 | 6 |
      | hybrid RRF k=60 | 0.85 | 1.00 | 60 |
      | hybrid DBSF | 0.89 | 1.00 | 44 |

    - **Seen for later:** hit@30 = 1.00, so a relevant chunk is always among the candidates: everything the reranker can gain is there. `lay_language` is the weakest type (2/8 at hit@5), as expected for vocabulary mismatch. RRF below dense alone, DBSF above: a 2–4 question difference, to confirm with the real metrics in step 22.
    - Patient filter checked: Q007 with `patient_ipp = 4100002` → only that patient's chunks.
- [x] 13. **`rag/rerank.py`:** call `RERANK_URL` and keep the top_k results.
    - `rerank(question, chunks, config)` → the `top_k` best chunks, each with a `rerank_score` (the `search_score` is kept).
    - `[reranker] model = "none"` keeps the search order: measures what the reranker brings, without a second code path.
    - The reranker reads the full chunk text, patient line and headings included.
    - One `requests.Session` reused for every call; `raise_for_status()` so a server error stops the run instead of returning nothing.
    - Tested on the 100 questions (baseline: hybrid RRF, 30 candidates, top 5; quick relevance check), stored as temporary experiment `search_hybrid_rrf_rerank`:

      | | hit@5 | MRR@5 | ms / question |
      |---|---|---|---|
      | search only | 0.85 | 0.702 | 50 |
      | search + bge-reranker | **0.95** | **0.833** | 50 + 101 |

    - Gains: `lay_language` 2→6/8, `abbreviation` 6→8, `cross_patient` 6→8, `negation` 7→8, `table` 7→8. 11 questions gained, 1 lost (Q013).
    - **Seen for later**, the 5 misses:
      - `multi_hop` Q013, Q037, Q078: the reranker scores each chunk against the whole question, so it favours one side (the notice, or the wrong part of the record); the second fact is at rank 9–11. → question decomposition (§8), larger `top_k` (§3).
      - `lay_language` Q088, Q106 (« opéré de la vésicule » / cholécystectomie): the right chunk is at rank 6. → `top_k` 10 (§3), query rewriting (§8).
      - The header chunk of every document holds the « Document fictif généré pour un cours… » disclaimer, and it sometimes ranks in the top 2 (Q013, Q078): noise worth noting in `RAPPORT.md`.
- [x] 14. **`rag/generate.py`:** read the prompt file, build the messages with numbered extracts (`[1] ...`, `[2] ...`, then `Question : ...`), and pass temperature, top_p, max_tokens and `extra_body` top_k from the config.
    - `generate(question, chunks, config)` → `{answer, prompt_tokens, completion_tokens, finish_reason}`; `load_system_prompt(config)` is what the runner will store with the experiment; `build_messages()` can be inspected on its own.
    - New `[generation]` keys in both TOMLs: `model = "qwen3.5-9b"` (§8 "another LLM") and `top_k = 40` (sampling, llama-server's default made explicit).
    - System message = the prompt file; user message = `Extraits :` + numbered chunks, best first, then `Question : ...` last.
    - No `seed`: at temperature > 0 the 3 runs of step 23 must be able to differ. A warning is logged when an answer is cut at `max_tokens`.
    - Tested on 8 questions (dose, table, cross_patient, multi_hop, lay_language, drug_info, 2 × out_of_corpus), `strict` vs `minimal`:
      - `strict`: 5/5 answerable questions right, short (27–54 tokens, 0.4–1.1 s); « Je ne sais pas. » on both out_of_corpus **and** on Q013 (multi_hop), where the kaliémie chunk was not in the top 5: the right refusal given the extracts.
      - `minimal`: same facts, but long Markdown answers (40–434 tokens, up to 5.4 s), and its refusals never say « Je ne sais pas » (« il n'est pas possible de déterminer… »).
    - **Seen for later:**
      - Q007: chunk [1] says ramipril 10 mg, the LLM still answers 5 mg from the later extracts: the `temporal` trap is handled by generation here.
      - `is_refusal()` only detects the `strict` phrase: with `minimal`, refusals would be counted as answers. Step 24 needs a broader detection (or the judge) for prompts that don't impose the phrase.
      - With `minimal`, the LLM cites the « Document fictif… » disclaimer in its reasoning (Q013): more evidence that this header chunk is noise.
      - `strict` rule 3 (« aucune déduction ») may hurt `multi_hop`, which needs to combine two extracts (hyperkaliémie + contre-indication): check in step 24.
- [x] 15. **`rag/pipeline.py`:** `answer(question, config) -> (answer, chunks)`. Scripts, evaluation and the future API all call this.
    - `retrieve(question, config)` → `{chunks, timings}`: search + rerank only, for the retrieval-only experiments.
    - `answer(question, config)` → `{answer, chunks, latency_ms, timings, prompt_tokens, completion_tokens, finish_reason}`; `timings` has `search_ms`, `rerank_ms`, `generate_ms`.
    - A dict, not a tuple: the runner and the API pick the fields they need, and adding one (e.g. the §6 filter used) breaks no caller.
    - Timed with `time.perf_counter()` (monotonic, made for durations).
    - Tested on Q007: right answer (5 mg); 94 + 110 + 1005 ms, generation is ~83 % of the latency. RRF search scores ≈ 0.03 (= 1/61 + 1/62), rerank scores 0.76–0.98.
    - Step 26 (filters) will plug in here, between the question and `search()`.
- [x] 16. **`scripts/ask.py`:** a few lines. Use it as a smoke test.
    - `uv run python scripts/ask.py "question"` or `--id Q013` (question from the eval set, shows the expected answer too).
    - `--config` (default `experiments/baseline.toml`), `--full` (whole chunk text), `--retrieve-only` (no generation).
    - Shows the answer, then each chunk: rank, rerank and search scores, document, chunk index, and its headings + text on one line; then the time of each stage and the tokens.
    - **The tool for the 3 failure analyses of `RAPPORT.md`:** `--id <question> --full` shows whether the failure comes from retrieval (right chunk missing), reranking (right chunk low) or generation (right chunk there, wrong answer).
    - **Seen:** for an out_of_corpus question that names a patient (Q019, no reranker), the top 5 are all that patient's header chunks: the name matches, nothing else does. More evidence that the header chunk (identity + disclaimer) attracts name-only questions.
- [x] 17. **`evaluation/metrics.py`:** `judge_chunk()`, hit@k, MRR, recall@k and nDCG@k (from notebook 06).
    - `judge_chunk(chunk, question)` → the evidence items the chunk proves: **same document** and every snippet in the text (whitespace and Unicode normalised, case kept).
    - `retrieval_scores(chunks, question, top_k)` → `{"hit@5": ..., "mrr@5": ..., "recall@5": ..., "ndcg@5": ...}` for every k of `K_VALUES = (3, 5, 10)` with k ≤ `top_k`; empty for `out_of_corpus`.
    - hit and MRR: grade 2 only. Recall: share of the distinct **facts** found (interchangeable sources count once). nDCG: grades 1 and 2, linear gain, each evidence item credited once (overlapping duplicates can't push it above 1).
    - Unit tests in `tests/test_metrics.py` (12 hand-made cases, `uv run pytest`); `pytest` added as a dev dependency.
    - Data checked: grade 2 always has a fact number, grade 1 never; 74 one-fact questions, 26 with 2–5 facts; all 212 evidence items match the baseline chunks.
    - Baseline retrieval (hybrid RRF, 30 candidates, bge-reranker), 100 questions:

      | k | hit | MRR | recall | nDCG |
      |---|---|---|---|---|
      | 3 | 0.910 | 0.823 | 0.855 | 0.784 |
      | 5 | **0.950** | **0.833** | **0.913** | **0.816** |
      | 10 | 0.990 | 0.838 | 0.953 | 0.842 |

    - **Seen for later:** recall@5 is 0.97 on one-fact questions but **0.74 on multi-fact** ones; by type `multi_hop` 0.58, `lay_language` 0.75, `cross_notice` 0.77 are the weak spots. hit@5 hides this (0.95). Going from 5 to 10 chunks gains +0.04 recall: a number for step 22.
- [x] 18. **`evaluation/judge.py`:** the DeepEval metrics (faithfulness, answer relevancy, contextual precision and recall, GEval correctness) plus `is_refusal()`. The judge model is served on `JUDGE_URL`.
    - **Dependencies:** `deepeval>=4.3` (the API of notebook 06: `SingleTurnParams`, `LocalModel(base_url=...)`) requires `transformers<5.18`, as notebook 06 says; `transformers` went from 5.21 to 5.17. Checked: same chunks (1173 / 1207), tests pass.
    - **Telemetry off:** DeepEval sends usage data to an external service by default. `judge.py` sets `DEEPEVAL_TELEMETRY_OPT_OUT=YES` before importing it; also in `.env` / `.env.example`. `.deepeval/` in `.gitignore`.
    - `JUDGE_MODEL=qwen3.8-27b` in `.env` and `settings.judge_model`.
    - `judge_sample(questions)` → 26 questions: the first 2 ids of each type, `out_of_corpus` excluded (13 types × 2). Deterministic: every experiment is judged on the same questions.
    - `judge_answer(question, answer, chunks, judge_model=None)` → `{metric: (value, reason)}`: `contextual_precision`, `contextual_recall`, `faithfulness` (`penalize_ambiguous_claims`), `answer_relevancy`, `correctness` (GEval vs the expected answer). A failed metric gives `(None, "judge error: ...")`, the others are kept. Another judge's scores are named `faithfulness (gemma4-26b-a4b)` etc., so they sit next to the main judge's.
    - New metric objects for each answer (a DeepEval metric keeps the state of its last run); `async_mode=False`, one call at a time (llama-server runs with `-np 1`).
    - Fixed from notebook 06: a missing comma in the GEval steps glued two steps into one string.
    - `is_refusal()` and `answer_scores(answer)` → `{"refusal": 0/1}` are in `metrics.py` (no LLM, computed for all 106 questions). Tests added (15 in total).
    - Tested with the judge on 4 answers (3 real ones from step 14, 1 wrong on purpose): real Q007 → faithfulness 1, correctness 1; **wrong « 10 mg » → faithfulness 0, correctness 0**; Q088 → correctness 0.8 (type of surgery missing); Q013 « Je ne sais pas. » → faithfulness 1, correctness 0 (justified by the extracts, but wrong).
    - **Cost: ~60 s per answer** (5 metrics, several judge calls each), so ~26 min per experiment for the 26 questions. Possible speed-up: start the judge with `-np 4 -c 32768` (4 parallel slots of 8k) and `async_mode=True`; to measure before relying on it.
- [x] 18b. **Judge in a separate pass:** the judge (about 20 GB) doesn't fit on the GPU next to the generator. So:
    - `run_experiment.py` does retrieval, generation and deterministic metrics, and stores answers and chunks **with their text** (faithfulness needs it);
    - `scripts/judge_experiment.py <experiment_id>` reads the answers back from PostgreSQL and writes the judge scores, once the generator is stopped and the judge is started;
    - bonus: old experiments can be re-judged later (second judge, §8) without generating again.
    - `uv run python scripts/judge_experiment.py baseline` (name = its latest finished run) or `... 12 13 --judge-model gemma4-26b-a4b`; `--force` judges again.
    - `runner.judge_experiment(id, judge_model=None, force=False)`: judges the 26-question sample, one commit per question; **resumes** an interrupted pass (skips the questions whose 5 metrics are already stored); skips questions without an answer; refuses an experiment that isn't `done` or has no generated answers.
    - New `db.py` functions and SQL files: `load_experiment`, `find_experiment`, `load_answers`, `load_score_keys`.
- [x] 19. **`evaluation/runner.py`:** `run_experiment(config)` does the following:
    - inserts an `experiments` row with the full config **and the prompt text**;
    - loops over the 106 questions and stores answers and latency (generation can be skipped for retrieval-only experiments);
    - computes the deterministic scores on the 100 questions with ground truth;
    - sets the status to `done` or `failed` and records the duration.
    - `uv run python scripts/run_experiment.py experiments/baseline.toml [other.toml ...]`.
    - Calls `index_corpus(config)` first: the experiment can't run on a missing or incomplete index (nothing rebuilt if it is complete).
    - New `[generation] enabled = true` in both TOMLs; `false` = retrieval only (`retrieve()` instead of `answer()`, `answer` NULL, no `refusal` score).
    - Scores per question: `hit/mrr/recall/ndcg@k` (k ≤ `top_k`) for the 100 questions with evidence, `refusal` for all 106 when generating.
    - Stored chunks: `document`, `chunk_index`, `search_score`, `rerank_score`, `text` (the metadata is in `hopital_documents.json`).
    - One commit per question; any exception, Ctrl+C included, marks the experiment `failed` before stopping.
    - Shared logging setup in `src/hospital_rag/logs.py`, used by `ingest.py`, `run_experiment.py` and `judge_experiment.py`.
    - Tested: retrieval-only run of the baseline, 106 questions in 17 s, stored means identical to step 17 (hit@5 0.950, MRR@5 0.833, recall@5 0.913, nDCG@5 0.816); judge pass on a 2-answer test experiment (same scores as step 18, 24 questions skipped), second pass resumes and judges nothing; errors on a retrieval-only experiment and an unknown name. Test experiments deleted.
    - Not stored yet: token counts per answer (only `latency_ms`). Worth adding if `RAPPORT.md` needs the cost in tokens.

- [x] **Extra: HTML report of the metrics** (done after step 12)
    - `uv run python scripts/report.py` → `data/reports/metrics.html` (self-contained, open it in a browser; re-run after each experiment). `data/reports/` is in `.gitignore`.
    - `evaluation/report.py` reads every experiment and score (`sql/report_*.sql`); the page (`templates/report.html`) computes the means itself.
    - Sections: **leaderboard** (mean per metric, ± between runs of the same name, difference with `baseline`, the config keys that differ from it, best per column, sortable), **by question type** (heatmap, hardest types first), **compare two experiments** (per type, then the questions improved and degraded), **runs not finished**.
    - Runs with the same `name` are grouped as repeats (the 3 runs per temperature of step 23). The reference is the experiment named `baseline`, or the first one until it exists.
    - **To delete before step 20:** 5 temporary experiments from the step 12 and 13 tests (`search_hybrid_rrf`, `search_dense`, `search_sparse`, `search_hybrid_dbsf`, `search_hybrid_rrf_rerank`; retrieval only, quick relevance check): `DELETE FROM experiments WHERE name LIKE 'search\_%';` (cascade removes their scores).

## Phase 2 — The baseline

- [x] 20. Run `uv run python scripts/run_experiment.py experiments/baseline.toml`, then the judge pass.
    - [x] Temporary `search_*` experiments deleted.
    - [x] **Run pass** (experiment 10, 102 s, no warning, no answer cut at `max_tokens`):
      - retrieval: hit@5 0.950, MRR@5 0.833, recall@5 0.913, nDCG@5 0.816 (same as step 17);
      - refusals: **6/6 out_of_corpus refused**; **7/100 answerable questions wrongly refused**: 6 `multi_hop` (Q012, Q013, Q071, Q072, Q078, Q089, recall@5 0 or 0.5: one fact missing) and 1 `buried` (Q017, recall@5 = 1);
      - latency: median 944 ms, p90 1179 ms, max 1982 ms; answers: median 110 characters.
    - **Seen:** Q017 is a generation failure caused by `strict.txt` rule 2 (« ou seulement une partie de la réponse » → refuse): the model says the appendix « était perforé » and still ends with « Je ne sais pas. ». The 6 `multi_hop` refusals are the same rule applied to a partial retrieval. Candidate for step 24: a prompt that answers the part it can and says what is missing.
    - [x] **Judge pass** (1644 s = 27 min, 0 judge failures), 26-question sample:

      | contextual_precision | contextual_recall | faithfulness | answer_relevancy | correctness |
      |---|---|---|---|---|
      | 0.876 | 0.942 | 0.881 | 0.910 | 0.850 |

      - correctness by type: `multi_hop` **0.00** (its 2 sample questions, Q012 and Q013, are among the wrong refusals), `buried` 0.65 (Q017), `abbreviation` 0.80, 9 types at 0.90–1.00.
      - The 3 answers under 0.5 all have faithfulness ≥ 0.67: the model doesn't invent, it refuses or under-answers. That points at retrieval (missing fact) and at `strict.txt` rule 2, not at hallucination.
      - **Read with care:** 2 questions per type, so one question moves a type's mean by 0.5 and the overall mean by ~0.04. Small differences between experiments on judge metrics are noise.

## Phase 3 — Experiments, one change at a time, one TOML each

- [x] 21. **§2 Chunking:** 128, 256 and 512 tokens × overlap 0 and about 15%. Needs reindexing. Watch `split`, `buried` and `table`; note the number of chunks.
    - 5 TOMLs = `baseline.toml` with only `[chunking]` changed and `[generation] enabled = false` (checked with `diff`): `chunks_256_overlap_0`, `chunks_256_overlap_32`, `chunks_128_overlap_0`, `chunks_128_overlap_16`, `chunks_128_overlap_32`. Experiments 11–15, ~18 s each; indexes 5–8 s each.

      | Experiment | Chunks | hit@5 | MRR@5 | recall@3 | recall@5 | nDCG@5 |
      |---|---|---|---|---|---|---|
      | **baseline (512/0)** | 1173 | 0.950 | **0.833** | **0.856** | 0.913 | **0.816** |
      | 256/0 | 1189 | 0.950 | 0.835 | 0.856 | 0.913 | 0.817 |
      | 256/32 | 1207 | **0.970** | 0.805 | 0.847 | **0.926** | 0.783 |
      | 128/0 | 1456 | 0.900 | 0.783 | 0.837 | 0.871 | 0.805 |
      | 128/16 | 1608 | 0.900 | 0.783 | 0.843 | 0.872 | 0.794 |
      | 128/32 | 1717 | 0.960 | 0.802 | 0.810 | 0.914 | 0.795 |

      recall@5 by type: `split` 1.00 everywhere except **128/0 and 128/16: 0.50** (Q009, Q016, Q023, Q059 lost, the 4 misses predicted by the step 9 coverage test), **recovered at 128/32**; `buried` and `table` 1.00 everywhere; `lay_language` 0.75 → 1.00 with overlap (256/32, 128/16); `cross_notice` 0.77 → 0.68–0.71 with overlap.
    - **Conclusions:**
      - 256/0 = baseline (same results, 16 more chunks): HybridChunker follows the sections, the size limit doesn't bite above 256 (as measured in step 9).
      - 128 tokens breaks the `split` questions; overlap 16 doesn't repair them, **overlap 32 does**: overlap works when it is large enough to carry the cut sentence.
      - Overlap reshuffles more than it improves: 256/32 gains 4 questions and loses 4 (recall@5 +0.013, but MRR −0.028, nDCG −0.033). It helps `lay_language` (the previous chunk's end brings context) and hurts `cross_notice` (list questions need many distinct chunks; overlapping ones repeat each other).
      - Smaller chunks cost more for nothing: 128/32 has 46 % more chunks than the baseline for the same recall and a lower MRR/nDCG.
      - **Kept: the baseline chunking (512/0).** No configuration beats it on more than noise (4 questions up, 4 down out of 100), so none goes to the judge (the exercise judges only "les configurations retenues").
    - Note: `answers.latency_ms` of the baseline includes generation (~960 ms); the retrieval-only runs are ~160 ms. Per-stage timings are not stored, only the total.
- [x] 22. **§3 Retrieval parameters:** top_k 3, 5, 10; candidates 10, 30, 50; RRF `k` vs DBSF. No reindexing.
    - 7 TOMLs, each = `baseline.toml` with one `[search]` key changed + `enabled = false`: `top_k_3`, `top_k_10`, `candidates_10`, `candidates_50`, `rrf_k_10`, `rrf_k_100`, `fusion_dbsf` (experiments 16–22, 12–22 s each, baseline index reused). Made with a helper that edits a key inside one named section (`top_k` exists in `[search]` and `[generation]`), checked with `diff`.

      | Experiment | recall@3 | hit@5 | MRR@5 | recall@5 | nDCG@5 | recall@10 | retrieval ms |
      |---|---|---|---|---|---|---|---|
      | baseline | 0.856 | 0.950 | 0.833 | 0.913 | 0.816 | – | ≈ 160 |
      | top_k_3 | 0.856 | – | – | – | – | – | 171 |
      | top_k_10 | 0.856 | 0.950 | 0.833 | 0.913 | 0.816 | **0.953** | 152 |
      | candidates_10 | 0.872 | 0.950 | 0.824 | 0.916 | 0.806 | – | **106** |
      | candidates_50 | = baseline | | | | | | 201 |
      | rrf_k_10, rrf_k_100, fusion_dbsf | = baseline | | | | | | ≈ 160 |

    - **Conclusions:**
      - **Fusion (RRF k=10/60/100, DBSF) and candidates 50 change nothing**, on all 100 questions. In step 12, without reranker, DBSF beat RRF (hit@5 0.89 vs 0.85); with the reranker, the difference disappears: fusion only decides which candidates reach the reranker, and the relevant chunks are among the 30 either way (hit@30 = 1.00). The reranker re-scores everything, so the fusion's ranking is thrown away.
      - **candidates 10**: +3 / −2 questions (noise), but the reranker is ~35 % faster (106 vs ≈ 160 ms): the rerank cost grows with the number of candidates (10 → 106 ms, 30 → ≈ 160, 50 → 201).
      - **top_k 3** loses facts: recall@3 0.856 vs recall@5 0.913.
      - **top_k 10** gains facts: recall@10 **0.953** (+0.04): `lay_language` 0.75 → 1.00, `cross_notice` 0.77 → 0.89, `multi_hop` 0.58 → 0.67, `cross_patient` → 1.00 (Q037, Q057, Q058, Q078, Q088, Q092, Q106). The @3/@5 scores are identical to the baseline, as expected (the reranker sorts the same candidates).
      - **Kept: baseline retrieval parameters**, except that **`top_k_10` is the one candidate worth testing with generation + judge**: 2× more prompt tokens, does the LLM use the extra facts or get distracted? (`[generation] enabled = true` → run pass with the generator, then judge pass.)
    - [ ] `top_k_10` with generation + judge: to do with the generator (and reconsidered with step 24: the wrong refusals of `strict.txt` rule 2 are on the same `multi_hop` questions).
- [x] 23. **§3 Generation parameters:** temperature 0, 0.3, 0.7, 1, plus top_p and top_k. Judge only, **3 runs each**; look at the spread, not just the mean, and watch the refusals.
    - TOMLs: `temperature_0.3`, `temperature_0.7`, `temperature_1` (one `[generation]` key changed), plus `top_k_10_gen` (step 22's follow-up: `[search] top_k = 10` with generation).
    - ⚠️ **Correction (3 Nov):** `prompts/strict.txt` had an uncommitted edit adding a patient rule (« Si la question mentionne un patient, n'utilise que les passage qui concernent ce patient. »). Every run of this step (23–33) used it; the baseline (10) didn't. Fixed:
      - the edited text is now `prompts/patient.txt` (exactly as run, typo included) and `strict.txt` is back to its committed version, identical to experiment 10's;
      - experiment 23, run as "baseline", is relabelled **`prompt_patient`** (`experiments/prompt_patient.toml` reproduces its config exactly): it is the §4 patient variant;
      - experiments 24–33 have a `note` in their config: their reference is `prompt_patient` (23), not `baseline` (10);
      - **guard added** in `runner.check_same_config()`: a run is refused if its name already ran with another config (prompt text included). It now refuses `temperature_*` and `top_k_10_gen` under their names, as it should.
    - [x] Run pass (generator): `prompt_patient` (23), `top_k_10_gen` (24), `temperature_0.3` (25, 28, 31), `temperature_0.7` (26, 29, 32), `temperature_1` (27, 30, 33). ~2 min each. All with the patient-rule prompt.

      | Experiment | Runs | out_of_corpus refused | Wrong refusals (/100) | Identical answers across runs | Refusal flips | Median length (chars) | Median latency |
      |---|---|---|---|---|---|---|---|
      | prompt_patient (T = 0) | 1 | 6/6 | 7 | – | – | 112 | 970 ms |
      | temperature_0.3 | 3 | 6/6/6 | 7/6/7 | 33/106 | 1 | 113 | 1034 ms |
      | temperature_0.7 | 3 | 6/6/6 | 6/6/6 | 17/106 | 2 | 118 | 1057 ms |
      | temperature_1 | 3 | 6/6/6 | 7/7/9 | 11/106 | 3 | 126 | 1065 ms |
      | top_k_10_gen | 1 | 6 | **5** | – | – | 110 | 1167 ms |

    - **Temperature 0 is reproducible** (measured in step 24): a second `baseline` run (34), 6 days and a reboot after the first (10), gives **identical answers on 106/106 questions**. The "45/106 answers differ" seen first was entirely the prompt change. The 3/10 differences seen when re-asking questions were probably the same cause: the prompt file was edited again (« passage » → « passages ») at 11:13, during that check.
    - **Temperature** (compared with `prompt_patient`, same prompt):
      - Refusals of out_of_corpus questions: 6/6 at every temperature.
      - Wrong refusals: 6–9 everywhere, always the same `multi_hop` questions (+ Q017, Q058 at T = 1): temperature doesn't fix the rule 2 problem.
      - Dispersion grows with temperature: identical answers across the 3 runs 33 → 17 → 11 /106 from T = 0.3 to 1. At T = 1, one answer (Q058, run 33) rambled to 2074 characters and was cut at `max_tokens = 512`.
      - Answers get slightly longer and slower as temperature rises (112 → 126 characters, 970 → 1065 ms).
      - **Not judged** (decision of 3 Nov): the deterministic results already show what temperature does here, and a factual medical RAG keeps temperature 0. **Kept: temperature 0.**
    - [x] Judge pass: experiments 24 (`top_k_10_gen`, 2377 s + 93 s to retry Q004's faithfulness timeout) and 23 (`prompt_patient`, 1445 s).

      | Metric | baseline (10) | prompt_patient (23) | top_k_10_gen (24, patient prompt) |
      |---|---|---|---|
      | contextual_precision | 0.876 | 0.876 | 0.865 |
      | contextual_recall | 0.942 | 0.942 | 0.923 |
      | faithfulness | 0.881 | 0.942 | 0.942 |
      | answer_relevancy | 0.910 | 0.923 | 0.935 |
      | correctness | 0.850 | 0.877 | 0.838 |

      - contextual_precision / recall identical for 10 and 23: same chunks → same scores, the judge is stable when its input doesn't change. Once (Q015 faithfulness) a score changed with the same answer text: the judge isn't perfectly deterministic.
      - **Judge noise: not measured yet** (10 vs 23 differ by the prompt). The second `baseline` run of step 24, judged, will give it.
    - **top_k_10_gen vs prompt_patient** (its real reference): correctness 0.838 vs 0.877, faithfulness equal, relevancy 0.935 vs 0.923; 5 wrong refusals instead of 7; +20 % latency, twice the prompt. No gain in answer quality on the sample. **Kept: `top_k = 5`.** A clean re-test needs a new name (e.g. `top_k_10_strict`), the guard refuses `top_k_10_gen`.
    - `top_p` / sampling `top_k`: only matter at temperature > 0; not tested, since temperature 0 is kept.

- [x] 24. **§4 Prompts:** (partly: judge pass stopped, see below) `minimal`, `strict`, `citations`, `patient` (each variant = `strict.txt` + one rule), at temperature 0.
    - Count refusals both ways: questions with an answer that were refused, and `out_of_corpus` questions that were answered.
    - Check citations without an LLM: `cited_passages()` + `judge_chunk()`.
    - Prompts (each = committed `strict.txt` with **one** change, checked with `diff`):
      - `patient.txt`: + rule 3 « Si la question mentionne un patient, n'utilise que les passage qui concernent ce patient. » (already run and judged: experiment 23 `prompt_patient`);
      - `citations.txt`: + rule 5 « Après chaque information, indique entre crochets le numéro de l'extrait… [2] … [1][3]. Ne cite rien après « Je ne sais pas. ». »;
      - `partial.txt`: rule 2 rewritten: « Je ne sais pas. » only when the extracts contain **no** element of answer; otherwise give the part, then say in one sentence what the extracts don't allow to say;
      - `minimal.txt`: unchanged.
    - TOMLs: `prompt_minimal`, `prompt_citations`, `prompt_partial` (+ `prompt_patient`, which reproduces experiment 23).
    - New in `metrics.py`: `cited_passages()`; `answer_scores(answer, chunks, question)` now also gives `citation_valid` (cited numbers that exist) and `citation_relevant` (cited chunks that prove part of the answer, `judge_chunk`), only when the answer cites something. 18 tests.
    - Known limit: `refusal` only detects « Je ne sais pas »; `minimal`'s refusals ("il n'est pas possible…") won't be counted: read them with the judge.
    - [x] Run pass (generator): `baseline` #2 (34), `prompt_minimal` (35), `prompt_citations` (36), `prompt_partial` (37). 100–234 s each.

      | Run | out_of_corpus refused | Wrong refusals (/100) | Median length (chars) | Median latency |
      |---|---|---|---|---|
      | baseline #1 (10) / #2 (34) | 6/6 · 6/6 | 7 · 7 (same questions) | 110 · 110 | 944 · 946 ms |
      | prompt_patient (23) | 6/6 | 7 | 114 | 986 ms |
      | prompt_minimal (35) | 0/6 (not detected) | 0 (not detected) | **407** | **1935 ms** |
      | prompt_citations (36) | 6/6 | 7 | **93** | 925 ms |
      | prompt_partial (37) | 6/6 | **3** | 222 | 1259 ms |

      - **baseline #1 vs #2: 106/106 identical answers**: temperature 0 is reproducible (step 23 corrected).
      - **citations:** 93/93 non-refusal answers cite an extract; `citation_valid` **1.000** (no invented number); `citation_relevant` **0.890**. Some "irrelevant" citations are a limit of the metric, not of the answer: Q088/Q106 cite the operation report whose **patient line** holds the date, but the evidence snippet (« cholécystectomie ») is elsewhere; Q037 cites the discharge letter instead of the notice. Also the shortest answers of all prompts.
      - **partial:** 7 → 3 wrong refusals, out_of_corpus still 6/6 refused. But reading the answers: only **Q017** is truly answered (« perforé… péritonite localisée », full expected answer); **Q072** gives half (current metformin dose, contrast instructions missing: retrieval); **Q012 and Q089 are refusals in other words** (« il n'est pas possible de déterminer… »), which break the prompt's own rule and that `is_refusal()` can't see. Real gain: 1 question, plus 1 half. The judge's correctness decides.
      - **minimal:** answers 4× longer, latency 2×, one answer cut at 512 tokens (Q072); its refusals never use the sentence, so 0/6 "refused" is a detection artefact.
      - Unanswerable for every prompt: Q013, Q071, Q078, Q089 (+ Q012, Q072 partly): the second fact isn't in the 5 chunks, a **retrieval** failure (`multi_hop`), for step 27 (question decomposition).
    - [x] Judge pass, **stopped on 4 Nov** (step 24 skipped from here, to go on with step 26): judged 38, 39, 40 (26/26); `prompt_partial` (37) **15/26 only**, its judge means can't be compared with the others (`judge_experiment.py prompt_partial` completes it); `baseline` #2 (34), `prompt_citations` (36), `prompt_minimal` (35) not judged. The judge's own noise is therefore not measured.

      | Run | contextual_precision | contextual_recall | faithfulness | answer_relevancy | correctness |
      |---|---|---|---|---|---|
      | baseline (10) | 0.876 | 0.942 | 0.881 | 0.910 | 0.850 |
      | prompt_patient (23) | 0.876 | 0.942 | 0.942 | 0.923 | 0.877 |
      | combined (38) | 0.867 | 0.942 | 0.965 | 0.949 | 0.881 |
      | combined_v2 (39) | 0.867 | 0.942 | 1.000 | 0.917 | 0.873 |
      | combined_v2_top_k_10 (40) | 0.865 | 0.923 | 0.923 | 0.906 | 0.842 |

      - On the sample, the combined prompts' scores are close to the baseline and to each other; without the judge-noise measurement, gaps of a few hundredths can't be called real.
      - Not concluded: which single prompt change helps (the single-change runs 35–37 aren't judged).
    - [x] **Combination** (requested 3 Nov): `prompts/combined.txt` = `strict.txt` + rule 2 of `partial` + the patient rule (typo « passage » fixed) + the citation rule (7 rules), with chunks **256/0**: `experiments/combined_chunks_256_overlap_0.toml`, experiment **38** (109 s). Four changes at once vs the baseline: it says whether the combination is better, not which change does it (the single-change runs 14, 23, 36, 37 do).

      | Run | recall@5 | nDCG@5 | out_of_corpus refused | Wrong refusals | citation_valid | citation_relevant | Median length | Median latency |
      |---|---|---|---|---|---|---|---|---|
      | baseline (10) | 0.913 | 0.816 | 6/6 | 7 | – | – | 110 | 944 ms |
      | citations (36) | 0.913 | 0.816 | 6/6 | 7 | 1.000 | 0.890 | 93 | 925 ms |
      | partial (37) | 0.913 | 0.816 | 6/6 | 3 | – | – | 222 | 1259 ms |
      | **combined 256/0 (38)** | 0.913 | 0.817 | 6/6 | 5 | 1.000 | 0.851 | 117 | 994 ms |

      - Retrieval identical to the baseline (256/0 ≈ 512/0, as in step 21).
      - 95/95 non-refusal answers cite an extract, no invented number.
      - The citation rule keeps `partial`'s answers short (117 characters vs 222) and its refusals cleaner: Q089 is now a plain « Je ne sais pas. », Q072 ends with it. So "5 wrong refusals" is more honest than `partial`'s "3", not worse.
      - Q017 answered with the full expected answer and its source (« perforé… péritonite localisée [1] »). Q012 is still a refusal in other words.
      - Judge: `uv run python scripts/judge_experiment.py combined_chunks_256_overlap_0` (with the judge).
    - [ ] **combined_v2** (3 Nov): rule 3 softened, « n'utilise que » → « utilise **en priorité** » les passages du patient (a notice concerns no patient, the strict rule could make the model ignore it on `multi_hop`). Prompts are versioned: `prompts/combined.txt` = experiment 38's text (restored), `prompts/combined_v2.txt` = the new one; once a prompt has run, it isn't edited, a new file is made.
      - `experiments/combined_v2.toml`: chunks 256/0 + `combined_v2.txt`, top_k 5 → vs 38: effect of the rule 3 edit.
      - `experiments/combined_v2_top_k_10.toml`: same + `[search] top_k = 10` → vs `combined_v2`: effect of 10 chunks.
      - [x] Run pass (run by hand, 3 Nov): experiments **39** (`combined_v2`, 111 s) and **40** (`combined_v2_top_k_10`, 136 s).

        | Run | recall@5 | recall@10 | out_of_corpus refused | Wrong refusals | citation_valid | citation_relevant | Median length | Median latency |
        |---|---|---|---|---|---|---|---|---|
        | combined (38) | 0.913 | – | 6/6 | 5 | 1.000 | 0.851 | 118 | 994 ms |
        | combined_v2 (39) | 0.913 | – | 6/6 | 5 | 1.000 | 0.868 | 120 | 992 ms |
        | combined_v2_top_k_10 (40) | 0.913 | 0.953 | 5/6 → **6/6** (see below) | **4** | 1.000 | 0.774 | 124 | 1250 ms |

        - **Rule 3 edit (38 → 39): no deterministic change** (same refusals, same questions); the judge decides.
        - **10 chunks (39 → 40):** Q078 now answered with the expected precaution (the morphine notice was at rank 9) → 4 wrong refusals. The "5/6 out_of_corpus" is a detection artefact: Q045 is refused in other words (« Les extraits ne contiennent aucune information… »), no hallucination. `citation_relevant` drops (0.868 → 0.774): more chunks cited that aren't evidence. Latency +26 %.
        - Seen: with 5 chunks, Q045 got « Je ne sais pas. [1][5] », a citation after the refusal, against rule 6.
      - [ ] Judge pass: `uv run python scripts/judge_experiment.py combined_v2 combined_v2_top_k_10` (with the judge, ~55 min).
- [x] 25. **§5 Models:** (4 Nov; the step 24 judge pass ran in parallel, the judge doesn't need the embedder or reranker)
    - TOMLs, one change each + `enabled = false`: `embedder_qwen3`, `embedder_qwen3_no_instruction`, `embedder_harrier`, `reranker_qwen3`.
    - Query instruction, **in English** as Qwen3's model card advises for multilingual use: « Given a question from hospital staff, retrieve the passages of patient records and drug information sheets that answer it ». `embedder_qwen3_no_instruction` measures what it is worth.
    - Served names = folder names (`qwen3-embedding-0.6b`, `harrier-oss-v1-0.6b`): `chunk.py` loads each model's own tokenizer, so the chunks are counted in its tokens and each embedder gets its own collection.
    - Cost reference: bge-m3 and bge-reranker-v2-m3 ≈ 1.55 GB GPU each (vLLM, fp16).
    - Results (experiments 41–48, ~20 s each; indexes 7.2–7.3 s, **1173 chunks for every embedder**: the sections are short, the tokenizer doesn't change the cuts). Added: the same embedders **without reranker** (`no_rerank_*`), because with it every embedder gives the same result.

      | Run | hit@5 | MRR@5 | recall@5 | nDCG@5 | lay_language | abbreviation | negation (MRR) | multi_hop | GPU | Retrieval ms |
      |---|---|---|---|---|---|---|---|---|---|---|
      | **bge-m3 + bge-reranker (baseline)** | 0.950 | **0.833** | **0.913** | **0.816** | 0.75 | 1.00 | 0.94 | 0.58 | 1.55 + 1.55 GB | ≈ 160 |
      | qwen3-emb + instr. + bge-reranker (41) | 0.950 | 0.828 | 0.913 | 0.813 | 0.75 | 1.00 | 0.94 | 0.58 | 2.62 + 1.55 | 193 |
      | qwen3-emb, no instr. + bge-reranker (42) | 0.950 | 0.833 | 0.913 | 0.816 | 0.75 | 1.00 | 0.94 | 0.58 | 2.62 + 1.55 | 189 |
      | harrier + instr. + bge-reranker (45) | 0.950 | 0.833 | 0.913 | 0.814 | 0.75 | 1.00 | 0.94 | 0.58 | 2.67 + 1.55 | 199 |
      | bge-m3 + **qwen3-reranker** (48) | 0.940 | 0.789 | 0.896 | 0.783 | 0.75 | 0.75 | **1.00** | **0.62** | 1.55 + 2.66 | 235 |
      | bge-m3, no reranker (47) | 0.850 | **0.707** | **0.803** | **0.675** | 0.25 | **0.75** | 0.67 | 0.50 | 1.55 | 53 |
      | qwen3-emb + instr., no reranker (43) | 0.830 | 0.685 | 0.771 | 0.668 | 0.25 | 0.38 | 0.75 | 0.42 | 2.62 | 52 |
      | qwen3-emb, no instr., no reranker (44) | 0.850 | 0.679 | 0.787 | 0.664 | 0.38 | 0.50 | 0.75 | 0.50 | 2.62 | 53 |
      | harrier + instr., no reranker (46) | 0.850 | 0.661 | 0.788 | 0.639 | 0.25 | 0.38 | 0.64 | 0.62 | 2.67 | 53 |

    - **Conclusions:**
      - **With the reranker, the embedder doesn't matter**: Qwen3 and Harrier give the baseline's results on all 100 questions (same lesson as the fusion in step 22: the embedder only picks the 30 candidates, the reranker re-sorts them).
      - **Alone, bge-m3 is the best embedder** here (recall@5 0.803 vs 0.771–0.788, MRR 0.707 vs 0.66–0.69), clearly on `abbreviation` (0.75 vs 0.38–0.50), with **40 % less GPU** (1.55 vs 2.6 GB). The Qwen3 and Harrier models run in bf16; bge-m3 in fp16 with fewer parameters.
      - **The query instruction doesn't help Qwen3 here** (recall@5 0.771 with, 0.787 without), unlike its model card's 1–5 %. Possible causes: an English instruction for French questions, or an instruction not specific enough; one wording only was tried.
      - **Qwen3-Reranker is worse overall** (recall@5 0.896, MRR 0.789, nDCG 0.783 vs 0.913 / 0.833 / 0.816; MRR better on 13 questions, worse on 20), better on `negation` (MRR 1.00) and `multi_hop` (0.62), worse on `abbreviation` (0.75). It is slower (235 ms) and 70 % bigger. It runs with the template's default instruction (« Given a web search query… »); a task-specific instruction might help, not tried.
      - **Kept: bge-m3 + bge-reranker-v2-m3** (best quality, smallest, fastest).
    - [x] Back to the baseline setup: bge-m3 on 8001 and bge-reranker on 8002 (used again from step 26 on).
    - Embedders: Harrier-OSS-v1-0.6b and Qwen3-Embedding-0.6B, each with its `query_instruction` and its own collection. Read the vector size from the model config.
    - Reranker: Qwen3-Reranker-0.6B with the special vLLM command (see [Services](#swap-models-for-5)).
    - Record indexing time, latency per question and GPU memory (`nvidia-smi`).
- [x] 26. **§6 Filters (`rag/filters.py`):** (4–5 Nov)
    - The LLM extracts JSON; the code maps the name to an IPP using `hopital_documents.json`. Never trust an ID written by the LLM.
    - If the filtered search returns nothing, fall back to an unfiltered search.
    - No patient filter for questions without a patient (`drug_info`, `cross_notice`, `cross_patient`).
    - Measure extraction accuracy against `patient_ipp` (correct, wrong, missed), the effect per question type, then compare with a plain name match that uses no LLM.
    - `rag/filters.py`: `patient_from_names()` (the 60 known full names, accents and case folded, whole words; none if 0 or 2+ names: 8 last names are shared, a last name alone is ambiguous), `patient_from_llm()` (the LLM returns `{"patient_name": ...}` in JSON, the code looks it up among the known patients, never trusts an identifier), `patient_filter(ipp, scope)`, `question_filter(question, config)`.
    - `scope = "patient_or_notice"` (default): this patient's records **or** any drug notice; `"patient_only"` excludes the notices.
    - Optional `[filters]` section (`patient = "none" | "names" | "llm"`, `scope`): configs without it are unchanged, the guard still accepts `baseline`.
    - `pipeline.retrieve()`: filter → search → **fallback without filter if the filtered search finds nothing** → rerank; returns `patient_ipp`, `fallback`, and `filter_ms` in the timings.
    - `metrics.filter_scores()`: `filter_correct`, `filter_wrong`, `filter_missed`, `filter_fallback`, stored for every question when a filter is on. `tests/test_filters.py`: 8 tests (26 in total).
    - TOMLs: `filter_names`, `filter_names_patient_only`, `filter_llm` (retrieval only).
    - Results, experiments **49** (`filter_names`) and **50** (`filter_names_patient_only`):
      - extraction by name matching: **106/106 correct**, 0 wrong, 0 missed, 0 fallback; no LLM, < 1 ms.
      - retrieval: `patient_or_notice` = baseline on all 100 questions; `patient_only` loses 5 `multi_hop` questions (recall@5 0.58 → 0.42: the notices are excluded, as predicted), recall@5 0.893.
      - **other patients' chunks in the top 5**: baseline **43/425 (10.1 %)** in 26/85 questions (`buried`, `split`, `dose` most), **0 with the filter**. Recall doesn't see them (they are never the evidence): they are distractors for the LLM. Their effect needs a run with generation.
    - [x] `filter_llm` (experiment **51**): the LLM also extracts **106/106** correctly (0 wrong, 0 missed), same recall@5 (0.913), but retrieval latency **355 ms vs 151 ms** for name matching (+200 ms and one more LLM call per question). **On this corpus, name matching does as well with no LLM and no latency**: kept for the filter.
    - [x] `filter_names_gen` (experiment **52**: baseline + name filter, with generation): out_of_corpus 6/6 refused; wrong refusals 7 → 8; latency 944 → 912 ms.
      - Of the 26 questions that had other patients' chunks in the baseline top 5, **none had a wrong answer because of them**: the patient line and the LLM already ignored them. 22 answers changed, mostly in form (with only one patient's records, the model stops repeating the patient's name).
      - **One regression, Q010** (« Mme Françoise Gauthier est-elle allergique à la pénicilline ? »): both runs had her two chunks « Pas d'allergie médicamenteuse connue », but the filtered run answered « Je ne sais pas. ». In the baseline, another patient's chunk (« Allergie à la pénicilline », 4100016) put the word « pénicilline » in the context and made the link easier; without it, `strict.txt` rule 3 (« aucune déduction ») blocks "no drug allergy → not allergic to penicillin". A distractor that helped by chance: the cause is the prompt's strictness, not the filter.
    - **Conclusions:** extraction is solved (both methods perfect, names free); the filter removes all other patients' chunks (10 % → 0) at no cost, without improving recall or the answers measurably here, since the baseline wasn't fooled by them. In a medical system it is a **safety** gain (the wrong patient's data never reaches the LLM), worth keeping in the final configuration; `scope` must keep the notices (`patient_only` costs 5 `multi_hop` questions). Not judged.
- [x] 27. **§8:** pick at least two. Cheapest: removing the patient line, reranker threshold. Most instructive: contextual retrieval, parent-child. A second judge from another family (Gemma4-26B-A4B) measures the self-preference bias of the Qwen judge.
    - **Option 1, reranker threshold** (`[reranker] threshold`, optional; below it, « Je ne sais pas. » without calling the LLM, `finish_reason = "threshold"`):
      - Studied on the stored rerank scores first (no run needed): best score of the 6 out_of_corpus questions **0.018–0.504** (Q045 at 0.504: the patient and a metformin notice both exist), answerable questions down to **0.067** (`cross_patient`, `multi_hop`: the answer is spread over several chunks, no single chunk matches the whole question). The distributions overlap.
      - With `strict.txt` the LLM already refuses 6/6 out_of_corpus: any threshold only adds wrong refusals (0.10: +1, 0.30: +5, 0.50: +10). With `minimal` (which never refuses with the sentence), catching 3/6 costs 4 wrong refusals.
      - Only useful value here: **0.05** (Q075, unknown patient, score 0.018, refused without an LLM call; no wrong refusal). Confirmed by `reranker_threshold_0.05` (experiment **53**): 6/6, 7 wrong refusals, as the baseline.
      - **Conclusion: the prompt is a better refusal mechanism than the rerank score** (the LLM reads what the chunks say; the score only measures similarity). The threshold only saves an LLM call on hopeless questions.
    - **Temperature 0 is not reproducible after all** (seen with experiment 53): same command, binary, chunks (106/106), prompt and settings as the baseline, yet 22/106 answers differ (wording); two runs of today (52, 53) differ on 11 of 76 questions with identical chunks. The answer depends on the server state left by previous requests (llama-server reuses its cache) and on the session; experiment 34's 106/106 was a case where that state matched. Refusal counts don't move. **Recommendation, not applied** (it would shift every new run slightly vs the stored ones): send `cache_prompt: false` from `generate.py`, then check two runs match.
    - **Option 2, question decomposition** (`rag/decompose.py`, optional `[decomposition] enabled`, `max_subquestions = 3`): the LLM returns `{"questions": [...]}` (self-contained, names repeated); each sub-question gets its own search + rerank; `interleave()` takes the best of each in turn, without duplicates, up to `top_k` (5, as the baseline). Simple questions come back unchanged. Stored score `subquestions`. 3 tests (29 in total).
      - `decomposition` (retrieval only, experiment **54**) and `decomposition_gen` (experiment **55**):

        | Run | hit@5 | MRR@5 | recall@5 | nDCG@5 | multi_hop recall@5 | cross_notice | Retrieval | Full answer |
        |---|---|---|---|---|---|---|---|---|
        | baseline | 0.950 | 0.833 | 0.913 | 0.816 | 0.58 | 0.77 | ≈ 150 ms | 944 ms |
        | decomposition | **0.970** | 0.811 | **0.922** | 0.802 | **0.75** | 0.81 | 734 ms | 1697 ms |

      - **The target is hit:** 4 of the `multi_hop` questions no prompt could answer are now answered correctly: Q013 (suxaméthonium avoided, kaliémie 6,0), Q072 (stop metformin, resume 48 h after), Q078 (reduce and space the morphine doses), Q089 (no ramipril with sacubitril/valsartan, 36 h).
      - **But it splits too much:** 45/106 questions decomposed, not only the 12 `multi_hop` (8 `split`, 5 `negation`, 5 `temporal`…); recall lost on 4 questions (Q052 `temporal`, Q067, Q086, Q090); new wrong refusals Q037, Q052, Q086, Q090. Net: 7 wrong refusals before and after, but different ones. Q071's decomposition gave two near-identical sub-questions.
      - Cost: one more LLM call per question, latency +80 %.
      - **Conclusion:** decomposition fixes the `multi_hop` retrieval failure; to keep, it should only split questions that need it (a stricter prompt, or only questions naming both a patient and a drug). Not judged.
    - **Follow-up, 5 Nov: `combined_v2_decomposition`** (`combined_v2.toml` + `[decomposition]`, one change; run and judge by hand: experiment **56**, 201 s, 26/26 judged). `decomposition_gen` (55) was judged too (correctness 0.877, faithfulness 0.920).

      | Metric | baseline (10) | combined_v2 (39) | decomposition_gen (55) | **combined_v2_decomposition (56)** |
      |---|---|---|---|---|
      | recall@5 / `multi_hop` | 0.913 / 0.58 | 0.913 / 0.58 | 0.922 / 0.75 | 0.922 / 0.75 |
      | faithfulness | 0.881 | **1.000** | 0.920 | 0.987 |
      | answer_relevancy | 0.910 | 0.917 | 0.919 | **0.936** |
      | correctness | 0.850 | 0.873 | 0.877 | **0.881** |
      | out_of_corpus refused / wrong refusals | 6/6 / 7 | 6/6 / 5 | 6/6 / 7 | 6/6 / **3** |
      | citations valid / relevant | – | 1.000 / 0.868 | – | 1.000 / 0.885 |
      | median latency | 944 ms | 992 ms | 1697 ms | 1785 ms |

      - **The two changes add up**: the prompt keeps the answers faithful and sourced, decomposition brings the missing facts; wrong refusals 5 → **3**, which neither reached alone. Newly answered, with sources: Q013 (correctness 0.00 → 0.70), Q072, Q089, Q078 (half).
      - **Still there**: the over-splitting regressions (Q052 `temporal` recall 1.0 → 0.0 and a new wrong refusal; Q086, Q090 recall 1.0 → 0.5; Q017 correctness 1.00 → 0.50, Q020 0.80 → 0.60) and Q071 (bad decomposition).
      - **Best configuration on quality**, at +80 % latency; `combined_v2` stays the efficient choice. Next: split only the questions that need it.
    - Plan for avoiding redundant chunks (Jaccard, MinHash + LSH, MMR, and others): `deduplicate_plan.md`.
    - [x] **Deduplication, step 1** (5 Nov): redundancy metrics `redundancy@k`, `near_duplicates@k`, `documents@k` (Jaccard on chunk bodies, within one patient or notice), computed for every new run and for the 47 stored ones (`scripts/rescore.py --all`). Main finding: the near-duplicates are **header chunks** (Q005: 4 of its 5 chunks are the patient's headers), so boilerplate removal is next. Details: `deduplicate_plan.md`, *Step 1 results*.
    - [x] **Deduplication, step 2** (5 Nov): `[chunking] strip_header = true` drops each document's header chunks (structural rule: title-only heading and only header lines; line frequency would have deleted « Pas d'allergie médicamenteuse connue », found in 18.6 % of documents). Run **57** (retrieval only): recall@5 0.913 → **0.943**, hit@5 0.950 → **0.990**, MRR 0.833 → 0.843, redundancy@5 0.051 → 0.007; 4 questions better (Q013, Q078, Q088, Q106), 0 worse; 24 % fewer chunks. Details: `deduplicate_plan.md`, *Step 2 results*.

## Phase 4 — Deliverables

- [x] 28. **(Bonus) `ui/app.py`:** a read-only Streamlit app that imports from `src/`. It shows a leaderboard per metric, a two-experiment comparison (per question type, questions improved or degraded) and question details.
    - `uv run streamlit run ui/app.py` (http://localhost:8501).
    - Read-only **enforced**: `db.connect(read_only=True)` sets the connection read-only, PostgreSQL refuses any write (checked: `ReadOnlySqlTransaction`).
    - Imports from `src/`: `evaluation.report.load_report_data()` (same data as the HTML report), `evaluation.runner.flatten()` (config differences), `evaluation.metrics.judge_chunk()` (the ✓ on relevant chunks), and two new `db.py` functions with their SQL files (`question_answers`, `question_scores`).
    - Refusals are split into "out_of_corpus refused" (higher is better) and "wrong refusals" (lower is better) so that both can be ranked.
    - Data cached 5 min (`st.cache_data`), with a "Reload from the database" button.
    - Tested headless with `streamlit.testing.AppTest`: the 3 pages without exception, leaderboard 39 rows, numbers equal to the report's; the server answers its health check.
- [x] 29. **`README.md`:** install, `docker compose`, model servers (copy the [Services](#services) section), ingest, run, ask.
    - Sections: requirements (GPU, models folder), installation and `.env`, model servers (with which commands need which server), usage (ingest, ask, run, judge, report, tests), the experiment file reference (every key, the optional sections), the "one name = one config" rule and "never edit a used prompt", the layout, and a troubleshooting table of the problems met during the exercise.
    - Checked: every script accepts the options shown; the README's TOML example parses to exactly `baseline.toml`.
- [x] 30. **`RAPPORT.md`:**
    - a comparison table built from the §7 SQL query;
    - the best config, chosen on quality, cost and latency;
    - what changed nothing or made results worse;
    - three failures of the best config analysed: where it fails (retrieval, rerank or generation) and the fix to try.
    - Written 5 Nov. Best configuration: **`combined_v2`** (faithfulness 1.000, correctness 0.873, 5 wrong refusals, 6/6 out_of_corpus refused, citations 100 % valid, latency ≈ 1 s), with the name filter and a restricted decomposition as next additions. The comparison table is generated from the database (46 runs, 39 configurations, grouped by section).
    - The three failures: **Q013** (ranking: the kaliémie at rank 12; fixed by decomposition), **Q012** (retrieval of a *chained* question: the notice at rank 23, and decomposition writes « cet antibiotique »: needs iterative retrieval), **Q034** (generation: complete retrieval, faithful but incomplete answer, the 1–3 sentence rule).
    - Plus a "lessons on measuring a RAG" section.

**Tip:** commit after each module in phase 1 and after each experiment's TOML in phase 3, so the git history mirrors the experiment log.

---

## How question decomposition works

Option of §8 ("Décomposer la question"), measured in step 27. Code: `src/hospital_rag/rag/decompose.py`, called by `rag/pipeline.py` `retrieve()`. Off unless the experiment's TOML has:

```toml
[decomposition]
enabled = true              # the LLM splits a question into sub-questions, one search + rerank each
max_subquestions = 3
```

### The problem it solves

A `multi_hop` question needs two facts from two different documents, e.g. « En cas d'intubation en urgence, le suxaméthonium pourrait-il être utilisé chez M. Pierre Perrin ? » needs **the patient's kaliémie** (his lab report) **and** the drug's contra-indications (the notice). The reranker scores each chunk against the *whole* question, so it favours one side: in the baseline, the notice is ranked first and the lab result is at rank 11, out of the top 5. No prompt can fix that: the fact isn't in the extracts (step 24).

### The four stages

1. **Decompose** (`decompose()`): one LLM call (the generator, temperature 0, JSON output) with a prompt asking for self-contained sub-questions, at most `max_subquestions`, repeating the patient's or the drug's name so each one makes sense alone. A question that needs one piece of information comes back unchanged, as a single sub-question.

   ```text
   Q013 → « Quel est l'état clinique et les antécédents de M. Pierre Perrin dans son dossier patient ? »
          « Quelles sont les contre-indications et les précautions d'emploi du suxaméthonium selon la notice du médicament ? »
   Q007 → « Quelle dose de ramipril prend actuellement M. Bruno Martin ? »   (unchanged: one fact)
   ```

   If the LLM's answer isn't valid JSON with a `questions` list, the original question is used: decomposition can make a search worse, never break it.

2. **Search each sub-question** (`search()`): the usual hybrid search, `[search] candidates` (30) candidates per sub-question.

3. **Rerank each sub-question against itself** (`rerank()`): the record's candidates are scored against the record sub-question, the notice's against the notice sub-question. This is the key difference with the baseline: no sub-question has to compete with the other for the reranker's attention.

4. **Interleave** (`interleave()`): the best chunk of each sub-question in turn (1st of the first, 1st of the second, 2nd of the first…), skipping duplicates, until `[search] top_k` (5) chunks. The total stays at 5, as in the baseline, so the comparison is fair and the prompt isn't longer. Generation then gets the **original** question and these 5 chunks.

With one sub-question, stages 2–3 are exactly the baseline's search and rerank, and stage 4 is skipped.

### What is stored

`retrieve()` returns `subquestions` (and `decompose_ms` in the timings); the runner stores a `subquestions` score (their number) for every question, so the report shows which questions were split.

### Results (step 27, experiments 54 and 55)

- **Target reached:** `multi_hop` recall@5 0.58 → 0.75; 4 `multi_hop` questions no prompt could answer are answered correctly (Q013, Q072, Q078, Q089).
- **Over-splitting:** 45 of 106 questions decomposed, not only the 12 `multi_hop` ones; recall lost on 4 questions, wrong refusals unchanged in number (7) but on different questions. Q071 got two near-identical sub-questions.
- **Cost:** one more LLM call per question; retrieval ≈ 150 → 734 ms, full answer 944 → 1697 ms (+80 %).
- **To keep it:** split only the questions that need it (a stricter decomposition prompt, or only questions naming both a patient and a drug).

## Services

### Hardware constraint

One GPU: RTX PRO 4000 Blackwell, **24 GB**. The embedder, the reranker and the generator must fit together; the judge runs on its own afterwards (step 18b). Before starting, stop anything else that uses the GPU (`nvidia-smi`), for example a leftover `llama-server`.

### Model choice

| Role | Model | Server | Port | Approx. VRAM |
|---|---|---|---|---|
| Embedder (baseline) | `~/models/embedder/bge-m3` | vLLM | 8001 | ~2.4 GB |
| Reranker (baseline) | `~/models/reranker/bge-reranker-v2-m3` | vLLM | 8002 | ~2.4 GB |
| Generator | `~/models/qwen3.5-9b/llama/Qwen3.5-9B-UD-Q4_K_XL.gguf` | llama-server | 8000 | ~7–8 GB |
| Judge | `~/models/qwen3.8-27b/llama/Qwen3.8-27B-UD-Q4_K_XL.gguf` | llama-server, alone | 8003 | ~20 GB |
| Second judge (§8) | `~/models/gemma4-26b-a4b/llama/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` | llama-server, alone | 8003 | ~19 GB |

The VRAM figures are estimates from the file sizes: check them with `nvidia-smi` after each start.

Why these:

- **Embedder and reranker:** `~/models/` holds exactly the models the exercise names. bge-m3 and bge-reranker-v2-m3 are the reference: multilingual, good in French, 8k tokens, and bge-m3 needs no query instruction. The Qwen3 and Harrier models are for §5 only.
- **Generator, Qwen3.5-9B Q4:** fits next to the embedder and reranker with room left for context, which a 27B doesn't. Fast enough for 106 questions × dozens of experiments, and a 9B model follows the « Je ne sais pas » rule reliably. The GGUF with llama-server also matches the exercise's `extra_body={"top_k": 40}`, a llama-server parameter.
- **Judge, Qwen3.8-27B:** the strongest model available, and it only scores the 26-question sample, so speed matters less. Same family as the generator, so it may favour its answers: the Gemma second judge (§8) measures that.
- **Not used:** diffusiongemma, muse-glimmer-30b, nemotron3-nano-4b; glm-ocr and paddleocr-vl (Docling's built-in OCR is enough for the 24 imaging PDFs; point it at `~/models/docling` so it runs offline); gemma4-12b (vLLM only; a candidate for the "other LLM" test in §8).

### Start order

One terminal per server. **Start the two vLLM servers first, then llama-server**: vLLM reserves a fixed share of the GPU when it starts, while llama-server only takes what it needs. The ports match `.env` (`LLM_URL` 8000, `EMBEDDINGS_URL` 8001, `RERANK_URL` 8002).

### Terminal 1 — embedder (port 8001)

```bash
~/vllm_server/.venv/bin/vllm serve ~/models/embedder/bge-m3 \
  --runner pooling --served-model-name bge-m3 --port 8001 \
  --dtype float16 --max-model-len 2048 \
  --gpu-memory-utilization 0.10 --enforce-eager
```

### Terminal 2 — reranker (port 8002)

```bash
~/vllm_server/.venv/bin/vllm serve ~/models/reranker/bge-reranker-v2-m3 \
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
~/vllm_server/.venv/bin/vllm serve ~/models/embedder/qwen3-embedding-0.6b \
  --runner pooling --served-model-name qwen3-embedding-0.6b --port 8001 \
  --max-model-len 2048 --gpu-memory-utilization 0.10 --enforce-eager

~/vllm_server/.venv/bin/vllm serve ~/models/embedder/harrier-oss-v1-0.6b \
  --runner pooling --served-model-name harrier-oss-v1-0.6b --port 8001 \
  --max-model-len 2048 --gpu-memory-utilization 0.10 --enforce-eager
```

Reranker (terminal 2), the command from the exercise with local paths:

```bash
~/vllm_server/.venv/bin/vllm serve ~/models/reranker/qwen3-reranker-0.6b \
  --runner pooling --served-model-name qwen3-reranker --port 8002 \
  --max-model-len 2048 --gpu-memory-utilization 0.10 --enforce-eager \
  --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' \
  --chat-template ~/models/reranker/qwen3_reranker.jinja
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
