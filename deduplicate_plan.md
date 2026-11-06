# Plan — avoiding redundant chunks

How Jaccard similarity, MinHash + LSH and Maximal Marginal Relevance (MMR) would apply to this RAG, what else to consider, and in which order to try them. Each step below is one experiment, compared with one change at a time, as in `PLAN.md`. Done: steps 1 (redundancy metrics) and 2 (header removal, kept everywhere); step 3 (cap per document) measured and **not kept**.

## The three techniques, and when they act

| Technique | What it does | When it acts |
|---|---|---|
| **Jaccard similarity** | measures how much two texts overlap: shared shingles (groups of words) divided by all shingles | a measure, used by the two others |
| **MinHash + LSH** | estimates Jaccard **without comparing every pair**, and puts near-duplicates in the same buckets | **at indexing time** (ETL): find duplicates in the corpus |
| **MMR** (Maximal Marginal Relevance) | picks the top k so that each new chunk is both relevant **and** different from those already picked | **at query time**: diversify the chunks sent to the LLM |

## The redundancy actually measured in this corpus

1. **Overlap duplicates**: with 256/32 chunks, neighbouring chunks repeat each other: MRR −0.03, and `cross_notice` recall 0.77 → 0.68: list questions need **many distinct** chunks, the duplicates took their places (`PLAN.md` step 21).
2. **Boilerplate**: every document's header chunk carries the « Document fictif généré pour un cours… » disclaimer and the hospital's header; those chunks rank high on name-only questions (Q019) and on Q013, and the LLM once used the disclaimer as an argument (steps 13–16).
3. **The same fact in several documents**: a consultation and a prescription both say « ramipril 5 mg »; interchangeable sources can fill the top 5 with one piece of information.
4. **Decomposition's interleaving** removes exact duplicates (same document and chunk index), not near-duplicates.

### ⚠️ Identical text is not always redundant

« Allergie alimentaire à l'arachide (œdème des lèvres en 2018). » appears word for word in **three patients'** records; for Q002 (« Quels patients ont une allergie à l'arachide ? ») each one is a **separate fact**. Naive deduplication would break the `cross_patient` questions. **Any deduplication must be scoped to one document, or to one patient, never across patients.**

## 1. Jaccard: the similarity measure

```text
J(A, B) = |shingles(A) ∩ shingles(B)| / |shingles(A) ∪ shingles(B)|
```

For example with 3-word shingles on the normalised text (the same `normalize()` as `judge_chunk()`). Two details matter here:

- **On the chunk's body, not its prefix.** Every chunk starts with the patient line and the headings; including them makes all of a patient's chunks look alike. Strip the lines before the text, as `ask.py` does for its one-line view.
- **Keep the numbers.** « Ramipril 10 mg par jour » and « Ramipril 5 mg par jour » have a high Jaccard but **different doses**: the `temporal` trap. A similarity doesn't know that a changed number changes the meaning; whatever uses it must be checked on the `temporal` and `dose` questions.

## 2. MinHash + LSH: deduplication at indexing time

**How it works.** Each chunk gets a signature of, say, 128 hash values (for each hash function, the smallest hash of its shingles); the share of equal values between two signatures estimates their Jaccard. LSH cuts the signatures into bands: two chunks that share one band land in the same bucket, and only those pairs are compared.

**Is it needed here?** Not at this size: 1173 chunks are ≈ 690,000 pairs, which exact Jaccard handles in seconds. MinHash + LSH matters with millions of chunks, where comparing every pair is impossible. Its value in this exercise is to **implement it** (`datasketch`) **and check it finds the same pairs as exact Jaccard**: an approximate method validated against the exact one.

**What to use it for:**
- **Boilerplate removal**: find the text repeated across many documents (the disclaimer, the hospital header) and remove it from the chunks. A simpler method does as well here: count how many documents each line appears in, and drop the lines found in more than, say, 20 % of them.
- **Merging near-identical chunks within one document** (what overlap creates).

It changes the chunks, so: new `[chunking]` options (e.g. `strip_boilerplate = true`), **to add to `collection_name()`**, a new collection, and `ingest.py` before the run.

## 3. MMR: diversifying what the LLM receives

The reranker scores the 30 candidates; instead of taking the 5 best, MMR picks them one by one:

```text
next = argmax over the remaining candidates c of
       λ · relevance(c)  −  (1 − λ) · max over the already picked s of similarity(c, s)
```

- `relevance`: the rerank score (already between 0 and 1);
- `similarity`: Jaccard on the bodies, or the cosine of the dense vectors (Qdrant returns them with `with_vectors=True`);
- `λ = 1`: today's behaviour; lower values favour diversity.

**Where:** **after the reranker**, on its 30 candidates: the reranker is the best judge of relevance, MMR only rearranges its choice. A new step in `pipeline.retrieve()` (and in the per-sub-question loop when decomposition is on), with an optional section, off by default so every existing config is unchanged:

```toml
[diversity]
method = "mmr"          # none (default) or mmr
lambda = 0.7
similarity = "jaccard"  # jaccard (on the chunk body) or cosine (dense vectors)
```

**What to watch:**
- **Should improve**: `cross_notice` and `cross_patient` (lists, which need distinct chunks), `multi_hop` (the second fact is often pushed out by a near-duplicate of the first) → **recall@5**, the metric for "more distinct facts".
- **Risks**: `temporal` and `dose` (MMR could drop the newer dose's chunk because it "looks like" the older one); MRR and nDCG fall by construction (the 2nd chunk is no longer the 2nd most relevant).

## A metric to add first: redundancy

Recall can't show that duplicates were removed (the patient filter taught the same lesson: some effects need their own metric). Add to `metrics.retrieval_scores()`:

- **`redundancy@5`**: the mean Jaccard between the pairs of the 5 chunks (bodies only); lower is better;
- **`documents@5`**: the number of distinct documents among the 5.

Computed on every run from now on, and recomputable on the stored runs (their chunks and texts are in `answers.chunks`), so the existing experiments get the measure too. The report and the dashboard rank "higher is better": `redundancy@5` needs the same treatment as "wrong refusals" there.

## Other techniques worth trying

| Technique | Idea | Why here |
|---|---|---|
| **Cap per document** | at most 2 chunks from the same document in the top 5 | the simplest form of diversity, and the **baseline MMR must beat**: if MMR does no better, its complexity isn't worth it |
| **Merge adjacent chunks** | if chunks i and i+1 of one document are both kept, send them as one passage | removes the overlap duplication and restores the paragraph; a light version of §8's parent–child |
| **Boilerplate removal** | strip the lines repeated across documents before chunking | the redundancy measured most clearly here (header chunks ranking high); possibly the biggest gain |
| **Exact hash first** | the hash of the normalised body, scoped per patient, before any fuzzy method | catches the exact duplicates in milliseconds; MinHash only for what's left |
| **Semantic deduplication** | the cosine of the dense vectors above a threshold (e.g. 0.95) | catches paraphrases Jaccard misses, with the same caution about numbers |
| **Context compression** | keep only the relevant sentences of each chunk (an LLM or an extractive model) | removes the redundancy *inside* the chunks; more expensive |

## Order of work

Each step is one change, measured with the deterministic metrics on 100 questions (retrieval only, ≈ 20 s), and with generation + judge only for what is kept.

1. ✅ **The `redundancy@5` and `documents@5` metrics**, recomputed on the stored runs: they tell which configurations are redundant today (expected: the overlap ones, `top_k_10`, decomposition). **Done (5 Nov)**, see *Step 1 results* below.
2. ✅ **Boilerplate removal at indexing**: done as **header removal** (`strip_header`), not by line frequency: see *Step 2 results*, and *Step 2 with generation and the judge* (kept on every configuration).
3. **Cap per document** (2, then 1 per document): the simple baseline for diversity. **Done (6 Nov), not kept**: cap 2 changes nothing, cap 1 loses 8 questions; see *Step 3 results*.
4. **MMR after the reranker**, Jaccard on the bodies, λ = 0.7 and 0.5 (`mmr_lambda_0.7`, `mmr_lambda_0.5`): compared with step 3 on `cross_notice`, `cross_patient`, `multi_hop` (recall@5) and on `temporal`, `dose` (no loss allowed).
5. **MinHash + LSH**, as a demonstration: same pairs as exact Jaccard on the 1173 chunks? Then used for the index-time deduplication if the corpus were to grow.
6. The best of 2–4 with generation and the judge, on top of the best configuration (`combined_v2_decomposition`, or `combined_v2` if latency matters).

## Step 1 results (5 Nov)

**Implemented** in `evaluation/metrics.py`, computed by the runner on every question of every new run, and for the 47 stored runs with `uv run python scripts/rescore.py --all` (from their stored chunks, no model needed):

- `chunk_body()`: the chunk without its patient line and its headings (up to 2 leading lines of ≤ 6 words without final punctuation);
- `shingles()` (3 words, lowercase, normalised) and `jaccard()`;
- `scope()`: the patient (IPP prefix of the file name) or the notice; **chunks of different scopes are never redundant**;
- `redundancy_scores()`: `redundancy@k` (mean Jaccard over the pairs of the same scope), `near_duplicates@k` (chunks with Jaccard ≥ 0.8 to an earlier one of the same scope), `documents@k` (distinct documents), for k = 5 and 10 (≤ `top_k`).
- Report and dashboard: `redundancy@` and `near_duplicates@` ranked lower-is-better (best = minimum, a decrease shown as an improvement).
- 8 tests, among them: the same allergy line in two patients' records is **not** redundant; shared headings don't make chunks redundant; a changed number still counts as similar (the `temporal` trap, documented by a test).

**What it shows:**

| Configuration | redundancy@5 | near_duplicates@5 | documents@5 |
|---|---|---|---|
| no reranker (bge-m3) | 0.083 | 0.49 | 3.73 |
| 256/32 (overlap) | **0.077** | 0.16 | 3.31 |
| baseline | 0.051 | 0.27 | 3.59 |
| combined_v2 (256/0) | 0.045 | 0.26 | 3.57 |
| 128/0 | 0.034 | 0.15 | 3.39 |
| decomposition | 0.032 | 0.19 | 3.55 |
| **combined_v2_decomposition** | **0.027** | 0.17 | 3.54 |

- **Overlap is the most redundant setting** among the reranked runs, as expected; the reranker itself *reduces* redundancy (0.083 → 0.051); **decomposition's interleaving diversifies** the chunks (the lowest redundancy).
- **The redundancy is concentrated**: `negation` (redundancy@5 0.22, 1.4 near-duplicates per question), `out_of_corpus` (0.15), `lay_language` (0.11); 0 for `cross_patient`, `cross_notice`, `buried`, `split`.
- **What the near-duplicates are: header chunks.** For Q005 (« Mme Catherine Boyer a-t-elle été transfusée… ? ») 4 of the 5 chunks sent to the LLM are this patient's header chunks (#0: hospital header, identity, disclaimer) from four documents; for Q019, all 5. For Q026, the same allergy line in two documents, plus three headers.
- **So step 2 (boilerplate removal) comes first**: it targets the measured source of redundancy. MMR (step 4) would only push these header chunks down; removing their boilerplate makes them disappear from the candidates.

## Step 2 results (5 Nov)

**Why not line frequency.** Measured on the 280 documents: the disclaimer is in 91 % of them, the « Rédigé par » and « Service de » lines in 9–20 %, but « Pas d'allergie médicamenteuse connue. » is in **18.6 %**, and it's evidence (Q010). A frequency threshold can't tell boilerplate from a frequent fact.

**The structural rule instead** (`etl/chunk.py`, `is_header()`): a document's **leading** chunks are dropped while they have **no section heading** (only the title) **and** every line is a header line (hospital, identity, date, author, disclaimer). Both conditions are needed: in the 24 PDFs every section has the title as its only heading (the heading test alone would delete the imaging reports), and the line test alone could match a header-like line inside a section. Checked on every document at 512, 256 and 128 tokens: all 280 start with such a chunk (2 or more when a small budget splits the header). The identity and the date stay in the patient line (built from the metadata); the author and the patient's age are lost, which no golden question needs.

- Option `[chunking] strip_header = true` (off by default: existing configs unchanged), in `collection_name()` (`__no-header`). 5 tests (`tests/test_chunk.py`, 42 in total).
- Evidence coverage unchanged at every size (grade 2 174/174, grade 1 38/38); no document left without a chunk; **24 % fewer chunks** (1173 → 893 at 512/0).
- Experiment `strip_header` (baseline + the option, retrieval only), run **57**:

| Metric | baseline | strip_header |
|---|---|---|
| hit@5 | 0.950 | **0.990** |
| MRR@5 | 0.833 | **0.843** |
| recall@5 | 0.913 | **0.943** |
| nDCG@5 | 0.816 | **0.828** |
| redundancy@5 | 0.051 | **0.007** |
| near_duplicates@5 | 0.274 | **0.028** |
| header chunks among the 530 sent to the LLM | 74 | **0** |
| chunks / indexing time | 1173 / 7.2 s | **893 / 3.8 s** |

- **4 questions better, 0 worse** (recall@5 and MRR@5): `multi_hop` **Q013, Q078** (the questions decomposition was needed for) and `lay_language` **Q088, Q106** (the ones `top_k = 10` was needed for). `multi_hop` recall@5 0.58 → 0.67, `lay_language` 0.75 → 1.00.
- **Why**: the header chunks matched every question naming a patient, on the name alone, and took places in the top 5; without them, those places go to content.
- **The largest retrieval gain of the exercise, at no cost**: no LLM call, no latency, half the indexing time.

## Step 2 with generation and the judge (5 Nov)

The same option (`strip_header = true`, one line added) on the three references, with generation, then the judge on the 26-question sample: runs **59** (`strip_header_gen`, vs `baseline` 10), **60** (`combined_v2_strip_header`, vs `combined_v2` 39), **61** (`combined_v2_decomposition_strip_header`, vs `combined_v2_decomposition` 56). (Run 58 failed after 1 s, before any question.)

| | baseline | + strip | combined_v2 | + strip | c_v2_decomp | + strip |
|---|---|---|---|---|---|---|
| run | 10 | 59 | 39 | 60 | 56 | 61 |
| hit@5 | 0.950 | **0.990** | 0.950 | **0.990** | 0.970 | **0.990** |
| recall@5 | 0.913 | **0.943** | 0.913 | **0.943** | 0.922 | **0.942** |
| MRR@5 | 0.833 | 0.843 | 0.835 | 0.844 | 0.813 | 0.817 |
| redundancy@5 | 0.051 | **0.007** | 0.045 | **0.007** | 0.027 | **0.006** |
| near_duplicates@5 | 0.274 | 0.028 | 0.255 | 0.028 | 0.170 | 0.028 |
| wrong refusals (/100) | 7 | 6 | 5 | **3** | 3 | **2** |
| out_of_corpus refused | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| faithfulness | 0.881 | 0.978 | 1.000 | 0.987 | 0.987 | 0.974 |
| answer relevancy | 0.910 | 0.926 | 0.917 | 0.923 | 0.936 | **0.962** |
| correctness | 0.850 | 0.842 | 0.873 | **0.881** | **0.881** | 0.873 |
| median latency | 944 ms | 970 ms | 992 ms | 1003 ms | 1785 ms | 1799 ms |

**Retrieval: the gain carries over to every configuration**, identically (the retrieval of run 59 is run 57's; decomposition adds nothing on top: 0.942 vs 0.943). The redundancy drops to ≈ 0.007 everywhere: there is almost nothing left for MMR to remove.

**The judge's scores don't move beyond their noise** (±0.01 on correctness, one question changing by 0.2 moves the mean by 0.008). The one large move, baseline faithfulness 0.881 → 0.978, is **not** the option: for Q003, Q015 and Q020 the two runs retrieve **the same chunks** (indices shifted by one, the header chunk gone), only the wording of the answer changes (temperature 0 isn't reproducible, PLAN step 27), and the judge scores the new wording higher (in run 10 it penalised « calcul » read as "calculation").

**The wrong refusals are where it shows**, question by question:

- **`combined_v2` + strip (60) refuses 3, like decomposition (56) does**: Q013 and Q078 are no longer refused (the second fact now reaches the top 5). The remaining three, **Q071, Q072, Q089**, are *chained* `multi_hop` questions: the drug isn't named in the question (« sa corticothérapie », « son traitement du diabète », « ajouter du ramipril » to a patient on sacubitril/valsartan), so no search on the question alone finds the notice. Header removal can't fix that; only decomposition does.
- **Decomposition + strip (61) refuses 2**: it answers Q072 (metformin stopped for 48 h) and Q089 (ramipril + sacubitril/valsartan contraindicated) correctly, and Q037 is no longer refused. Still refused: **Q071** (everywhere: « corticothérapie » never becomes « prednisone ») and **Q052** (decomposition's over-splitting, see RAPPORT).
- **Not a refusal, but not an answer either**: Q013 in run 60 retrieves the suxamethonium notice's *indication* chunks and the bilan, but not the hyperkalaemia contraindication; the answer says « les extraits ne permettent pas de déterminer » (correctness 0). Runs 56 and 61 answer it.
- Q061 in run 59 is a prompt artefact, not retrieval: the right answer followed by « Je ne sais pas. » (`strict.txt`); `combined_v2.txt` fixes it, with the same chunks.

**Conclusions:**

1. **`strip_header` is kept everywhere**: better retrieval at every configuration, the redundancy gone, 24 % fewer chunks, no extra latency, nothing lost.
2. **It halves the case for decomposition.** `combined_v2_strip_header` has the correctness of `combined_v2_decomposition` (0.881), the same number of wrong refusals (3), faithfulness 0.987, at **1.0 s instead of 1.8 s**. What decomposition still adds is the chained questions (Q072, Q089), and it still over-splits Q052.
3. **New best trade-off: `combined_v2_strip_header`.** On quality alone, `combined_v2_decomposition_strip_header` is ahead (2 wrong refusals, answer relevancy 0.962), at +80 % latency.

**Next** (written before the gated decomposition): steps 3–4 (per-document cap, MMR) have little redundancy left to act on (near_duplicates@5 0.028: 3 chunks over 106 questions); they remain worth one retrieval-only run each, on top of `strip_header`, for `cross_*` diversity rather than for duplicates. The bigger remaining lever is the chained `multi_hop` questions, i.e. a decomposition that only triggers when the question needs a second fact (RAPPORT, *Improvements*).

## Step 3: preparation (6 Nov)

**Implemented**: `rag/diversity.py` and an optional section, off when absent (every existing configuration is unchanged):

```toml
[diversity]
method = "cap"              # none (default) or cap
max_per_document = 2
```

- **Where**: after the reranker, in `pipeline.retrieve()`. With `[diversity]`, `rerank()` returns **every** candidate ranked (`top_n = 30` instead of 5: the reranker scores all 30 anyway, so it costs nothing), and `cap_per_document()` walks down that ranking, skipping a chunk whose document already has `max_per_document` chunks, until 5. With decomposition, the cap applies to the interleaved order (after `interleave()`), so two sub-questions can't fill the top 5 with one document between them.
- **Never fewer chunks**: if the cap leaves fewer than 5 (every candidate from one or two documents), the skipped chunks fill the gap in ranked order.
- 5 tests (`tests/test_diversity.py`, 51 in total). Checked on 10 questions: without `[diversity]`, the chunks are identical to run 57's.

**What the stored runs say before running it**: the cap at 2 acts only when one document holds 3 or more of the 5 chunks. That happens on 21 questions in run 57 (26 in run 60), mostly `imaging`, `buried`, `abbreviation`, `lay_language`: questions answered by **one** report, whose several sections are legitimately the evidence (Q003, Q014, Q044: 3 chunks of one imaging report). And the two failures that motivated this step had **exactly 2** chunks of one notice (Q052 in run 61: ramipril; Q080 in run 62: metformin): a cap at 2 doesn't touch them, only a cap at 1 does.

**So two experiments**, each `strip_header` + `[diversity]` (retrieval only, ≈ 20 s, embedder + reranker; compared with run **57**):

| Experiment | max_per_document | Expected |
|---|---|---|
| `strip_header_cap_2` | 2 | few changes; watch `imaging` and `buried` (one-report questions) for losses, `cross_*` for gains |
| `strip_header_cap_1` | 1 | 5 distinct documents per question; gains on lists (`cross_notice`, `cross_patient`), losses likely on `split` and `buried` (an answer spread over two sections of one document) |

Metrics: recall@5 and MRR@5 by type, `documents@5` (should rise to ≤ 5), and the questions improved / degraded. The Q052 / Q080 effect needs decomposition, hence the generator: if the cap at 1 doesn't cost too much on its own, one run of the gated configuration + cap 1, with generation.

## Step 3 results (6 Nov)

Runs **63** (`strip_header_cap_2`) and **64** (`strip_header_cap_1`), retrieval only, vs run **57** (`strip_header`):

| Metric | 57: no cap | 63: cap 2 | 64: cap 1 |
|---|---|---|---|
| hit@5 | **0.990** | **0.990** | 0.930 |
| recall@5 | **0.943** | **0.943** | 0.891 |
| MRR@5 | **0.843** | **0.843** | 0.827 |
| nDCG@5 | 0.828 | **0.831** | 0.806 |
| documents@5 | 3.64 | 3.84 | 5.00 |
| near_duplicates@5 | 0.028 | 0.028 | 0.038 |
| chunks of another patient (single-patient questions) | 51 / 425 (31 q.) | 54 / 425 (33 q.) | **121 / 425 (62 q.)** |

- **Cap 2: no effect.** The top 5 changes on some questions (documents@5 3.64 → 3.84) but no question's recall or MRR moves: the third chunk of a document it removes is never the evidence, and the chunk that replaces it isn't either. 3 more chunks of other patients.
- **Cap 1: worse, 8 questions lost, 1 gained** (Q037, recall 0 → 0.5). Lost: `abbreviation` Q034, Q074 (1.00 → 0.75 for the type), `lay_language` Q029, Q099 (→ 0.75), `cross_patient` Q057, `cross_notice` Q058, `multi_hop` Q071. Even the list questions, which it was meant to help, lose.
- **Why: a document's best-ranked chunk is often not its evidence.** In Q057 (« Quels patients ont reçu un stent sur l'artère interventriculaire antérieure ? »), each coronarography report's chunk 0 (indication) ranks above its chunk 1 (where the stent went): the cap keeps chunk 0 and drops the evidence. Q058: the furosemide notice's chunk 5 ranks above its chunk 6, the evidence. Q034: the second consultation chunk with « VEMS à 48 % » is dropped. The cap assumes "one chunk per document is enough"; with sections as chunks, the reranker's favourite section and the answering section are often two different ones.
- **And the places it frees go to other patients.** For a question about one patient, the next-best documents are other patients' records: other-patient chunks more than double (51 → 121), in 62 questions instead of 31 (Q034: two chunks of patient 4100036 replace Mme Dupont's evidence). Diversity across documents works against the patient focus; it would need the patient filter (`[filters] patient = "names"`) to be safe.
- **What it means for Q052 / Q080** (two chunks of one notice pushing out a record fact, under decomposition): a cap at 2 can't reach them, a cap at 1 costs far more than it could recover. Their fix belongs to the interleaving (the whole question's ranking first), not to a diversity step.

**Conclusion and consequence for step 4 (MMR).** After header removal, the remaining duplicates are too few (near_duplicates@5 0.028) for diversity to help: cap 2 has nothing to remove, cap 1 removes evidence. MMR is a smoother version of the same trade (relevance against similarity to the chunks already picked, and chunks of one document are the most similar): it would need a λ close to 1 to avoid cap 1's losses, which is cap 2's "no effect". Step 4 is therefore **not worth running for this corpus**; step 5 (MinHash + LSH) remains as a learning exercise only. The redundancy problem was the header chunks, and step 2 solved it.
