# Rapport — Exercice 3 : industrialiser et optimiser le RAG

The RAG of the Centre Hospitalier de Valmont, moved out of the notebooks into a Python application (`src/hospital_rag/`), then used to measure, one change at a time, the optimisations of sections 2 to 8 of the exercise. 46 runs of 39 configurations are stored in PostgreSQL with their full configuration and their scores, question by question. The step-by-step log, with every intermediate result, is `PLAN.md`; how to reproduce everything is `README.md`.

## Summary

- **Best configuration: `combined_v2`**: the baseline's models and retrieval, with a system prompt combining four rules (partial answers allowed, the patient's extracts first, a citation after each fact). Faithfulness **1.000** and correctness **0.873** on the judge's sample (baseline 0.881 / 0.850), **5** wrong refusals instead of 7, every out-of-corpus question still refused, citations that always point to an existing extract, the same latency (≈ 1 s).
- **Most retrieval settings change nothing at the end of the pipeline**: the reranker re-sorts 30 candidates that always contain the relevant chunks (hit@30 = 1.00), so the fusion method, its `k`, the embedder and the number of candidates above 30 are all invisible.
- **The remaining failures are `multi_hop` questions** whose second fact isn't retrieved. Question decomposition fixes 4 of them, at the cost of +80 % latency and over-splitting.
- **Measurement matters as much as optimisation**: three findings of this work are about the measures themselves (the reranker hides upstream changes, the refusal metric can be fooled by wording, temperature 0 is not reproducible with llama-server).

## Method

- **Reference (`baseline`)**: 512-token chunks with the patient line, bge-m3, hybrid search (dense + BM25, RRF k = 60), 30 candidates, bge-reranker-v2-m3, 5 chunks, Qwen3.5-9B (Q4, llama-server) at temperature 0, prompt `prompts/strict.txt`.
- **One change at a time**: every experiment is `experiments/baseline.toml` with one key changed (checked with `diff`); the runner refuses a run whose name already ran with another configuration, prompt text included.
- **Deterministic metrics on every run**, on the 100 questions with ground truth: hit, MRR, recall and nDCG at 3, 5 and 10, with `judge_chunk()` (right document, every snippet); refusals on the 106 questions (`out_of_corpus` refused = good, other questions refused = wrong); citation validity and relevance; patient-filter quality.
- **LLM judge** (DeepEval, Qwen3.8-27B) on a fixed sample of 26 questions (2 per type, `out_of_corpus` excluded), only for the changes that touch generation and for the configurations kept: contextual precision and recall, faithfulness, answer relevancy, correctness.
- **Limits**: one run per configuration in most cases; 26 judged questions, so one question moves a judge mean by ≈ 0.04; the judge's own noise wasn't measured (step 24's judge pass was stopped before the second baseline run).

## Comparison of the experiments

Retrieval metrics on 100 questions, refusals on 106, judge on 26 (only for fully judged runs). "(retrieval)": retrieval-only run, no generation.

| Experiment | Runs | hit@5 | MRR@5 | recall@5 | nDCG@5 | out_of_corpus refused | wrong refusals | faithfulness | correctness | median latency |
|---|---|---|---|---|---|---|---|---|---|---|
| **Reference** | | | | | | | | | | |
| `baseline` | 2 | 0.950 | 0.833 | 0.913 | 0.816 | 6/6 | 7 | 0.881 | 0.850 | 945 ms |
| **§2 Chunking (retrieval only)** | | | | | | | | | | |
| `chunks_256_overlap_0` | 1 | 0.950 | 0.835 | 0.913 | 0.817 | – | – | – | – | 156 ms (retrieval) |
| `chunks_256_overlap_32` | 1 | 0.970 | 0.805 | 0.926 | 0.783 | – | – | – | – | 163 ms (retrieval) |
| `chunks_128_overlap_0` | 1 | 0.900 | 0.783 | 0.871 | 0.805 | – | – | – | – | 154 ms (retrieval) |
| `chunks_128_overlap_16` | 1 | 0.900 | 0.783 | 0.872 | 0.794 | – | – | – | – | 161 ms (retrieval) |
| `chunks_128_overlap_32` | 1 | 0.960 | 0.802 | 0.914 | 0.795 | – | – | – | – | 160 ms (retrieval) |
| **§3 Retrieval parameters (retrieval only)** | | | | | | | | | | |
| `top_k_10` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | – | – | – | – | 153 ms (retrieval) |
| `candidates_10` | 1 | 0.950 | 0.824 | 0.916 | 0.806 | – | – | – | – | 109 ms (retrieval) |
| `candidates_50` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | – | – | – | – | 204 ms (retrieval) |
| `rrf_k_10` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | – | – | – | – | 158 ms (retrieval) |
| `rrf_k_100` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | – | – | – | – | 160 ms (retrieval) |
| `fusion_dbsf` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | – | – | – | – | 159 ms (retrieval) |
| **§3 Generation parameters (patient-rule prompt, see PLAN step 23)** | | | | | | | | | | |
| `prompt_patient` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | 6/6 | 7 | 0.942 | 0.877 | 986 ms |
| `temperature_0.3` | 3 | 0.950 | 0.833 | 0.913 | 0.816 | 6/6 | 6.7 | – | – | 1034 ms |
| `temperature_0.7` | 3 | 0.950 | 0.833 | 0.913 | 0.816 | 6/6 | 6 | – | – | 1057 ms |
| `temperature_1` | 3 | 0.950 | 0.833 | 0.913 | 0.816 | 6/6 | 7.7 | – | – | 1065 ms |
| `top_k_10_gen` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | 6/6 | 5 | 0.942 | 0.838 | 1167 ms |
| **§4 Prompts** | | | | | | | | | | |
| `prompt_minimal` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | 0/6 | 0 | – | – | 1935 ms |
| `prompt_citations` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | 6/6 | 7 | – | – | 925 ms |
| `prompt_partial` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | 6/6 | 3 | – | – | 1259 ms |
| `combined_chunks_256_overlap_0` | 1 | 0.950 | 0.835 | 0.913 | 0.817 | 6/6 | 5 | 0.965 | 0.881 | 994 ms |
| `combined_v2` | 1 | 0.950 | 0.835 | 0.913 | 0.817 | 6/6 | 5 | 1.000 | 0.873 | 992 ms |
| `combined_v2_top_k_10` | 1 | 0.950 | 0.835 | 0.913 | 0.817 | 5/6 | 4 | 0.923 | 0.842 | 1250 ms |
| **§5 Embedders and rerankers (retrieval only)** | | | | | | | | | | |
| `embedder_qwen3` | 1 | 0.950 | 0.828 | 0.913 | 0.813 | – | – | – | – | 193 ms (retrieval) |
| `embedder_qwen3_no_instruction` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | – | – | – | – | 189 ms (retrieval) |
| `embedder_harrier` | 1 | 0.950 | 0.833 | 0.913 | 0.814 | – | – | – | – | 199 ms (retrieval) |
| `reranker_qwen3` | 1 | 0.940 | 0.789 | 0.896 | 0.783 | – | – | – | – | 235 ms (retrieval) |
| `no_rerank_bge_m3` | 1 | 0.850 | 0.707 | 0.803 | 0.675 | – | – | – | – | 53 ms (retrieval) |
| `no_rerank_qwen3` | 1 | 0.830 | 0.685 | 0.771 | 0.668 | – | – | – | – | 52 ms (retrieval) |
| `no_rerank_qwen3_no_instruction` | 1 | 0.850 | 0.679 | 0.787 | 0.664 | – | – | – | – | 53 ms (retrieval) |
| `no_rerank_harrier` | 1 | 0.850 | 0.661 | 0.788 | 0.639 | – | – | – | – | 53 ms (retrieval) |
| **§6 Filters** | | | | | | | | | | |
| `filter_names` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | – | – | – | – | 151 ms (retrieval) |
| `filter_names_patient_only` | 1 | 0.960 | 0.835 | 0.893 | 0.805 | – | – | – | – | 128 ms (retrieval) |
| `filter_llm` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | – | – | – | – | 355 ms (retrieval) |
| `filter_names_gen` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | 6/6 | 8 | – | – | 912 ms |
| **§8 Options** | | | | | | | | | | |
| `reranker_threshold_0.05` | 1 | 0.950 | 0.833 | 0.913 | 0.816 | 6/6 | 7 | – | – | 949 ms |
| `decomposition` | 1 | 0.970 | 0.811 | 0.922 | 0.802 | – | – | – | – | 734 ms (retrieval) |
| `decomposition_gen` | 1 | 0.970 | 0.811 | 0.922 | 0.802 | 6/6 | 7 | – | – | 1697 ms |

Notes on the table:
- **`prompt_minimal` "0/6"** and **`combined_v2_top_k_10` "5/6"** are detection artefacts: these answers refuse in other words (« il n'est pas possible de déterminer… », « Les extraits ne contiennent aucune information… »); refusals are only counted when the answer contains « Je ne sais pas ». `prompt_partial`'s "3" is partly the same artefact (2 of its 4 "answers" are refusals in other words).
- **§3 generation runs** (`prompt_patient`, temperatures, `top_k_10_gen`) used an uncommitted edit of `strict.txt` adding a patient rule; their reference is `prompt_patient`, not `baseline` (`PLAN.md` step 23).
- **`top_k_3`**: only @3 metrics: recall@3 0.856 (baseline recall@5 0.913).
- **`prompt_partial`** was judged on 15/26 questions only; `prompt_minimal`, `prompt_citations` and the temperature runs weren't judged.

## The best configuration: `combined_v2`

`experiments/combined_v2.toml`: chunks 256/0 (equivalent to the baseline's 512/0: same chunks, same retrieval) and `prompts/combined_v2.txt`, the baseline prompt with three changes:

1. « Je ne sais pas. » only when the extracts contain **no** element of answer; otherwise give the part they support and say what is missing (from `partial.txt`);
2. if the question names a patient, use **first** the extracts about that patient (a softer version of `patient.txt`: a drug notice concerns no patient, and the strict version could make the model ignore it);
3. after each piece of information, the number of the extract it comes from, `[2]` (from `citations.txt`).

**Why, on quality:**

| | baseline | combined_v2 |
|---|---|---|
| faithfulness (judge) | 0.881 | **1.000** |
| correctness (judge) | 0.850 | **0.873** |
| answer relevancy (judge) | 0.910 | 0.917 |
| out_of_corpus refused | 6/6 | 6/6 |
| wrong refusals | 7 | **5** |
| citations to an existing extract | – | **100 %** |
| cited extracts that hold part of the answer | – | 86.8 % |

The gain in faithfulness is the most robust: every claim of every judged answer is backed by the extracts, and the citations let a reader check it. The correctness gain (+0.023) is below what one question can move on 26, so it is a tendency, not a proof.

**Why, on cost and response time:** the same models and the same retrieval as the baseline, so the same GPU footprint (≈ 11 GB for embedder, reranker and generator); a longer system prompt (7 rules instead of 5) but shorter answers (citations keep them to the point: median 120 characters); median latency **992 ms** vs 944 ms.

**What it doesn't fix:** the 5 remaining wrong refusals are all `multi_hop` questions whose second fact isn't in the 5 chunks (Q013, Q071, Q072, Q078, Q089): a retrieval problem no prompt can solve.

**Recommended next additions** (each measured on its own, not yet in combination with `combined_v2`):
- the **patient filter by name matching** (`[filters] patient = "names"`): free (no LLM, < 1 ms, 106/106 patients right), it removes the 10 % of chunks that belong to another patient: no measured gain in the answers, but in a medical system the wrong patient's data should never reach the LLM;
- **question decomposition**, once restricted to the questions that need it (see below).

## What changed nothing, or made results worse

**No effect** (identical results on the 100 questions, or a few questions up and down):
- **Fusion method and its parameter** (RRF k = 10 / 60 / 100, DBSF): identical. Without the reranker, DBSF did beat RRF (hit@5 0.89 vs 0.85); with it, the difference disappears: fusion only chooses which 30 candidates reach the reranker, and the relevant ones are among them either way.
- **Embedder** (Qwen3-Embedding-0.6B, Harrier-OSS-v1-0.6b): identical to bge-m3 behind the reranker. Measured alone, bge-m3 is the best (recall@5 0.803 vs 0.771–0.788, clearly on `abbreviation`) and 40 % smaller.
- **Query instruction for Qwen3**: worse, not better, without reranker (recall@5 0.771 with, 0.787 without), unlike the model card's +1–5 % (one English wording tried for French questions).
- **Candidates 50**: identical to 30, 30 % slower to rerank. Candidates 10: noise (+3 / −2 questions), 30 % faster.
- **Chunk size 256 vs 512**: same chunks (1189 vs 1173): HybridChunker already cuts at each section and the sections are short (85 tokens on average).
- **Temperature 0.3 / 0.7 / 1**: same refusals; only more dispersion (identical answers between 3 runs: 33 → 17 → 11 /106).
- **Patient filter, on the scores**: same recall and same answers, as the patient line already lets the reranker and the LLM ignore the other patients.
- **Reranker threshold**: with `strict.txt` the LLM already refuses 6/6; the best rerank scores of answerable and out-of-corpus questions overlap (0.067 vs 0.018–0.504), so any useful threshold only adds wrong refusals. Only 0.05 is harmless, and only saves one LLM call.

**Worse:**
- **128-token chunks without enough overlap**: `split` recall 1.00 → 0.50 (4 questions cut in two); overlap 16 doesn't repair it, overlap 32 does.
- **Overlap at 256**: +1 recall point but MRR −0.03 and nDCG −0.03 (overlapping chunks compete for the top 5), `cross_notice` 0.77 → 0.68.
- **Qwen3-Reranker-0.6B**: recall@5 0.896 and MRR 0.789 vs 0.913 / 0.833, ≈ 50 % slower, 70 % bigger (with its template's default instruction).
- **Patient filter that excludes the notices** (`patient_only`): −5 `multi_hop` questions (recall 0.58 → 0.42).
- **LLM patient extraction**: as right as name matching (106/106), but +200 ms and one LLM call per question.
- **`minimal` prompt**: answers 4× longer, latency ×2, no clean refusal.
- **10 chunks to the LLM** (`top_k_10_gen`, `combined_v2_top_k_10`): +4 recall points, but correctness and faithfulness down on the sample and +26 % latency.
- **Question decomposition, as built**: fixes 4 `multi_hop` questions but splits 45/106 questions, loses 4 others, +80 % latency.

## Three failures of the best configuration

Analysed with `scripts/ask.py --id <question> --config experiments/combined_v2.toml --full` and the stored ranks.

### 1. Q013 (`multi_hop`): a ranking failure, fixed by question decomposition

> « En cas d'intubation en urgence, le suxaméthonium pourrait-il être utilisé chez M. Pierre Perrin ? »
> Expected: to avoid, contra-indicated in hyperkalaemia (kaliémie 6,0 mmol/L on 04/07/2025).
> `combined_v2`: « …ils ne contiennent aucune information sur son utilisation spécifique chez M. Pierre Perrin… » (correctness 0.00, faithfulness 1.00).

- **Where it fails: reranking.** The answer needs two facts: the patient's kaliémie (his consultation) and the contra-indication (the notice). The search finds both (ranks 10 and 4), but the reranker scores each chunk against the *whole* question and pushes the patient's chunk to rank 12, out of the top 5. The answer is faithful to what it got: the model doesn't invent, the fact just isn't there.
- **What fixes it: question decomposition** (§8). Split into « l'état clinique de M. Pierre Perrin » and « les contre-indications du suxaméthonium », each sub-question is reranked against itself; both facts reach the top 5 and the answer becomes right: « Non… car il présente une hyperkaliémie à 6,0 mmol/L. La notice indique que ce médicament… doit être évité chez les patients dont la kaliémie est élevée » (run 55). To keep, it must only split the questions that need it.

### 2. Q012 (`multi_hop`): a chained question, which decomposition can't fix

> « Quels effets indésirables de l'antibiotique prescrit à la sortie de Mme Colette Chevalier sont les plus courants ? »
> Expected: diarrhée, nausées, candidoses, éruptions cutanées (amoxicilline-acide clavulanique).
> `combined_v2`: « Les extraits ne permettent pas de répondre… ils ne listent aucun effet indésirable pour l'amoxicilline-acide clavulanique… » (correctness 0.00).

- **Where it fails: retrieval, of the second fact.** The discharge letter is found first, but the notice's « Effets indésirables » section is at rank **16** in the search and **23** after reranking: the question never names the drug, so neither BM25 nor the embedder can link it to the amoxicillin notice. The answer is again faithful, and also a refusal in other words, which the refusal metric doesn't count.
- **Why decomposition doesn't help:** its second sub-question is « Quels sont les effets indésirables les plus courants de **cet** antibiotique ? »: the drug's name is only known once the first sub-question is answered. The question is *chained*, not just double.
- **What could fix it: iterative retrieval**: answer the first sub-question, put its answer (« amoxicilline-acide clavulanique ») into the second, then search. One more LLM call for chained questions only. A cheaper partial fix: let the decomposition prompt refuse pronouns (« cet antibiotique ») so that it at least flags chained questions.

### 3. Q034 (`abbreviation`): a complete retrieval, an incomplete answer

> « Quelle est la sévérité de la bronchopneumopathie chronique obstructive de Mme Béatrice Dupont ? »
> Expected: stade GOLD 3, VEMS à 48 % de la théorique.
> `combined_v2`: « La sévérité de la BPCO de Mme Béatrice Dupont est classée au stade GOLD 3 [4]. » (correctness 0.70, faithfulness 1.00).

- **Where it fails: generation.** Retrieval is perfect (recall@5 = 1.00; the abbreviation trap, « BPCO » in the record and the full name in the question, is solved by the reranker) and the answer is right and sourced, but shorter than the expected answer: it gives the stage without the measurement that defines it. Q008 fails the same way (the drug and dose, without « 1 comprimé le matin »). Both come from rule 5 (« une à trois phrases ») and the model's terseness, which the citation rule reinforces.
- **What could fix it: one prompt rule**, « donne aussi les valeurs chiffrées qui justifient la réponse », measured with the judge's correctness, while watching that answers don't get padded. It is also partly a matter of the golden dataset: the question asks for a severity, the expected answer adds the value behind it; a judge only checking that the asked fact is there would score 1.

## Lessons on measuring a RAG

1. **The last stage hides the earlier ones.** With a reranker over 30 candidates, fusion, embedder and candidate count all looked useless; measured without the reranker, they differ clearly. Measure each stage where it acts.
2. **A metric can be fooled by wording.** `partial` went from 7 to 3 wrong refusals mostly by refusing in other words; `minimal` "refused" nothing. Read the answers behind a count before trusting it.
3. **Recall doesn't see distractors.** The patient filter left every score unchanged, yet it removed the 10 % of chunks from other patients that reached the LLM. Safety needs its own metric.
4. **Temperature 0 is not reproducible with llama-server**: same configuration, same chunks, yet up to 22/106 answers worded differently between runs (the server's state between requests). Retrieval scores are exactly reproducible; differences between answer-level results must exceed that noise.
5. **Store the inputs, not just the outputs.** Storing the prompt *text* with every run is what revealed that 11 runs had used an edited prompt file; the same-name guard now prevents it.

## Reproduce

```bash
uv run python scripts/ingest.py experiments/combined_v2.toml
uv run python scripts/run_experiment.py experiments/combined_v2.toml
uv run python scripts/judge_experiment.py combined_v2
uv run python scripts/report.py
```
