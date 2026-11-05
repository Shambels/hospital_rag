"""Hand-made cases for the retrieval metrics: each expected value can be checked on paper.

Run: uv run pytest
"""
import math

import pytest

from hospital_rag.evaluation.metrics import (
    answer_scores, chunk_body, cited_passages, is_refusal, jaccard, judge_chunk, redundancy_scores, retrieval_scores, shingles,
)

# Two complementary facts (0 and 1); fact 0 has two interchangeable sources; one useful-only passage (grade 1)
QUESTION = {
    "evidence": [
        {"document": "a.docx", "snippets": ["Ramipril 5 mg"], "grade": 2, "fact": 0},
        {"document": "b.docx", "snippets": ["ramipril 5 mg par jour"], "grade": 2, "fact": 0},
        {"document": "notice.docx", "snippets": ["hyperkaliémie"], "grade": 2, "fact": 1},
        {"document": "c.docx", "snippets": ["à 5 mg"], "grade": 1, "fact": None},
    ]
}


def chunk(document, text):
    return {"document": document, "text": text}


NOISE = chunk("x.docx", "Metformine 1000 mg")


def test_judge_chunk_needs_the_right_document():
    # The same snippet in another patient's record proves nothing
    assert judge_chunk(chunk("a.docx", "- Ramipril 5 mg : 1 cp"), QUESTION) == [0]
    assert judge_chunk(chunk("other_patient.docx", "- Ramipril 5 mg : 1 cp"), QUESTION) == []


def test_judge_chunk_needs_every_snippet():
    question = {"evidence": [{"document": "bilan.docx", "snippets": ["HbA1c", "8,5"], "grade": 2, "fact": 0}]}
    assert judge_chunk(chunk("bilan.docx", "HbA1c, Résultat = 8,5."), question) == [0]
    assert judge_chunk(chunk("bilan.docx", "HbA1c, Résultat = 7,2."), question) == []


def test_judge_chunk_ignores_whitespace_changes():
    assert judge_chunk(chunk("a.docx", "Ramipril\n 5   mg"), QUESTION) == [0]


def test_one_fact_found_at_rank_2():
    scores = retrieval_scores([NOISE, chunk("a.docx", "Ramipril 5 mg"), NOISE], QUESTION, top_k=3)
    assert scores["hit@3"] == 1
    assert scores["mrr@3"] == 0.5
    assert scores["recall@3"] == 0.5  # fact 0 found, fact 1 (the notice) missing


def test_interchangeable_sources_count_as_one_fact():
    chunks = [chunk("a.docx", "Ramipril 5 mg"), chunk("b.docx", "ramipril 5 mg par jour"), NOISE]
    assert retrieval_scores(chunks, QUESTION, top_k=3)["recall@3"] == 0.5


def test_all_facts_found():
    chunks = [chunk("notice.docx", "Contre-indication : hyperkaliémie"), chunk("b.docx", "ramipril 5 mg par jour"), NOISE]
    scores = retrieval_scores(chunks, QUESTION, top_k=3)
    assert scores["recall@3"] == 1
    assert scores["mrr@3"] == 1


def test_grade_1_alone_is_not_a_hit():
    scores = retrieval_scores([chunk("c.docx", "réduit à 5 mg"), NOISE, NOISE], QUESTION, top_k=3)
    assert scores["hit@3"] == 0
    assert scores["recall@3"] == 0
    assert scores["ndcg@3"] > 0  # but nDCG gives it some credit


def test_ndcg_perfect_ranking_is_1():
    chunks = [chunk("a.docx", "Ramipril 5 mg"), chunk("b.docx", "ramipril 5 mg par jour"),
              chunk("notice.docx", "hyperkaliémie"), chunk("c.docx", "à 5 mg"), NOISE]
    assert retrieval_scores(chunks, QUESTION, top_k=5)["ndcg@5"] == pytest.approx(1)


def test_ndcg_by_hand():
    # Only the grade-1 chunk, at rank 2: DCG = 1/log2(3); ideal top 3 = grades 2, 2, 2
    scores = retrieval_scores([NOISE, chunk("c.docx", "à 5 mg"), NOISE], QUESTION, top_k=3)
    ideal = 2 / math.log2(2) + 2 / math.log2(3) + 2 / math.log2(4)
    assert scores["ndcg@3"] == pytest.approx((1 / math.log2(3)) / ideal)


def test_duplicate_chunks_are_credited_once():
    # Two overlapping chunks proving the same evidence item must not push nDCG above 1
    same = chunk("a.docx", "Ramipril 5 mg")
    chunks = [same, same, same, chunk("b.docx", "ramipril 5 mg par jour"), chunk("notice.docx", "hyperkaliémie")]
    assert retrieval_scores(chunks, QUESTION, top_k=5)["ndcg@5"] <= 1


def test_metrics_only_up_to_top_k():
    scores = retrieval_scores([NOISE] * 5, QUESTION, top_k=5)
    assert set(scores) == {f"{m}@{k}" for m in ("hit", "mrr", "recall", "ndcg") for k in (3, 5)}


def test_out_of_corpus_has_no_retrieval_score():
    assert retrieval_scores([NOISE] * 5, {"evidence": []}, top_k=5) == {}


def test_refusal_is_the_strict_sentence_whatever_the_case():
    assert is_refusal("Je ne sais pas.")
    assert is_refusal("JE NE SAIS PAS")
    assert not is_refusal("M. Bruno Martin prend 5 mg de ramipril par jour.")


def test_refusal_misses_other_wordings():
    # Known limit, see answer_scores(): the minimal prompt refuses without the sentence
    assert not is_refusal("Il n'est pas possible de déterminer le groupe sanguin de Mme Marie Morel.")


def test_answer_scores():
    assert answer_scores("Je ne sais pas.", [], QUESTION) == {"refusal": 1.0}
    assert answer_scores("5 mg par jour.", [], QUESTION) == {"refusal": 0.0}


def test_cited_passages():
    assert cited_passages("5 mg [2], réduit en avril [1][3].") == {1, 2, 3}
    assert cited_passages("5 mg par jour.") == set()


def test_citation_scores():
    chunks = [NOISE, chunk("a.docx", "Ramipril 5 mg"), NOISE]
    # [2] exists and is relevant, [3] exists but is noise, [7] doesn't exist
    scores = answer_scores("5 mg par jour [2][3][7].", chunks, QUESTION)
    assert scores["citation_valid"] == pytest.approx(2 / 3)
    assert scores["citation_relevant"] == 0.5


def test_no_citation_scores_without_citations():
    assert set(answer_scores("5 mg par jour.", [NOISE], QUESTION)) == {"refusal"}


def test_chunk_body_strips_the_patient_line_and_the_headings():
    text = ("Patient : M. Bruno MARTIN (IPP 4100002) — Ordonnance du 13/04/2025\nOrdonnance\nPrescription\n"
            "- Ramipril 5 mg : 1 comprimé le matin.")
    assert chunk_body(text) == "- Ramipril 5 mg : 1 comprimé le matin."
    assert chunk_body("Fiche médicament : Metformine\nEffets indésirables\nNausées, diarrhée.") == "Nausées, diarrhée."


def test_chunk_body_keeps_a_body_without_final_punctuation():
    # a header chunk's first body line is long: it isn't a heading
    text = "Compte rendu opératoire\nCentre Hospitalier de Valmont — Service de Chirurgie orthopédique\nDate : 21/03/2026"
    assert chunk_body(text).startswith("Centre Hospitalier")


def test_jaccard():
    assert jaccard(shingles("a b c d"), shingles("a b c d")) == 1
    assert jaccard(shingles("a b c d"), shingles("e f g h")) == 0
    assert jaccard(shingles("a b c d"), shingles("a b c e")) == 1 / 3  # {abc, bcd} vs {abc, bce}


def body(document, text, patient="Patient : X (IPP 1) — Doc du 01/01/2025\nDoc\nSection\n"):
    return {"document": document, "text": patient + text}


def test_same_text_in_the_same_patient_is_redundant():
    same = "Allergie alimentaire à l'arachide (œdème des lèvres en 2018)."
    chunks = [body("4100003_a.docx", same), body("4100003_b.docx", same)] + [body(f"notice_{i}.docx", f"texte {i} a b c") for i in range(3)]
    scores = redundancy_scores(chunks, top_k=5)
    assert scores["near_duplicates@5"] == 1
    assert scores["redundancy@5"] == 1 / 10  # one identical pair out of 10
    assert scores["documents@5"] == 5


def test_same_text_in_two_patients_is_not_redundant():
    # Q002: three patients with the same allergy line are three facts, not duplicates
    same = "Allergie alimentaire à l'arachide (œdème des lèvres en 2018)."
    chunks = [body("4100003_a.docx", same), body("4100017_a.docx", same), body("4100027_a.docx", same)]
    scores = redundancy_scores(chunks, top_k=5)
    assert scores["redundancy@5"] == 0 and scores["near_duplicates@5"] == 0


def test_shared_headings_dont_make_chunks_redundant():
    chunks = [body("4100002_a.docx", "Ramipril 10 mg par jour, amlodipine 10 mg."),
              body("4100002_a.docx", "Hyperkaliémie à 5,8 mmol/L sur le bilan du 04/04/2025.")]
    assert redundancy_scores(chunks, top_k=5)["redundancy@5"] == 0


def test_a_changed_number_still_counts_as_similar():
    # the temporal trap: lexically close, different meaning; what uses the metric must be checked on temporal questions
    chunks = [body("4100002_a.docx", "Poursuite du ramipril 10 mg par jour le matin."),
              body("4100002_b.docx", "Poursuite du ramipril 5 mg par jour le matin.")]
    assert 0.3 < redundancy_scores(chunks, top_k=5)["redundancy@5"] < 0.8


def test_redundancy_only_up_to_top_k():
    assert set(redundancy_scores([NOISE] * 3, top_k=3)) == set()
    assert {k.split("@")[1] for k in redundancy_scores([NOISE] * 10, top_k=10)} == {"5", "10"}
