"""Hand-made cases for the retrieval metrics: each expected value can be checked on paper.

Run: uv run pytest
"""
import math

import pytest

from hospital_rag.evaluation.metrics import answer_scores, is_refusal, judge_chunk, retrieval_scores

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
    assert answer_scores("Je ne sais pas.") == {"refusal": 1.0}
    assert answer_scores("5 mg par jour.") == {"refusal": 0.0}
