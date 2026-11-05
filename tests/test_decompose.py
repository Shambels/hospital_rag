"""Merging the rankings of several sub-questions.

Run: uv run pytest
"""
from hospital_rag.rag.decompose import decompose, interleave


def chunk(document, index=0):
    return {"document": document, "chunk_index": index}


def test_interleave_takes_the_best_of_each_in_turn():
    record = [chunk("dossier", 1), chunk("dossier", 2), chunk("dossier", 3)]
    notice = [chunk("notice", 1), chunk("notice", 2)]
    merged = interleave([record, notice], top_k=4)
    assert merged == [chunk("dossier", 1), chunk("notice", 1), chunk("dossier", 2), chunk("notice", 2)]


def test_interleave_skips_duplicates():
    a = [chunk("x", 1), chunk("y", 1)]
    b = [chunk("x", 1), chunk("z", 1)]
    assert interleave([a, b], top_k=5) == [chunk("x", 1), chunk("y", 1), chunk("z", 1)]


def test_decomposition_off_by_default():
    # no LLM call: configs without [decomposition] keep one search per question
    assert decompose("Quelle dose ?", {"generation": {"model": "x"}}) == ["Quelle dose ?"]
