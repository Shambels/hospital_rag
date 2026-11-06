"""The cap per document (deduplicate_plan.md, step 3).

Run: uv run pytest
"""
import pytest

from hospital_rag.rag.diversity import cap_per_document, diversify


def chunk(document, index):
    return {"document": document, "chunk_index": index}


def test_cap_skips_a_third_chunk_of_the_same_document():
    # the Q080 shape: two chunks of one notice would push out the record's third chunk
    ranked = [chunk("ordonnance", 0), chunk("notice", 4), chunk("notice", 3), chunk("notice", 2),
              chunk("consultation", 5), chunk("consultation", 3)]
    assert cap_per_document(ranked, top_k=5, max_per_document=2) == [
        chunk("ordonnance", 0), chunk("notice", 4), chunk("notice", 3), chunk("consultation", 5), chunk("consultation", 3)]


def test_cap_keeps_the_ranked_order():
    ranked = [chunk("a", 0), chunk("b", 0), chunk("a", 1), chunk("c", 0)]
    assert cap_per_document(ranked, top_k=3, max_per_document=2) == [chunk("a", 0), chunk("b", 0), chunk("a", 1)]


def test_cap_never_gives_fewer_chunks():
    # every candidate from one document: the skipped chunks fill the gap, best first
    ranked = [chunk("a", i) for i in range(6)]
    assert cap_per_document(ranked, top_k=5, max_per_document=2) == [chunk("a", i) for i in range(5)]


def test_no_diversity_section_means_the_reranker_order():
    ranked = [chunk("a", i) for i in range(6)]
    assert diversify(ranked, {"search": {"top_k": 5}}) == ranked[:5]


def test_unknown_method_is_an_error():
    with pytest.raises(ValueError):
        diversify([], {"search": {"top_k": 5}, "diversity": {"method": "mmr"}})
