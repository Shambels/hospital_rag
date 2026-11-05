"""The RAG: question -> chunks -> answer. Scripts, evaluation and the API call this module, never the steps directly."""
import time

from hospital_rag.rag.decompose import decompose, interleave
from hospital_rag.rag.filters import question_filter
from hospital_rag.rag.generate import generate
from hospital_rag.rag.rerank import rerank
from hospital_rag.rag.search import search


def elapsed_ms(start):
    return (time.perf_counter() - start) * 1000


def retrieve(question, config):
    """The top_k chunks for a question (filter, search, then rerank), with the time of each stage.

    Enough on its own for the retrieval metrics: experiments that only change retrieval skip generation.
    With [decomposition] enabled, each sub-question gets its own search and rerank, then the rankings are interleaved.
    patient_ipp is the patient the filter used (None: no filter); fallback is True when the filtered search found
    nothing and was run again without the filter.
    """
    start = time.perf_counter()
    query_filter, patient_ipp = question_filter(question, config)
    filter_ms = elapsed_ms(start)

    start = time.perf_counter()
    subquestions = decompose(question, config)  # [question] when decomposition is off
    decompose_ms = elapsed_ms(start)

    search_ms = rerank_ms = 0.0
    fallback = False
    rankings = []
    for subquestion in subquestions:
        start = time.perf_counter()
        candidates = search(subquestion, config, query_filter)
        if query_filter is not None and not candidates:  # a wrong filter is worse than no filter
            fallback = True
            candidates = search(subquestion, config)
        search_ms += elapsed_ms(start)

        start = time.perf_counter()
        rankings.append(rerank(subquestion, candidates, config))  # each sub-question ranks its own candidates
        rerank_ms += elapsed_ms(start)

    chunks = rankings[0] if len(rankings) == 1 else interleave(rankings, config["search"]["top_k"])
    return {
        "chunks": chunks,
        "patient_ipp": patient_ipp,
        "fallback": fallback,
        "subquestions": subquestions,
        "timings": {"filter_ms": filter_ms, "decompose_ms": decompose_ms, "search_ms": search_ms, "rerank_ms": rerank_ms},
    }


def answer(question, config):
    """The full RAG: {answer, chunks, latency_ms, timings, prompt_tokens, completion_tokens, finish_reason}."""
    result = retrieve(question, config)

    start = time.perf_counter()
    generation = generate(question, result["chunks"], config)
    result["timings"]["generate_ms"] = elapsed_ms(start)

    return {
        **generation,
        **result,
        "latency_ms": sum(result["timings"].values()),
    }
