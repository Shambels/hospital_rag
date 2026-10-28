"""The RAG: question -> chunks -> answer. Scripts, evaluation and the API call this module, never the steps directly."""
import time

from hospital_rag.rag.generate import generate
from hospital_rag.rag.rerank import rerank
from hospital_rag.rag.search import search


def elapsed_ms(start):
    return (time.perf_counter() - start) * 1000


def retrieve(question, config):
    """The top_k chunks for a question (search, then rerank), with the time of each stage.

    Enough on its own for the retrieval metrics: experiments that only change retrieval skip generation.
    """
    start = time.perf_counter()
    candidates = search(question, config)
    search_ms = elapsed_ms(start)

    start = time.perf_counter()
    chunks = rerank(question, candidates, config)
    rerank_ms = elapsed_ms(start)
    return {"chunks": chunks, "timings": {"search_ms": search_ms, "rerank_ms": rerank_ms}}


def answer(question, config):
    """The full RAG: {answer, chunks, latency_ms, timings, prompt_tokens, completion_tokens, finish_reason}."""
    result = retrieve(question, config)

    start = time.perf_counter()
    generation = generate(question, result["chunks"], config)
    result["timings"]["generate_ms"] = elapsed_ms(start)

    return {
        **generation,
        "chunks": result["chunks"],
        "timings": result["timings"],
        "latency_ms": sum(result["timings"].values()),
    }
