"""Reranking: the search candidates -> the [search] top_k best, scored by a cross-encoder."""
from functools import cache

import requests

from hospital_rag.config import settings

NO_RERANKER = "none"  # [reranker] model = "none": keep the search order, to measure what the reranker brings


@cache
def get_session():
    """One HTTP session for every call: the connection to the server is reused."""
    return requests.Session()


def rerank(question, chunks, config):
    """The top_k chunks the reranker finds most relevant, best first, each with its rerank_score added."""
    model = config["reranker"]["model"]
    top_k = config["search"]["top_k"]
    if model == NO_RERANKER or not chunks:
        return chunks[:top_k]

    response = get_session().post(
        settings.rerank_url,
        # The full chunk text, patient line and headings included: they tell the reranker whose record it is
        json={"model": model, "query": question, "documents": [chunk["text"] for chunk in chunks], "top_n": top_k},
        timeout=60,
    )
    response.raise_for_status()
    return [
        {**chunks[result["index"]], "rerank_score": result["relevance_score"]}
        for result in response.json()["results"]  # already sorted, best first
    ]
