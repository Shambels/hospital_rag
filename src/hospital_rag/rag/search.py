"""Search: question -> candidate chunks from Qdrant, with dense, sparse (BM25) or hybrid search."""
from qdrant_client import models

from hospital_rag.etl.index import bm25, collection_name, embed, get_qdrant


def query_text(question, config):
    """Harrier and Qwen3-Embedding expect a task instruction before the question; documents get none."""
    instruction = config["embedder"]["query_instruction"]
    return f"Instruct: {instruction}\nQuery: {question}" if instruction else question


def fusion_query(search):
    """How the hybrid search merges the dense and sparse rankings."""
    if search["fusion"] == "rrf":
        return models.RrfQuery(rrf=models.Rrf(k=search["rrf_k"]))
    if search["fusion"] == "dbsf":
        return models.FusionQuery(fusion=models.Fusion.DBSF)
    raise ValueError(f"unknown fusion: {search['fusion']!r}")


def search(question, config, query_filter=None):
    """The [search] candidates best chunks for a question: payload dicts plus the search score, best first.

    query_filter (a models.Filter, see filters.py) also applies to the prefetches of the hybrid search.
    """
    params = config["search"]
    method = params["method"]
    limit = params["candidates"]
    # The BM25 vector uses the bare question: the instruction is for the embedder, its words would only add noise
    sparse = bm25(question)

    if method == "sparse":
        query, using, prefetch = sparse, "sparse", None
    else:
        dense = embed([query_text(question, config)], config["embedder"]["model"])[0]
        if method == "dense":
            query, using, prefetch = dense, "dense", None
        elif method == "hybrid":
            query, using = fusion_query(params), None
            prefetch = [
                models.Prefetch(query=dense, using="dense", limit=limit),
                models.Prefetch(query=sparse, using="sparse", limit=limit),
            ]
        else:
            raise ValueError(f"unknown search method: {method!r}")

    points = get_qdrant().query_points(
        collection_name(config),
        query=query,
        using=using,
        prefetch=prefetch,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points
    return [{**point.payload, "search_score": point.score} for point in points]
