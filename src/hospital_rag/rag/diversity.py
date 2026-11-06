"""Diversity (deduplicate_plan.md, step 3): choose the top_k chunks among the reranked candidates so that one
document can't fill them. Off unless the experiment has a [diversity] section:

[diversity]
method = "cap"              # none (default) or cap
max_per_document = 2

Runs after the reranker (and after the interleaving of the sub-questions): the reranker stays the judge of relevance,
this step only skips the chunks of a document that already has its share.
"""

NONE = "none"


def enabled(config):
    return config.get("diversity", {}).get("method", NONE) != NONE


def cap_per_document(ranked, top_k, max_per_document):
    """The top_k chunks in ranked order, at most max_per_document per document.

    If the cap leaves fewer than top_k chunks (every candidate from one or two documents), the skipped ones fill the
    gap, in ranked order: the cap never gives the LLM fewer chunks than without it.
    """
    kept, skipped, per_document = [], [], {}
    for chunk in ranked:
        document = chunk["document"]
        if per_document.get(document, 0) < max_per_document:
            per_document[document] = per_document.get(document, 0) + 1
            kept.append(chunk)
            if len(kept) == top_k:
                return kept
        else:
            skipped.append(chunk)
    return kept + skipped[:top_k - len(kept)]


def diversify(ranked, config):
    """The top_k chunks the LLM gets, from a ranking longer than top_k (all the reranked candidates)."""
    top_k = config["search"]["top_k"]
    params = config.get("diversity", {})
    method = params.get("method", NONE)
    if method == NONE:
        return ranked[:top_k]
    if method == "cap":
        return cap_per_document(ranked, top_k, params.get("max_per_document", 2))
    raise ValueError(f"unknown [diversity] method: {method!r}")
