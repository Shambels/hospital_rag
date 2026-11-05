"""Question decomposition (§8): a question that needs several facts -> self-contained sub-questions, one search each.

For multi_hop questions (a fact of the record + a fact of a notice), the reranker scores each chunk against the
whole question and favours one side; each sub-question gets its own search instead.
[decomposition] enabled = false (default) | true; max_subquestions (default 3).
"""
import json
import logging

from hospital_rag.rag.generate import get_llm

logger = logging.getLogger(__name__)

PROMPT = """Tu prépares la recherche documentaire d'un assistant hospitalier (dossiers patients et notices de médicaments).
Si la question demande plusieurs informations de sources différentes (par exemple le traitement d'un patient dans
son dossier ET une précaution d'emploi dans la notice du médicament), découpe-la en sous-questions autonomes, au plus {n}.
Chaque sous-question doit se comprendre seule : répète le nom du patient ou du médicament quand il le faut.
Si la question ne demande qu'une information, renvoie-la telle quelle, seule.
Réponds uniquement par un objet JSON : {{"questions": ["...", "..."]}}"""


def decompose(question, config):
    """The sub-questions of a question; [question] itself when it needs one search or the LLM's answer is unusable."""
    params = config.get("decomposition", {})
    if not params.get("enabled", False):
        return [question]
    n = params.get("max_subquestions", 3)
    response = get_llm().chat.completions.create(
        model=config["generation"]["model"],
        messages=[{"role": "system", "content": PROMPT.format(n=n)}, {"role": "user", "content": question}],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    try:
        subquestions = json.loads(response.choices[0].message.content)["questions"]
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("unusable decomposition for: %s", question)
        return [question]
    subquestions = [q.strip() for q in subquestions if isinstance(q, str) and q.strip()][:n]
    return subquestions or [question]


def interleave(rankings, top_k):
    """Merge several rankings into top_k chunks: the best of each sub-question in turn, without duplicates."""
    merged, seen = [], set()
    for rank in range(max(map(len, rankings), default=0)):
        for ranking in rankings:
            if rank < len(ranking):
                chunk = ranking[rank]
                key = (chunk["document"], chunk["chunk_index"])
                if key not in seen:
                    seen.add(key)
                    merged.append(chunk)
                if len(merged) == top_k:
                    return merged
    return merged
