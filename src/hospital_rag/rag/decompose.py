"""Question decomposition (§8): a question that needs several facts -> self-contained sub-questions, one search each.

For multi_hop questions (a fact of the record + a fact of a notice), the reranker scores each chunk against the
whole question and favours one side; each sub-question gets its own search instead.
[decomposition] enabled = false (default) | true; max_subquestions (default 3);
gate = false (default) | true: split only the questions that need a patient's record AND a drug notice
(PLAN.md, *Gated decomposition*).
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


# The gate (PLAN.md, *Gated decomposition*): a yes/no question asked before splitting. Without it, the decomposition
# prompt splits 45 of the 106 questions (any question naming a patient and a drug), and the split loses the evidence
# of one-source questions (Q052). The criterion is the corpus's structure: a second fact means a second *kind* of
# source, record + notice.
GATE_PROMPT = """Tu prépares la recherche documentaire d'un assistant hospitalier. Le corpus contient deux sortes de
documents : les dossiers des patients (comptes rendus, ordonnances, bilans) et les notices des médicaments (indications,
posologie, contre-indications, interactions, effets indésirables, précautions d'emploi).
Pour répondre à la question, faut-il combiner une information du dossier d'un patient ET une information générale de la
notice d'un médicament ?
- Oui : par exemple, savoir si un médicament convient à ce patient compte tenu de son état ou de son traitement, ou
  connaître les effets indésirables, la surveillance ou les précautions d'un médicament qu'il reçoit.
- Non : quand tout se trouve dans le dossier du patient (un traitement, une dose, une date, un résultat, la raison d'une
  modification), ou tout dans les notices (aucun patient en jeu).
Réponds uniquement par un objet JSON : {"deux_sources": true} ou {"deux_sources": false}"""


def needs_two_sources(question, config):
    """The gate: True when the question needs a patient's record and a drug notice. False if the answer is unusable:
    the whole question, searched once, is the safe default (the best configuration without decomposition)."""
    response = get_llm().chat.completions.create(
        model=config["generation"]["model"],
        messages=[{"role": "system", "content": GATE_PROMPT}, {"role": "user", "content": question}],
        temperature=0.0,
        max_tokens=20,
        response_format={"type": "json_object"},
    )
    try:
        verdict = json.loads(response.choices[0].message.content)["deux_sources"]
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("unusable gate answer for: %s", question)
        return False
    return verdict is True


def decompose(question, config):
    """The sub-questions of a question; [question] itself when it needs one search or the LLM's answer is unusable."""
    params = config.get("decomposition", {})
    if not params.get("enabled", False):
        return [question]
    if params.get("gate", False) and not needs_two_sources(question, config):
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
