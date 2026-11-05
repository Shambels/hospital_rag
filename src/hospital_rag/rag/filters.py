"""Metadata filters from the question (§6): which patient is it about? The LLM may extract, the code decides.

[filters] patient = "none" (default) | "names" (match the 60 known names, no LLM) | "llm" (the LLM extracts a name,
the code looks it up among the known patients). scope = "patient_or_notice" (default) | "patient_only".
"""
import json
import logging
import re
import unicodedata
from functools import cache

from qdrant_client import models

from hospital_rag.config import settings

logger = logging.getLogger(__name__)

NO_FILTER = {"patient": "none", "scope": "patient_or_notice"}  # configs without [filters] keep working unchanged

EXTRACTION_PROMPT = """Tu extrais le patient nommé dans une question sur des dossiers médicaux.
Réponds uniquement par un objet JSON : {"patient_name": "Prénom Nom"} si la question nomme un patient précis,
{"patient_name": null} sinon (question sur un médicament, sur plusieurs patients, ou sans nom)."""


def fold(text):
    """Lowercase without accents, so « Mme Michèle BERNARD » and « michele bernard » match."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()


def strip_civility(name):
    return re.sub(r"^(M\.|Mme)\s+", "", name.strip())


@cache
def known_patients():
    """{folded full name: IPP} for the 60 patients of hopital_documents.json. Full names: 8 last names are shared."""
    with open(settings.documents_path, encoding="utf-8") as f:
        return {fold(strip_civility(d["patient_name"])): d["patient_ipp"] for d in json.load(f) if d["patient_ipp"]}


def patient_from_names(question):
    """The IPP of the one known patient whose full name is in the question; None if there is none, or several."""
    text = fold(question)
    found = {ipp for name, ipp in known_patients().items() if re.search(rf"\b{re.escape(name)}\b", text)}
    return found.pop() if len(found) == 1 else None


def patient_from_llm(question, config):
    """The LLM extracts a name; it is kept only if it is a known patient. Never trust an identifier written by the LLM."""
    from hospital_rag.rag.generate import get_llm  # only this method needs the LLM

    response = get_llm().chat.completions.create(
        model=config["generation"]["model"],
        messages=[{"role": "system", "content": EXTRACTION_PROMPT}, {"role": "user", "content": question}],
        temperature=0.0,
        max_tokens=50,
        response_format={"type": "json_object"},
    )
    try:
        name = json.loads(response.choices[0].message.content).get("patient_name")
    except (json.JSONDecodeError, AttributeError):
        logger.warning("unreadable extraction for: %s", question)
        return None
    return known_patients().get(fold(strip_civility(name))) if isinstance(name, str) else None


def patient_filter(ipp, scope):
    """The Qdrant filter for one patient. patient_or_notice keeps the drug notices, which concern no patient:
    a multi_hop question needs the patient's record *and* a notice."""
    patient = models.FieldCondition(key="patient_ipp", match=models.MatchValue(value=ipp))
    if scope == "patient_only":
        return models.Filter(must=[patient])
    if scope == "patient_or_notice":
        notice = models.FieldCondition(key="doc_type", match=models.MatchValue(value="notice"))
        return models.Filter(should=[patient, notice])
    raise ValueError(f"unknown filter scope: {scope!r}")


def question_filter(question, config):
    """(Qdrant filter or None, the IPP extracted or None) for a question, as [filters] says."""
    params = {**NO_FILTER, **config.get("filters", {})}
    method = params["patient"]
    if method == "none":
        return None, None
    if method == "names":
        ipp = patient_from_names(question)
    elif method == "llm":
        ipp = patient_from_llm(question, config)
    else:
        raise ValueError(f"unknown patient extraction method: {method!r}")
    return (patient_filter(ipp, params["scope"]) if ipp else None), ipp
