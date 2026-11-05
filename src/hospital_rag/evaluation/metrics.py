"""Retrieval metrics against the golden dataset: no LLM, and the same ground truth for every chunking.

A question's evidence lists, document by document, the snippets that contain the answer:
grade 2 = contains the answer or a necessary part of it (with a fact number), grade 1 = useful, not necessary.
Same fact number = interchangeable sources; different numbers = complementary facts, all needed.
"""
import math
import re
import unicodedata

K_VALUES = (3, 5, 10)  # computed for every k <= [search] top_k, so experiments with different top_k stay comparable


def normalize(text):
    """Unicode NFC and single spaces: a chunk boundary or an overlap can change whitespace, never the words."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def judge_chunk(chunk, question):
    """Indices of the question's evidence items this chunk proves: the right document and every snippet in its text.

    The document must match: six patients share the same drugs, the same snippet in another record proves nothing.
    """
    text = normalize(chunk["text"])
    return [
        i for i, evidence in enumerate(question["evidence"])
        if evidence["document"] == chunk["document"] and all(normalize(s) in text for s in evidence["snippets"])
    ]


def hit(matches, evidence, k):
    """1 if a chunk of the top k contains the answer or a necessary part of it (grade 2), else 0."""
    return float(any(evidence[i]["grade"] == 2 for ranked in matches[:k] for i in ranked))


def mrr(matches, evidence, k):
    """1 / rank of the first grade-2 chunk in the top k, 0 if none: is the best chunk at the top?"""
    for rank, ranked in enumerate(matches[:k], 1):
        if any(evidence[i]["grade"] == 2 for i in ranked):
            return 1 / rank
    return 0.0


def recall(matches, evidence, k):
    """Share of the needed facts found in the top k. A fact counts once, whichever of its sources was found."""
    needed = {e["fact"] for e in evidence if e["grade"] == 2}
    found = {evidence[i]["fact"] for ranked in matches[:k] for i in ranked if evidence[i]["grade"] == 2}
    return len(found) / len(needed)


def ndcg(matches, evidence, k):
    """Ranking quality with graded relevance: gain = grade / log2(rank + 1), divided by the best possible ranking.

    Each evidence item is credited once and each chunk brings at most one: two overlapping chunks proving the same
    item don't count twice, so the score stays between 0 and 1.
    """
    credited, dcg = set(), 0.0
    for rank, ranked in enumerate(matches[:k], 1):
        new = [i for i in ranked if i not in credited]
        if new:
            best = max(new, key=lambda i: evidence[i]["grade"])
            credited.add(best)
            dcg += evidence[best]["grade"] / math.log2(rank + 1)
    ideal = sorted((e["grade"] for e in evidence), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return dcg / idcg


METRICS = {"hit": hit, "mrr": mrr, "recall": recall, "ndcg": ndcg}


def is_refusal(answer):
    """The system prompt asks for this exact sentence when the passages do not answer the question."""
    return "je ne sais pas" in answer.lower()


def filter_scores(extracted_ipp, question, fallback):
    """How well the patient was extracted, against the question's patient_ipp (every question, when a filter is on).

    filter_correct = the right patient, or none for a question without one; filter_wrong = another patient (the worst
    case: the right records are excluded); filter_missed = the question names a patient, none was extracted.
    """
    expected = question["patient_ipp"]
    return {
        "filter_correct": float(extracted_ipp == expected),
        "filter_wrong": float(extracted_ipp is not None and extracted_ipp != expected),
        "filter_missed": float(expected is not None and extracted_ipp is None),
        "filter_fallback": float(fallback),
    }


def cited_passages(answer):
    """Numbers of the passages cited in the answer, written like [2]."""
    return {int(number) for number in re.findall(r"\[(\d+)\]", answer)}


def answer_scores(answer, chunks, question):
    """Deterministic answer checks.

    refusal (every question) = 1 if the answer is a refusal: its mean is the right-refusal rate on out_of_corpus
    questions and the wrong-refusal rate on the others. Only reliable with prompts that impose the sentence.
    When the answer cites passages ([2]): citation_valid = share of cited numbers that exist (1..number of chunks),
    citation_relevant = share of the existing cited chunks that prove part of the answer (judge_chunk, any grade).
    """
    scores = {"refusal": float(is_refusal(answer))}
    cited = cited_passages(answer)
    if cited:
        existing = [n for n in cited if 1 <= n <= len(chunks)]
        scores["citation_valid"] = len(existing) / len(cited)
        if existing and question["evidence"]:
            scores["citation_relevant"] = sum(bool(judge_chunk(chunks[n - 1], question)) for n in existing) / len(existing)
    return scores


def retrieval_scores(chunks, question, top_k):
    """The scores of one question, as {metric: value} (e.g. "recall@5"). Empty for a question without evidence."""
    evidence = question["evidence"]
    if not evidence:  # out_of_corpus: nothing to find, these questions test refusals instead
        return {}
    matches = [judge_chunk(chunk, question) for chunk in chunks]
    return {
        f"{name}@{k}": metric(matches, evidence, k)
        for k in K_VALUES if k <= top_k
        for name, metric in METRICS.items()
    }
