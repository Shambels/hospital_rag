"""Merging the rankings of several sub-questions.

Run: uv run pytest
"""
from hospital_rag.rag.decompose import decompose, interleave


def chunk(document, index=0):
    return {"document": document, "chunk_index": index}


def test_interleave_takes_the_best_of_each_in_turn():
    record = [chunk("dossier", 1), chunk("dossier", 2), chunk("dossier", 3)]
    notice = [chunk("notice", 1), chunk("notice", 2)]
    merged = interleave([record, notice], top_k=4)
    assert merged == [chunk("dossier", 1), chunk("notice", 1), chunk("dossier", 2), chunk("notice", 2)]


def test_interleave_skips_duplicates():
    a = [chunk("x", 1), chunk("y", 1)]
    b = [chunk("x", 1), chunk("z", 1)]
    assert interleave([a, b], top_k=5) == [chunk("x", 1), chunk("y", 1), chunk("z", 1)]


def test_decomposition_off_by_default():
    # no LLM call: configs without [decomposition] keep one search per question
    assert decompose("Quelle dose ?", {"generation": {"model": "x"}}) == ["Quelle dose ?"]


class FakeLLM:
    """Stands for the generator: answers each call with the next prepared JSON, and records the system prompts."""

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []
        self.chat = self.completions = self

    def create(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        message = type("Message", (), {"content": self.answers.pop(0)})
        return type("Response", (), {"choices": [type("Choice", (), {"message": message})]})


GATED = {"generation": {"model": "x"}, "decomposition": {"enabled": True, "gate": True}}


def test_gate_keeps_a_one_source_question_whole(monkeypatch):
    llm = FakeLLM('{"deux_sources": false}')
    monkeypatch.setattr("hospital_rag.rag.decompose.get_llm", lambda: llm)
    assert decompose("Pourquoi la dose a-t-elle été réduite ?", GATED) == ["Pourquoi la dose a-t-elle été réduite ?"]
    assert len(llm.prompts) == 1  # the decomposition itself is never called


def test_gate_lets_a_two_source_question_be_split(monkeypatch):
    llm = FakeLLM('{"deux_sources": true}', '{"questions": ["Traitement de M. X ?", "Précautions du médicament Y ?"]}')
    monkeypatch.setattr("hospital_rag.rag.decompose.get_llm", lambda: llm)
    assert decompose("Peut-on donner Y à M. X ?", GATED) == ["Traitement de M. X ?", "Précautions du médicament Y ?"]


def test_unusable_gate_answer_means_no_split(monkeypatch):
    # a "yes" in words isn't the JSON asked for: the safe default is the whole question
    monkeypatch.setattr("hospital_rag.rag.decompose.get_llm", lambda: FakeLLM("oui"))
    assert decompose("Peut-on donner Y à M. X ?", GATED) == ["Peut-on donner Y à M. X ?"]


def test_without_gate_every_question_goes_to_the_decomposition(monkeypatch):
    llm = FakeLLM('{"questions": ["A ?", "B ?"]}')
    monkeypatch.setattr("hospital_rag.rag.decompose.get_llm", lambda: llm)
    assert decompose("A et B ?", {"generation": {"model": "x"}, "decomposition": {"enabled": True}}) == ["A ?", "B ?"]
