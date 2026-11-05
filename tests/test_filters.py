"""Patient extraction without LLM, on the real patient list (data/hopital_documents.json).

Run: uv run pytest
"""
from hospital_rag.evaluation.metrics import filter_scores
from hospital_rag.rag.filters import patient_filter, patient_from_names


def test_full_name_gives_the_ipp():
    assert patient_from_names("Quelle dose de ramipril prend actuellement M. Bruno Martin ?") == "4100002"


def test_case_and_accents_dont_matter():
    assert patient_from_names("quelle dose prend MME MICHELE BERNARD ?") == "4100003"


def test_last_name_alone_is_ambiguous():
    # 8 last names are shared between patients: a last name alone must not pick one
    assert patient_from_names("Quelle dose prend M. Martin ?") is None


def test_unknown_patient_gives_no_filter():
    assert patient_from_names("Quel traitement a été prescrit à M. Kevin Durand pour son asthme ?") is None


def test_two_patients_give_no_filter():
    assert patient_from_names("M. Bruno Martin et Mme Michèle Bernard prennent-ils du ramipril ?") is None


def test_question_without_patient():
    assert patient_from_names("Quels sont les effets indésirables de la metformine ?") is None


def test_patient_or_notice_keeps_the_notices():
    f = patient_filter("4100002", "patient_or_notice")
    assert f.must is None and len(f.should) == 2  # this patient's records OR any drug notice


def test_filter_scores():
    question = {"patient_ipp": "4100002"}
    assert filter_scores("4100002", question, False) == {
        "filter_correct": 1.0, "filter_wrong": 0.0, "filter_missed": 0.0, "filter_fallback": 0.0}
    assert filter_scores("4100003", question, False)["filter_wrong"] == 1.0
    assert filter_scores(None, question, False)["filter_missed"] == 1.0
    assert filter_scores(None, {"patient_ipp": None}, False)["filter_correct"] == 1.0
