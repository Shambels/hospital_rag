"""Header detection for strip_header: what is dropped, and what must be kept.

Run: uv run pytest
"""
from types import SimpleNamespace

from hospital_rag.etl.chunk import is_header


def chunk(text, headings):
    return SimpleNamespace(text=text, meta=SimpleNamespace(headings=headings))


HEADER = ("Centre Hospitalier de Valmont — Service de Néphrologie\n"
          "Patient : M. Bruno MARTIN, né le 16/10/1958 (66 ans) — IPP 4100002\nDate : 13/04/2025\n"
          "Rédigé par : Dr Anne Leclercq, cardiologue\n"
          "*Document fictif généré pour un cours. Patients, médecins et données sont inventés.*")


def test_a_docx_header_is_a_header():
    assert is_header(chunk(HEADER, ["Ordonnance"]))


def test_a_pdf_header_with_its_wrapped_disclaimer_is_a_header():
    pdf = ("Centre Hospitalier de Valmont - Service de Imagerie médicale Patient : Mme Marie MOREL, née le 22/04/1967 "
           "Date : 17/01/2026 Rédigé par : Dr X *Document fictif généré pour un cours. ne jamais utiliser\ncomme référence médicale.")
    assert is_header(chunk(pdf, ["Compte rendu de radiographie du bassin"]))


def test_a_pdf_section_with_the_title_as_only_heading_is_kept():
    # PDF sections come out with the title as their only heading: the heading test alone would drop them
    assert not is_header(chunk("Appendice augmenté de taille, mesuré à 12 mm de diamètre.", ["Compte rendu de scanner"]))


def test_a_section_is_kept_even_if_a_line_looks_like_a_header():
    assert not is_header(chunk("Date : 13/04/2025", ["Ordonnance", "Prescription"]))


def test_a_header_line_mixed_with_content_is_kept():
    assert not is_header(chunk(HEADER + "\nRamipril 5 mg : 1 comprimé le matin.", ["Ordonnance"]))
