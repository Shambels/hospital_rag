"""Docling: convert the corpus (DOCX, PDF) to DoclingDocument JSON files, once."""
import logging

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from hospital_rag.config import ocr_settings, settings

logger = logging.getLogger(__name__)

SUFFIXES = {".docx", ".pdf"}


def get_converter(ocr=False):
    """A converter for DOCX and PDF that only uses the local Docling models.

    The PDFs of the corpus have a text layer, so OCR is off by default: Docling reads the text directly.
    """
    pdf_options = PdfPipelineOptions(
        artifacts_path=settings.docling_models_path,  # layout and table models: no download
        do_ocr=ocr,
        do_table_structure=True,
        accelerator_options=AcceleratorOptions(
            device=ocr_settings.ocr_device,
            num_threads=ocr_settings.ocr_num_threads,
        ),
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.DOCX, InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
    )


def conversion_path(document):
    """Where the conversion of a document is stored: data/conversions/{name without extension}.json."""
    return settings.conversions_path / f"{document.stem}.json"


def convert_documents(force=False, ocr=False):
    """Convert every document of the corpus that has no conversion yet. Returns the names of the failures."""
    settings.conversions_path.mkdir(parents=True, exist_ok=True)
    documents = sorted(p for p in settings.corpus_path.iterdir() if p.suffix.lower() in SUFFIXES)
    todo = [p for p in documents if force or not conversion_path(p).exists()]
    logger.info("%d documents, %d to convert", len(documents), len(todo))
    if not todo:
        return []

    converter = get_converter(ocr=ocr)
    failures = []
    for i, document in enumerate(todo, 1):
        try:
            result = converter.convert(document)
        except Exception:
            logger.exception("conversion failed: %s", document.name)
            failures.append(document.name)
            continue
        target = conversion_path(document)
        tmp = target.with_suffix(".json.tmp")
        result.document.save_as_json(tmp)
        tmp.replace(target)  # atomic: a crash never leaves a half-written file that would be skipped next time
        logger.info("[%d/%d] %s", i, len(todo), document.name)
    return failures


def load_document(name):
    """The converted DoclingDocument of a document, by file name (e.g. "4100002_2025-04-13_ordonnance.docx")."""
    return DoclingDocument.load_from_json(settings.conversions_path / f"{name.rsplit('.', 1)[0]}.json")
