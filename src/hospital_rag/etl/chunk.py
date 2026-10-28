"""Chunking: converted documents -> chunks ready to embed. Depends on [chunking] and [embedder] of the config."""
import json
from datetime import date
from functools import cache

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from hospital_rag.config import settings
from hospital_rag.etl.convert import load_document


@cache
def get_tokenizer(model):
    """The embedder's own tokenizer: chunk sizes are counted in the tokens the embedder will see."""
    return AutoTokenizer.from_pretrained(settings.models_path / "embedder" / model)


def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_metadata():
    """The record of each document in hopital_documents.json, by file name."""
    with open(settings.documents_path, encoding="utf-8") as f:
        return {m["document"]: m for m in json.load(f)}


def patient_line(meta):
    """Who and what the document is about, e.g. "Patient : M. Bruno MARTIN (IPP 4100002) — Ordonnance du 13/04/2025".

    Built from hopital_documents.json, which is reliable for every format (the PDF headers are merged into one
    paragraph by Docling). None for a drug notice: no patient, and its headings already name the drug.
    """
    if meta["patient_ipp"] is None:
        return None
    day = date.fromisoformat(meta["date"]).strftime("%d/%m/%Y")  # the format used in the documents
    return f"Patient : {meta['patient_name']} (IPP {meta['patient_ipp']}) — {meta['title']} du {day}"


def add_overlap(texts, tokenizer, overlap):
    """Prefix each chunk of a document with the last `overlap` tokens of the previous chunk."""
    result = []
    for i, text in enumerate(texts):
        if i == 0 or overlap == 0:
            result.append(text)
        else:
            previous_ids = tokenizer.encode(texts[i - 1], add_special_tokens=False)
            result.append(tokenizer.decode(previous_ids[-overlap:]) + " " + text)
    return result


def chunk_document(meta, config):
    """The chunks of one document: dicts with the document's metadata, chunk_index and text.

    text = patient line + headings + (overlap + chunk text), and stays under [chunking] max_tokens.
    """
    tokenizer = get_tokenizer(config["embedder"]["model"])
    max_tokens = config["chunking"]["max_tokens"]
    overlap = config["chunking"]["overlap"]
    line = patient_line(meta) if config["chunking"]["patient_line"] else None

    # HybridChunker counts headings + text; the overlap and the patient line are added afterwards, so reserve them
    reserved = overlap + (count_tokens(line, tokenizer) + 1 if line else 0)  # +1: the newline
    chunker = HybridChunker(tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=max_tokens - reserved))
    chunks = list(chunker.chunk(load_document(meta["document"])))

    # Overlap on the chunk text only, so a heading is never repeated in the middle of the next chunk
    texts = add_overlap([chunk.text for chunk in chunks], tokenizer, overlap)
    return [
        {**meta, "chunk_index": i, "text": "\n".join(p for p in [line, *(chunk.meta.headings or []), text] if p)}
        for i, (chunk, text) in enumerate(zip(chunks, texts))
    ]


def chunk_corpus(config):
    """The chunks of every document of the corpus, in the order of hopital_documents.json."""
    return [chunk for meta in load_metadata().values() for chunk in chunk_document(meta, config)]
