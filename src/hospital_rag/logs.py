"""Logging for the command-line entry points: our progress lines, without the libraries' noise."""
import logging
import warnings


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ["httpx", "httpx2", "docling", "deepeval"]:  # one line per HTTP request, per Docling stage...
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # "Token indices sequence length is longer than...": HybridChunker tokenizes whole documents to measure them
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")  # local Qdrant, plain http
