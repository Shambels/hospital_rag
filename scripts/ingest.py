"""Convert the corpus (once), then chunk and index it for one or more configurations.

Usage: uv run python scripts/ingest.py experiments/baseline.toml [experiments/other.toml ...] [--force]
"""
import argparse
import sys
import tomllib

from hospital_rag.etl.convert import convert_documents
from hospital_rag.etl.index import index_corpus
from hospital_rag.logs import setup_logging

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("configs", nargs="+", help="experiment TOML files")
parser.add_argument("--force", action="store_true", help="rebuild the collections even if they are complete")
args = parser.parse_args()

setup_logging()

failures = convert_documents()
if failures:
    sys.exit(f"{len(failures)} documents failed to convert, nothing indexed: {failures}")

for path in args.configs:
    with open(path, "rb") as f:
        config = tomllib.load(f)
    stats = index_corpus(config, force=args.force)
    print(f"{path}: {stats['collection']} — {stats['chunks']} chunks, "
          f"{'indexed in %.1f s' % stats['seconds'] if stats['created'] else 'already indexed'}")
