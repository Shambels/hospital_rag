"""Compute the redundancy metrics of stored runs from their stored chunks. Needs PostgreSQL only.

An experiment is given by its id, or by its name (all its finished runs); --all for every finished run.

Usage:
  uv run python scripts/rescore.py --all
  uv run python scripts/rescore.py baseline 39
"""
import argparse

from hospital_rag import db
from hospital_rag.evaluation.runner import rescore_redundancy
from hospital_rag.logs import setup_logging

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("experiments", nargs="*", help="experiment ids or names")
parser.add_argument("--all", action="store_true", help="every finished experiment")
args = parser.parse_args()
if not args.all and not args.experiments:
    parser.error("give experiments or --all")
setup_logging()

with db.connect(read_only=True) as conn:
    runs = [(eid, name) for eid, name, _, _, _, status in db.list_experiments(conn) if status == "done"]
ids = [eid for eid, name in runs if args.all or str(eid) in args.experiments or name in args.experiments]
for experiment_id in ids:
    rescore_redundancy(experiment_id)
print(f"{len(ids)} runs rescored")
