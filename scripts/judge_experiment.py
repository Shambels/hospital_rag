"""Judge the stored answers of experiments with the LLM judge (26-question sample). Needs the judge on JUDGE_URL.

An experiment is given by its id, or by its name (its most recent finished run).
An interrupted pass resumes where it stopped; --force judges everything again.

Usage:
  uv run python scripts/judge_experiment.py baseline
  uv run python scripts/judge_experiment.py 12 13 --judge-model gemma4-26b-a4b
"""
import argparse
import sys

from hospital_rag import db
from hospital_rag.evaluation.runner import judge_experiment
from hospital_rag.logs import setup_logging

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("experiments", nargs="+", help="experiment ids or names")
parser.add_argument("--judge-model", help="served name of another judge (default: JUDGE_MODEL)")
parser.add_argument("--force", action="store_true", help="judge again the questions already judged")
args = parser.parse_args()
setup_logging()

with db.connect() as conn:
    ids = [int(e) if e.isdigit() else db.find_experiment(conn, e) for e in args.experiments]
for given, experiment_id in zip(args.experiments, ids):
    if experiment_id is None:
        sys.exit(f"no finished experiment named {given!r}")
    judge_experiment(experiment_id, judge_model=args.judge_model, force=args.force)
