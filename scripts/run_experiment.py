"""Run one or more experiments: retrieval, generation, deterministic scores, results in PostgreSQL.

Needs the embedder, the reranker and (unless [generation] enabled = false) the generator.
Then judge the answers with scripts/judge_experiment.py, once the judge has replaced the generator.

Usage: uv run python scripts/run_experiment.py experiments/baseline.toml [experiments/other.toml ...]
"""
import argparse
import tomllib

from hospital_rag.evaluation.runner import run_experiment
from hospital_rag.logs import setup_logging

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("configs", nargs="+", help="experiment TOML files")
args = parser.parse_args()
setup_logging()

for path in args.configs:
    with open(path, "rb") as f:
        config = tomllib.load(f)
    print(f"{path}: experiment {run_experiment(config)}")
