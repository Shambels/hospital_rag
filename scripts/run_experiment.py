"""Run one experiment: retrieval, generation, evaluation, results in PostgreSQL.

Usage: uv run python scripts/run_experiment.py experiments/baseline.toml
"""
import sys
import tomllib

from hospital_rag.evaluation.runner import run_experiment

with open(sys.argv[1], "rb") as f:
    config = tomllib.load(f)
run_experiment(config)