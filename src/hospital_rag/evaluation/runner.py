"""Experiments: run one configuration on the eval set and store everything in PostgreSQL, then judge it.

Two passes, because the generator and the judge don't fit on the GPU together:
run_experiment (generator) -> answers + deterministic scores; judge_experiment (judge) -> judge scores.
"""
import copy
import json
import logging
import time

from hospital_rag import db
from hospital_rag.config import settings
from hospital_rag.etl.index import index_corpus
from hospital_rag.evaluation.metrics import answer_scores, retrieval_scores
from hospital_rag.rag.generate import load_system_prompt
from hospital_rag.rag.pipeline import answer, retrieve

logger = logging.getLogger(__name__)

STORED_CHUNK_FIELDS = ["document", "chunk_index", "search_score", "rerank_score", "text"]


def load_questions():
    with open(settings.eval_path, encoding="utf-8") as f:
        return json.load(f)


def stored_config(config):
    """The config as stored with the experiment: the TOML, plus the prompt text (the file may change later)."""
    stored = copy.deepcopy(config)
    if config["generation"]["enabled"]:
        stored["generation"]["system_prompt_text"] = load_system_prompt(config)
    return stored


def score_rows(question, scores):
    """{metric: value} or {metric: (value, reason)} -> rows for db.insert_scores()."""
    rows = []
    for metric, value in scores.items():
        value, reason = value if isinstance(value, tuple) else (value, None)
        rows.append({"question_id": question["id"], "question_type": question["type"],
                     "metric": metric, "value": value, "reason": reason})
    return rows


def run_experiment(config):
    """Every question of the eval set through the RAG: answers, retrieval scores and refusals. Returns the experiment id."""
    index_corpus(config)  # the collection must exist and be complete; nothing is rebuilt if it is
    generating = config["generation"]["enabled"]
    questions = load_questions()
    top_k = config["search"]["top_k"]

    with db.connect() as conn:
        db.create_tables(conn)
        experiment_id = db.start_experiment(conn, config["name"], stored_config(config))
        logger.info("experiment %d: %s (%d questions, generation %s)",
                    experiment_id, config["name"], len(questions), "on" if generating else "off")
        start = time.time()
        try:
            for i, question in enumerate(questions, 1):
                result = answer(question["question"], config) if generating else retrieve(question["question"], config)
                chunks = [{field: chunk.get(field) for field in STORED_CHUNK_FIELDS} for chunk in result["chunks"]]
                latency_ms = result["latency_ms"] if generating else sum(result["timings"].values())
                db.insert_answers(conn, experiment_id, [{
                    "question_id": question["id"], "chunks": chunks,
                    "answer": result["answer"] if generating else None, "latency_ms": latency_ms,
                }])
                scores = retrieval_scores(result["chunks"], question, top_k)
                if generating:
                    scores |= answer_scores(result["answer"])
                db.insert_scores(conn, experiment_id, score_rows(question, scores))
                logger.info("[%d/%d] %s %s %.0f ms", i, len(questions), question["id"],
                            f"recall@5={scores['recall@5']:.2f}" if "recall@5" in scores else "", latency_ms)
        except BaseException:  # Ctrl+C included: the experiment must not stay 'running' forever
            db.finish_experiment(conn, experiment_id, "failed", time.time() - start)
            raise
        db.finish_experiment(conn, experiment_id, "done", time.time() - start)
        logger.info("experiment %d done in %.0f s", experiment_id, time.time() - start)
    return experiment_id


def judge_experiment(experiment_id, judge_model=None, force=False):
    """Judge the stored answers of an experiment on the 26-question sample. Resumes where an interrupted pass stopped."""
    from hospital_rag.evaluation.judge import METRIC_NAMES, judge_answer, judge_sample, metric_name  # loads DeepEval

    with db.connect() as conn:
        row = db.load_experiment(conn, experiment_id)
        if row is None:
            raise ValueError(f"no experiment {experiment_id}")
        _, name, _, status = row
        if status != "done":
            raise ValueError(f"experiment {experiment_id} ({name}) is {status!r}, not 'done'")
        answers = db.load_answers(conn, experiment_id)
        done = set() if force else db.load_score_keys(conn, experiment_id)
        metrics = [metric_name(m, judge_model) for m in METRIC_NAMES]

        sample = judge_sample(load_questions())
        if all(answers.get(q["id"], (None, None))[1] is None for q in sample):
            raise ValueError(f"experiment {experiment_id} ({name}) has no generated answers: nothing to judge")
        start = time.time()
        for i, question in enumerate(sample, 1):
            if all((question["id"], m) in done for m in metrics):
                logger.info("[%d/%d] %s already judged", i, len(sample), question["id"])
                continue
            chunks, generated = answers.get(question["id"], (None, None))
            if generated is None:
                logger.warning("[%d/%d] %s has no answer: skipped", i, len(sample), question["id"])
                continue
            scores = judge_answer(question, generated, chunks, judge_model)
            db.insert_scores(conn, experiment_id, score_rows(question, scores))
            logger.info("[%d/%d] %s %s", i, len(sample), question["id"],
                        " ".join(f"{m}={'–' if v is None else f'{v:.2f}'}" for m, (v, _) in scores.items()))
        logger.info("experiment %d (%s) judged in %.0f s", experiment_id, name, time.time() - start)
