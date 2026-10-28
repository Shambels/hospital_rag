"""PostgreSQL: connection, tables, inserts. The only module that runs SQL; the queries are in sql/."""
from importlib.resources import files

import psycopg
from psycopg.types.json import Jsonb

from hospital_rag.config import settings


def load_sql(name):
    """Read a query from src/hospital_rag/sql/, wherever the program is started from."""
    return files("hospital_rag").joinpath("sql", f"{name}.sql").read_text(encoding="utf-8")


def connect():
    """Open a connection. Use it as `with connect() as conn:`: commit on success, rollback on error."""
    return psycopg.connect(settings.postgres_url)


def create_tables(conn):
    """Create the tables if they don't exist yet. Safe to call before every run."""
    conn.execute(load_sql("schema"))


def start_experiment(conn, name, config):
    """Insert an experiment with status 'running' and return its id."""
    row = conn.execute(load_sql("start_experiment"), (name, Jsonb(config))).fetchone()
    conn.commit()  # visible right away, even if the run crashes later
    return row[0]


def finish_experiment(conn, experiment_id, status, duration_s):
    """Mark an experiment as 'done' or 'failed' and record how long it took."""
    conn.execute(load_sql("finish_experiment"), (status, duration_s, experiment_id))
    conn.commit()


def insert_answers(conn, experiment_id, answers):
    """Insert dicts with keys question_id, chunks, answer and latency_ms."""
    with conn.cursor() as cur:
        cur.executemany(
            load_sql("insert_answers"),
            [(experiment_id, a["question_id"], Jsonb(a["chunks"]), a["answer"], a["latency_ms"]) for a in answers],
        )
    conn.commit()


def insert_scores(conn, experiment_id, scores):
    """Insert dicts with keys question_id, question_type, metric, value and reason (optional).

    A score already stored for the same question and metric is replaced: re-running the judge overwrites it.
    """
    with conn.cursor() as cur:
        cur.executemany(
            load_sql("insert_scores"),
            [
                (experiment_id, s["question_id"], s["question_type"], s["metric"], s["value"], s.get("reason"))
                for s in scores
            ],
        )
    conn.commit()


def compare_experiments(conn):
    """Mean of each metric for each experiment: rows of (name, metric, mean, n), best first."""
    return conn.execute(load_sql("compare_experiments")).fetchall()
