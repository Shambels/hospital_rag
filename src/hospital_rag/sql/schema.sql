-- The three tables of an experiment. IF NOT EXISTS: safe to run before every experiment.

CREATE TABLE IF NOT EXISTS experiments (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    config      JSONB NOT NULL,                  -- the whole TOML file, with the prompt text
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    duration_s  REAL,
    status      TEXT NOT NULL DEFAULT 'running'  -- running, done, failed
);

CREATE TABLE IF NOT EXISTS answers (
    experiment_id  INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    question_id    TEXT NOT NULL,
    chunks         JSONB NOT NULL,               -- document, chunk index, score and text of each retrieved chunk
    answer         TEXT,                         -- NULL when the experiment skips generation
    latency_ms     REAL,
    PRIMARY KEY (experiment_id, question_id)
);

CREATE TABLE IF NOT EXISTS scores (
    experiment_id  INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    question_id    TEXT NOT NULL,
    question_type  TEXT NOT NULL,
    metric         TEXT NOT NULL,                -- hit@5, mrr, recall@5, ndcg@5, faithfulness...
    value          REAL,                         -- NULL when the judge failed
    reason         TEXT,                         -- the judge's explanation, if any
    PRIMARY KEY (experiment_id, question_id, metric)
);
