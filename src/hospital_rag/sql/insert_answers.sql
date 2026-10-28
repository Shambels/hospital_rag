-- Parameters: experiment_id, question_id, chunks (JSONB), answer, latency_ms.
INSERT INTO answers (experiment_id, question_id, chunks, answer, latency_ms)
VALUES (%s, %s, %s, %s, %s);
