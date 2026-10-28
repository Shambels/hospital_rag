-- Parameters: experiment_id, question_id, question_type, metric, value, reason.
-- A score already stored for the same question and metric is replaced: re-running the judge overwrites it.
INSERT INTO scores (experiment_id, question_id, question_type, metric, value, reason)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (experiment_id, question_id, metric)
DO UPDATE SET value = EXCLUDED.value, reason = EXCLUDED.reason;
