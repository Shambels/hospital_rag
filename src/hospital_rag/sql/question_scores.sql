-- Parameters: question id. Every score of one question, with the judge's explanation when there is one.
SELECT experiment_id, metric, value, reason
FROM scores
WHERE question_id = %s
ORDER BY experiment_id, metric;
