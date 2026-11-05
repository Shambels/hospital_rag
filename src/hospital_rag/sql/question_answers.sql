-- Parameters: question id. The answer, chunks and latency of every finished experiment for one question.
SELECT e.id, e.name, a.answer, a.chunks, a.latency_ms
FROM answers a
JOIN experiments e ON e.id = a.experiment_id
WHERE a.question_id = %s AND e.status = 'done'
ORDER BY e.id;
