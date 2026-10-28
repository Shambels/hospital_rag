-- Parameters: experiment id. Returns question_id, chunks, answer for each question.
SELECT question_id, chunks, answer
FROM answers
WHERE experiment_id = %s
ORDER BY question_id;
