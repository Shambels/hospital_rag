-- Every score of every experiment, for the HTML report. No parameters.
SELECT experiment_id, question_id, question_type, metric, value
FROM scores
ORDER BY experiment_id, question_id, metric;
