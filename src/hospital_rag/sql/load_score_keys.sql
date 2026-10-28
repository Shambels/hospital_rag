-- Parameters: experiment id. Returns the (question_id, metric) pairs already scored, with a non-NULL value.
SELECT question_id, metric
FROM scores
WHERE experiment_id = %s AND value IS NOT NULL;
