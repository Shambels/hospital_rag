-- Parameters: experiment name. Returns the id of its most recent finished run.
SELECT id
FROM experiments
WHERE name = %s AND status = 'done'
ORDER BY started_at DESC, id DESC
LIMIT 1;
