-- Parameters: experiment id. Returns id, name, config, status.
SELECT id, name, config, status
FROM experiments
WHERE id = %s;
