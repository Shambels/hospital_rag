-- Every experiment, oldest first. No parameters.
SELECT id, name, config, started_at, duration_s, status
FROM experiments
ORDER BY started_at, id;
