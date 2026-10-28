-- Parameters: status ('done' or 'failed'), duration_s, experiment id.
UPDATE experiments
SET status = %s, duration_s = %s
WHERE id = %s;
