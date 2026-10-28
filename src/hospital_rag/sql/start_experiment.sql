-- Parameters: name, config (JSONB). Returns the new experiment id.
INSERT INTO experiments (name, config)
VALUES (%s, %s)
RETURNING id;
