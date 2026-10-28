-- Mean of each metric for each experiment, best first. No parameters.
-- n counts the non-NULL values: lower than expected when the judge failed on some questions.
SELECT e.name, s.metric, round(avg(s.value)::numeric, 3) AS mean, count(s.value) AS n
FROM scores s
JOIN experiments e ON e.id = s.experiment_id
GROUP BY e.name, s.metric
ORDER BY s.metric, mean DESC;
