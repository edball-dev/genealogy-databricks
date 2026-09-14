-- id: DQ-004
-- title: gold_source_coverage.event_id NULL
-- severity: critical
-- guards_bug: 1218420283884341
-- known_failing: true
-- existing_asana_task: 1218420283884341
-- description: >
--   All event_id values in gold_source_coverage are NULL — open bug, unknown
--   downstream impact on coverage reporting. This check exists so the day
--   it's fixed, any regression back to NULL is caught automatically instead
--   of silently reintroducing the same bug.

SELECT *
FROM genealogy.gold_source_coverage
WHERE event_id IS NULL;
