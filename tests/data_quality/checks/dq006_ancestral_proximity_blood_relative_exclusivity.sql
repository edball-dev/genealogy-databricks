-- id: DQ-006
-- title: gold_ancestral_proximity is_blood_relative mutual exclusivity
-- severity: critical
-- guards_bug:
-- known_failing: false
-- existing_asana_task:
-- description: >
--   Design invariant from Notebook 08: a NOT EXISTS guard on spouse INSERTs
--   is supposed to guarantee that any given person_id gets exactly one
--   is_blood_relative value, by construction. If this ever returns rows,
--   that guard has regressed.

SELECT person_id, COUNT(DISTINCT is_blood_relative) AS distinct_flags
FROM genealogy.gold_ancestral_proximity
GROUP BY person_id
HAVING COUNT(DISTINCT is_blood_relative) > 1;
