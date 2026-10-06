-- id: DQ-022
-- title: Fact conflict review rows have a valid status and key
-- severity: critical
-- known_failing:
-- existing_asana_task:
-- description: >
--   silver_fact_conflict_review rows must use a known review_status
--   (FALSE_POSITIVE, RESOLVED, ACCEPTED) and carry a reviewed_tree_value
--   and reviewed_at. A row with a typo'd status or no tree snapshot would
--   either never apply or apply forever, silently hiding a real conflict.

SELECT file_id, person_gedcom_id, fact_type, transcript_value, review_status, reviewed_tree_value, reviewed_at
FROM genealogy.silver_fact_conflict_review
WHERE review_status NOT IN ('FALSE_POSITIVE', 'RESOLVED', 'ACCEPTED')
   OR reviewed_at IS NULL
   OR (reviewed_tree_value IS NULL AND fact_type <> 'occupation' AND review_status IS NOT NULL
       AND EXISTS (SELECT 1 FROM genealogy.gold_fact_comparison fc
                   WHERE fc.file_id = silver_fact_conflict_review.file_id
                     AND fc.person_gedcom_id = silver_fact_conflict_review.person_gedcom_id
                     AND fc.fact_type = silver_fact_conflict_review.fact_type
                     AND fc.tree_value IS NOT NULL));
