-- id: DQ-023
-- title: Fact conflict reviews that no longer apply (stale or orphaned)
-- severity: info
-- known_failing:
-- existing_asana_task:
-- description: >
--   Reviews in silver_fact_conflict_review whose key no longer matches a
--   CONFLICT row in gold_fact_comparison with the same tree value. STALE =
--   the tree value changed since review (conflict reopened by design);
--   ORPHANED = the key vanished after a rebuild (transcript value or file_id
--   changed, or the conflict no longer exists). Informational: a nonzero
--   count is expected after tree/transcript corrections, but a sudden jump
--   after a notebook_03 rebuild means reviews are being silently dropped.

SELECT r.file_id, r.person_gedcom_id, r.fact_type, r.transcript_value, r.review_status,
       r.reviewed_tree_value, fc.tree_value AS current_tree_value,
       CASE WHEN fc.file_id IS NULL THEN 'ORPHANED' ELSE 'STALE' END AS reason
FROM genealogy.silver_fact_conflict_review r
LEFT JOIN genealogy.gold_fact_comparison fc
  ON  fc.file_id          = r.file_id
  AND fc.person_gedcom_id = r.person_gedcom_id
  AND fc.fact_type        = r.fact_type
  AND fc.transcript_value <=> r.transcript_value
  AND fc.status           = 'CONFLICT'
WHERE fc.file_id IS NULL
   OR (r.reason_code IS DISTINCT FROM 'TREE_GAP' AND NOT (r.reviewed_tree_value <=> fc.tree_value));
