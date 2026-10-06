-- id: DQ-024
-- title: TREE_GAP reviews missing from the tree-gap worklist
-- severity: critical
-- known_failing:
-- existing_asana_task: 1219234333541171
-- description: >
--   Every TREE_GAP review (legitimate occupation the tree should record) whose
--   comparison row is still CONFLICT or TRANSCRIPT_ONLY must appear in
--   genealogy.gold_housekeeping_tree_gap_occupations, otherwise Ed is never told
--   to add it in Ancestry and SIGNAL_TRANSCRIPT_ONLY_FACTS stays silent. Also
--   fails if the view is dropped by a notebook_03 rebuild (query errors). Rows
--   that have become MATCH (tree gained the occupation) are correctly absent.

SELECT fc.person_gedcom_id, fc.file_id, fc.transcript_value, fc.status
FROM genealogy.gold_fact_comparison_reviewed fc
WHERE fc.review_reason_code = 'TREE_GAP'
  AND fc.fact_type = 'occupation'
  AND fc.status IN ('CONFLICT', 'TRANSCRIPT_ONLY')
  AND NOT EXISTS (
    SELECT 1 FROM genealogy.gold_housekeeping_tree_gap_occupations w
    WHERE w.person_gedcom_id = fc.person_gedcom_id
      AND w.file_id = fc.file_id
      AND w.occupation_as_written <=> fc.transcript_value);
