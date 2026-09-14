-- id: DQ-002
-- title: Manual document-person overrides outstanding
-- severity: info
-- guards_bug: 1218420283786561
-- known_failing:
-- existing_asana_task:
-- description: >
--   silver_document_person_override was reduced from 23 rows to 0 once the
--   BUG-001 matching-algorithm fix landed (2026-09-13) — the mechanism is
--   retained for genuine future Sr/Jr-style disambiguation, so a nonzero
--   count is informational, not necessarily a regression. Reported at info
--   severity: logged to data_quality_results for trend visibility, never
--   files an Asana task.

SELECT file_id, file_name, person_gedcom_id, notes
FROM genealogy.silver_document_person_override;
