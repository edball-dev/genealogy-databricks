-- id: DQ-010
-- title: Document-match exceptions outstanding (Cell 5/5c declined to force-match)
-- severity: info
-- guards_bug: 1218506594311016
-- known_failing:
-- existing_asana_task:
-- description: >
--   silver_document_match_exception (added 2026-09-15) holds files Cell 5
--   (zero year-plausible candidates at all) or Cell 5c (sibling consensus
--   disagrees with this specific file's own year) declined to force-match
--   onto a chronologically implausible candidate -- the fix for the root
--   cause behind the Elizabeth Balls / Henry Easter wrong-person matches
--   (task 1218506594311016). These need manual review, the same as any
--   other unmatched file: resolve via a silver_person_alias addition (the
--   common case) or a silver_document_person_override row (last resort) if
--   no tree/transcript signal can resolve it -- see the same DQ-002 pattern
--   for the override table.
--
--   Reported at info severity: logged to data_quality_results for trend
--   visibility, never files an Asana task. A nonzero count is expected and
--   not itself a regression -- this table exists precisely because some
--   documents in the corpus can't be safely auto-matched. Watch for
--   unexpected growth between runs (more files landing here than were added
--   to the corpus), not the raw count.

SELECT file_id, file_name, doc_type_detected, year, surname, forename,
       candidate_count, reason, candidates_considered, logged_at
FROM genealogy.silver_document_match_exception;
