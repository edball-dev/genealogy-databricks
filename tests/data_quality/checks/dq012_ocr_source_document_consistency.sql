-- id: DQ-012
-- title: ocr_transcriptions rows whose source file is gone/deleted from Drive
-- severity: warning
-- guards_bug:
-- known_failing: true
-- existing_asana_task: 1218567581863400
-- description: >
--   ocr_transcriptions.file_id should stay consistent with
--   staging_google_drive.documents (the Fivetran Drive sync ocr_pipeline.ipynb
--   reads from) in two ways: every file_id we transcribed should still exist
--   there (an orphan means the Drive file was hard-deleted from the sync
--   entirely, not just soft-deleted), and any that does exist should not be
--   flagged _fivetran_deleted = true (soft-deleted in Drive but we still hold
--   a transcription for it). Live validation (2026-09-16): 0 orphans, but 17
--   rows across 9 distinct file_ids (9 pages of
--   "7_June_1890_-_Conveyance.pdf" plus 8 single-page files) point at
--   documents Fivetran has marked deleted. This may be legitimate (a
--   duplicate/renamed file removed from Drive after OCR, or a source cleaned
--   up post-transcription) rather than a pipeline bug, so this is warning,
--   not critical -- but it is exactly the kind of drift DQ-005's duplicate-
--   filename bug came from, so it is worth periodic manual review rather than
--   silent accumulation. known_failing: true until the 9 current files are
--   triaged (confirm each is an intentional cleanup, not a mistaken deletion
--   that should be restored/re-pointed).
--
--   Triage (2026-09-17, Ed): 7 of the 9 file_ids (all 15 of the original 17
--   rows -- the whole "7_June_1890_-_Conveyance.pdf" plus EASTOE_Charles x2,
--   EDMISTON_David x2, BALL_Harriet, MADDOX_Samuel) turned out to be from a
--   `_Possibles` Drive folder that should never have been Fivetran-synced in
--   the first place -- unconfirmed documents Ed hadn't yet verified as his
--   family, now moved out of the synced volume. Two of those (EDMISTON_David
--   x2, EASTOE_Charles x2 -- 4 file_ids total) had already been auto-matched
--   onto real tree people (David Edmiston @I999886834@, Charles Eastoe
--   @I_967017894@) and fact-extracted/compared against the tree, so the
--   cleanup wasn't just this table: also deleted the matching rows from
--   silver_document_person (4), gold_transcript_facts (12), gold_fact_comparison
--   (4), and silver_document_match_exception (1, MADDOX_Samuel). Confirmed
--   live: all five tables at 0 rows for these 7 file_ids post-delete.
--   gold_transcript_facts/gold_fact_comparison/silver_document_match_exception
--   had to be added to the Databricks MCP's execute_write_sql allow-list
--   first (edball-dev/databricks-mcp#2) -- ocr_transcriptions and
--   silver_document_person were already writable.
--
--   The remaining 2 file_ids (HALLAM_Samuel_1875_Friends with Thomas
--   Palmer.pdf, STIRLING_Marion_1889_Death.jpg) are unrelated to `_Possibles`
--   -- both from confirmed Family_ folders, both successfully transcribed
--   back in March -- and are a different, still-open question: Ed confirms
--   HALLAM was renamed in Drive (to
--   "..._Nottingham_Journal_16_August_1875_0004_Clip.jpg") and STIRLING is
--   still present in Drive, unrenamed and undeleted. Both point at stale
--   Fivetran staging data (last _fivetran_synced 2026-04-27 and 2026-05-31
--   respectively) rather than a real deletion -- expected to self-clear on
--   the next Fivetran resync of staging_google_drive.documents, not
--   something to fix in this repo. known_failing stays true until that
--   resync happens and this check goes green.

SELECT t.file_id, t.file_name, t.page_index,
       d.file_id IS NULL AS source_missing,
       COALESCE(d._fivetran_deleted, false) AS source_deleted
FROM genealogy.ocr_transcriptions t
LEFT JOIN staging_google_drive.documents d ON t.file_id = d.file_id
WHERE d.file_id IS NULL OR d._fivetran_deleted = true;
