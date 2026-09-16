-- id: DQ-012
-- title: ocr_transcriptions rows whose source file is gone/deleted from Drive
-- severity: warning
-- guards_bug:
-- known_failing: true
-- existing_asana_task:
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

SELECT t.file_id, t.file_name, t.page_index,
       d.file_id IS NULL AS source_missing,
       COALESCE(d._fivetran_deleted, false) AS source_deleted
FROM genealogy.ocr_transcriptions t
LEFT JOIN staging_google_drive.documents d ON t.file_id = d.file_id
WHERE d.file_id IS NULL OR d._fivetran_deleted = true;
