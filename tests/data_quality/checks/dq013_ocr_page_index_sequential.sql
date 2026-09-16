-- id: DQ-013
-- title: ocr_transcriptions page_index not sequential/gapless/unique per file_id
-- severity: critical
-- guards_bug:
-- known_failing: false
-- description: >
--   Each file_id in ocr_transcriptions should have page_index running 1..N
--   with no gaps and no duplicates, matching pages_bytes' 1-based enumerate()
--   in ocr_pipeline.ipynb Cell 22 -- a gap means a page silently failed to
--   write (partial append) and a duplicate means the same page was
--   transcribed and appended twice (e.g. a rerun that bypassed the
--   ocr_processing_log idempotency check). Either breaks any downstream
--   consumer that assumes contiguous pages (e.g. reconstructing full-document
--   text in page order). Live validation (2026-09-16): 0 violations across
--   1002 rows.

SELECT file_id, ANY_VALUE(file_name) AS file_name,
       COUNT(*) AS row_count,
       COUNT(DISTINCT page_index) AS distinct_pages,
       MIN(page_index) AS min_page, MAX(page_index) AS max_page
FROM genealogy.ocr_transcriptions
GROUP BY file_id
HAVING MIN(page_index) <> 1
    OR MAX(page_index) <> COUNT(DISTINCT page_index)
    OR COUNT(DISTINCT page_index) <> COUNT(*);
