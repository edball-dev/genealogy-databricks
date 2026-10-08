-- Databricks notebook source
-- MAGIC %md
-- MAGIC # notebook_02 schema changes and one-off clean-up (Asana 1218420653302709)
-- MAGIC
-- MAGIC DDL and deletes for the page-level notebook_02 redesign. Applied separately by Ed, not run by notebook_02.
-- MAGIC Run Cell 1 and 2 before the first notebook_02 run. Cell 3 is the one-off delete of known-wrong facts: review the preview first.

-- COMMAND ----------

-- Cell 1: new key columns on gold_transcript_facts (nullable: legacy rows keep NULL)
ALTER TABLE genealogy.gold_transcript_facts ADD COLUMNS (
  page_index   INT COMMENT 'ocr_transcriptions page the fact was read from; NULL on legacy (pre page-level) rows',
  person_index INT COMMENT 'silver_transcript_person_mention.person_index the fact is about; NULL on legacy rows'
);

-- COMMAND ----------

-- Cell 2: persisted per-page extraction status so skipped/errored pages are never silently re-billed.
-- status: DONE | NO_PEOPLE | REJECTED (subject rejected for every linked person) | ERROR
CREATE TABLE IF NOT EXISTS genealogy.silver_fact_extraction_status (
  file_id       STRING    COMMENT 'ocr_transcriptions.file_id',
  page_index    INT       COMMENT 'ocr_transcriptions.page_index',
  status        STRING    COMMENT 'DONE | NO_PEOPLE | REJECTED | ERROR',
  detail        STRING    COMMENT 'Error text or rejection reason',
  n_calls       INT       COMMENT 'Gemini calls made for this page (chunks)',
  n_facts       INT       COMMENT 'Fact rows written',
  attempted_at  TIMESTAMP
)
USING DELTA
COMMENT 'Per-page outcome of notebook_02 fact extraction. A page with any row here is not retried unless notebook_02 is run with retry_errors=true (ERROR only) or the row is deleted.';

-- COMMAND ----------

-- Cell 3: preview then delete the known-wrong legacy facts so they are re-extracted.
-- Piggin 1871 census, Cuthbertson 1841 census, Palmer 1926 'Death of son' clippings.
-- Deleting is scoped to the (file, person) pairs named; nothing else is touched.
-- Alternative that deletes nothing by hand: run notebook_02 with force_file_ids=<these file_ids>,
-- which deletes the legacy (page_index IS NULL) facts for those files itself after a successful extraction.
SELECT gtf.file_id, ot.file_name, gtf.person_gedcom_id, COUNT(*) AS fact_rows
FROM genealogy.gold_transcript_facts gtf
JOIN (SELECT DISTINCT file_id, file_name FROM genealogy.ocr_transcriptions) ot ON ot.file_id = gtf.file_id
WHERE gtf.page_index IS NULL
  AND (ot.file_name LIKE 'PIGGIN_Reuben Rogers_1871_Census%'
    OR ot.file_name LIKE 'CUTHBERTSON_George (Sr)_1841_Census%'
    OR ot.file_name LIKE 'PALMER_Samuel_1926%')
GROUP BY gtf.file_id, ot.file_name, gtf.person_gedcom_id
ORDER BY ot.file_name;
