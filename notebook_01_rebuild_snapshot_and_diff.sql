-- Databricks notebook source
-- MAGIC %md
-- MAGIC # notebook_01 rebuild: snapshot, then diff (Asana 1219284071167246)
-- MAGIC
-- MAGIC Cell 5 of notebook_01 TRUNCATEs `silver_document_person` and repopulates it; Cells 5b to 5i then add rows. The age gate, the abbreviation list and Cell 5h3 only reach rows that do not exist yet, so a rebuild is the way old household mislinks go. Run this notebook around it:
-- MAGIC
-- MAGIC 1. Cells 1 and 2 below: snapshot `silver_document_person` and `gold_transcript_facts` (DEEP CLONE, so the snapshot is independent of the live tables). Cell cannot be re-run once the snapshots exist: it is `CREATE TABLE`, not `CREATE OR REPLACE`, so a second run fails rather than overwriting the pre-rebuild state.
-- MAGIC 2. Run notebook_01 in order: Cell 1 (aliases, already run), Cell 4d, then Cell 5, 5b, 5c, 5d, 5e, 5f/5g/5h, 5h2, 5h3, 5i. Re-run Cells 6 to 8 afterwards so `gold_source_coverage` follows the new links.
-- MAGIC 3. Cells 3 to 6 below: the diff. Every cell is read-only.
-- MAGIC 4. Cell 7 is the way back, commented out.
-- MAGIC
-- MAGIC Row identity for the diff is `(file_id, person_gedcom_id)`. A row is CHANGED when the same person is still linked to the file but the mention (`page_index`, `person_index`), `match_method` or `match_confidence` differs. A person moved to a different mention shows as CHANGED, not REMOVED and ADDED.

-- COMMAND ----------

-- Cell 1: snapshot silver_document_person (fails if the snapshot already exists: do not overwrite the pre-rebuild state)
CREATE TABLE genealogy.silver_document_person_pre_rebuild DEEP CLONE genealogy.silver_document_person;

-- COMMAND ----------

-- Cell 2: snapshot gold_transcript_facts (Cell 5d rewrites person_gedcom_id on single-mapping files, so facts need a before-image too)
CREATE TABLE genealogy.gold_transcript_facts_pre_rebuild DEEP CLONE genealogy.gold_transcript_facts;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Run notebook_01 Cell 4d to 5i now, then continue with Cell 3
-- MAGIC
-- MAGIC Cell 3 is the headline: how many links are REMOVED, ADDED or CHANGED, split by whether DQ-025 already flagged the old row. REMOVED or CHANGED rows with `dq025_flagged_before = false` are the ones DQ-025 could not see.

-- COMMAND ----------

-- Cell 3: summary of the change set
WITH
-- DQ-025 (tests/data_quality/checks/dq025_census_mention_age_inconsistent.sql) evaluated against the SNAPSHOT
census AS (
  SELECT file_id, MAX(TRY_CAST(year AS INT)) AS doc_year
  FROM genealogy.ocr_transcriptions
  WHERE doc_type_detected = 'Census'
  GROUP BY file_id
),
dq025_before AS (
  SELECT sdp.file_id, sdp.person_gedcom_id
  FROM genealogy.silver_document_person_pre_rebuild sdp
  JOIN census c ON c.file_id = sdp.file_id
  JOIN genealogy.gold_person_life pl ON pl.person_gedcom_id = sdp.person_gedcom_id
  JOIN genealogy.silver_transcript_person_mention m
    ON m.file_id = sdp.file_id AND m.person_index = sdp.person_index AND m.page_index <=> sdp.page_index
  WHERE sdp.match_confidence IN ('HIGH', 'MEDIUM')
    AND m.override_person_gedcom_id IS NULL
    AND c.doc_year IS NOT NULL AND m.age_years IS NOT NULL
    AND TRY_CAST(pl.birth_year AS INT) IS NOT NULL
    AND ABS(c.doc_year - m.age_years - TRY_CAST(pl.birth_year AS INT)) > 5
  UNION
  SELECT sdp.file_id, sdp.person_gedcom_id
  FROM genealogy.silver_document_person_pre_rebuild sdp
  JOIN census c ON c.file_id = sdp.file_id
  WHERE sdp.match_confidence IN ('HIGH', 'MEDIUM')
    AND sdp.person_index IS NULL
    AND EXISTS (SELECT 1 FROM genealogy.silver_transcript_person_mention mm WHERE mm.file_id = sdp.file_id)
),
before_links AS (
  SELECT file_id, person_gedcom_id,
         MIN(page_index) AS page_index, MIN(person_index) AS person_index,
         MIN(match_method) AS match_method, MIN(match_confidence) AS match_confidence, COUNT(*) AS n_rows
  FROM genealogy.silver_document_person_pre_rebuild
  GROUP BY file_id, person_gedcom_id
),
after_links AS (
  SELECT file_id, person_gedcom_id,
         MIN(page_index) AS page_index, MIN(person_index) AS person_index,
         MIN(match_method) AS match_method, MIN(match_confidence) AS match_confidence, COUNT(*) AS n_rows
  FROM genealogy.silver_document_person
  GROUP BY file_id, person_gedcom_id
),
diff AS (
  SELECT COALESCE(b.file_id, a.file_id) AS file_id,
         COALESCE(b.person_gedcom_id, a.person_gedcom_id) AS person_gedcom_id,
         CASE WHEN a.file_id IS NULL THEN 'REMOVED'
              WHEN b.file_id IS NULL THEN 'ADDED'
              ELSE 'CHANGED' END AS change_type,
         b.match_method AS method_before, a.match_method AS method_after,
         d.file_id IS NOT NULL AS dq025_flagged_before
  FROM before_links b
  FULL OUTER JOIN after_links a
    ON a.file_id = b.file_id AND a.person_gedcom_id = b.person_gedcom_id
  LEFT JOIN dq025_before d
    ON d.file_id = b.file_id AND d.person_gedcom_id = b.person_gedcom_id
  WHERE a.file_id IS NULL OR b.file_id IS NULL
     OR NOT (a.page_index <=> b.page_index AND a.person_index <=> b.person_index
             AND a.match_method <=> b.match_method AND a.match_confidence <=> b.match_confidence
             AND a.n_rows = b.n_rows)
)
SELECT change_type, COALESCE(method_before, method_after) AS match_method, dq025_flagged_before, COUNT(*) AS links
FROM diff
GROUP BY change_type, COALESCE(method_before, method_after), dq025_flagged_before
ORDER BY change_type, links DESC;

-- COMMAND ----------

-- Cell 4: row-level detail, so each REMOVED / CHANGED link can be checked by eye. ADDED rows are listed too.
-- Same CTEs as Cell 3 (Databricks notebooks cannot share a CTE across cells).
WITH
before_links AS (
  SELECT file_id, person_gedcom_id,
         MIN(page_index) AS page_index, MIN(person_index) AS person_index,
         MIN(match_method) AS match_method, MIN(match_confidence) AS match_confidence, COUNT(*) AS n_rows
  FROM genealogy.silver_document_person_pre_rebuild
  GROUP BY file_id, person_gedcom_id
),
after_links AS (
  SELECT file_id, person_gedcom_id,
         MIN(page_index) AS page_index, MIN(person_index) AS person_index,
         MIN(match_method) AS match_method, MIN(match_confidence) AS match_confidence, COUNT(*) AS n_rows
  FROM genealogy.silver_document_person
  GROUP BY file_id, person_gedcom_id
)
SELECT CASE WHEN a.file_id IS NULL THEN 'REMOVED' WHEN b.file_id IS NULL THEN 'ADDED' ELSE 'CHANGED' END AS change_type,
       COALESCE(b.file_id, a.file_id) AS file_id,
       (SELECT MAX(file_name) FROM genealogy.ocr_transcriptions o WHERE o.file_id = COALESCE(b.file_id, a.file_id)) AS file_name,
       COALESCE(b.person_gedcom_id, a.person_gedcom_id) AS person_gedcom_id,
       pl.display_name, pl.birth_year,
       b.match_method AS method_before, a.match_method AS method_after,
       b.match_confidence AS conf_before, a.match_confidence AS conf_after,
       b.person_index AS idx_before, a.person_index AS idx_after
FROM before_links b
FULL OUTER JOIN after_links a
  ON a.file_id = b.file_id AND a.person_gedcom_id = b.person_gedcom_id
LEFT JOIN genealogy.gold_person_life pl
  ON pl.person_gedcom_id = COALESCE(b.person_gedcom_id, a.person_gedcom_id)
WHERE a.file_id IS NULL OR b.file_id IS NULL
   OR NOT (a.page_index <=> b.page_index AND a.person_index <=> b.person_index
           AND a.match_method <=> b.match_method AND a.match_confidence <=> b.match_confidence
           AND a.n_rows = b.n_rows)
ORDER BY change_type, file_name, pl.display_name;

-- COMMAND ----------

-- Cell 5: facts that now point at the wrong person (read-only).
-- A link with a NULL page_index (most of them) covers every page of its file; the fact's page_index is the OCR page, so it is not compared then.
-- A page-level fact carries the person_gedcom_id it had when it was extracted. If that person is no longer linked to the file
-- (REMOVED), or is now linked to a different mention than the fact's person_index (CHANGED), the fact is stale.
SELECT gtf.file_id,
       (SELECT MAX(file_name) FROM genealogy.ocr_transcriptions o WHERE o.file_id = gtf.file_id) AS file_name,
       gtf.person_gedcom_id, gtf.person_index, COUNT(*) AS stale_facts
FROM genealogy.gold_transcript_facts gtf
WHERE gtf.page_index IS NOT NULL
  AND gtf.person_gedcom_id IS NOT NULL
  AND NOT EXISTS (
    SELECT 1 FROM genealogy.silver_document_person sdp
    WHERE sdp.file_id = gtf.file_id
      AND sdp.person_gedcom_id = gtf.person_gedcom_id
      AND (sdp.page_index IS NULL OR sdp.page_index = gtf.page_index)
      AND sdp.person_index = gtf.person_index
  )
GROUP BY gtf.file_id, gtf.person_gedcom_id, gtf.person_index
ORDER BY file_name, stale_facts DESC;

-- COMMAND ----------

-- Cell 6: files that need a forced notebook_02 run (paste the file_id column into force_file_ids).
-- (a) files with stale facts (Cell 5), and (b) files where a person is newly linked to a mention whose facts are stored unlinked
-- (person_gedcom_id NULL), since notebook_02 skips pages that already have a status row.
WITH stale AS (
  SELECT DISTINCT gtf.file_id
  FROM genealogy.gold_transcript_facts gtf
  WHERE gtf.page_index IS NOT NULL AND gtf.person_gedcom_id IS NOT NULL
    AND NOT EXISTS (
      SELECT 1 FROM genealogy.silver_document_person sdp
      WHERE sdp.file_id = gtf.file_id AND sdp.person_gedcom_id = gtf.person_gedcom_id
        AND (sdp.page_index IS NULL OR sdp.page_index = gtf.page_index) AND sdp.person_index = gtf.person_index
    )
),
newly_linked AS (
  SELECT DISTINCT gtf.file_id
  FROM genealogy.gold_transcript_facts gtf
  JOIN genealogy.silver_document_person sdp
    ON sdp.file_id = gtf.file_id AND (sdp.page_index IS NULL OR sdp.page_index = gtf.page_index) AND sdp.person_index = gtf.person_index
  WHERE gtf.page_index IS NOT NULL AND gtf.person_gedcom_id IS NULL
    AND sdp.match_confidence IN ('HIGH', 'MEDIUM')
    -- only links the rebuild created: facts for a link the age gate rejected are stored unlinked on purpose, and that link already existed before
    AND NOT EXISTS (
      SELECT 1 FROM genealogy.silver_document_person_pre_rebuild b
      WHERE b.file_id = sdp.file_id AND b.person_gedcom_id = sdp.person_gedcom_id AND b.person_index <=> sdp.person_index
    )
)
SELECT file_id, MAX(reason) AS reason
FROM (
  SELECT file_id, 'stale_facts' AS reason FROM stale
  UNION ALL
  SELECT file_id, 'newly_linked_unlinked_facts' AS reason FROM newly_linked
) u
GROUP BY file_id
ORDER BY file_id;

-- COMMAND ----------

-- Cell 7: the way back. Leave commented out unless the diff shows the rebuild is wrong.
-- INSERT OVERWRITE genealogy.silver_document_person SELECT * FROM genealogy.silver_document_person_pre_rebuild;
-- INSERT OVERWRITE genealogy.gold_transcript_facts SELECT * FROM genealogy.gold_transcript_facts_pre_rebuild;
-- Once the rebuild is accepted, drop the snapshots:
-- DROP TABLE genealogy.silver_document_person_pre_rebuild;
-- DROP TABLE genealogy.gold_transcript_facts_pre_rebuild;
