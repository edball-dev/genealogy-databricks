-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Notebook 03 v1.5: Fact Comparison — Tree vs Transcript
-- MAGIC
-- MAGIC Compares gold_transcript_facts against gold_person_life and gold_person_event_timeline.
-- MAGIC Run after notebook_02_fact_extraction has completed.
-- MAGIC
-- MAGIC ## Summary
-- MAGIC
-- MAGIC Refreshes the gold_fact_comparison table: `mode=full` (default) deletes and refills every row as before; `mode=scoped` recompares only the files in the `file_ids` widget plus the stale files (facts extracted, or document matched, after the file's comparison rows were written), leaving every other file's rows untouched. Every row carries `refreshed_at`. Scoped mode needs one full refresh first (rows written before v1.5 have no `refreshed_at`, so staleness cannot be judged). Tree-side changes (gold_person_life, the event timeline) need a full refresh. (This notebook creates no views: a view recreate here would transfer view ownership to whoever runs it). The views that read it live in views_document_and_fact_review and must already exist; run that notebook only on definition changes. Ed's review marks come from silver_fact_conflict_review (prerequisite: notebook_fact_conflict_review_init.sql), applied by the view gold_fact_comparison_reviewed. Consumers counting or listing conflicts should use its is_open_conflict flag. Reason code TREE_GAP (a legitimate occupation the tree should record) is never an open conflict and applies even if the tree's occupation list later changes; `is_tree_gap` is TRUE only while the tree still lacks the value and self-clears when the comparison becomes MATCH. The worklist view gold_housekeeping_tree_gap_occupations is likewise in views_document_and_fact_review.
-- MAGIC
-- MAGIC The occupation comparison (Cell 4) is many-to-many: the transcript value is matched against all of the person's tree occupations (MATCH if any matches; tree_value shows them pipe-separated), military ranks (Pte, Cpl, Sgt etc.) are excluded, and occupation conflicts get conflict_severity = LOW. All transcript CTEs use DISTINCT to stop multi-page PDFs fanning out into duplicate comparison rows.
-- MAGIC
-- MAGIC ## Version history
-- MAGIC
-- MAGIC - v1.5 (2026-10-17): `mode` and `file_ids` widgets (Asana 1218716867008994, step 3). Cell 0 builds `tmp_fc_scope`; Cell 1 deletes the scoped files' rows instead of TRUNCATE (in full mode every file is in scope, same result); Cells 2 to 5 read only scoped files' facts and write `refreshed_at`.
-- MAGIC - v1.4 (2026-10-11): view DDL moved out to views_document_and_fact_review. This notebook only
-- MAGIC         refreshes the gold_fact_comparison table (a view recreate here would transfer view ownership
-- MAGIC         to whoever runs it). Prerequisite: the views exist (run that notebook only on definition changes).
-- MAGIC - v1.3: TREE_GAP review outcome (task 1219234333541171). Reason code TREE_GAP = legitimate
-- MAGIC         occupation the tree should record. Never an open conflict; applies even if the tree's
-- MAGIC         occupation list later changes; `is_tree_gap` is TRUE only while the tree still lacks the
-- MAGIC         value (self-clears when the comparison becomes MATCH). The worklist view
-- MAGIC         `gold_housekeeping_tree_gap_occupations` is likewise in views_document_and_fact_review.
-- MAGIC - v1.2: view gold_fact_comparison_reviewed (now in views_document_and_fact_review), applying Ed's review marks from
-- MAGIC         silver_fact_conflict_review (see notebook_fact_conflict_review_init.sql — prerequisite).
-- MAGIC         Consumers counting/listing conflicts should use its is_open_conflict flag.
-- MAGIC - v1.1: Add DISTINCT to all transcript CTEs to prevent page-level fan-out from
-- MAGIC         multi-page PDFs producing duplicate comparison rows per fact per document.
-- MAGIC         Cell 4 (occupation): rewrite as many-to-many comparison — transcript value
-- MAGIC         matched against all tree occupations for the person; MATCH if any tree
-- MAGIC         occupation matches; tree_value shows all occupations pipe-separated.
-- MAGIC         Cell 4: add military rank exclusion filter (Pte, Cpl, Sgt etc.).
-- MAGIC         Cell 4: add conflict_severity = LOW for occupation conflicts (was NULL).
-- MAGIC - v1: Initial version

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 0 — Run mode and scope
-- MAGIC
-- MAGIC Widgets: `mode` (`full` default, or `scoped`) and `file_ids` (comma-separated `file_id`s, scoped mode only).
-- MAGIC
-- MAGIC - **full**: every file is in scope, so Cells 1 to 5 behave as they did before scoping.
-- MAGIC - **scoped**: the files in `file_ids` plus the stale files. A file is stale when its facts (`gold_transcript_facts.extracted_at`) or its document match (`ocr_transcriptions.last_matched_at`, stamped by notebook_01) are newer than its latest `gold_fact_comparison.refreshed_at`, or it has facts but no comparison rows and the facts arrived after the oldest refresh. So a scoped notebook_01 run followed by a scoped notebook_03 run recompares exactly the files that were rematched.
-- MAGIC - Limits: needs one full refresh first (older rows have no `refreshed_at`); a file whose facts yield no comparison rows (low confidence, or a person not in the tree) is re-listed only until the next full refresh; changes on the tree side need a full refresh.
-- MAGIC
-- MAGIC DQ-031 lists the stale files (same SQL as below: change them together). Creates the temp view `tmp_fc_scope`.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import re
-- MAGIC
-- MAGIC dbutils.widgets.dropdown("mode", "full", ["full", "scoped"], "Run mode")
-- MAGIC dbutils.widgets.text("file_ids", "", "Comma-separated file_ids (scoped mode)")
-- MAGIC MODE = dbutils.widgets.get("mode")
-- MAGIC MANUAL_IDS = [f.strip() for f in dbutils.widgets.get("file_ids").split(",") if f.strip()]
-- MAGIC
-- MAGIC # Drive file ids are alphanumeric with - and _; refuse anything else before it reaches SQL.
-- MAGIC bad = [f for f in MANUAL_IDS if not re.fullmatch(r"[A-Za-z0-9_-]+", f)]
-- MAGIC if bad:
-- MAGIC     raise ValueError(f"Not valid file_ids: {bad}")
-- MAGIC if MODE == "full" and MANUAL_IDS:
-- MAGIC     raise ValueError("file_ids is only used with mode=scoped; clear it or switch mode.")
-- MAGIC
-- MAGIC # Stale files: facts or document match newer than the file's comparison rows. Mirrored by DQ-031; change them together.
-- MAGIC STALE_SQL = r"""WITH f AS (
-- MAGIC   SELECT file_id, MAX(extracted_at) AS facts_at
-- MAGIC   FROM genealogy.gold_transcript_facts
-- MAGIC   WHERE confidence IN ('high', 'medium')
-- MAGIC   GROUP BY file_id
-- MAGIC ),
-- MAGIC m AS (
-- MAGIC   SELECT file_id, MAX(last_matched_at) AS matched_at
-- MAGIC   FROM genealogy.ocr_transcriptions
-- MAGIC   GROUP BY file_id
-- MAGIC ),
-- MAGIC c AS (
-- MAGIC   SELECT file_id, MAX(refreshed_at) AS compared_at
-- MAGIC   FROM genealogy.gold_fact_comparison
-- MAGIC   GROUP BY file_id
-- MAGIC ),
-- MAGIC base AS (
-- MAGIC   SELECT MIN(refreshed_at) AS oldest_refresh FROM genealogy.gold_fact_comparison
-- MAGIC )
-- MAGIC SELECT f.file_id, f.facts_at, m.matched_at, c.compared_at
-- MAGIC FROM f
-- MAGIC LEFT JOIN m ON m.file_id = f.file_id
-- MAGIC LEFT JOIN c ON c.file_id = f.file_id
-- MAGIC CROSS JOIN base
-- MAGIC WHERE base.oldest_refresh IS NOT NULL
-- MAGIC   AND (
-- MAGIC     -- compared before, and its facts or its match changed since
-- MAGIC     (c.compared_at IS NOT NULL AND (f.facts_at > c.compared_at OR m.matched_at > c.compared_at))
-- MAGIC     -- never compared (no rows), but facts arrived after the last full refresh
-- MAGIC     OR (c.compared_at IS NULL AND f.facts_at > base.oldest_refresh)
-- MAGIC   )"""
-- MAGIC
-- MAGIC if MODE == "scoped":
-- MAGIC     stale = {r["file_id"] for r in spark.sql(STALE_SQL).collect()}
-- MAGIC     scope_ids = sorted(stale | set(MANUAL_IDS))
-- MAGIC     print(f"scoped: {len(MANUAL_IDS)} named, {len(stale)} stale, {len(scope_ids)} in scope")
-- MAGIC else:
-- MAGIC     # Includes ids that only survive in the comparison table, so the delete in Cell 1 clears orphans as TRUNCATE did.
-- MAGIC     scope_ids = sorted({r["file_id"] for r in spark.sql(
-- MAGIC         "SELECT file_id FROM genealogy.gold_transcript_facts UNION SELECT file_id FROM genealogy.gold_fact_comparison").collect()
-- MAGIC         if r["file_id"]})
-- MAGIC     print(f"full: {len(scope_ids)} file(s) in scope")
-- MAGIC
-- MAGIC # Materialised from the driver: Cells 1 to 5 delete from and insert into the tables the scope is read from.
-- MAGIC spark.createDataFrame([(f,) for f in scope_ids], "file_id string").createOrReplaceTempView("tmp_fc_scope")
-- MAGIC if MODE == "scoped" and not scope_ids:
-- MAGIC     print("Nothing to recompare: every cell below will be a no-op.")

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 1 — Clear the scoped files' rows in gold_fact_comparison
-- MAGIC
-- MAGIC A scoped delete in place of TRUNCATE. In full mode `tmp_fc_scope` holds every file, so this clears the table as before.

-- COMMAND ----------

-- %sql
DELETE FROM genealogy.gold_fact_comparison WHERE file_id IN (SELECT file_id FROM tmp_fc_scope);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 2 — Birth year comparison

-- COMMAND ----------

-- %sql
INSERT INTO genealogy.gold_fact_comparison
  (person_gedcom_id, display_name, fact_type,
   tree_value, transcript_value, file_id, file_name, source_doc_type,
   status, conflict_severity, notes, refreshed_at)

WITH transcript_birth_year AS (
  -- DISTINCT collapses multiple pages of the same document extracting the same value
  SELECT DISTINCT
    gtf.person_gedcom_id,
    gtf.file_id,
    gtf.fact_value  AS transcript_value,
    gtf.fact_year   AS transcript_year,
    gtf.confidence,
    gtf.source_doc_type,
    ot.file_name
  FROM genealogy.gold_transcript_facts gtf
  JOIN genealogy.ocr_transcriptions ot ON gtf.file_id = ot.file_id
  WHERE gtf.file_id IN (SELECT file_id FROM tmp_fc_scope)
    AND gtf.fact_type = 'birth_year'
    AND gtf.confidence IN ('high','medium')
)

SELECT
  p.person_gedcom_id,
  p.display_name,
  'birth_year'                        AS fact_type,
  CAST(p.birth_year AS STRING)        AS tree_value,
  t.transcript_value,
  t.file_id,
  t.file_name,
  t.source_doc_type,
  CASE
    WHEN p.birth_year IS NULL                              THEN 'TRANSCRIPT_ONLY'
    WHEN ABS(p.birth_year - t.transcript_year) <= 1       THEN 'MATCH'
    ELSE                                                        'CONFLICT'
  END AS status,
  CASE
    WHEN p.birth_year IS NOT NULL AND ABS(p.birth_year - t.transcript_year) > 3  THEN 'HIGH'
    WHEN p.birth_year IS NOT NULL AND ABS(p.birth_year - t.transcript_year) IN (2,3) THEN 'MEDIUM'
    WHEN p.birth_year IS NOT NULL AND ABS(p.birth_year - t.transcript_year) = 1  THEN 'LOW'
    ELSE NULL
  END AS conflict_severity,
  CASE
    WHEN p.birth_year IS NULL
      THEN 'Birth year missing from tree -- transcript suggests ' || t.transcript_value
    WHEN ABS(p.birth_year - t.transcript_year) > 3
      THEN 'Significant discrepancy: tree=' || p.birth_year || ' transcript=' || t.transcript_value
    ELSE NULL
  END AS notes,
  CURRENT_TIMESTAMP() AS refreshed_at
FROM transcript_birth_year t
JOIN genealogy.gold_person_life p ON t.person_gedcom_id = p.person_gedcom_id;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 3 — Birth place comparison

-- COMMAND ----------

-- %sql
INSERT INTO genealogy.gold_fact_comparison
  (person_gedcom_id, display_name, fact_type,
   tree_value, transcript_value, file_id, file_name, source_doc_type,
   status, conflict_severity, notes, refreshed_at)

WITH transcript_birth_place AS (
  -- DISTINCT collapses multiple pages of the same document extracting the same value
  SELECT DISTINCT
    gtf.person_gedcom_id,
    gtf.file_id,
    gtf.fact_value AS transcript_value,
    gtf.confidence,
    gtf.source_doc_type,
    ot.file_name
  FROM genealogy.gold_transcript_facts gtf
  JOIN genealogy.ocr_transcriptions ot ON gtf.file_id = ot.file_id
  WHERE gtf.file_id IN (SELECT file_id FROM tmp_fc_scope)
    AND gtf.fact_type = 'birth_place'
    AND gtf.confidence IN ('high','medium')
)

SELECT
  p.person_gedcom_id,
  p.display_name,
  'birth_place'          AS fact_type,
  p.birth_place          AS tree_value,
  t.transcript_value,
  t.file_id,
  t.file_name,
  t.source_doc_type,
  CASE
    WHEN p.birth_place IS NULL THEN 'TRANSCRIPT_ONLY'
    WHEN UPPER(TRIM(p.birth_place)) = UPPER(TRIM(t.transcript_value)) THEN 'MATCH'
    WHEN UPPER(p.birth_place) LIKE CONCAT('%', UPPER(SPLIT(t.transcript_value,',')[0]), '%')
      OR UPPER(t.transcript_value) LIKE CONCAT('%', UPPER(SPLIT(p.birth_place,',')[0]), '%')
      THEN 'MATCH'
    ELSE 'CONFLICT'
  END AS status,
  CASE
    WHEN p.birth_place IS NOT NULL
     AND UPPER(TRIM(p.birth_place)) != UPPER(TRIM(t.transcript_value)) THEN 'MEDIUM'
    ELSE NULL
  END AS conflict_severity,
  CASE
    WHEN p.birth_place IS NULL
      THEN 'Birth place missing from tree -- transcript says: ' || t.transcript_value
    ELSE NULL
  END AS notes,
  CURRENT_TIMESTAMP() AS refreshed_at
FROM transcript_birth_place t
JOIN genealogy.gold_person_life p ON t.person_gedcom_id = p.person_gedcom_id;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 4 — Occupation comparison
-- MAGIC
-- MAGIC Many-to-many: each transcript occupation is checked against ALL tree occupations
-- MAGIC for that person. Status is MATCH if any tree occupation matches. tree_value shows
-- MAGIC all tree occupations pipe-separated. Military ranks excluded from transcript values.

-- COMMAND ----------

-- %sql
INSERT INTO genealogy.gold_fact_comparison
  (person_gedcom_id, display_name, fact_type,
   tree_value, transcript_value, file_id, file_name, source_doc_type,
   status, conflict_severity, notes, refreshed_at)

WITH transcript_occupations AS (
  -- DISTINCT collapses multi-page fan-out; rank filter removes military ranks
  -- that Gemini may extract as occupations from military records
  SELECT DISTINCT
    gtf.person_gedcom_id,
    gtf.file_id,
    gtf.fact_value  AS transcript_value,
    gtf.confidence,
    gtf.source_doc_type,
    ot.file_name
  FROM genealogy.gold_transcript_facts gtf
  JOIN genealogy.ocr_transcriptions ot ON gtf.file_id = ot.file_id
  WHERE gtf.file_id IN (SELECT file_id FROM tmp_fc_scope)
    AND gtf.fact_type = 'occupation'
    AND gtf.confidence IN ('high','medium')
    AND UPPER(TRIM(gtf.fact_value)) NOT IN (
      'PTE', 'PRIVATE', 'CPL', 'CPL.', 'CORPL', 'CORPORAL',
      'SGT', 'SERGEANT', 'LT', 'LIEUTENANT', 'CAPT', 'CAPTAIN',
      'L/CPL', 'LCPL', 'GNRL', 'GENERAL', '2ND LT'
    )
),

tree_occupations AS (
  SELECT DISTINCT
    person_gedcom_id,
    event_value AS tree_occupation
  FROM genealogy.gold_person_event_timeline
  WHERE UPPER(event_type) IN ('OCCU','OCCUPATION')
    AND event_value IS NOT NULL
),

-- For each transcript occupation, check whether it matches ANY tree occupation.
-- Collapses the tree-side fan-out via GROUP BY, preserving all tree values for display.
occupation_match AS (
  SELECT
    t.person_gedcom_id,
    t.file_id,
    t.transcript_value,
    t.confidence,
    t.source_doc_type,
    t.file_name,
    COLLECT_LIST(tocc.tree_occupation)  AS all_tree_occupations,
    MAX(CASE
      WHEN UPPER(TRIM(tocc.tree_occupation)) = UPPER(TRIM(t.transcript_value))              THEN 1
      WHEN UPPER(tocc.tree_occupation) LIKE CONCAT('%', UPPER(LEFT(t.transcript_value,6)), '%') THEN 1
      ELSE 0
    END) AS any_match
  FROM transcript_occupations t
  JOIN genealogy.gold_person_life p ON t.person_gedcom_id = p.person_gedcom_id
  LEFT JOIN tree_occupations tocc ON t.person_gedcom_id = tocc.person_gedcom_id
  GROUP BY
    t.person_gedcom_id, t.file_id, t.transcript_value,
    t.confidence, t.source_doc_type, t.file_name
)

SELECT
  p.person_gedcom_id,
  p.display_name,
  'occupation'                                AS fact_type,
  ARRAY_JOIN(om.all_tree_occupations, ' | ')  AS tree_value,
  om.transcript_value,
  om.file_id,
  om.file_name,
  om.source_doc_type,
  CASE
    WHEN om.all_tree_occupations = ARRAY()    THEN 'TRANSCRIPT_ONLY'
    WHEN om.any_match = 1                     THEN 'MATCH'
    ELSE                                           'CONFLICT'
  END AS status,
  CASE
    WHEN om.all_tree_occupations != ARRAY()
     AND om.any_match = 0                     THEN 'LOW'
    ELSE NULL
  END AS conflict_severity,
  CASE
    WHEN om.all_tree_occupations = ARRAY()
      THEN 'Occupation in transcript not in tree: ' || om.transcript_value
    ELSE NULL
  END AS notes,
  CURRENT_TIMESTAMP() AS refreshed_at
FROM occupation_match om
JOIN genealogy.gold_person_life p ON om.person_gedcom_id = p.person_gedcom_id;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 5 — Death year comparison

-- COMMAND ----------

-- %sql
INSERT INTO genealogy.gold_fact_comparison
  (person_gedcom_id, display_name, fact_type,
   tree_value, transcript_value, file_id, file_name, source_doc_type,
   status, conflict_severity, notes, refreshed_at)

WITH transcript_death_year AS (
  -- DISTINCT collapses multiple pages of the same document extracting the same value
  SELECT DISTINCT
    gtf.person_gedcom_id,
    gtf.file_id,
    gtf.fact_value  AS transcript_value,
    gtf.fact_year   AS transcript_year,
    gtf.confidence,
    gtf.source_doc_type,
    ot.file_name
  FROM genealogy.gold_transcript_facts gtf
  JOIN genealogy.ocr_transcriptions ot ON gtf.file_id = ot.file_id
  WHERE gtf.file_id IN (SELECT file_id FROM tmp_fc_scope)
    AND gtf.fact_type = 'death_year'
    AND gtf.confidence IN ('high','medium')
)

SELECT
  p.person_gedcom_id,
  p.display_name,
  'death_year'                       AS fact_type,
  CAST(p.death_year AS STRING)       AS tree_value,
  t.transcript_value,
  t.file_id,
  t.file_name,
  t.source_doc_type,
  CASE
    WHEN p.death_year IS NULL                             THEN 'TRANSCRIPT_ONLY'
    WHEN ABS(p.death_year - t.transcript_year) <= 1      THEN 'MATCH'
    ELSE                                                       'CONFLICT'
  END AS status,
  CASE
    WHEN p.death_year IS NOT NULL AND ABS(p.death_year - t.transcript_year) > 2  THEN 'HIGH'
    WHEN p.death_year IS NOT NULL AND ABS(p.death_year - t.transcript_year) = 2  THEN 'MEDIUM'
    ELSE NULL
  END AS conflict_severity,
  NULL AS notes,
  CURRENT_TIMESTAMP() AS refreshed_at
FROM transcript_death_year t
JOIN genealogy.gold_person_life p ON t.person_gedcom_id = p.person_gedcom_id;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 6 — Summary of comparison results

-- COMMAND ----------

-- %sql
SELECT
  status,
  conflict_severity,
  fact_type,
  COUNT(*) AS n
FROM genealogy.gold_fact_comparison
GROUP BY status, conflict_severity, fact_type
ORDER BY
  CASE status WHEN 'CONFLICT' THEN 1 WHEN 'TRANSCRIPT_ONLY' THEN 2
              WHEN 'MATCH' THEN 3 ELSE 4 END,
  CASE conflict_severity WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 3 ELSE 4 END,
  n DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 7 — Priority conflicts to review

-- COMMAND ----------

-- %sql
SELECT
  fc.person_gedcom_id,
  fc.display_name,
  fc.fact_type,
  fc.tree_value,
  fc.transcript_value,
  fc.conflict_severity,
  fc.file_name,
  fc.source_doc_type,
  fc.notes
FROM genealogy.gold_fact_comparison fc
WHERE fc.status = 'CONFLICT'
   OR (fc.status = 'TRANSCRIPT_ONLY' AND fc.fact_type IN ('birth_year','death_year','birth_place'))
ORDER BY
  CASE conflict_severity WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 WHEN 'LOW' THEN 3 ELSE 4 END,
  display_name;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 8 — Verify a specific person
-- MAGIC
-- MAGIC Substitute a real gedcom_id from your tree.

-- COMMAND ----------

-- %sql
SELECT
  fc.fact_type,
  fc.tree_value,
  fc.transcript_value,
  fc.status,
  fc.conflict_severity,
  fc.file_name,
  fc.notes
FROM genealogy.gold_fact_comparison fc
WHERE fc.person_gedcom_id = '@I123@'
ORDER BY fc.fact_type;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 10 — Open vs reviewed conflict counts

-- COMMAND ----------

-- %sql
SELECT fact_type, conflict_severity,
       SUM(CASE WHEN is_open_conflict THEN 1 ELSE 0 END)                         AS open_conflicts,
       SUM(CASE WHEN status = 'CONFLICT' AND review_applies THEN 1 ELSE 0 END)  AS reviewed_conflicts
FROM genealogy.gold_fact_comparison_reviewed
WHERE status = 'CONFLICT'
GROUP BY fact_type, conflict_severity
ORDER BY fact_type, conflict_severity;


