-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Notebook 03: Fact Comparison — Tree vs Transcript
-- MAGIC
-- MAGIC Compares gold_transcript_facts against gold_person_life and gold_person_event_timeline.
-- MAGIC Run after notebook_02_fact_extraction has completed.
-- MAGIC
-- MAGIC **Change log:**
-- MAGIC - v1: Initial version
-- MAGIC - v1.1: Add DISTINCT to all transcript CTEs to prevent page-level fan-out from
-- MAGIC         multi-page PDFs producing duplicate comparison rows per fact per document.
-- MAGIC         Cell 4 (occupation): rewrite as many-to-many comparison — transcript value
-- MAGIC         matched against all tree occupations for the person; MATCH if any tree
-- MAGIC         occupation matches; tree_value shows all occupations pipe-separated.
-- MAGIC         Cell 4: add military rank exclusion filter (Pte, Cpl, Sgt etc.).
-- MAGIC         Cell 4: add conflict_severity = LOW for occupation conflicts (was NULL).

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 1 — Truncate and refresh gold_fact_comparison

-- COMMAND ----------

-- %sql
TRUNCATE TABLE genealogy.gold_fact_comparison;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 2 — Birth year comparison

-- COMMAND ----------

-- %sql
INSERT INTO genealogy.gold_fact_comparison
  (person_gedcom_id, display_name, fact_type,
   tree_value, transcript_value, file_id, file_name, source_doc_type,
   status, conflict_severity, notes)

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
  WHERE gtf.fact_type = 'birth_year'
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
  END AS notes
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
   status, conflict_severity, notes)

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
  WHERE gtf.fact_type = 'birth_place'
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
  END AS notes
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
   status, conflict_severity, notes)

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
  WHERE gtf.fact_type = 'occupation'
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
  END AS notes
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
   status, conflict_severity, notes)

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
  WHERE gtf.fact_type = 'death_year'
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
  NULL AS notes
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
