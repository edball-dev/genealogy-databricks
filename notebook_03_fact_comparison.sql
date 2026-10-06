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
-- MAGIC - v1.2: Cell 9 creates view gold_fact_comparison_reviewed, applying Ed's review marks from
-- MAGIC         silver_fact_conflict_review (see notebook_fact_conflict_review_init.sql — prerequisite).
-- MAGIC         Consumers counting/listing conflicts should use its is_open_conflict flag.
-- MAGIC - v1.3: TREE_GAP review outcome (task 1219234333541171). Reason code TREE_GAP = legitimate
-- MAGIC         occupation the tree should record. Never an open conflict; applies even if the tree's
-- MAGIC         occupation list later changes; `is_tree_gap` is TRUE only while the tree still lacks the
-- MAGIC         value (self-clears when the comparison becomes MATCH). Cell 11 adds the worklist view
-- MAGIC         `gold_housekeeping_tree_gap_occupations`.

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

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 9 — gold_fact_comparison_reviewed (applies review marks)
-- MAGIC
-- MAGIC Prerequisite: `silver_fact_conflict_review` exists (notebook_fact_conflict_review_init).
-- MAGIC
-- MAGIC Key: file_id + person_gedcom_id + fact_type + transcript_value (null-safe). The latest review
-- MAGIC per key wins. `review_applies` is TRUE only while the tree value is unchanged since the
-- MAGIC review (`reviewed_tree_value`); a changed tree value reopens the conflict, and a changed
-- MAGIC transcript value or new file_id is simply a new key (open). Reviewed rows stay visible here.
-- MAGIC `is_open_conflict` is the single definition of "conflict still needing attention".
-- MAGIC
-- MAGIC `TREE_GAP` reviews (legitimate occupation missing from the tree) are the exception to the
-- MAGIC tree-value snapshot rule: adding one occupation changes `tree_value`, which must not reopen the
-- MAGIC person's other TREE_GAP rows as conflicts. They stay applied until the comparison is a MATCH.

-- COMMAND ----------

-- %sql
CREATE OR REPLACE VIEW genealogy.gold_fact_comparison_reviewed AS
WITH latest_review AS (
  SELECT * EXCEPT (rn)
  FROM (
    SELECT r.*,
           ROW_NUMBER() OVER (
             PARTITION BY file_id, person_gedcom_id, fact_type, transcript_value
             ORDER BY reviewed_at DESC NULLS LAST
           ) AS rn
    FROM genealogy.silver_fact_conflict_review r
  )
  WHERE rn = 1
)
SELECT
  fc.*,
  rv.review_status,
  rv.reason_code        AS review_reason_code,
  rv.notes              AS review_notes,
  rv.reviewed_at,
  rv.reviewed_tree_value,
  COALESCE(rv.review_status IS NOT NULL
           AND (rv.reason_code = 'TREE_GAP' OR rv.reviewed_tree_value <=> fc.tree_value), FALSE)
                                                                              AS review_applies,
  (fc.status = 'CONFLICT'
   AND NOT COALESCE(rv.review_status IS NOT NULL
                    AND (rv.reason_code = 'TREE_GAP' OR rv.reviewed_tree_value <=> fc.tree_value), FALSE))
                                                                              AS is_open_conflict,
  -- Legitimate occupation the tree lacks: reviewed TREE_GAP and the comparison still isn't a MATCH.
  -- Clears on its own once the tree gains the occupation (status flips to MATCH on the next rebuild).
  COALESCE(rv.reason_code = 'TREE_GAP'
           AND fc.fact_type = 'occupation'
           AND fc.status IN ('CONFLICT', 'TRANSCRIPT_ONLY'), FALSE)           AS is_tree_gap
FROM genealogy.gold_fact_comparison fc
LEFT JOIN latest_review rv
  ON  rv.file_id          = fc.file_id
  AND rv.person_gedcom_id = fc.person_gedcom_id
  AND rv.fact_type        = fc.fact_type
  AND rv.transcript_value <=> fc.transcript_value;

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


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell 11 — gold_housekeeping_tree_gap_occupations (Housekeeping worklist)
-- MAGIC
-- MAGIC One row per document occupation Ed has marked `TREE_GAP` that the tree still lacks, so he can add
-- MAGIC a dated occupation in Ancestry. Rows vanish by themselves once the tree matches (no review step).
-- MAGIC `NOT_AN_OCCUPATION` rows (students, club roles) are deliberately absent. Feeds
-- MAGIC `SIGNAL_TRANSCRIPT_ONLY_FACTS` via `is_tree_gap`.

-- COMMAND ----------

-- %sql
CREATE OR REPLACE VIEW genealogy.gold_housekeeping_tree_gap_occupations AS
SELECT
  fc.person_gedcom_id,
  fc.display_name,
  fc.file_id,
  fc.file_name,
  ot.year               AS document_year,
  fc.source_doc_type,
  fc.transcript_value   AS occupation_as_written,
  fc.tree_value         AS tree_occupations,
  fc.review_notes,
  fc.reviewed_at
FROM genealogy.gold_fact_comparison_reviewed fc
LEFT JOIN genealogy.ocr_transcriptions ot ON ot.file_id = fc.file_id
WHERE fc.is_tree_gap;
