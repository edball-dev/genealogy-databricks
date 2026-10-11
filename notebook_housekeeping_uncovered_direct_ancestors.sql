-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Housekeeping — uncovered citations for direct ancestors
-- MAGIC
-- MAGIC Creates `genealogy.gold_housekeeping_uncovered_direct_ancestors`: one row per citation on a direct
-- MAGIC ancestor (generation 1-4) that has no matched document (`gold_source_coverage.coverage_status = 'UNCOVERED'`).
-- MAGIC Once the open matching fixes clear, every remaining row is a genuinely missing document to
-- MAGIC download, locate or OCR.
-- MAGIC
-- MAGIC **Depends on:** `gold_source_coverage` (notebook_01 Cell 8), `gold_research_person_signals`, `silver_person`.
-- MAGIC It is a view, so it recomputes live — no refresh step.
-- MAGIC
-- MAGIC `uncovered_count` is the person's total uncovered citations (`SIGNAL_UNCOVERED_SOURCES` fires at 2+).
-- MAGIC `ancestry_person_url` deep-links into Ancestry tree 8316967.

-- COMMAND ----------

CREATE OR REPLACE VIEW genealogy.gold_housekeeping_uncovered_direct_ancestors AS
SELECT
  sc.person_gedcom_id,
  trim(concat_ws(' ', p.given_name, p.surname))                         AS display_name,
  CAST(s.generation_depth AS INT)                                       AS generation_depth,
  sc.source_title,
  sc.source_category,
  sc.citation_date,
  COUNT(*) OVER (PARTITION BY sc.person_gedcom_id)                      AS uncovered_count,
  sc.citation_url,
  concat('https://www.ancestry.co.uk/family-tree/person/tree/8316967/person/',
         regexp_extract(sc.person_gedcom_id, '(\\d+)', 1), '/facts')    AS ancestry_person_url,
  current_timestamp()                                                   AS refreshed_at
FROM genealogy.gold_source_coverage sc
JOIN genealogy.gold_research_person_signals s ON s.person_gedcom_id = sc.person_gedcom_id
JOIN genealogy.silver_person p                ON p.person_gedcom_id = sc.person_gedcom_id
WHERE sc.coverage_status = 'UNCOVERED'
  AND s.is_direct_ancestor
  AND CAST(s.generation_depth AS INT) BETWEEN 1 AND 4;
