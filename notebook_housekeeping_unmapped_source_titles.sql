-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Housekeeping — unmapped source titles
-- MAGIC
-- MAGIC Creates `genealogy.gold_housekeeping_unmapped_source_titles`: one row per source title that
-- MAGIC `gold_source_coverage` categorised as `OTHER` (no row in `ref_source_doctype_map`, or an explicit `OTHER` row).
-- MAGIC These are the candidates to map. `citations_without_record_id` counts citations that go fully invisible
-- MAGIC (`NOT_APPLICABLE`) rather than showing as UNCOVERED, so work those first.
-- MAGIC
-- MAGIC Add mappings with `INSERT INTO genealogy.ref_source_doctype_map` via the MCP (see `notebook_ref_source_doctype_map`),
-- MAGIC then run `refresh_gold_source_coverage`, notebook 03 and notebook B. Do not map a combined-collection title
-- MAGIC (`looks_combined`) to one category.
-- MAGIC
-- MAGIC **Depends on:** `gold_source_coverage` (materialised, so it reflects the last `refresh_gold_source_coverage` run).
-- MAGIC View-only notebook: running it makes the executing identity the view owner (see `CLAUDE.md`).
-- MAGIC
-- MAGIC ## History
-- MAGIC - 2026-10-16: created (Asana 1218767063521380).

-- COMMAND ----------

CREATE OR REPLACE VIEW genealogy.gold_housekeeping_unmapped_source_titles AS
SELECT
  sc.source_title,
  COUNT(*)                                                         AS citations,
  COUNT(DISTINCT sc.person_gedcom_id)                              AS people,
  SUM(CASE WHEN sc.ancestry_record_id IS NULL THEN 1 ELSE 0 END)   AS citations_without_record_id,
  SUM(CASE WHEN sc.coverage_status = 'UNCOVERED' THEN 1 ELSE 0 END) AS uncovered_citations,
  sc.source_title RLIKE '(?i)(marriage|birth|death|baptism|burial|christening|banns|certificate|register|census)' AS looks_vital,
  sc.source_title RLIKE '(?i)(baptisms?,? (and )?(marriages|burials)|marriages?,? (and )?burials|baptism, marriage)' AS looks_combined,
  current_timestamp()                                              AS refreshed_at
FROM genealogy.gold_source_coverage sc
WHERE sc.source_category = 'OTHER'
GROUP BY sc.source_title;
