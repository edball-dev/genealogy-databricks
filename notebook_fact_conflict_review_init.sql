-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Fact conflict review — one-off DDL (needs Ed's sign-off / run)
-- MAGIC
-- MAGIC Durable record of fact conflicts Ed has reviewed. `gold_fact_comparison` is TRUNCATEd and
-- MAGIC rebuilt by `notebook_03_fact_comparison`, so review marks live here and are applied by the
-- MAGIC view `genealogy.gold_fact_comparison_reviewed` (created at the end of notebook 03).
-- MAGIC
-- MAGIC **Run order:** this notebook (once) → grants → notebook_03 → notebook_B.
-- MAGIC
-- MAGIC **Statuses:** `FALSE_POSITIVE` (not a real conflict), `RESOLVED` (real; fixed at source —
-- MAGIC tree, transcript or match corrected), `ACCEPTED` (genuine discrepancy in the record itself,
-- MAGIC assessed, nothing further to do — optional, Ed to confirm).
-- MAGIC
-- MAGIC **Reason codes (suggested):** PLACE_GRANULARITY, LIFE_STAGE_OCCUPATION, TRANSCRIPTION_ERROR,
-- MAGIC WRONG_PERSON_MATCH, TREE_CORRECTED, CENSUS_AGE_DRIFT, OTHER.
-- MAGIC
-- MAGIC **Conversational review:** Claude lists open conflicts (`gold_fact_comparison_reviewed WHERE
-- MAGIC is_open_conflict`) grouped by person; Ed says what to mark; Claude writes rows with
-- MAGIC `execute_write_sql` as a single-line `INSERT ... SELECT` (bulk-friendly, e.g. all LOW
-- MAGIC occupation conflicts for one person), copying `tree_value` into `reviewed_tree_value`.

-- COMMAND ----------

CREATE TABLE IF NOT EXISTS genealogy.silver_fact_conflict_review (
  file_id             STRING NOT NULL,
  person_gedcom_id    STRING NOT NULL,
  fact_type           STRING NOT NULL,
  transcript_value    STRING,
  reviewed_tree_value STRING COMMENT 'tree_value at review time; review stops applying if the tree value changes',
  review_status       STRING NOT NULL COMMENT 'FALSE_POSITIVE | RESOLVED | ACCEPTED',
  reason_code         STRING,
  notes               STRING,
  reviewed_at         TIMESTAMP
)
USING DELTA
COMMENT 'Durable review marks for gold_fact_comparison conflicts. Key = file_id + person_gedcom_id + fact_type + transcript_value. Survives notebook_03 rebuilds.';

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Grants (run by Ed; Claude has no DDL access)
-- MAGIC Also add `genealogy.silver_fact_conflict_review` to the databricks-mcp write allow-list.

-- COMMAND ----------

GRANT SELECT, MODIFY ON TABLE genealogy.silver_fact_conflict_review TO `3a592309-eaa3-472c-8bac-9ddddd2af1ff`;
