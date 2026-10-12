-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Housekeeping — citations without a source
-- MAGIC
-- MAGIC Creates `genealogy.gold_housekeeping_citations_without_source`: one row per source-less citation
-- MAGIC (`source_xref IS NULL`) in the Ancestry tree, for working through the cleanup (Asana 1219402948881812).
-- MAGIC Covers all people, not just direct ancestors; sort by `has_content DESC, generation_depth`.
-- MAGIC
-- MAGIC - `scope = 'PERSON'`: person-root citations (`silver_person_source`).
-- MAGIC - `scope = 'EVENT'`: event citations (`silver_event_source`). Events on `@I` records show that person; events on
-- MAGIC   `@F` family records show the husband and wife (`silver_event_participant`, `data_source = 'recorded'` only), with
-- MAGIC   the lowest `generation_depth` of the two and `is_direct_ancestor` true if either is one.
-- MAGIC - `has_content` is true when `citation_page`, `citation_text` or `citation_url` is populated (e.g. a real transcription), so those sort first.
-- MAGIC
-- MAGIC **Depends on:** `silver_person_source`, `silver_event_source`, `silver_event`, `silver_event_participant`, `silver_person`,
-- MAGIC `gold_research_person_signals`. It is a view, so it recomputes live — no refresh step.
-- MAGIC
-- MAGIC `ancestry_person_url` deep-links into Ancestry tree 8316967 (`@I_123@` maps to `-123`); for `@F` rows it links the husband.
-- MAGIC
-- MAGIC Five `@F` census events have no `silver_event_participant` rows: `display_name` falls back to the family id and `ancestry_person_url` is NULL.

-- COMMAND ----------

CREATE OR REPLACE VIEW genealogy.gold_housekeeping_citations_without_source AS
WITH person_names AS (
  SELECT person_gedcom_id,
         trim(concat_ws(' ', first(given_name), first(surname)))        AS display_name
  FROM genealogy.silver_person
  GROUP BY person_gedcom_id
),
fam_spouses AS (
  SELECT ep.event_id,
         concat_ws(' & ',
           collect_list(CASE WHEN ep.role = 'husband' THEN pn.display_name END),
           collect_list(CASE WHEN ep.role = 'wife'    THEN pn.display_name END)) AS display_name,
         min(CAST(s.generation_depth AS INT))                           AS generation_depth,
         max(coalesce(s.is_direct_ancestor, false))                     AS is_direct_ancestor,
         min(CASE WHEN ep.role = 'husband' THEN ep.person_id END)       AS husband_id
  FROM genealogy.silver_event_participant ep
  LEFT JOIN person_names pn ON pn.person_gedcom_id = ep.person_id
  LEFT JOIN genealogy.gold_research_person_signals s ON s.person_gedcom_id = ep.person_id
  WHERE ep.data_source = 'recorded'
    AND ep.role IN ('husband', 'wife')
  GROUP BY ep.event_id
),
src AS (
  SELECT 'PERSON'                                  AS scope,
         ps.person_gedcom_id                       AS person_gedcom_id,
         ps.person_gedcom_id                       AS record_xref,
         pn.display_name,
         CAST(s.generation_depth AS INT)           AS generation_depth,
         s.is_direct_ancestor,
         CAST(NULL AS STRING)                      AS event_type,
         CAST(NULL AS STRING)                      AS event_date,
         CAST(NULL AS STRING)                      AS event_place,
         ps.citation_note, ps.citation_text, ps.citation_page, ps.citation_url,
         ps.citation_root_line_no,
         ps.person_gedcom_id                       AS url_person_id
  FROM genealogy.silver_person_source ps
  LEFT JOIN person_names pn ON pn.person_gedcom_id = ps.person_gedcom_id
  LEFT JOIN genealogy.gold_research_person_signals s ON s.person_gedcom_id = ps.person_gedcom_id
  WHERE ps.source_xref IS NULL
    AND ps.person_gedcom_id IS NOT NULL

  UNION ALL

  SELECT 'EVENT',
         CASE WHEN e.record_xref LIKE '@I%' THEN e.record_xref END,
         e.record_xref,
         CASE WHEN e.record_xref LIKE '@I%' THEN pn.display_name ELSE coalesce(fs.display_name, e.record_xref) END,
         CASE WHEN e.record_xref LIKE '@I%' THEN CAST(s.generation_depth AS INT) ELSE fs.generation_depth END,
         CASE WHEN e.record_xref LIKE '@I%' THEN s.is_direct_ancestor ELSE fs.is_direct_ancestor END,
         e.event_type,
         CAST(e.event_date AS STRING),
         e.event_place,
         es.citation_note, es.citation_text, es.citation_page, es.citation_url,
         es.citation_root_line_no,
         CASE WHEN e.record_xref LIKE '@I%' THEN e.record_xref ELSE fs.husband_id END
  FROM genealogy.silver_event_source es
  JOIN genealogy.silver_event e ON e.event_id = es.event_id
  LEFT JOIN person_names pn ON pn.person_gedcom_id = e.record_xref
  LEFT JOIN genealogy.gold_research_person_signals s ON s.person_gedcom_id = e.record_xref
  LEFT JOIN fam_spouses fs ON fs.event_id = es.event_id
  WHERE es.source_xref IS NULL
)
SELECT
  scope,
  person_gedcom_id,
  record_xref,
  display_name,
  generation_depth,
  is_direct_ancestor,
  event_type,
  event_date,
  event_place,
  left(citation_note, 120)                                              AS citation_note,
  left(citation_text, 120)                                              AS citation_text,
  (citation_page IS NOT NULL OR citation_text IS NOT NULL OR citation_url IS NOT NULL) AS has_content,
  citation_root_line_no,
  concat('https://www.ancestry.co.uk/family-tree/person/tree/8316967/person/',
         regexp_replace(regexp_extract(url_person_id, '^@I(_?\\d+)@$', 1), '^_', '-'), '/facts')    AS ancestry_person_url,
  current_timestamp()                                                   AS refreshed_at
FROM src;
