-- ============================================================
-- gold_person_fact_summary (view)
--
-- Run manually as the view owner (Ed) only when the definition changes. If run by
-- another identity, follow with:
--   ALTER VIEW genealogy.gold_person_fact_summary OWNER TO `<owner>`;
--
-- Change 2026-10-16 (Asana: SIGNAL_MIGRANT false positive for England + United Kingdom):
--   num_countries now treats England, Scotland, Wales and United Kingdom as one
--   country group ('Great Britain'), so a move inside Great Britain no longer
--   fires SIGNAL_MIGRANT. Ireland is deliberately NOT in the group and still counts
--   as a separate country. Observed effect: people with num_countries > 1 fall
--   from 446 to 347. Everything else is unchanged from the previous definition.
--   After running: re-run notebook B (gold_research_person_signals is materialised).
-- ============================================================

CREATE OR REPLACE VIEW genealogy.gold_person_fact_summary (
  person_gedcom_id,
  given_name,
  surname,
  birth_date,
  birth_place,
  death_date,
  death_place,
  has_given_name,
  has_surname,
  num_parents,
  has_birth_event,
  birth_after_1837,
  birth_date_precision,
  has_1841_census,
  has_1851_census,
  has_1861_census,
  has_1871_census,
  has_1881_census,
  has_1891_census,
  has_1901_census,
  has_1911_census,
  has_1921_census,
  has_1939_register,
  num_marriages,
  marriage_after_1837,
  earliest_marriage_year,
  marriage_date_precision,
  num_occupations,
  num_residences,
  num_military,
  num_child_births,
  max_days_between_child_births,
  num_migration,
  num_countries,
  has_death_event,
  death_after_1837,
  death_date_precision,
  num_family_units)
WITH SCHEMA COMPENSATION
AS SELECT
  p.person_gedcom_id,
  p.given_name,
  p.surname,
  p.birth_date,
  p.birth_place,
  p.death_date,
  p.death_place,
  IF(p.given_name IS NOT NULL, 1, 0)                                        AS has_given_name,
  IF(p.surname IS NOT NULL, 1, 0)                                           AS has_surname,
  COUNT(DISTINCT r.person_id_1) FILTER (WHERE r.relationship_type = 'parent')
                                                                            AS num_parents,
  p.has_birth_event,
  CASE WHEN (p.birth_date >= '1837-01-01') THEN 1 ELSE 0 END               AS birth_after_1837,
  MAX(CASE WHEN event_type IN ('BIRT', 'BAPM', 'CHR')
            AND event_role IN ('principal')
           THEN d.date_precision ELSE NULL END)                             AS birth_date_precision,
  -- Census flags
  CASE WHEN COUNT(DISTINCT es.source_xref)
    FILTER (WHERE s.source_title LIKE '1841%Census') > 0 THEN 1 ELSE 0 END AS has_1841_census,
  CASE WHEN COUNT(DISTINCT es.source_xref)
    FILTER (WHERE s.source_title LIKE '1851%Census') > 0 THEN 1 ELSE 0 END AS has_1851_census,
  CASE WHEN COUNT(DISTINCT es.source_xref)
    FILTER (WHERE s.source_title LIKE '1861%Census') > 0 THEN 1 ELSE 0 END AS has_1861_census,
  CASE WHEN COUNT(DISTINCT es.source_xref)
    FILTER (WHERE s.source_title LIKE '1871%Census') > 0 THEN 1 ELSE 0 END AS has_1871_census,
  CASE WHEN COUNT(DISTINCT es.source_xref)
    FILTER (WHERE s.source_title LIKE '1881%Census') > 0 THEN 1 ELSE 0 END AS has_1881_census,
  CASE WHEN COUNT(DISTINCT es.source_xref)
    FILTER (WHERE s.source_title LIKE '1891%Census') > 0 THEN 1 ELSE 0 END AS has_1891_census,
  CASE WHEN COUNT(DISTINCT es.source_xref)
    FILTER (WHERE s.source_title LIKE '1901%Census') > 0 THEN 1 ELSE 0 END AS has_1901_census,
  CASE WHEN COUNT(DISTINCT es.source_xref)
    FILTER (WHERE s.source_title LIKE '1911%Census') > 0 THEN 1 ELSE 0 END AS has_1911_census,
  CASE WHEN COUNT(DISTINCT es.source_xref)
    FILTER (WHERE s.source_title LIKE '1921%Census') > 0 THEN 1 ELSE 0 END AS has_1921_census,
  CASE WHEN COUNT(DISTINCT es.source_xref)
    FILTER (WHERE s.source_title = '1939 England and Wales Register') > 0
    THEN 1 ELSE 0 END                                                       AS has_1939_register,
  -- Marriage
  COUNT(DISTINCT t.event_id) FILTER (WHERE event_type = 'MARR')            AS num_marriages,
  CASE WHEN MAX(t.event_date_parsed) FILTER (WHERE event_type = 'MARR') > '1837-01-01'
       THEN 1 ELSE 0 END                                                    AS marriage_after_1837,
  -- earliest marriage year: used in SIGNAL_IMPRECISE_DATES guard
  YEAR(MIN(t.event_date_parsed) FILTER (WHERE event_type = 'MARR'))        AS earliest_marriage_year,
  -- marriage date precision: used in SIGNAL_IMPRECISE_DATES
  MAX(CASE WHEN event_type = 'MARR'
            AND event_role = 'principal'
           THEN d.date_precision ELSE NULL END)                             AS marriage_date_precision,
  -- Occupation / residence / military
  COUNT(DISTINCT t.event_id) FILTER (WHERE event_type = 'OCCU')            AS num_occupations,
  COUNT(DISTINCT t.event_id) FILTER (WHERE event_type = 'RESI')            AS num_residences,
  COUNT(DISTINCT t.event_id) FILTER (WHERE event_type = '_MILT')           AS num_military,
  -- Children
  COUNT(DISTINCT t.event_id)
    FILTER (WHERE event_type IN ('BIRT', 'BAPM', 'CHR')
              AND event_role IN ('father', 'mother'))                       AS num_child_births,
  MAX(days_since_previous_same_event)
    FILTER (WHERE event_type IN ('BIRT', 'BAPM', 'CHR')
              AND event_role IN ('father', 'mother'))                       AS max_days_between_child_births,
  -- Migration
  COUNT(DISTINCT t.event_id)
    FILTER (WHERE event_type IN ('EMIG', 'IMMI', 'NATU')
               OR (event_type = 'EVEN' AND event_value IN ('Departure', 'Arrival')))
                                                                            AS num_migration,
  -- num_countries: England, Scotland, Wales and United Kingdom count as one group
  -- (Great Britain), so moves inside Great Britain are not migration. Ireland stays separate.
  COUNT(DISTINCT CASE WHEN t.event_country_inferred IN ('England', 'Scotland', 'Wales', 'United Kingdom')
                      THEN 'Great Britain'
                      ELSE t.event_country_inferred END)                    AS num_countries,
  -- Death
  p.has_death_event,
  CASE WHEN (p.death_date >= '1837-01-01') THEN 1 ELSE 0 END               AS death_after_1837,
  MAX(CASE WHEN event_type IN ('DEAT', 'BUR', 'CREM')
            AND event_role IN ('principal')
           THEN d.date_precision ELSE NULL END)                             AS death_date_precision,
  -- Family units: count of distinct family units from silver_family.
  -- Used by SIGNAL_MULTIPLE_SPOUSES in place of num_marriages, which
  -- overcounts because MARR events appear on both spouses' timelines.
  COALESCE(MAX(fam.num_family_units), 0)                                    AS num_family_units

FROM genealogy.gold_person_life p
LEFT JOIN genealogy.gold_person_event_timeline t  ON t.person_gedcom_id = p.person_gedcom_id
LEFT JOIN genealogy.silver_event_date d           ON d.event_id = t.event_id
LEFT JOIN genealogy.silver_event_source es        ON es.event_id = t.event_id
                                       AND t.event_role = 'principal'
LEFT JOIN genealogy.silver_source s               ON s.record_xref = es.source_xref
LEFT JOIN genealogy.silver_relationship r         ON r.person_id_2 = p.person_gedcom_id
-- Pre-aggregated family unit count: one row per person, so MAX() is safe
LEFT JOIN (
  SELECT person_gedcom_id, COUNT(*) AS num_family_units
  FROM (
    SELECT husband_gedcom_id AS person_gedcom_id FROM genealogy.silver_family
    UNION ALL
    SELECT wife_gedcom_id    AS person_gedcom_id FROM genealogy.silver_family
  )
  GROUP BY person_gedcom_id
) fam ON fam.person_gedcom_id = p.person_gedcom_id
GROUP BY
  p.person_gedcom_id,
  p.given_name,
  p.surname,
  p.birth_date,
  p.birth_place,
  p.has_birth_event,
  p.death_date,
  p.death_place,
  p.has_death_event;
