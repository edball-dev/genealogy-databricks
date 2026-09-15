-- id: DQ-009
-- title: gold_person_life year sanity
-- severity: critical
-- guards_bug:
-- known_failing: true
-- existing_asana_task: 1218492852523020
-- description: >
--   birth_year/death_year should be plausible calendar years (not a parser
--   placeholder) and death_year should never be before birth_year.
--   Originally scoped (Asana task 1218448003154261) as "outside 1500-2026,
--   or death_year < birth_year" — live testing (2026-09-14) found that
--   range would false-positive on real, confirmed medieval-era ancestors
--   in this tree (e.g. Ninian of Glendonwyn, b. 1482; Sir Simon of
--   Glendonwyn, b. 1378; Archibald 4th Earl of Douglas, b. 1368), so the
--   floor was widened to 100 -- low enough to never flag genuine
--   genealogical data, but still well above the specific corruption
--   pattern this check exists to catch (year = 1). Upper bound is the
--   current year, not a hardcoded 2026.
--
--   Live violations (15 rows, 2026-09-14) split into two distinct causes,
--   not one: 14 rows had birth_year or death_year = 1. Traced (not to
--   05a_parse_dates.ipynb, despite the initial guess -- see Asana task
--   1218448497347847 for the correction) to genealogy.gold_event, a
--   Databricks-only view at the time, always taking
--   silver_event_date.date_start, which is the year-1 sentinel for a
--   "BEF <year>" range instead of the real year held in date_end. One of
--   the 14, Whyte (@I_967017974@), had birth_year = 1 from a "BEF 1631"
--   *birth* date, confirming the bug wasn't death-date-specific -- it was
--   in the shared view logic, hit from either side.
--
--   Fixed and source-controlled in 07_materialise_gold_tables.ipynb
--   Cells 1-4, applied to production and confirmed live 2026-09-15:
--   violations dropped from 15 to 1. The remaining row, James Pratt
--   (@I332511987559@), has death_year (1825) one year before birth_year
--   (1826) from "abt 1826" vs "11 Dec 1826" -- a plausible approximate-date
--   rounding artifact, not the BEF bug; not code-fixable without checking
--   the actual source record, tracked separately at existing_asana_task
--   above. known_failing stays true because this is now the sole remaining
--   cause, tracked elsewhere rather than a fresh regression.

SELECT person_gedcom_id, display_name, birth_year, death_year, birth_date_raw, death_date_raw
FROM genealogy.gold_person_life
WHERE (birth_year IS NOT NULL AND (birth_year < 100 OR birth_year > YEAR(CURRENT_DATE())))
   OR (death_year IS NOT NULL AND (death_year < 100 OR death_year > YEAR(CURRENT_DATE())))
   OR (birth_year IS NOT NULL AND death_year IS NOT NULL AND death_year < birth_year);
