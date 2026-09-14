-- id: DQ-009
-- title: gold_person_life year sanity
-- severity: critical
-- guards_bug:
-- known_failing: true
-- existing_asana_task: 1218448497347847
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
--   Live violations (15 rows) split into two distinct causes, not one:
--   14 rows have birth_year or death_year = 1, all tracing to the known
--   date-parser bug tracked by existing_asana_task 1218448497347847
--   ("Bef./BEF YYYY" qualifiers parse to year 1 instead of the real year,
--   in 05a_parse_dates.ipynb). This check adds one new data point to that
--   bug: Whyte (@I_967017974@) has birth_year = 1 from a "BEF 1631"
--   *birth* date, confirming the bug isn't death-date-specific -- it's
--   the shared qualifier-parsing logic, wherever it's called from.
--   The 15th row, James Pratt (@I332511987559@), has death_year (1825)
--   one year before birth_year (1826) from "abt 1826" vs "11 Dec 1826" --
--   a plausible approximate-date rounding artifact, not the same bug;
--   flagged for manual review rather than folded into the parser-bug
--   count.

SELECT person_gedcom_id, display_name, birth_year, death_year, birth_date_raw, death_date_raw
FROM genealogy.gold_person_life
WHERE (birth_year IS NOT NULL AND (birth_year < 100 OR birth_year > YEAR(CURRENT_DATE())))
   OR (death_year IS NOT NULL AND (death_year < 100 OR death_year > YEAR(CURRENT_DATE())))
   OR (birth_year IS NOT NULL AND death_year IS NOT NULL AND death_year < birth_year);
