-- id: DQ-008
-- title: gold_dna_coverage.dna_matches_total reconciliation
-- severity: critical
-- guards_bug: 1218420159647669
-- known_failing: false
-- existing_asana_task: 1218420159647669
-- description: >
--   gold_dna_coverage.dna_matches_total per branch should equal the count of
--   distinct people with a 'DNA Match' row in gold_person_dna for that
--   branch. Fixed 2026-09-15 (triaged on Asana task 1218420159647669): the
--   check previously recomputed via a raw COUNT(*) of gold_person_dna rows,
--   but gold_person_dna is one row per person x DNA citation, so a person
--   matched via more than one kit (e.g. both ed_ancestry and dad_ancestry)
--   produces multiple rows for the same person. gold_dna_coverage
--   deliberately dedupes with COUNT(DISTINCT person_gedcom_id) — this check
--   now matches that semantic instead of flagging every multi-kit match as
--   a false positive (was firing on 7/8 branches).

SELECT c.branch, c.dna_matches_total AS reported_total, COUNT(DISTINCT d.person_gedcom_id) AS recomputed_total
FROM genealogy.gold_dna_coverage c
LEFT JOIN genealogy.gold_person_dna d
  ON d.branch = c.branch AND d.dna_role = 'DNA Match'
GROUP BY c.branch, c.dna_matches_total
HAVING c.dna_matches_total != COUNT(DISTINCT d.person_gedcom_id);
