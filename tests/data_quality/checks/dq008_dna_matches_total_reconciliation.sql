-- id: DQ-008
-- title: gold_dna_coverage.dna_matches_total reconciliation
-- severity: critical
-- guards_bug: 1218420159647669
-- known_failing: true
-- existing_asana_task: 1218420159647669
-- description: >
--   gold_dna_coverage.dna_matches_total per branch should equal the count of
--   'DNA Match' rows in gold_person_dna for that branch. Asana task
--   1218420159647669 documents this as a Ball-only overcount (24 vs 21
--   expected). Live testing (2026-09-14) found the discrepancy affects ALL
--   8 branches, and in the opposite direction to what's documented —
--   dna_matches_total under-reports vs a raw recount, not overcounts. This
--   needs Ed to update or re-scope the Asana task; don't treat the existing
--   task description as accurate. Root cause not yet diagnosed (possibly
--   dedup-by-match-name vs raw-row-count — tested and it doesn't cleanly
--   explain the gap either, see chat history).

SELECT c.branch, c.dna_matches_total AS reported_total, COUNT(*) AS recomputed_total
FROM genealogy.gold_dna_coverage c
LEFT JOIN genealogy.gold_person_dna d
  ON d.branch = c.branch AND d.dna_role = 'DNA Match'
GROUP BY c.branch, c.dna_matches_total
HAVING c.dna_matches_total != COUNT(*);
