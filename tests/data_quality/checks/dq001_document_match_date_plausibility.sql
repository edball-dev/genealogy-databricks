-- id: DQ-001
-- title: Document match date plausibility (BUG-001 regression)
-- severity: critical
-- guards_bug: 1218420283786561
-- known_failing: true
-- existing_asana_task: 1218448497347847
-- description: >
--   A document's parsed year must fall within the matched person's plausible
--   lifespan (birth_year .. death_year, or birth_year .. birth_year+110 if no
--   death_year), with a symmetric 2yr buffer on both ends. Regression guard
--   for the year_plausible fix in notebook_01_document_matching.ipynb Cell 5
--   (2026-09-13). Also structurally covers the same-name/different-generation
--   collision check scoped in Asana task 1218430419299788 (Eliza Easter x2,
--   John/William Pearson x2+) — any such collision that mis-attaches a
--   document will show up here as an implausible date, so no separate check
--   was built for that task. There is deliberately no DQ-003: that GID was
--   reserved for the collision check, which turned out to be structurally
--   identical to this one and was folded in here instead of duplicated.
--
--   Check-design fix (2026-09-15): the buffer was previously applied only on
--   the late side (`death_year + 2`), not the early side. The matching
--   algorithm this check guards (notebook_01_document_matching.ipynb Cell 5)
--   already tolerates `birth_date - 2` years when scoring year_plausible, so
--   an unbuffered lower bound here made the check stricter than the code it
--   guards and produced a false positive (John Orme, @I332787034185@: 1735
--   doc vs 1736 birth — inside the matcher's own tolerance, but outside this
--   check's). Added the matching `- 2` on the lower bound so the check
--   mirrors the matcher's actual tolerance window instead of an arbitrary
--   tighter one. Confirmed live: violations dropped 10 -> 9, only John
--   Orme's row cleared.
--
--   Triage of the remaining 9 (2026-09-14/15, see task 1218447137381460):
--   6 rows (David Edmiston x5, Jonathan Croxon x1) are not a document-
--   matching problem — gold_person_life.death_year=1 for both, traced to a
--   date-parser bug that mis-parses "Bef. YYYY" qualifiers into year 1
--   (05a_parse_dates.ipynb), tracked separately at existing_asana_task above.
--   The remaining 3 (Elizabeth Balls, Thomas Thorpe, Henry Easter) have clean
--   birth/death years with no parsing issue, and the matched document's year
--   is 4-12 years outside lifespan — genuine candidates for a wrong-person
--   match or mis-parsed document year, filed separately for manual review at
--   Asana task 1218488426839724. known_failing is set to true because both
--   causes are tracked elsewhere rather than being a fresh regression in the
--   matching algorithm itself (0 of 9 rows point to a regression there).

SELECT DISTINCT dp.file_id, dp.person_gedcom_id, dp.display_name, dp.match_method,
       dp.match_confidence, t.year AS doc_year, pl.birth_year, pl.death_year
FROM genealogy.silver_document_person dp
JOIN genealogy.ocr_transcriptions t ON t.file_id = dp.file_id
JOIN genealogy.gold_person_life pl ON pl.person_gedcom_id = dp.person_gedcom_id
WHERE t.year RLIKE '^[0-9]{4}'
  AND CAST(SUBSTRING(t.year, 1, 4) AS INT) NOT BETWEEN
      pl.birth_year - 2
      AND COALESCE(pl.death_year, pl.birth_year + 110) + 2;
