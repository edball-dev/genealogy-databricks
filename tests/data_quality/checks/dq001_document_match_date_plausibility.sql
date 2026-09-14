-- id: DQ-001
-- title: Document match date plausibility (BUG-001 regression)
-- severity: critical
-- guards_bug: 1218420283786561
-- known_failing: false
-- existing_asana_task:
-- description: >
--   A document's parsed year must fall within the matched person's plausible
--   lifespan (birth_year .. death_year, or birth_year .. birth_year+110 if no
--   death_year, +2yr buffer for registration/probate lag). Regression guard
--   for the year_plausible fix in notebook_01_document_matching.ipynb Cell 5
--   (2026-09-13). Also structurally covers the same-name/different-generation
--   collision check scoped in Asana task 1218430419299788 (Eliza Easter x2,
--   John/William Pearson x2+) — any such collision that mis-attaches a
--   document will show up here as an implausible date, so no separate check
--   was built for that task. There is deliberately no DQ-003: that GID was
--   reserved for the collision check, which turned out to be structurally
--   identical to this one and was folded in here instead of duplicated.

SELECT DISTINCT dp.file_id, dp.person_gedcom_id, dp.display_name, dp.match_method,
       dp.match_confidence, t.year AS doc_year, pl.birth_year, pl.death_year
FROM genealogy.silver_document_person dp
JOIN genealogy.ocr_transcriptions t ON t.file_id = dp.file_id
JOIN genealogy.gold_person_life pl ON pl.person_gedcom_id = dp.person_gedcom_id
WHERE t.year RLIKE '^[0-9]{4}'
  AND CAST(SUBSTRING(t.year, 1, 4) AS INT) NOT BETWEEN
      pl.birth_year
      AND COALESCE(pl.death_year, pl.birth_year + 110) + 2;
