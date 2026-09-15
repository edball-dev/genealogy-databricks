-- id: DQ-001
-- title: Document match date plausibility (BUG-001 regression)
-- severity: critical
-- guards_bug: 1218420283786561
-- known_failing: false
-- existing_asana_task:
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
--   6 rows (David Edmiston x5, Jonathan Croxon x1) were not a document-
--   matching problem — gold_person_life.death_year=1 for both, traced not to
--   a parser bug but to genealogy.gold_event (a Databricks-only view at the
--   time) always taking silver_event_date.date_start, which is the year-1
--   sentinel for a "BEF <year>" range instead of the real year held in
--   date_end. Fixed and source-controlled in 07_materialise_gold_tables.ipynb
--   Cells 1-4, applied to production and confirmed live 2026-09-15: those 6
--   rows cleared, DQ-001 dropped from 9 to 3 violations. Full root-cause
--   trace and resolution at Asana task 1218448497347847 (now closed).
--
--   The remaining 3 (Elizabeth Balls, Thomas Thorpe, Henry Easter) had clean
--   birth/death years with no parsing issue, and the matched document's year
--   was 4-12 years outside lifespan. Manual review (task 1218488426839724,
--   closed) found two distinct causes:
--
--   Elizabeth Balls and Henry Easter were genuine wrong-person matches, not
--   data problems: the source documents spell a surname variant ("Balls" for
--   tree surname "Baulls"; "Easter" for tree surname "Eastoe") that had no
--   matching alias, so the matcher's fallback logic forced a match onto the
--   nearest same-literal-surname person even though every such candidate was
--   chronologically impossible. Fixed by adding a manually-curated
--   SURNAME_VARIANT row to silver_person_alias for each (see
--   notebook_01_document_matching.ipynb Cell 1) and re-pointing the two
--   silver_document_person rows directly. Confirmed live: DQ-001 dropped
--   from 3 to 1. The underlying matcher gaps that let a chronologically
--   impossible candidate win in the first place (Cell 5's zero-plausible-
--   candidates fallback; Cell 5c sibling-consensus with no year check at
--   all) were tracked at task 1218506594311016 and fixed 2026-09-15 (Cell 5's
--   NOT EXISTS fallback removed in favour of logging to the new
--   silver_document_match_exception table; Cell 5c now checks the file's own
--   year against the consensus person before applying it, same fallback).
--
--   Thomas Thorpe was CONFIRMED CORRECTLY matched all along — not a
--   mismatch. The 1894 doc is a probate/estate-sale notice ("...under the
--   Will of the late Mr. Thomas Thorpe"), which legitimately postdates his
--   1888 death by 6 years; this check had no exemption for a
--   NewspaperClipping like this one (tracked at 1218506783096280). Fixed
--   2026-09-15: the upper bound (death_year+2, or birth_year+110 with no
--   death_year) no longer applies to doc_type_detected = 'NewspaperClipping'
--   — that doc type can legitimately postdate death by years or decades
--   (obituaries, memorial notices, estate-sale auctions), so only this
--   check's lower (birth-side) bound still applies to it, mirroring the same
--   exemption added to notebook_01_document_matching.ipynb Cell 5/Cell 5c's
--   own year_plausible logic (task 1218506594311016) so this check never
--   drifts stricter than the algorithm it guards, the same principle behind
--   the 2026-09-14 lower-bound fix above.
--
--   Probate/Will were also considered for this exemption (per this task's
--   original proposed wording) but scoped back out after live validation:
--   removing their upper bound entirely turned 4 previously-clean Corner/
--   Cope Will matches into new LOW-confidence ties, because a candidate who
--   died decades before the document (e.g. John Corner, d.1742, vs a
--   CORNER_John_1760_Will.jpg document) became "plausible" once the upper
--   bound was gone. No live Probate/Will case in this corpus actually needed
--   the exemption — only Thorpe's NewspaperClipping did — so the fix stays
--   scoped to what was observed, not the broader original wording. Confirmed
--   live: Thomas Thorpe's row cleared, the 5 Corner/Cope Will rows are
--   unaffected, DQ-001 at 0 violations. known_failing flipped to false — a
--   future violation here is a genuine regression, not a known/accepted gap.

SELECT DISTINCT dp.file_id, dp.person_gedcom_id, dp.display_name, dp.match_method,
       dp.match_confidence, t.doc_type_detected, t.year AS doc_year, pl.birth_year, pl.death_year
FROM genealogy.silver_document_person dp
JOIN genealogy.ocr_transcriptions t ON t.file_id = dp.file_id
JOIN genealogy.gold_person_life pl ON pl.person_gedcom_id = dp.person_gedcom_id
WHERE t.year RLIKE '^[0-9]{4}'
  AND (
    CAST(SUBSTRING(t.year, 1, 4) AS INT) < pl.birth_year - 2
    OR (
      CAST(SUBSTRING(t.year, 1, 4) AS INT) > COALESCE(pl.death_year, pl.birth_year + 110) + 2
      AND t.doc_type_detected != 'NewspaperClipping'
    )
  );
