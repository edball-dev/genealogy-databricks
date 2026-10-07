-- id: DQ-025
-- title: Census match inconsistent with the page's own mention (likely OCR misread or wrong-person link)
-- severity: warning
-- guards_bug: 1218420653302709
-- known_failing:
-- existing_asana_task:
-- description: >
--   notebook_02 (page-level fact extraction) and notebook_01 Cell 5h2 both
--   check a census match against the mention row the matched person is linked
--   to (silver_document_person.person_index -> silver_transcript_person_mention),
--   using the age on the page: a mention whose age implies a birth year more
--   than 5 years from the tree's is not accepted. Two things can make that
--   check fire, and both need a person to look at the image:
--     * the OCR misread the name or age (found 2026-10-07: George Cuthbertson
--       Sr's 1841 head read as "Thos, 64" instead of "Geo, 69"; John
--       Ballantyne's 1901 age read as 51 instead of 57), or
--     * the tree person is not the person on the page (a wrong-person link,
--       e.g. a son linked as his father).
--   Left uncorrected, the person's facts from the page are stored unlinked
--   (notebook_02 logs FACT_EXTRACTION_LINK_REJECTED) or not linked at all, so
--   notebook_03 never compares them with the tree and the discrepancy goes
--   unseen.
--
--   Two reasons, one row each:
--     LINK_AGE_MISMATCH   - the person is linked to a census mention whose age
--                           implies a birth year >5 years from the tree's
--                           (the same rule notebook_02 and Cell 5h2 apply).
--     NO_MATCHING_MENTION - a HIGH/MEDIUM census match with no person_index
--                           although the file has mentions: Cell 5h2 found no
--                           mention with this name and a consistent age.
--
--   Reported at warning: a nonzero count is expected until the underlying
--   rows are reviewed (correct the transcript and mention rows as for
--   Cuthbertson 1841, or fix the match) and falls as that work is done. Watch
--   for growth after an OCR re-run or a notebook_01 rebuild. Census only:
--   other document types give an age at an event, not a true age, so the same
--   arithmetic does not apply.

WITH census AS (
  SELECT file_id, MAX(file_name) AS file_name, MAX(TRY_CAST(year AS INT)) AS doc_year
  FROM genealogy.ocr_transcriptions
  WHERE doc_type_detected = 'Census'
  GROUP BY file_id
)
SELECT sdp.file_id, c.file_name, sdp.person_gedcom_id, pl.display_name, sdp.match_method,
       'LINK_AGE_MISMATCH' AS reason,
       pl.birth_year AS tree_birth_year,
       m.name_raw AS mention_name, m.age_years AS mention_age,
       c.doc_year - m.age_years AS implied_birth_year
FROM genealogy.silver_document_person sdp
JOIN census c ON c.file_id = sdp.file_id
JOIN genealogy.gold_person_life pl ON pl.person_gedcom_id = sdp.person_gedcom_id
JOIN genealogy.silver_transcript_person_mention m
  ON m.file_id = sdp.file_id
 AND m.person_index = sdp.person_index
 AND m.page_index <=> sdp.page_index
WHERE sdp.match_confidence IN ('HIGH', 'MEDIUM')
  AND c.doc_year IS NOT NULL
  AND m.age_years IS NOT NULL
  AND TRY_CAST(pl.birth_year AS INT) IS NOT NULL
  AND ABS(c.doc_year - m.age_years - TRY_CAST(pl.birth_year AS INT)) > 5

UNION ALL

SELECT sdp.file_id, c.file_name, sdp.person_gedcom_id, pl.display_name, sdp.match_method,
       'NO_MATCHING_MENTION' AS reason,
       pl.birth_year AS tree_birth_year,
       CAST(NULL AS STRING) AS mention_name, CAST(NULL AS INT) AS mention_age,
       CAST(NULL AS INT) AS implied_birth_year
FROM genealogy.silver_document_person sdp
JOIN census c ON c.file_id = sdp.file_id
JOIN genealogy.gold_person_life pl ON pl.person_gedcom_id = sdp.person_gedcom_id
WHERE sdp.match_confidence IN ('HIGH', 'MEDIUM')
  AND sdp.person_index IS NULL
  AND EXISTS (
    SELECT 1 FROM genealogy.silver_transcript_person_mention mm
    WHERE mm.file_id = sdp.file_id
  );
