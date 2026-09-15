-- id: DQ-011
-- title: Age-bearing document types not getting age-based corroboration
-- severity: warning
-- guards_bug: 1218506594311016
-- known_failing:
-- existing_asana_task:
-- description: >
--   notebook_01_document_matching.ipynb Cell 5 extracts a document-stated
--   age (doc_stated_age) via doc-type-specific regex for Census,
--   BurialRegister, MonumentalInscription, NewspaperClipping, and
--   MilitaryRecord, then uses it to score doc_type_affinity (tier 0/1) when
--   disambiguating multiple same-name candidates. A silent extraction
--   failure -- a doc type added to that scoring list without a matching
--   regex branch, or a regex that stops matching after an OCR/transcription
--   format change -- doesn't error; it just falls through to the worst tier
--   (ELSE 2, no age evidence used) and the match gets decided by other
--   signals alone (ancestor proximity, tie-break ordering) instead. This is
--   exactly the gap that let EASTER_Henry_1848_Burial.jpg match the wrong
--   Henry for months: BurialRegister was never in Cell 5's age-extraction
--   list at all, so the register's own "Age 80" was never read, even though
--   the file had multiple same-name candidates (task 1218506594311016).
--
--   silver_document_person.notes only includes "doc_type_affinity=N" when
--   doc_type_affinity < 2 (see Cell 5's CONCAT_WS) -- so a row for an
--   age-bearing doc type with multiple candidates and no
--   "doc_type_affinity=" fragment in notes means age extraction produced no
--   usable evidence for that file. Restricted to candidate_count > 1: a
--   unique (candidate_count=1) match doesn't need age corroboration to be
--   correct, so a NULL/failed extraction there isn't informative.
--
--   Reported at warning: a nonzero count doesn't necessarily mean a wrong
--   match (ancestor proximity/uniqueness can still be correct on their own),
--   but it means this file's match isn't corroborated by the document's own
--   stated age and is worth a manual spot-check, and a sudden jump signals a
--   regex/extraction regression worth investigating before it produces a
--   wrong match like the Henry Easter case.

SELECT dp.file_id, dp.file_name, t.doc_type_detected, dp.person_gedcom_id,
       dp.display_name, dp.match_method, dp.match_confidence, dp.candidate_count, dp.notes
FROM genealogy.silver_document_person dp
JOIN genealogy.ocr_transcriptions t ON t.file_id = dp.file_id
WHERE t.doc_type_detected IN ('Census', 'BurialRegister', 'MonumentalInscription', 'NewspaperClipping', 'MilitaryRecord')
  AND dp.candidate_count > 1
  AND (dp.notes IS NULL OR dp.notes NOT LIKE '%doc_type_affinity=%');
