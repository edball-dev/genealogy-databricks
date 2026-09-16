-- id: DQ-017
-- title: ocr_transcriptions blank transcribed_text not flagged for review
-- severity: critical
-- guards_bug:
-- known_failing: false
-- description: >
--   A blank/whitespace-only transcribed_text (Gemini transcribing a blank or
--   illegible page) should always carry confidence_flag = true so it surfaces
--   for manual review rather than silently passing through downstream
--   text-length/coverage checks as an empty but "unflagged" row. Live
--   validation (2026-09-16): 2 blank rows exist
--   (SHEARER_Mary Ann_1891_Census.pdf p2, PEARSON_William_1921_WWI Pension
--   Index Card.pdf p4), both already correctly confidence='low'/flag=true --
--   0 violations. This is a narrower regression guard than DQ-016 (which
--   checks the confidence/flag pairing generally): it specifically protects
--   against a future prompt/parsing change that returns confidence='high'
--   for an empty transcription.

SELECT file_id, file_name, page_index, confidence, confidence_flag, confidence_notes
FROM genealogy.ocr_transcriptions
WHERE (transcribed_text IS NULL OR TRIM(transcribed_text) = '')
  AND confidence_flag = false;
