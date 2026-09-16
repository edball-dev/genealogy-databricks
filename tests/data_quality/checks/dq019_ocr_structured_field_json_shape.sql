-- id: DQ-019
-- title: ocr_transcriptions personal_names/locations/uncertain_entries malformed JSON
-- severity: warning
-- guards_bug:
-- known_failing: false
-- description: >
--   personal_names and locations are written as a JSON array of strings, and
--   uncertain_entries as a JSON array of {entry, field, reason} objects (the
--   shapes Gemini is prompted to return, per ocr_pipeline.ipynb's
--   _call_gemini_once) -- but the column is plain STRING with no schema
--   enforcement, and a malformed-JSON model response only gets caught at
--   write time for the top-level `transcription` key (extract_json's
--   try/except falls back to doc_type_detected='Other'/confidence='low' for
--   that case, per DQ-016/017's coverage) -- these three fields are stored
--   as-is with no equivalent fallback. A future prompt/model change that
--   breaks their shape would silently produce a string no downstream reader
--   can parse (e.g. a UI trying to render personal_names as a name list)
--   without ever surfacing here. Live validation (2026-09-16): 0 violations
--   across 1002 rows -- a regression guard, not a currently-known gap.

SELECT file_id, file_name, page_index, personal_names, locations, uncertain_entries
FROM genealogy.ocr_transcriptions
WHERE (personal_names IS NOT NULL AND from_json(personal_names, 'ARRAY<STRING>') IS NULL)
   OR (locations IS NOT NULL AND from_json(locations, 'ARRAY<STRING>') IS NULL)
   OR (uncertain_entries IS NOT NULL AND from_json(uncertain_entries, 'ARRAY<MAP<STRING,STRING>>') IS NULL);
