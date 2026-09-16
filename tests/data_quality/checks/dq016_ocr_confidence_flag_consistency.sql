-- id: DQ-016
-- title: ocr_transcriptions confidence_flag inconsistent with confidence value
-- severity: critical
-- guards_bug:
-- known_failing: false
-- description: >
--   confidence_flag is the mechanism the pipeline uses to surface
--   lower-confidence transcriptions for review (Gemini returns confidence as
--   high/medium/low; the write loop in ocr_pipeline.ipynb sets
--   confidence_flag = true whenever confidence is medium or low). If that
--   derivation ever drifts -- a future edit only flags "low" (matching the
--   table's own DDL comment, which already says "True if confidence is low"
--   and has fallen out of sync with the medium-or-low behaviour actually
--   implemented) or a manual backfill sets the columns independently -- a
--   medium/low-confidence row could silently stop being surfaced for manual
--   review. Live validation (2026-09-16): 0 violations across 1002 rows
--   (959 high/flag=false, 35 medium/flag=true, 8 low/flag=true) -- also
--   flagged the stale DDL comment for a documentation fix, tracked
--   separately since it isn't a data issue.

SELECT file_id, file_name, page_index, confidence, confidence_flag
FROM genealogy.ocr_transcriptions
WHERE (confidence IN ('medium', 'low') AND confidence_flag = false)
   OR (confidence = 'high' AND confidence_flag = true)
   OR confidence NOT IN ('high', 'medium', 'low')
   OR confidence IS NULL;
