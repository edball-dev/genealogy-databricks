-- id: DQ-005
-- title: ocr_transcriptions duplicate transcribed_text across file_ids
-- severity: critical
-- guards_bug: 1218420653319268
-- known_failing: true
-- existing_asana_task: 1218420653319268
-- description: >
--   Two different file_ids sharing identical transcribed_text is a likely
--   symptom of duplicate filenames in Drive being mis-processed as separate
--   documents. Open bug — mitigation is renaming duplicate files in Drive,
--   not a pipeline fix, so this may stay known_failing for a while.

SELECT transcribed_text, COUNT(DISTINCT file_id) AS distinct_files,
       COLLECT_SET(file_id) AS file_ids
FROM genealogy.ocr_transcriptions
WHERE transcribed_text IS NOT NULL AND LENGTH(transcribed_text) > 50
GROUP BY transcribed_text
HAVING COUNT(DISTINCT file_id) > 1;
