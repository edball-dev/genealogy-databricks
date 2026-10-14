-- id: DQ-028
-- title: Filename-embedded person id not found in the tree
-- severity: warning
-- guards_bug: 1219234684173800
-- known_failing:
-- existing_asana_task: 1219234684173800
-- description: >
--   A live Drive document whose current filename carries a bracketed person
--   id token (for example 'PALMER_Samuel_1921_Census [I_967018131].jpg')
--   is pinned to that tree person by notebook_01 Cell 5 (match_method
--   FILENAME_ID). The token is the GEDCOM id without the @ signs; [NONE] means
--   the document is not about anyone in the tree and is exempt. Every other
--   token must exist in gold_person_life, otherwise the file is left unmatched
--   and logged as FILENAME_ID_NOT_FOUND. This check surfaces that drift,
--   typically a typo or a tree id that changed after a GEDCOM re-import,
--   before it silently mismatches. Uses the same token regex as Cell 5 and
--   ocr_pipeline parse_filename; change them together.

SELECT d.file_id,
       REGEXP_EXTRACT(d._fivetran_file_path, '[^/]+$', 0) AS current_name,
       CONCAT('@', REGEXP_EXTRACT(d._fivetran_file_path, '\\[(NONE|I_?[0-9]+)\\]', 1), '@') AS pinned_person_gedcom_id
FROM workspace.staging_google_drive.documents d
WHERE d._fivetran_deleted = FALSE
  AND d._fivetran_file_path RLIKE '\\[(NONE|I_?[0-9]+)\\]'
  AND REGEXP_EXTRACT(d._fivetran_file_path, '\\[(NONE|I_?[0-9]+)\\]', 1) <> 'NONE'
  AND NOT EXISTS (
    SELECT 1 FROM workspace.genealogy.gold_person_life p
    WHERE p.person_gedcom_id = CONCAT('@', REGEXP_EXTRACT(d._fivetran_file_path, '\\[(NONE|I_?[0-9]+)\\]', 1), '@')
  );
