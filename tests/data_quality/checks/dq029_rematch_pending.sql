-- id: DQ-029
-- title: Files whose inputs changed since their last document match (rematch pending)
-- severity: info
-- guards_bug: 1218716867008994
-- known_failing:
-- existing_asana_task:
-- description: >
--   notebook_01 stamps ocr_transcriptions.last_matched_at on every file it considers. A file is
--   listed here when something that feeds its match has changed since: REVIEW_CORRECTION (a page's
--   reviewed_at is later), MENTION_CORRECTION (a silver_transcript_person_mention row's corrected_at
--   is later), NEW_FILE (a page has no last_matched_at yet) or ALIAS_ADDED (a silver_person_alias row's created_at / updated_at is later and the
--   alias could affect the file: its surname, a person it is matched to, or a mention surname).
--   These are the files a scoped run (notebook_01 mode=scoped, Cell 4f) queues automatically; run it,
--   then refresh_gold_source_coverage, notebook 03 and B. Files whose every page has content_flag are
--   not listed. NEW_FILE rows are normal between an OCR run and the next matching run; they should
--   not survive a nightly pipeline run.
--
--   Reported at info severity: a worklist, never files an Asana task. A nonzero count right after
--   an edit is expected; a count that stays up across days means corrections are not being
--   rematched. Aliases without audit values (rows older than 2026-10-17) and deleted aliases are
--   invisible to this check. The four trigger queries must stay identical to TRIGGERS in
--   notebook_01 Cell 4f; change them together.

WITH new_file AS (
  SELECT file_id, 'NEW_FILE' AS reason,
    CONCAT(CAST(COUNT(*) - COUNT(last_matched_at) AS STRING), ' of ', CAST(COUNT(*) AS STRING), ' page(s) never matched') AS detail
  FROM genealogy.ocr_transcriptions
  GROUP BY file_id
  HAVING COUNT(last_matched_at) < COUNT(*)
     AND NOT BOOL_AND(COALESCE(content_flag, FALSE))
),
review AS (
  SELECT file_id, 'REVIEW_CORRECTION' AS reason,
    CONCAT('reviewed_at ', CAST(MAX(reviewed_at) AS STRING), ' after last_matched_at ', CAST(MIN(last_matched_at) AS STRING)) AS detail
  FROM genealogy.ocr_transcriptions
  GROUP BY file_id
  HAVING COUNT(last_matched_at) = COUNT(*)
     AND MAX(reviewed_at) > MIN(last_matched_at)
     AND NOT BOOL_AND(COALESCE(content_flag, FALSE))
),
mention AS (
  SELECT m.file_id, 'MENTION_CORRECTION' AS reason,
    CONCAT('mention corrected_at ', CAST(MAX(m.corrected_at) AS STRING), ' after last_matched_at ', CAST(MIN(f.lm) AS STRING)) AS detail
  FROM genealogy.silver_transcript_person_mention m
  JOIN (
    SELECT file_id, MIN(last_matched_at) AS lm
    FROM genealogy.ocr_transcriptions
    GROUP BY file_id
    HAVING COUNT(last_matched_at) = COUNT(*) AND NOT BOOL_AND(COALESCE(content_flag, FALSE))
  ) f ON f.file_id = m.file_id
  WHERE m.corrected_at > f.lm
  GROUP BY m.file_id
),
changed AS (
  SELECT person_gedcom_id, UPPER(TRIM(alias_surname)) AS surname_key, alias_type,
         GREATEST(created_at, updated_at) AS changed_at
  FROM genealogy.silver_person_alias
  WHERE created_at IS NOT NULL OR updated_at IS NOT NULL
),
files AS (
  SELECT file_id, MIN(last_matched_at) AS lm, MIN(UPPER(TRIM(surname))) AS surname_key
  FROM genealogy.ocr_transcriptions
  GROUP BY file_id
  HAVING COUNT(last_matched_at) = COUNT(*) AND NOT BOOL_AND(COALESCE(content_flag, FALSE))
),
hits AS (
  -- the document is filed under the alias surname (Cell 5 name matching)
  SELECT f.file_id, c.alias_type, c.surname_key, c.changed_at
  FROM changed c JOIN files f ON f.surname_key = c.surname_key
  WHERE c.changed_at > f.lm
  UNION
  -- the alias belongs to a person the file is already matched to
  SELECT f.file_id, c.alias_type, c.surname_key, c.changed_at
  FROM changed c
  JOIN genealogy.silver_document_person d ON d.person_gedcom_id = c.person_gedcom_id
  JOIN files f ON f.file_id = d.file_id
  WHERE c.changed_at > f.lm
  UNION
  -- a mention in the file carries the alias surname (household and witness matching)
  SELECT f.file_id, c.alias_type, c.surname_key, c.changed_at
  FROM changed c
  JOIN genealogy.silver_transcript_person_mention m
    ON UPPER(TRIM(REGEXP_EXTRACT(m.name_raw, '(\\S+)$', 1))) = c.surname_key
  JOIN files f ON f.file_id = m.file_id
  WHERE c.changed_at > f.lm
),
alias_files AS (
  SELECT file_id, 'ALIAS_ADDED' AS reason,
    CONCAT(MAX(alias_type), ' alias ', MAX(surname_key), ' changed ', CAST(MAX(changed_at) AS STRING)) AS detail
  FROM hits
  GROUP BY file_id
)
SELECT file_id, reason, detail FROM new_file
UNION ALL
SELECT file_id, reason, detail FROM review
UNION ALL
SELECT file_id, reason, detail FROM mention
UNION ALL
SELECT file_id, reason, detail FROM alias_files;
