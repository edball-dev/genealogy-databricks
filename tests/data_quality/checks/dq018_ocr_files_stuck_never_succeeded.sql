-- id: DQ-018
-- title: Drive files that never produced a successful OCR transcription
-- severity: warning
-- guards_bug:
-- known_failing: false
-- existing_asana_task: 1218567602351792
-- description: >
--   ocr_pipeline.ipynb's own idempotency check (Cell 14) only excludes a
--   file_id from the next run's candidate list once ocr_processing_log has a
--   'success' or 'skipped' row for it -- a file that only ever logged
--   'error' or 'safety_blocked' stays a candidate and should get picked up
--   and retried on a subsequent run. This check looks for the ones that
--   never actually cleared: a file_id with every logged attempt at
--   error/safety_blocked and none at success/skipped means either it's
--   fallen out of the candidate list some other way (deleted from Drive
--   between attempts -- see DQ-012), or repeated reruns keep hitting the
--   same permanent failure (a corrupt/unreadable file, a 400 INVALID_ARGUMENT
--   from Vertex, a stale volume path from a Drive move) rather than the
--   transient 429/504 errors retrying would fix. Live validation
--   (2026-09-16): 1307 log rows / 1096 distinct file_ids, of which 40 never
--   succeeded -- error reasons span transient (429 RESOURCE_EXHAUSTED, 504
--   DEADLINE_EXCEEDED, some now years-stale e.g. last attempt 2026-05-22),
--   permanent-looking (400 INVALID_ARGUMENT "Provided image is not valid",
--   malformed JSON, a FileNotFoundError for a Drive path that moved), and 3
--   safety_blocked. Every one of these is a genealogy document Ed has that
--   simply isn't in ocr_transcriptions at all -- a silent coverage gap this
--   suite had no visibility into before. Reported at warning (not critical)
--   since a fresh pipeline run may clear some of these on its own; shipped
--   known_failing: true given the current backlog of 40. Distinct from
--   DQ-012 (which catches the opposite direction -- a transcription whose
--   source is now gone from Drive) and from the bronze-registry note that
--   deliberately excludes ocr_processing_log from Tier 1 row-count checks.
--   Parked files are excluded: Ed moves problem files out of Documents into
--   the Drive folder 'Documents _Skipped' (not synced by Fivetran) while he
--   fixes them (crop, convert, replace). Fivetran flags such a file
--   _fivetran_deleted = TRUE in staging_google_drive.documents (same
--   file_id), and flips it back to FALSE if the file is moved back. A file
--   flagged deleted is therefore not counted; if moved back and failing
--   again it reappears automatically. A file with no staging row at all
--   (never synced) is still counted, to avoid hiding genuine failures.

SELECT g.file_id,
       MAX_BY(g.file_name, g.processed_at) AS file_name,
       COUNT(*) AS attempt_count,
       MAX_BY(g.status, g.processed_at) AS latest_status,
       SUBSTRING(MAX_BY(g.error_message, g.processed_at), 1, 300) AS latest_error_summary,
       MIN(g.processed_at) AS first_attempted_at,
       MAX(g.processed_at) AS last_attempted_at
FROM genealogy.ocr_processing_log g
WHERE NOT EXISTS (
  SELECT 1 FROM staging_google_drive.documents d
  WHERE d.file_id = g.file_id AND d._fivetran_deleted = TRUE
)
GROUP BY g.file_id
HAVING SUM(CASE WHEN g.status IN ('success', 'skipped') THEN 1 ELSE 0 END) = 0;
