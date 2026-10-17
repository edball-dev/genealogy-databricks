-- id: DQ-030
-- title: Rematch queue health (pending rows stuck, unknown reason or unknown file)
-- severity: warning
-- guards_bug: 1218716867008994
-- known_failing:
-- existing_asana_task:
-- description: >
--   control_rematch_queue holds files waiting for a scoped notebook_01 run; Cell 5k closes each row
--   it handles (processed_at). STUCK_PENDING: a row older than a day that was never closed (a scoped
--   run failed part-way, or the rows were added and no run followed). UNKNOWN_REASON: a reason
--   outside MANUAL / REVIEW_CORRECTION / MENTION_CORRECTION / ALIAS_ADDED (a typo in a manual
--   INSERT). UNKNOWN_FILE: a file_id with no ocr_transcriptions row, which a run can never match.

SELECT file_id, reason, detail, created_at, 'STUCK_PENDING' AS problem
FROM genealogy.control_rematch_queue
WHERE processed_at IS NULL AND created_at < current_timestamp() - INTERVAL 1 DAY
UNION ALL
SELECT file_id, reason, detail, created_at, 'UNKNOWN_REASON'
FROM genealogy.control_rematch_queue
WHERE reason NOT IN ('MANUAL', 'REVIEW_CORRECTION', 'MENTION_CORRECTION', 'ALIAS_ADDED')
UNION ALL
SELECT q.file_id, q.reason, q.detail, q.created_at, 'UNKNOWN_FILE'
FROM genealogy.control_rematch_queue q
WHERE q.processed_at IS NULL
  AND NOT EXISTS (SELECT 1 FROM genealogy.ocr_transcriptions o WHERE o.file_id = q.file_id);
