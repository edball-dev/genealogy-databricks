-- id: DQ-031
-- title: Files whose facts or document match changed since their fact comparison (recompare pending)
-- severity: info
-- guards_bug: 1218716867008994
-- known_failing:
-- existing_asana_task:
-- description: >
--   notebook_03 writes gold_fact_comparison.refreshed_at for every row it produces. A file is listed here
--   when its facts (gold_transcript_facts.extracted_at, HIGH or MEDIUM confidence) or its document match
--   (ocr_transcriptions.last_matched_at, stamped by notebook_01) are newer than its latest comparison row,
--   or it has facts but no comparison rows and the facts arrived after the oldest refresh. These are the
--   files a scoped notebook_03 run (mode=scoped) recompares automatically; then run notebook B.
--
--   Reported at info severity: a worklist, never files an Asana task. Returns nothing until one full
--   notebook_03 refresh has populated refreshed_at (rows written before 2026-10-17 have none). A file
--   whose facts yield no comparison rows (low confidence, or a person not in the tree) stays listed
--   until the next full refresh. The SQL must stay identical to STALE_SQL in notebook_03 Cell 0;
--   change them together.

WITH f AS (
  SELECT file_id, MAX(extracted_at) AS facts_at
  FROM genealogy.gold_transcript_facts
  WHERE confidence IN ('high', 'medium')
  GROUP BY file_id
),
m AS (
  SELECT file_id, MAX(last_matched_at) AS matched_at
  FROM genealogy.ocr_transcriptions
  GROUP BY file_id
),
c AS (
  SELECT file_id, MAX(refreshed_at) AS compared_at
  FROM genealogy.gold_fact_comparison
  GROUP BY file_id
),
base AS (
  SELECT MIN(refreshed_at) AS oldest_refresh FROM genealogy.gold_fact_comparison
)
SELECT f.file_id, f.facts_at, m.matched_at, c.compared_at
FROM f
LEFT JOIN m ON m.file_id = f.file_id
LEFT JOIN c ON c.file_id = f.file_id
CROSS JOIN base
WHERE base.oldest_refresh IS NOT NULL
  AND (
    -- compared before, and its facts or its match changed since
    (c.compared_at IS NOT NULL AND (f.facts_at > c.compared_at OR m.matched_at > c.compared_at))
    -- never compared (no rows), but facts arrived after the last full refresh
    OR (c.compared_at IS NULL AND f.facts_at > base.oldest_refresh)
  );
