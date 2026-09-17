-- id: DQ-020
-- title: staging_google_drive.documents has two file_ids for the same Drive path
-- severity: info
-- guards_bug:
-- known_failing:
-- existing_asana_task:
-- description: >
--   staging_google_drive.documents (Fivetran) should have at most one file_id
--   per _fivetran_file_path -- two different file_ids sharing a path is the
--   early signature of the exact Drive file-identity churn DQ-012's
--   HALLAM/STIRLING investigation (2026-09-17) traced down manually: a Drive
--   rename or replacement that leaves the connector minting a second row
--   instead of updating the first, with the original permanently stuck
--   _fivetran_deleted rather than ever being reconciled. This check surfaces
--   that census directly, on the staging table itself, rather than waiting
--   for it to resurface as a DQ-012 violation once ocr_transcriptions already
--   points at the now-dead half of the pair.
--
--   Reported at info, matching the DQ-002/DQ-010 pattern: a nonzero count is
--   expected and not itself a regression -- once a pipeline table has been
--   repointed to the live file_id (confirm via DQ-012, which will show 0 for
--   an already-reconciled pair), the stale duplicate row simply stays in
--   staging_google_drive.documents forever (Fivetran never cleans up an old
--   tombstone), so this check's baseline will never return to zero and isn't
--   meant to. Its purpose is visibility for *new* occurrences worth a manual
--   look, not a per-row action item -- watch for the count growing between
--   runs, not the raw total. Live validation (2026-09-17): 13 duplicate
--   paths, including the already-resolved STIRLING pair (see DQ-012) --
--   the other 12 have not been triaged and may include further HALLAM/
--   STIRLING-style cases worth checking against their own ocr_transcriptions
--   rows.

SELECT _fivetran_file_path, COUNT(DISTINCT file_id) AS distinct_file_ids,
       COLLECT_SET(file_id) AS file_ids
FROM staging_google_drive.documents
GROUP BY _fivetran_file_path
HAVING COUNT(DISTINCT file_id) > 1;
