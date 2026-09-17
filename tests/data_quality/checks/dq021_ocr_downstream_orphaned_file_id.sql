-- id: DQ-021
-- title: silver/gold rows referencing a file_id no longer in ocr_transcriptions
-- severity: warning
-- guards_bug:
-- known_failing: true
-- existing_asana_task:
-- description: >
--   silver_document_person.file_id, gold_transcript_facts.file_id,
--   gold_fact_comparison.file_id, and silver_document_match_exception.file_id
--   are all documented as (informal) foreign keys back to
--   ocr_transcriptions.file_id -- none of the four are covered by any Tier 1
--   fk_integrity check, since ocr_transcriptions was deliberately excluded
--   from every Tier 1 registry (see tier1_bronze_registry.yaml's own note)
--   and gold-layer objects never get fk_integrity as a Tier 1 check type at
--   all (see task 1218582528489867, filed 2026-09-17, proposing a Tier 1
--   expansion to close this structurally). This check is the Tier 2 stopgap
--   for that gap: the exact multi-table cross-check that had to be done by
--   hand, once per table, during this session's _Possibles-folder and
--   HALLAM/STIRLING cleanups (DQ-012) -- automating it means a future
--   deletion that forgets one of these four tables gets caught instead of
--   sitting silently.
--
--   Reported at warning, matching the Tier 1 fk_integrity default (an
--   orphaned FK is usually recoverable/traceable, not an immediate hard
--   break). known_failing: true -- live validation (2026-09-17) found 7
--   rows in gold_transcript_facts, all one file_id
--   (1FStSOipFFHbRQrTDgkN2W0a7wABcUGZy,
--   Documents/Family_CUTHBERTSON/BALLANTYNE_Rachel_1809_Birth.jpg per
--   staging_google_drive.documents, extracted_at 2026-02-24) with zero trace
--   in ocr_processing_log, ocr_transcriptions, silver_document_person, or
--   gold_fact_comparison -- i.e. facts were extracted for a file that has no
--   record of ever being OCR'd or logged, root cause not yet understood.
--   silver_document_person, gold_fact_comparison, and
--   silver_document_match_exception were all clean (0 rows) at the same
--   check. Needs its own investigation (bucket B/D territory per the
--   failing-check-triage skill, not diagnosed further here) before this
--   flips to known_failing: false.

SELECT 'silver_document_person' AS source_table, t.file_id,
       t.person_gedcom_id AS context
FROM genealogy.silver_document_person t
LEFT ANTI JOIN genealogy.ocr_transcriptions o ON t.file_id = o.file_id
UNION ALL
SELECT 'gold_transcript_facts', t.file_id, t.fact_type
FROM genealogy.gold_transcript_facts t
LEFT ANTI JOIN genealogy.ocr_transcriptions o ON t.file_id = o.file_id
UNION ALL
SELECT 'gold_fact_comparison', t.file_id, t.fact_type
FROM genealogy.gold_fact_comparison t
LEFT ANTI JOIN genealogy.ocr_transcriptions o ON t.file_id = o.file_id
UNION ALL
SELECT 'silver_document_match_exception', t.file_id, t.reason
FROM genealogy.silver_document_match_exception t
LEFT ANTI JOIN genealogy.ocr_transcriptions o ON t.file_id = o.file_id;
