-- id: DQ-014
-- title: ocr_transcriptions doc_type_detected outside the controlled vocabulary
-- severity: critical
-- guards_bug:
-- known_failing: false
-- description: >
--   doc_type_detected is meant to be Gemini's classification constrained to
--   the DOC_TYPE_VOCABULARY list in ocr_pipeline.ipynb Cell 6 (19 values,
--   including the catch-all "Other") -- it is only enforced by prompt
--   instruction, not a CHECK constraint, so a model update or prompt drift
--   could silently emit a value outside the vocabulary (a new label, a typo,
--   free text) that downstream doc-type filters (DQ-011's IN-list,
--   notebook_01_document_matching.ipynb's age-extraction dispatch) would then
--   silently ignore rather than error on. Live validation (2026-09-16): 0
--   violations across 1002 rows, all 18 distinct values observed are in the
--   list (CriminalRecord is in the data despite not appearing in every
--   DOC_TYPE_MAP keyword-mapping cell -- it is a valid vocabulary member).

SELECT file_id, file_name, page_index, doc_type_detected
FROM genealogy.ocr_transcriptions
WHERE doc_type_detected IS NOT NULL
  AND doc_type_detected NOT IN (
    'Census', 'BirthCertificate', 'MarriageCertificate', 'DeathCertificate',
    'BaptismRegister', 'BurialRegister', 'Will', 'Probate',
    'NewspaperClipping', 'ElectoralRegister', 'MilitaryRecord', 'BMDIndex',
    'PoorLawRecord', 'MonumentalInscription', 'PassengerList', 'Directory',
    'CV', 'CriminalRecord', 'Other'
  );
