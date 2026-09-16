-- id: DQ-015
-- title: ocr_transcriptions surname/forename/year parsed from filename look implausible
-- severity: warning
-- guards_bug:
-- known_failing: true
-- existing_asana_task: 1218567855930502
-- description: >
--   surname/forename/year/doc_type are parsed from the filename by
--   parse_filename() in ocr_pipeline.ipynb (convention
--   SURNAME_Firstname_Year_Description.ext) -- parse_filename computes a
--   parse_ok flag internally but never writes it to the table, so a
--   non-conforming filename silently produces garbage in these columns
--   instead of surfacing as a parse failure. Live validation (2026-09-16)
--   found two real cases this exact pattern already produced:
--   "7_June_1890_-_Conveyance.pdf" (9 pages) does not follow the naming
--   convention at all, so parse_filename's "_".join split reads "7" as
--   surname and "June" as forename; and
--   "BALLS_Charles_1846-Dec_Birth.jpg" has a year token
--   ("1846-Dec") that fails the `^\d{4}(-\d{2}(-\d{2})?)?$` regex (a month
--   abbreviation instead of a numeric month), so year ends up NULL and
--   doc_type ends up "1846-Dec_Birth" instead of "Birth". Flags: surname or
--   forename that is blank, purely numeric, or a single character (not a
--   real name token); or year that is NULL while doc_type itself starts with
--   a 4-digit token (a strong signal the year regex rejected a real year
--   value rather than the file genuinely lacking one). Reported at warning,
--   not critical -- these are Drive filenames Ed controls and can rename,
--   not a pipeline bug -- but known_failing: true until the 10 current rows
--   (9 Conveyance pages + 1 Charles Balls row) are renamed in Drive and
--   reprocessed, or the parse_filename regex is extended to accept month
--   abbreviations.

SELECT file_id, file_name, surname, forename, year, doc_type, page_index
FROM genealogy.ocr_transcriptions
WHERE surname IS NULL OR TRIM(surname) = '' OR surname RLIKE '^[0-9]+$' OR LENGTH(TRIM(surname)) <= 1
   OR forename IS NULL OR TRIM(forename) = '' OR forename RLIKE '^[0-9]+$' OR LENGTH(TRIM(forename)) <= 1
   OR (year IS NULL AND doc_type RLIKE '^[0-9]{4}');
