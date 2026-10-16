# OCR pipeline and document matching

Applies when working on `ocr_pipeline`, `notebook_01_document_matching`, `notebook_02_fact_extraction`,
`refresh_gold_source_coverage`, `views_document_and_fact_review`, or anything touching
`ocr_transcriptions`, `silver_document_person`, `silver_transcript_person_mention` or `gold_source_coverage`.

## OCR pipeline

- Source: Fivetran -> Unity Catalog Volume `/Volumes/workspace/staging_google_drive/documents/`.
  Model: Vertex AI Gemini (GCP project `genealogy-488213`). Runs nightly at 2am UTC
  (`max_files=50`, `max_retries=3`).
- Person mentions are extracted in the same Gemini call as transcription (per page).
  `ocr_pipeline.ipynb` owns `silver_transcript_person_mention` (idempotent schema cell); never
  `CREATE OR REPLACE` it, as that would destroy real data. Key is `(file_id, page_index, person_index)`.
- Prompt is two-step: classify `doc_type_detected` first, then apply that type's extra roles
  on top of the shared `FAMILY_RELATION_ROLES` vocabulary.
- `ocr_transcriptions` has **one row per page**, not per file. Any per-file join, especially
  inside a `MERGE`, needs `GROUP BY file_id` / `MIN()` dedup first or Delta throws
  `DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW_IN_MERGE`.
- Review workflow columns on `ocr_transcriptions`: `content_flag`, `content_note`,
  `review_status` (NULL/PENDING/COMPLETE; auto-PENDING when `content_flag` or `confidence_flag`
  fires), `reviewed_at`, `review_notes`, `last_matched_at`.
- `ocr_processing_log.status`: `skipped` = not a real document; `error` = still retryable;
  `failed_permanent` = gave up after `max_retries`. Keep these meanings separate. A
  `failed_permanent` file blocks retry if replaced in place (same `file_id`).
- Parked files: Ed moves failing files to the Drive folder `Documents _Skipped` (not synced),
  fixes them, then moves them back. See the Fivetran notes in the root `CLAUDE.md`.
- Known limits: cropping printed instructions off 1921 census fronts fixes size and safety
  blocks. Large PDFs and some handwritten forms time out (504) whatever the format and need
  splitting or a pipeline change.
- `doc_type_detected` vocabulary: Census, BirthCertificate, MarriageCertificate, DeathCertificate,
  BaptismRegister, BurialRegister, Will, Probate, NewspaperClipping, ElectoralRegister,
  MilitaryRecord, BMDIndex, PoorLawRecord, MonumentalInscription, PassengerList, Directory, CV,
  CriminalRecord, TitheApportionment, AsylumRecord, PropertyConveyance, PersonalCorrespondence, Other.
- Notebook-level facts only: `max_files` (default 50) and `max_retries` (default 3) are widgets in
  `ocr_pipeline.ipynb`. Job schedule and parameters live in Databricks, not in this repo.
- Do not run the OCR Pipeline job through the MCP `run_job`: it processes up to 100 files and
  `max_files` is a task-level parameter.
- The Databricks MCP identity cannot read the `genealogy` secret scope, so a non-dry `notebook_02_fact_extraction` run (Gemini calls) has to be started by Ed in the Databricks UI. Dry runs work from the MCP.

## Matching rules (notebook_01)

- **Filename-embedded person id (`FILENAME_ID`)**: a bracketed token in the file's *current* Drive name
  (`... [I332310876241].png`, `[I_967018131]`; GEDCOM id without the `@`s) pins the primary subject: HIGH,
  no candidate ranking. Cell 5 reads the name from `staging_google_drive.documents` (basename of
  `_fivetran_file_path`, live rows, keyed on `file_id`), not the `ocr_transcriptions.file_name` snapshot, so a
  rename needs no re-OCR (observed: renaming keeps the `file_id`). An id not in `gold_person_life` logs
  `FILENAME_ID_NOT_FOUND` and does not fall back to name matching; `[NONE]` means no primary match and no
  exception. Household/secondary cells are unchanged and anchor on the pinned person. Cell 5c excludes
  `FILENAME_ID` rows from sibling votes. `parse_filename` strips the token before splitting. DQ-028 reports
  ids missing from the tree. After renaming files: re-run notebook_01, then a forced `notebook_02` run
  (`force_file_ids`) for only those files, then `refresh_gold_source_coverage`, notebook 03 and B.
  Retire stale `WRONG_PERSON_MATCH` review marks once the pinned match supersedes them.
- **Alias matching needs literal string equality** (`UPPER(TRIM(...))`). An alias does not help
  a document that spells the surname differently, however close.
- **`silver_person_alias` is never rebuilt.** Notebook_01 Cell 1 only does `CREATE TABLE IF NOT EXISTS`.
  Add manual aliases (DOCUMENT_NAME, SURNAME_VARIANT, nickname MARRIED_NAME) with `INSERT` via the MCP
  `execute_write_sql`; never put `CREATE OR REPLACE` or `TRUNCATE` on it. Derived MARRIED_NAME and
  FATHER_SURNAME rows come from `notebook_sync_derived_aliases` (job task `Sync_Derived_Aliases`, after
  `Materialise_Gold_Tables` in the GEDCOM job), which is insert-only; it never removes rows, so a stale
  derived row must be deleted by hand. DQ-026 reports derived rows that are missing or stale. After adding rows: re-run the matching cells, `refresh_gold_source_coverage`, notebook 03,
  then notebook B.
- **Spouse inference applies to marriage certificates only** (structurally unambiguous via
  `silver_family`), with a year guard of +/-2 of that family unit's marriage year.
  Cell 5e extends the idea to Census/BurialRegister with a second independent guard: the relative
  must appear in the document's own extracted `personal_names` and be alive in the document year.
  Confidence is MEDIUM, not HIGH.
- The id token must use an underscore or no separator (`[I_967018131]`, `[I332310876241]`). A hyphen (`[I-967018131]`) is silently ignored by the matcher and DQ-028, so the file falls back to name matching.
- `notebook_01` Cell 5 truncates `silver_document_person` on every run: a pinned match never needs a delete-and-reinsert, only a re-run.
- Mention-to-match link: `silver_document_person` has nullable `page_index` and `person_index`.
  Key is `(file_id, page_index, person_index)`, joined null-safe (`<=>`). NULL means no known
  mention. Cell 5e writes it directly; Cell 5h2 links other rows by first forename and last name
  token only when the person matches exactly one unclaimed mention in the file.
- A mention-level `override_person_gedcom_id` links the match to its mention but does not change
  `match_confidence`.
- `Resident` mentions (Cell 5g) match the primary's family pool only, with no whole-tree fallback, so
  lodgers stay unlinked. The pool has no children's spouses, so a son-in-law can match a same-named
  grandchild: a `Resident` with neither age nor date of birth does not link through grandparent or
  grandchild, and a name equal to the primary's own is skipped. Pin real exceptions with the
  mention-level override.
- Surname-first names are fixed at source: the `ocr_pipeline` prompt asks for forename-first and
  DQ-027 flags regressions. Correct existing rows with a scoped UPDATE on
  `(file_id, person_index, name_raw)`, not extra matching readings. A bare-surname Head (`Cope`) is
  linked by Cell 5h2.
- `silver_document_person_override` is deprecated but still wired into matching. Don't add rows;
  use aliases or the mention-level pin.
- A file whose only `silver_document_person` matches are LOW is excluded from `notebook_02`
  (scoping and mention linking use HIGH or MEDIUM only), so a forced run brings in 0 files. Raise
  the confidence before extracting. This can go once the LOW-matches task lands.
- Cell 5/5c log implausible forced matches to `silver_document_match_exception` instead of
  forcing them.
- The NewspaperClipping postdate-death exemption is scoped to NewspaperClipping only. Extending it
  to Probate/Will broke previously correct matches.
- A filename disambiguation marker such as "(Uncle)" should become a suffix-aware alias
  (`DOCUMENT_NAME` `alias_suffix`), not be stripped as noise, or one person's documents scatter
  across same-name candidates.
- A mention's age is comparable to document year for Census and certificates only. Adult census
  ages are rounded, and monuments, `PoorLawRecord` and `Other` give age at an event.
- A rule belonging to one stage can be wrongly ported to another. The Wife/Husband exclusion on
  marriage certificates is a matching-stage rule (Cell 5e), not an extraction-stage one.
- `notebook_02_fact_extraction` fact guard is scoped per `(file_id, person_gedcom_id)`; the prompt
  takes forename/surname from the matched person's own tree name.

## Source coverage

- `gold_source_coverage` is a materialised table (citation grain, no `event_id`). Run
  `refresh_gold_source_coverage` after document matching, `gold_person_event_timeline` and
  `v_source_doctype_map` exist. It truncates and reinserts, so it is safe to run alone.
- Only `BirthIndex`, `MarriageIndex`, `DeathIndex` categories give INDEX_ONLY, and `OTHER_NO_FILE`
  (or OTHER with no record id) gives NOT_APPLICABLE. Any other category with no matched document
  is UNCOVERED, so mapping a title to a typed category does not clear it.
- `v_source_doctype_map` is a view over the hand-maintained table `ref_source_doctype_map` (never
  rebuilt; created and seeded by `notebook_ref_source_doctype_map`). Add or change titles with `INSERT` /
  `UPDATE` via the MCP `execute_write_sql`, not by editing a notebook; the table must be on the MCP
  allow-list. Unmapped titles default to `OTHER`; `gold_housekeeping_unmapped_source_titles` lists them.
  Never map a combined-collection title (baptisms, marriages and burials in one) to a single category.
  After a change: `refresh_gold_source_coverage`, notebook 03, then notebook B.
- The category-to-doctype comparison is case-sensitive: an unmapped title (`OTHER`) can never
  match a file the OCR classified `Other`. Fix by mapping the title to the file's exact
  `doc_type_detected`; do not lower-case the join (every unmapped citation would then match any
  `Other` file).
- Citation-year fallback is asymmetric: Birth/Death fall back to `gold_person_life`'s year, Marriage
  has no such fallback for `citation_year` (the timeline's MARR year is only used when matching a
  document to a marriage citation). Prefer fixing the missing citation date in Ancestry.
- Titles with a real image (school admissions, railway, NZ cemetery) are deliberately left UNCOVERED.

## Fact conflict reviews

- `gold_fact_comparison` is rebuilt (TRUNCATE) by notebook 03, so review marks live in
  `silver_fact_conflict_review` (key: file_id, person_gedcom_id, fact_type, transcript_value) and
  are applied by `gold_fact_comparison_reviewed`. Use `is_open_conflict` from that view, never raw
  `status = 'CONFLICT'`.
- A review stops applying if the tree value changes (`reviewed_tree_value`). Reason code TREE_GAP
  keeps applying and clears only when the comparison becomes MATCH; it feeds
  `SIGNAL_TRANSCRIPT_ONLY_FACTS` and `gold_housekeeping_tree_gap_occupations`.
  NOT_AN_OCCUPATION appears nowhere. ACCEPTED means GPS assessed and resolved, not a general dismissal.
- Review conversationally: list open conflicts by person, Ed says what to mark, insert rows with
  `execute_write_sql` as a single-line `INSERT ... SELECT`.

## Writing through the MCP `execute_write_sql`

- Table names must be fully qualified (`workspace.genealogy.table`).
- Single statement, single line. A semicolon anywhere, even inside a string value, triggers a
  false multiple-statement error. The keyword filter is naive substring matching, so plain words
  such as "granted" or "merge" in string values are rejected.
- The allow-list lives in the databricks-mcp code, not here.
