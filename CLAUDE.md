# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@README.md

Topic detail lives in `.claude/rules/` (scoring, ocr-and-matching,
data-quality, dna-and-story). This file holds the rules that apply to
every change in this repo.

## Working agreements

- **Never guess table or column names.** If unsure which columns a table has,
  stop and ask, or check the live schema (Databricks MCP `describe_table`, or the schema
  doc that `notebook_schema_export.ipynb` writes to Drive, linked in the README).
- Notebook output is `.ipynb` with SQL/Python cells, markdown headers explaining
  context, and inline comments for design choices, so it can be imported directly.
  If the `databricks-notebook` skill is available, read it before writing cells
  (serverless SQL, join patterns, type coercion).
- When creating or changing files, give a Git commit message.
- Consider whether a data-quality check should be added, changed or retired.
- New tables get audit columns `created_at`, `created_by`, `updated_at`, `updated_by` (defaults on the `created_*` pair), and hand-maintained tables above all: a manual INSERT then records who and when. Existing tables get them when next touched (done so far: `silver_person_alias` via a one-off migration, `control_rematch_queue`). Writers set `updated_at` / `updated_by` on UPDATE; an MCP INSERT records the service principal as `created_by`.
- Prefer free solutions; if one costs money, set out the trade-off first.
- Ed is the only contributor and often works from an Android phone.

## Principles

- **Ancestry is the source of truth for the tree.** If a gap traces to an incomplete
  or inaccurate citation, fix the citation in Ancestry rather than adding notebook
  logic to paper over it. Code fixes are for accurate citations the logic fails to use.
- **Diagnose before fixing**: confirm the root cause with targeted SQL first.
- **Staleness is the first hypothesis.** When data looks missing or wrong, check when
  the table was last materialised before debugging logic. Views recompute live;
  `CREATE OR REPLACE TABLE` outputs do not. `DESCRIBE HISTORY` works on tables only.
  Before calling an upstream fix complete, trace forward to downstream tables.
- **Verify what a signal or column actually computes** by reading its source notebook;
  do not infer meaning from the name.
- **Business logic goes upstream in views**, not re-derived in the app or snapshot notebooks.
- Don't generalise connector (Fivetran) behaviour from one observed case.
- Scope any algorithm or check exemption to what has actually been observed, and
  keep a check and the algorithm it guards on the same tolerance (update them together).

## Views and notebooks

- **Never create or replace views in a data-refresh notebook.** `CREATE OR REPLACE VIEW`
  makes the executing identity the owner, which can break views owned by Ed
  (`INSUFFICIENT_PERMISSIONS`). Core view DDL lives in `views_core_gold`,
  `views_document_and_fact_review`, `views_story_plan` and `views_bronze`; run those
  only when a definition changes, as the owning account, or follow with
  `ALTER VIEW ... OWNER TO`. Exceptions already in the repo: `notebook_D_person_scores`,
  `notebook_ancestor_coverage_view` and the `notebook_housekeeping_*.sql` files are view-only
  notebooks, so the same ownership caveat applies when running them. `CREATE OR REPLACE TABLE`
  keeps owner and history.
- A Databricks git folder with uncommitted notebook output blocks `repo_checkout_branch` and `repo_pull`,
  and the MCP cannot discard it, so reset the folder in the Databricks UI. Scheduled jobs run whatever
  branch is checked out, so switch back to `main` straight after testing a branch.
- `run_notebook` (MCP) runs whole notebooks only, so a step that must run alone
  needs its own notebook.
- `run_notebook` starts a **new run on every call**; poll with `get_run` using the returned `run_id`,
  never by calling it again (a duplicate TRUNCATE + INSERT run is the risk). The free tier is meant to
  run one serverless job at a time, but two ran together on 2026-10-17 (an interrupted `run_notebook`
  call still launched a run, and overlapping notebook_01 rebuilds interleave deletes and inserts): check
  `list_active_runs` before every `run_notebook` and after any interrupted call, and cancel both if two
  are active. Observed durations: `notebook_01` about 6 minutes (full or a few files),
  `refresh_gold_source_coverage` about 14, notebook 03 about 1, notebook B about 1.
- **Refresh order after a date or year change**: `views_core_gold` (only if the
  definition changed), then notebook 07, `refresh_gold_source_coverage`, notebook 03,
  then notebook B. Score views from notebook D are live and need no rebuild.
  Check each table with `DESCRIBE HISTORY` afterwards.
- Hand-editing a notebook as raw JSON needs JSON escaping and SQL escaping checked
  independently (SQL apostrophes are doubled, `''`). When unsure, reword to avoid the
  apostrophe. `json.load()` will not catch invalid SQL. Edit `.ipynb` files from Python with
  `json.load` and `json.dump(indent=1, ensure_ascii=False)` plus a trailing newline (keeps diffs
  small), assert that each anchor text occurs exactly once, and build cell text with `.replace`, not
  `%` formatting (a `%python` cell breaks it).
- **Validating a rebuild or refactor of a table**: record the `DESCRIBE HISTORY` versions first, run,
  then diff with `LEFT ANTI JOIN` against `VERSION AS OF` (removed, added, changed). Live-data tests
  must be reversible: note the original values, label temporary rows `TEMP`, restore, then verify
  against time travel. The MCP cannot read printed output, so a notebook meant to be driven from it
  should end with `dbutils.notebook.exit(json)` (dry runs do); per-cell timings come from the commit
  times in `DESCRIBE HISTORY`.
- When saving a view's DDL to the repo from `SHOW CREATE`, qualify every table as
  `genealogy.<table>`; it prints unqualified names that only resolve in the original schema.
- Spark SQL serverless gotchas: don't batch correlated `NOT EXISTS` subqueries
  across checks into one query (optimizer bug; one FK check per query; it also hit table diffs, so
  use one `LEFT ANTI JOIN` per direction); `CONCAT` returning NULL silently breaks naive duplicate
  counts (use `GROUP BY ... HAVING COUNT(*) > 1`); `ALTER TABLE ... ADD COLUMNS IF NOT EXISTS` is
  unsupported (use a migration notebook that checks `DESCRIBE`); a CTE read by several later CTEs is
  re-expanded each time (queries returning zero rows had 0.5 to 6 million character plans), so build
  the shared step once into a `tmp_` table with `CREATE OR REPLACE TABLE` and drop it after the last reader.
- Always query live `ref_signal_weights` before editing scoring notebooks; weights drift.

## Schema gotchas

- Database is `workspace.genealogy`. Root person GEDCOM ID is `@I_967018310@`.
- Retained but unused (do not reference in new code): `gold_research_action`,
  `gold_research_signal_action`, `ref_intent_category_weights`.
- Retired views (do not reference): `gold_research_person_signals_pivoted`,
  `gold_person_integrity_priority`, `gold_person_narrative_priority`,
  `gold_person_integrity_next_best_actions`, `gold_person_narrative_next_best_actions`.
- Branch lives on `gold_person_scores`; do not join `gold_person_life` for branch filtering.
- `gold_person_event_timeline`: always filter `event_participant_source = 'recorded'`.
- `gold_generation_depth.person_id` and `gold_ancestral_proximity.person_id` hold GEDCOM IDs
  (rename pending). Path directions are opposite: `gold_generation_depth.path` is
  `[researcher, ..., ancestor]`; `gold_ancestral_proximity.path_to_ancestor` is
  `[ancestor, ..., person]`. `is_blood_relative` is NULL for unbranched people.
- `silver_event_source` has no direct `person_gedcom_id`; join via `silver_event_participant`.
  Person-root citations are in `silver_person_source`.
- `silver_event_date.date_start` is a year-1 sentinel for `BEF` dates and ABT is a +/-1 year range;
  never take `date_start` unconditionally. `gold_event` (see `views_core_gold`) resolves BEF to
  `date_end` and ABT to the range midpoint. `event_sequence_by_date` still orders by `date_start`.
- GEDCOM header lines (HEAD, SUBM, the Ancestry `1 SOUR`) have NULL `record_xref`. New
  person- or event-level builds must require `record_xref IS NOT NULL`. When a not-null check
  fails, compare silver with raw `bronze_gedcom_with_record` first: a 1:1 match means the gap
  is in the Ancestry tree, not the pipeline.
- GEDCOM IDs contain `@`; underscore-prefixed IDs become negative Ancestry IDs.
- Document filename convention: `SURNAME_Forename_YYYY_DocType.ext` (the date is the
  document date, not the birth date).
  Optional person-id token before the extension, e.g. `PALMER_Samuel_1921_Census [I_967018131].jpg`:
  the GEDCOM id without the `@`s pins the primary match (see `.claude/rules/ocr-and-matching.md`);
  `[NONE]` means the document is not about anyone in the tree.
- A child listed under two families of a parent with two spouses trips the silver
  uniqueness checks. Fix in Ancestry, re-download the GEDCOM, re-run the pipeline.
- A shared `ref_*` table synced by full DELETE + reinsert must always be synced from the
  complete dataset, never a filtered subset.

## Fivetran and Drive

- Moving a file out of the synced Drive `Documents` folder sets `_fivetran_deleted = true`
  on its `staging_google_drive.documents` row; moving it back clears the flag and keeps
  the same `file_id`. `ocr_processing_log` is not flagged by this.
- One Drive path can hold two live `file_id`s (different documents can share a filename),
  and deleted tombstones persist. Checks grouping staging rows by path must ignore deleted rows.
- Key on `file_id`, never `file_name`: `file_name` columns do not update after a rename.
- Before assuming ordinary staleness on a connector-fed table, group the table's sync-batch
  metadata by sync timestamp. If the fix is a data correction, repoint downstream rows
  (matches, facts, comparisons) to the corrected id rather than deleting and reprocessing.

## Repo layout and data flow

Source control for the Databricks pipeline behind Ed Ball's "Genealogy Research Assistant" / *We Are The Balls* family-history project. There is no build system: almost everything is a **Databricks notebook** (`.ipynb`, with `%sql` cells and PySpark; a few `.sql` / `.py` notebooks) that runs in a Databricks workspace against Unity Catalog `workspace.genealogy` (code usually writes `genealogy.<table>`). Notebooks are executed on Databricks (or via the Databricks MCP tools: `run_notebook`, `execute_sql`, `repo_pull`), not locally. Only `tests/data_quality/` is ordinary Python that runs locally/CI.

## Data flow (medallion layout)

Layers are encoded in table prefixes: `bronze_*`, `control_*`, `silver_*`, `gold_*`, `ref_*` (hand-maintained reference/config tables, e.g. `ref_signal_weights`).

1. **GEDCOM tree pipeline**, numbered notebooks, run in order: `01_ingest_gedcom` (Ancestry GEDCOM from a Fivetran-synced Drive volume → `bronze_gedcom`; idempotent, raises `SKIP_PIPELINE` if the file id was already ingested) → `02_enrich` → `03a`/`03b` (events, sources) → `04_process_silver_tables` → `05a_parse_dates` / `05b_normalise_places` → `06_create_canonical_places` → `07_materialise_gold_tables` → `08_generate_ancestral_depth_and_proximity`. Each notebook's markdown header states its run-order prerequisites; read it before running.
2. **Document/OCR pipeline** (scanned records on Google Drive): `ocr_pipeline` (nightly Gemini transcription + person-mention extraction → `ocr_transcriptions`, `ocr_processing_log`) → `notebook_01_document_matching` (matches mentions/documents to tree people; many lettered cells 5, 5b…5j that must run in order) → `notebook_02_fact_extraction.py` (Gemini facts → `gold_transcript_facts`) → `notebook_03_fact_comparison.sql` (transcript facts vs tree facts; reviewed via `silver_fact_conflict_review`).
3. **Signals / scores / planning layer**: `notebook_A_ref_actions_and_signals` → `notebook_B_person_signals` (`gold_research_person_signals`; needs Notebook A and 08 first) → `notebook_D_person_scores`, `notebook_scores_snapshot` (append-only history), plus `notebook_dna_load`/`notebook_dna_gold`, `notebook_week_plan`, `story_plan_*`, `person_story_status`.
4. `lib_*.ipynb` are shared helpers (`%run`-style): `lib_helper_utils` (e.g. `merge_place_canonicals`), `lib_gedcom_text_utils`.
5. `migration_*.ipynb` are one-off, dated schema migrations.

Cross-notebook dependencies are mostly implicit via table names — when changing a table's columns, grep for the table across all notebooks (and the DQ registries) before editing.


## Conventions

- **Notebook headers**: open with a short current-state summary (purpose, run order, what it
  builds, key design rules), then the version history below it. When you change a notebook,
  update the summary so it describes the code as it is now, and add a dated entry to the history
  rather than rewriting the history. Don't repeat in a header what `CLAUDE.md` or `.claude/rules/` already say.
- `.ipynb` JSON is edited directly; keep cell structure intact and `%sql` cells idempotent (`CREATE OR REPLACE`, `MERGE`).
- Many notebooks end with anomaly-check cells that should return 0 violations; preserve or extend them.

## Data-quality commands

Run from `tests/data_quality/` with `DATABRICKS_HOST`, `DATABRICKS_TOKEN` and
`DATABRICKS_WAREHOUSE` set (`ASANA_PAT` optional): `python run_checks.py` (add `--only`,
`--severity`, `--layer`, `--object` to narrow). Flags, formats and check types are in
`tests/data_quality/README.md`; behaviour rules are in `.claude/rules/data-quality.md`.
