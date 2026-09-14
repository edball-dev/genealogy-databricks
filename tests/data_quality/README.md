# Data quality test suite

Two-tier data quality checks against `workspace.genealogy` (Data Quality
Test Plan, Notion §7): **Tier 2** hand-written `checks/*.sql` files for
business rules, aggregate reconciliation, and regression guards; **Tier 1**
config-driven `registry/tier1_*.yaml` files for generic checks
(row-count-not-zero, freshness-vs-source, not-null, not-blank, uniqueness,
FK integrity) that would otherwise be dozens of near-identical files. Both
compile down to the same contract — a SELECT
that returns *violating rows*, zero rows means the check passes — so there
is one execution/results/Asana-dedup path (`run_checks.py`) for both, and
one authoring surface across all four run surfaces: ad hoc from Claude in
chat (via the Databricks MCP `execute_sql` tool — the SQL either format
generates is just a SELECT), the `data_quality_suite.ipynb` notebook
(native Databricks/scheduled runs), this repo's GitHub Actions workflow,
and this script run locally.

(A hand-rolled registry vs Soda Core v4 pilot decided this format — see
`SODA_PILOT_OUTCOME.md`.)

Out of scope for this suite: raw GEDCOM/tree-structure checks (covered by
GEDminer) and the `ref_research_resources` checks (tracked separately in
Asana).

## Running locally

```bash
pip install -r requirements.txt
export DATABRICKS_HOST=...          # e.g. dbc-xxxxxxxx-xxxx.cloud.databricks.com
export DATABRICKS_TOKEN=...
export DATABRICKS_WAREHOUSE=...     # HTTP path, e.g. /sql/1.0/warehouses/abc123
export ASANA_PAT=...                # optional locally — omit to skip Asana filing
python run_checks.py
```

All three Databricks env vars are required; the script exits immediately
with an error if any are missing. `ASANA_PAT` is optional — without it, a
newly-failing check is still recorded in `data_quality_results` but no
Asana task is filed (a warning is printed instead).

Filter which checks run (works across both tiers — Tier 1 check IDs are
derived, e.g. `T1-ROW_COUNT_NOT_ZERO-GOLD_EVENT`, not manually assigned):

```bash
python run_checks.py --only DQ-001,DQ-006
python run_checks.py --only T1-FRESHNESS_VS_SOURCE-GOLD_DNA_COVERAGE
python run_checks.py --severity critical
```

Exit code is non-zero only if a `critical`-severity check **newly** failed
(i.e. wasn't already `known_failing`) — a `known_failing` check that is
still failing does not turn the run red.

## Check file format

```sql
-- id: DQ-001
-- title: Document match date plausibility (BUG-001 regression)
-- severity: critical
-- guards_bug: 1218420283786561
-- known_failing: false
-- existing_asana_task:
-- description: >
--   Free text. Everything from "description:" to the blank line is folded
--   into one string — this is parsed with a plain line-prefix strip, not a
--   real YAML parser, so keep the header to this exact shape.

SELECT ...
```

| Field | Meaning |
|---|---|
| `id` | Stable ID (e.g. `DQ-001`). Referenced in `data_quality_results` and Asana task titles — never renumber. |
| `title` | Short human label, used in the Asana task title. |
| `severity` | `critical` \| `warning` \| `info`. `info` never files an Asana task, regardless of `known_failing` — it only ever logs to `data_quality_results`. |
| `guards_bug` | Asana task GID this check was written to catch a regression of. Optional. |
| `known_failing` | `true` if this check is *expected* to fail right now because it tracks an open, unfixed bug. |
| `existing_asana_task` | If `known_failing: true` and an Asana task already tracks this exact issue, its GID. The runner never creates a duplicate task for this check — it only logs and (if `ASANA_PAT` is set) adds a progress comment to the existing task. |
| `description` | Free text shown in the Asana task body and used for context. |

## Adding a new Tier 2 check

1. Add `checks/dqNNN_<short_name>.sql` with the header above. Pick the next
   unused `DQ-NNN` ID (see the note in `dq001_...sql` about why there's no
   `DQ-003`, and `dq007` being skipped — check history before assuming the
   next number is contiguous).
2. Validate the SQL against live data first (Databricks MCP `execute_sql`,
   or `python run_checks.py --only DQ-NNN` once the file exists) — don't
   ship a check whose baseline row count you haven't actually seen.
3. If the check is `known_failing: true`, decide whether there's already an
   open Asana task for the issue; set `existing_asana_task` if so.
4. Add the same check to `data_quality_suite.ipynb` (it loops over these
   same files, so this is usually automatic — confirm rather than assume).

## Tier 1 registry (`registry/tier1_*.yaml`)

Generic checks, config-driven instead of one `.sql` file per check.
`registry/tier1_gold_registry.yaml` (Phase 1 of the Test Plan) covers the
full gold layer — row-count and freshness; `registry/tier1_silver_registry.yaml`
(Phase 2, per Notion §8) covers the full silver layer — not-null,
uniqueness, and FK integrity. `run_checks.py` and the notebook load
**every** `registry/tier1_*.yaml` file (`REGISTRY_DIR.glob("tier1_*.yaml")`),
merging their objects into one check list — adding a new layer later means
adding a new `tier1_<layer>_registry.yaml` file, not editing an existing
one. These files are the checked-in source of truth;
`genealogy.ref_data_quality_registry` is a queryable Delta materialization
of all of them combined, re-synced (delete + re-insert) on every
`run_checks.py` run — same "config lives in `ref_*` tables" pattern as
`ref_signal_weights`, but git stays authoritative so registry changes go
through PR review like everything else. (The Delta table's schema grew in
Phase 2 — `run_checks.py`'s `ensure_registry_table` checks `DESCRIBE TABLE`
and runs `ALTER TABLE ... ADD COLUMNS` for any new columns rather than
assuming `CREATE TABLE IF NOT EXISTS` will add them; it won't, since the
table already exists.)

### Gold layer (`tier1_gold_registry.yaml`)

Each entry is a gold object (`table` or `view`) with a list of checks:

```yaml
- name: genealogy.gold_ancestral_proximity
  type: table
  checks:
    - check_type: row_count_not_zero
      severity: critical
    - check_type: freshness_vs_source
      severity: warning
      depends_on: [genealogy.gold_generation_depth, genealogy.silver_relationship]
```

| `check_type` | What it checks | Applies to |
|---|---|---|
| `row_count_not_zero` | The object has at least one row — catches an upstream join/filter bug silently emptying it. | Both `table` and `view` objects. |
| `freshness_vs_source` | The table's last Delta write (`DESCRIBE HISTORY`) isn't older than any table named in `depends_on`. | `table` objects only. **Never** name a `view` in `depends_on` — a view has no Delta log, so `DESCRIBE HISTORY` errors on one; a view is by construction always as fresh as whatever it reads, so freshness only needs checking at the nearest managed-table ancestor. |

`depends_on` names should be the *nearest* managed-table dependencies, not
necessarily the ultimate silver/bronze root — checking each hop in the
gold DAG separately (rather than one check against a flattened root) means
a staleness failure points at the specific stage that broke, not just "gold
is stale somewhere."

Severity defaults (confirmed by Ed, 2026-09-14, resolving Test Plan §11):
`critical` for `row_count_not_zero`, `warning` for `freshness_vs_source`.

**Known gaps, left as explicit `notes` rather than guessed at:**
`gold_fact_comparison` and `gold_transcript_facts`'s actual population
notebooks weren't located in this repo (only downstream reads / a sync
step) — freshness is deferred for both until that's traced.
`gold_scores_snapshot_branch`/`_person` are append-only historical logs,
where "freshness vs source" doesn't fit the same way it does a
CREATE-OR-REPLACE gold table — the right check ("did today's snapshot get
written") isn't built yet. `gold_research_action`/`_signal_action` are
static, hand-maintained reference lists with no freshness concept at all.

### Silver layer (`tier1_silver_registry.yaml`)

Covers all 28 managed `silver_*` tables (excludes the one silver *view*,
`silver_v_active_place_canonical` — no natural key/FK shape to check on a
view beyond what its underlying tables already cover — and
`silver_article_chunk_index`, a dead table per the `article-resource-search`
skill). Three check types, in addition to the two gold ones above:

```yaml
- name: genealogy.silver_relationship
  type: table
  checks:
    - check_type: not_null
      column: person_id_1
      severity: critical
    - check_type: uniqueness
      columns: [person_id_1, person_id_2, relationship_type]
      severity: critical
    - check_type: fk_integrity
      column: person_id_1
      ref_table: genealogy.silver_person
      ref_column: person_gedcom_id
      severity: warning
```

| `check_type` | What it checks | Config keys |
|---|---|---|
| `not_null` | `column` has no `NULL` values. | `column` |
| `uniqueness` | The combination of `columns` has no duplicate groups (`GROUP BY ... HAVING COUNT(*) > 1`). | `columns` (list — one column is fine as a single-item list) |
| `fk_integrity` | Every non-null `column` value in this table exists as `ref_column` in `ref_table` (`NOT EXISTS` anti-join). | `column`, `ref_table`, `ref_column` |

Severity defaults (confirmed by Ed, 2026-09-14): `critical` for
`not_null` and `uniqueness`, `warning` for `fk_integrity` — same reasoning
as the gold defaults (a null/duplicate in a key column is a hard
data-integrity break; an orphaned FK is usually recoverable/traceable and
shouldn't block a run by itself).

Keys and FK relationships were traced from each table's actual build
notebook (`CREATE TABLE`/`MERGE`/`.saveAsTable()` statements, Delta `NOT
NULL` DDL, and column comments where present) — never guessed from column
names alone.

**Validation gotcha — don't batch uniqueness checks with `CONCAT`.** Doing
a quick manual pre-check across many tables at once via
`COUNT(DISTINCT CONCAT(col1, '|', col2, ...))` will produce false-positive
"duplicates" whenever any of the concatenated columns is nullable: SQL
`CONCAT()` returns `NULL` if *any* argument is `NULL`, and `COUNT(DISTINCT
x)` doesn't count `NULL`s — so every row with a null in one of the
columns silently collapses out of the distinct count, making the row count
look larger than the distinct count even with zero real duplicates. This
produced apparent duplicates for `silver_event_source` (169 "dupes") and
`silver_person_source` (181 "dupes") that turned out to be **zero** real
duplicates once re-checked with a direct `GROUP BY ... HAVING COUNT(*) >
1` (which is what the shipped `uniqueness` check actually runs — the
production check type was never affected, only this ad hoc validation
shortcut).

**Validation gotcha — don't batch FK checks into one query.** Running
several correlated `NOT EXISTS` subqueries together in one `SELECT` (via
`UNION ALL` or scalar subqueries) intermittently triggers `[INTERNAL_ERROR]
The Spark SQL phase optimization failed` — validate each FK check as its
own standalone query, which is also exactly how `fk_integrity` runs in
production (one query per check).

**Findings from Phase 2 validation** (see the `failing-check-triage`
skill's classification): 8 duplicate `(person_id_1, person_id_2,
relationship_type)` rows in `silver_relationship` plus 10 duplicate rows in
`silver_event_participant`, both traced to duplicate family/person records
in the source GEDCOM — filed as Asana task `1218472474949479`
(`known_failing: true` on those two checks references it).
`silver_person_source`/`silver_event_source` have ~180/169 rows with a null
`source_xref` (root cause unconfirmed — could be a valid GEDCOM
inline-citation style or an extraction gap) — not filed as its own task per
the triage skill's bucket-B confidence bar, but referenced against the same
task with a "needs further triage" note rather than silently passed over.
All ~24 `fk_integrity` checks otherwise came back with **zero** orphaned
rows — a clean result, not a gap.

### Bronze layer (`tier1_bronze_registry.yaml`)

Covers the GEDCOM ingest pipeline's bronze/control tables — `bronze_gedcom`,
`bronze_gedcom_with_record` (view), `bronze_gedcom_enriched`,
`control_ingested_gedcom`, `control_latest_gedcom` (view) — traced from
`01_ingest_gedcom.ipynb`/`02_enrich_gedcom.ipynb`. Reuses the two gold check
types (`row_count_not_zero`, `freshness_vs_source`) rather than introducing
new ones — per Notion §8, bronze only needs "row-count-not-zero, freshness
vs last known ingest run," which is the same shape as gold's checks, just
one hop earlier in the DAG. `bronze_gedcom` itself is the pipeline's root
(no other in-warehouse table to compare freshness against) so it only gets
`row_count_not_zero`; `bronze_gedcom_enriched`'s `freshness_vs_source`
`depends_on` is flattened to `bronze_gedcom` (the managed table), never the
`bronze_gedcom_with_record` view it's actually built from — same
never-a-view-in-`depends_on` rule as gold. `ocr_transcriptions`/
`ocr_processing_log`/`ocr_token_usage` are bronze-shaped per §8's own
definition but deliberately excluded here — `ocr_transcriptions` already
has a dedicated Tier 2 check (DQ-005) and Notion §9 treats OCR output as
feeding the silver layer rather than a standalone bronze stage; revisit if
the OCR workflow gets its own Tier 1 pass.

### Ref layer (`tier1_ref_registry.yaml`)

Covers hand-maintained lookup/config tables: `ref_intent_category_weights`,
`ref_signal_weights`, `ref_week_plan`, and `genealogy.v_source_doctype_map`
(not `ref_`-prefixed, but functionally identical — its own definition is a
literal `CREATE OR REPLACE VIEW ... AS SELECT * FROM (VALUES (...), ...)`,
a hardcoded lookup, not a query over other tables). Excludes
`ref_research_resources` (out of scope for this whole suite, see the top of
this file) and `ref_data_quality_registry` (this suite's own
materialization of its registry files — checking it with itself would be
circular). One new check type beyond gold/silver's:

```yaml
- name: genealogy.ref_signal_weights
  type: table
  checks:
    - check_type: row_count_not_zero
      severity: critical
    - check_type: not_blank
      column: signal_code
      severity: critical
    - check_type: uniqueness
      columns: [signal_code]
      severity: critical
```

| `check_type` | What it checks | Config keys |
|---|---|---|
| `not_blank` | `column` is never `NULL` **or** an empty/whitespace string (`column IS NULL OR TRIM(column) = ''`). | `column` |

`not_blank` is distinct from `not_null` because ref-layer keys are
hand-typed strings (`signal_code`, `source_title`, ...) rather than
auto-generated GEDCOM/UUID ids — an accidental blank string is a real,
separate risk from a `NULL` one, and matches §8's own wording for this
layer ("no-blank-key-values"), not silver's "not-null." A column with no
string-blank equivalent (e.g. `ref_week_plan.week_commencing`, a `DATE`)
still uses plain `not_null`.

Severity: `critical` throughout this file (`row_count_not_zero`,
`not_blank`, `uniqueness`) — these are small, hand-maintained tables where
a missing/blank/duplicate key silently breaks downstream scoring or signal
logic per §8's own rationale, not a "recoverable" issue the way a silver FK
orphan is. No `fk_integrity` checks here — none of these tables reference
another table's key.

### Adding a new Tier 1 entry

1. Add an entry to the relevant `registry/tier1_<layer>_registry.yaml`
   (or create a new `tier1_<layer>_registry.yaml` file for a layer that
   doesn't have one yet — the loader picks up any file matching that glob
   automatically). Trace `depends_on`/keys/FKs from the table's actual
   build notebook (grep the `.ipynb` files for `CREATE (OR REPLACE) TABLE
   genealogy.<name>` / `MERGE INTO` and read its `FROM`/`JOIN` clauses and
   `NOT NULL`/comment DDL) — don't guess lineage or keys.
2. Validate live before merging: run the check's SQL by hand via the
   Databricks MCP `execute_sql` tool first (see `build_tier1_check_sql` in
   `run_checks.py` for the exact template per `check_type`), the same way
   Tier 2 checks get validated. `python run_checks.py --only
   T1-<CHECK_TYPE>-<TABLE>[-<LABEL>]` also works once the entry exists
   (uniqueness/not_null/fk_integrity IDs get a `-<COLUMN(S)>` suffix so
   multiple checks of the same type on one table don't collide — see
   `tier1_check_label` in `run_checks.py`).
3. No `data_quality_suite.ipynb` change needed — the notebook loads every
   `registry/tier1_*.yaml` file the same way `run_checks.py` does (confirm
   rather than assume, same as Tier 2).

## Results table

Every run writes one row per check (both tiers) to
`genealogy.data_quality_results` (created automatically if missing) — this
is the trend-tracking source of truth, shared by every run surface. See
`run_checks.py` for the exact schema.
