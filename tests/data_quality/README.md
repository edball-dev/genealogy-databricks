# Data quality test suite

Two-tier data quality checks against `workspace.genealogy` (Data Quality
Test Plan, Notion §7): **Tier 2** hand-written `checks/*.sql` files for
business rules, aggregate reconciliation, and regression guards; **Tier 1**
a config-driven `registry/tier1_gold_registry.yaml` for generic checks
(row-count-not-zero, freshness-vs-source) that would otherwise be dozens of
near-identical files. Both compile down to the same contract — a SELECT
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

## Tier 1 registry (`registry/tier1_gold_registry.yaml`)

Generic checks for the gold layer (Phase 1 of the Test Plan — Phase 2 will
add a silver-layer registry with not-null/uniqueness/FK checks, per Notion
§8). `registry/tier1_gold_registry.yaml` is the checked-in source of
truth; `genealogy.ref_data_quality_registry` is a queryable Delta
materialization of it, re-synced (delete + re-insert) on every
`run_checks.py` run — same "config lives in `ref_*` tables" pattern as
`ref_signal_weights`, but git stays authoritative so registry changes go
through PR review like everything else.

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

### Adding a new Tier 1 entry

1. Add an entry to `registry/tier1_gold_registry.yaml`. Trace
   `depends_on` from the table's actual build notebook (grep the `.ipynb`
   files for `CREATE (OR REPLACE) TABLE genealogy.<name>` and read its
   `FROM`/`JOIN` clauses) — don't guess lineage.
2. Validate live before merging: run the row-count and (if applicable)
   freshness SQL by hand via the Databricks MCP `execute_sql` tool first
   (see `build_tier1_check_sql` in `run_checks.py` for the exact template),
   the same way Tier 2 checks get validated. `python run_checks.py --only
   T1-<CHECK_TYPE>-<TABLE>` also works once the entry exists.
3. No `data_quality_suite.ipynb` change needed — the notebook should call
   `run_checks.py`'s Tier 1 path the same way it calls the Tier 2 one
   (confirm rather than assume, same as Tier 2).

## Results table

Every run writes one row per check (both tiers) to
`genealogy.data_quality_results` (created automatically if missing) — this
is the trend-tracking source of truth, shared by every run surface. See
`run_checks.py` for the exact schema.
