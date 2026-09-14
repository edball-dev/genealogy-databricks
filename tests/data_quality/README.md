# Data quality test suite

Known-bug regression checks against `workspace.genealogy`. Each check is a
`.sql` file whose query returns *violating rows* — zero rows means the check
passes. This keeps one authoring surface for the check logic across four
run surfaces: ad hoc from Claude in chat (via the Databricks MCP
`execute_sql` tool), the `data_quality_suite.ipynb` notebook (native
Databricks/scheduled runs), this repo's GitHub Actions workflow, and this
script run locally.

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

Filter which checks run:

```bash
python run_checks.py --only DQ-001,DQ-006
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

## Adding a new check

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

## Results table

Every run writes one row per check to `genealogy.data_quality_results`
(created automatically if missing) — this is the trend-tracking source of
truth, shared by every run surface. See `run_checks.py` for the exact
schema.
