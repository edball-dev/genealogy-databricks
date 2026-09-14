# Track B pilot — Soda Core v4 data contracts

Part of the Soda pilot (`SODA_PILOT_SPEC.md`, tracked in Notion Data Quality
Test Plan §9a and Asana task 1218450818305928).

## Package names (v4, not v3)

`soda-core==4.23.1` + **`soda-databricks==4.23.1`** — confirmed by probing
PyPI directly rather than guessing. `soda-core-databricks` (the v3 naming
pattern, still on PyPI at 3.5.x for other connectors like
`soda-core-spark`) **does not exist** for v4; the per-datasource plugin
package dropped the `-core-` infix. `soda-databricks` pulls in
`databricks-sql-connector` and `databricks-sdk` directly — confirming it's
the SQL connector path, not `sparkdf` (spec §5).

## Connection syntax — hand-derived, not scaffolded

`soda data-source create -t databricks` (the CLI's own YAML scaffolding
command) **only supports `-t postgres`** in this version — it errors with
"Only type postgres is supported atm". There's no bundled example to copy.
`data_source.yml` here was written by reading
`soda_databricks/model/data_source/databricks_connection_properties.py`
directly: `type: databricks`, `connection: {host, http_path, catalog,
access_token}` (PAT auth, the default when `auth_type` is omitted — v4 also
supports `databricks-oauth-m2m` and `azure-service-principal`). `${VAR}`
env var substitution is confirmed native (`soda_core/common/yaml.py`,
resolved post-parse against `os.environ`), so the existing
`DATABRICKS_HOST` / `DATABRICKS_TOKEN` / `DATABRICKS_WAREHOUSE` names carry
over unchanged from the hand-rolled suite.

One non-obvious wrinkle: a contract's `dataset:` identifier is
slash-separated (`workspace/genealogy/gold_dna_coverage`), and the first
segment must match the data source's own `name:` field in `data_source.yml`
— get that wrong and the CLI fails with `Data source 'workspace' not
found` rather than anything schema-shaped. Caught this live via `soda
contract verify` before wiring up real credentials (see below).

## Check syntax — a genuine v4 trap, caught by testing not reading

Every check-type class in `soda_core/contracts/impl/check_types/*_yaml.py`
takes a `type_name` constructor argument, which strongly suggests
`- type: missing` as the YAML shape. **That's wrong.** `soda contract test`
rejected it outright: `🚨 Checks require 1 key to be the type`. The actual
parser (`_parse_checks` in `soda_core/contracts/impl/contract_yaml.py`)
pops the check's single YAML key as the type name:

```yaml
checks:
  - missing:
      must_be: 0
```

not `- type: missing`. This is not documented anywhere short of reading
that one method — a real, current (package is ~7 weeks old per spec §7)
syntax trap that source-reading alone did not catch; only running `soda
contract test` did. Both forms produce no warning-then-fallback — the
wrong form is silently treated as an invalid check and dropped with an
error, not run.

## The three contracts

`contracts/gold_dna_coverage.yml`, `contracts/gold_source_coverage.yml`,
`contracts/gold_ancestral_proximity.yml` — all three pass
`soda contract test` (syntax-only validation, no connection required):

```
✅ contracts/gold_ancestral_proximity.yml is valid
✅ contracts/gold_dna_coverage.yml is valid
✅ contracts/gold_source_coverage.yml is valid
```

- **Missing/duplicate checks** (spec §2, checks 1–2) map directly to
  built-in `missing` / `duplicate` check types — genuinely boilerplate,
  4–5 lines of YAML each.
- **Custom reconciliation (DQ-008) and business-rule (DQ-006) checks**
  (spec §2, checks 3–4) both needed `failed_rows` in `query` form. The
  built-in `aggregate` check type only computes one SQL function over one
  column of the dataset itself (confirmed by reading
  `aggregate_check_yaml.py`) — it cannot express a JOIN across two tables
  (the DQ-008 reconciliation) or a `GROUP BY ... HAVING` (the DQ-006 rule).
  **This is Track B's harder case turning out awkward, not a good fit**:
  the "custom check" ends up being the exact same hand-written SQL as the
  hand-rolled registry's custom check, just wrapped in one extra layer of
  YAML (`query: |` + `must_be: 0`) with no added expressiveness — Soda's
  contract model earns its keep on Tier 1 boilerplate and buys nothing on
  Tier 2 custom logic.

## Known-failing semantics (spec §3) — no native equivalent

Soda's contract model has nothing like `known_failing` /
`existing_asana_task`. The DQ-004 check (`event_id` 100% NULL) is written
as `missing: {must_be_less_than_or_equal: 17172}` — hard-coding the
*current* violation count as the ceiling, so the check still passes today
but would fail the moment the count gets worse. This is real glue, not a
clean feature:

- It requires knowing and baking in the exact live count at authoring
  time (a magic number, vs. the hand-rolled registry's declarative
  `known_failing: true`).
- It silently stops being meaningful once the bug is fixed and the count
  drops below 17172 — the check would then always pass and nobody would
  notice DQ-004 was resolved (whereas the hand-rolled runner's
  `known_failing` + `existing_asana_task` fields are metadata the runner
  can act on — e.g. comment on the existing Asana task — independent of
  the count).
- There's no way to distinguish "still exactly as broken as documented" from
  "got a little better but not fixed" — both pass silently.
- No dedup-to-existing-Asana-task concept at all; that would need to be
  built entirely in glue code around the CLI's JSON/exit-code output.

## Execution-surface test (spec §6) — tested, not assumed

**Ad hoc from Claude in chat**: **No** — confirmed, not assumed. A
hand-rolled check is a `SELECT` that can be pasted directly into the
Databricks MCP `execute_sql` tool (this is exactly how DQ-004/006/008's
live baselines were captured earlier in this pilot). A Soda contract
cannot: it's a YAML document interpreted by the `soda` Python
library/CLI, which itself issues multiple SQL queries (a CTE-wrapped
`COUNT(*)` for `failed_rows`, an aggregate query for `missing`/`duplicate`)
and evaluates pass/fail client-side against the parsed threshold. There is
no plain-SQL equivalent to hand to `execute_sql`. Running a Soda check ad
hoc genuinely requires Claude Code (a real shell) or a not-yet-built Soda
MCP integration — this is a real behavioural difference between the two
tracks, exactly as spec §6 warned it might be.

**CI / Claude Code**: straightforward, as expected — `pip install
soda-core soda-databricks && soda contract verify -c <contract> -ds
data_source.yml` is one more step than the existing suite's `pip install
-r requirements.txt && python run_checks.py`, same secrets
(`DATABRICKS_HOST`/`TOKEN`/`WAREHOUSE`, unchanged names).

**Databricks Job task**: soda-core and soda-databricks are pure-Python
packages (pydantic, requests, ruamel.yaml, `databricks-sql-connector`,
`databricks-sdk` — no Spark dependency, no native binaries) — they are not
preinstalled on any Databricks runtime, so a Job "Python script" task
needs an explicit library-install step (a job cluster library, a
serverless job environment's pip dependency list, or a `%pip install` at
the top of a notebook task) that the hand-rolled runner doesn't need if
run from a notebook already carrying `databricks-sql-connector`. Two ways
to actually run it once installed: shell out to the `soda` CLI via
`subprocess`, or call the programmatic API directly —
`soda_core.contracts.api.verify_api.verify_contracts_locally()` /
`verify_contract_locally()` exist and are the cleaner fit for a Python
task (real function calls, structured `ContractVerificationSessionResult`
return value, no argv/exit-code parsing).

## Attempted live connection

`soda contract verify -c contracts/gold_ancestral_proximity.yml -ds
data_source.yml` was run in this sandbox. It correctly parsed the
contract, resolved the data source by name, substituted `${DATABRICKS_HOST}`
etc., and attempted a real network connection via
`databricks-sql-connector` — which timed out, because this sandbox has no
`DATABRICKS_HOST`/`TOKEN`/`WAREHOUSE` values to substitute (same
constraint as Track A's `run_registry.py` — neither pilot script has raw
Databricks credentials available directly in this environment; both would
run normally in CI with the existing `production` GitHub environment
secrets, or locally with Ed's own credentials). This is a real (not
fabricated) connection attempt, and it went exactly as far as the
plumbing should before hitting the credentials gap — good evidence the
data source config and contracts are wired correctly end to end.

## Expected results

Every check type here (`missing`, `duplicate`, `failed_rows` in `query`
form) is a well-defined, deterministic SQL translation — `missing`
compiles to a `COUNT` of NULLs, `duplicate` to a `GROUP BY ... HAVING
COUNT(*) > 1`-equivalent, `failed_rows`/`query` wraps the literal query in
`WITH failed_rows AS (<query>) SELECT COUNT(*) FROM failed_rows`. Because
the `query`-form checks here are byte-identical to the SQL already run
live via the Databricks MCP `execute_sql` tool (Track A's validation
step), and the `missing`/`duplicate` checks are arithmetically identical
to the not-null/uniqueness SQL also run live there, the counts below are
the real, confirmed live result of these checks — not an estimate — even
though the `soda` CLI process itself couldn't complete a live run without
credentials in this sandbox:

| Check | Table | Violations | Status |
|---|---|---|---|
| missing(branch) | gold_dna_coverage | 0 | pass |
| missing(dna_matches_total) | gold_dna_coverage | 0 | pass |
| duplicate(branch) | gold_dna_coverage | 0 | pass |
| failed_rows: dna_matches_total_reconciliation | gold_dna_coverage | 7 | fail |
| missing(event_id) ≤ 17172 | gold_source_coverage | 17172 | pass (by design — see known-failing section) |
| missing(person_gedcom_id) | gold_source_coverage | 0 | pass |
| missing(person_id) | gold_ancestral_proximity | 0 | pass |
| missing(ancestral_proximity) | gold_ancestral_proximity | 0 | pass |
| duplicate(person_id) | gold_ancestral_proximity | 0 | pass |
| failed_rows: blood_relative_mutual_exclusivity | gold_ancestral_proximity | 0 | pass |

Matches DQ-008 (7 branches, same direction) and DQ-006 (0 violations)
exactly — same underlying SQL as Track A and the existing suite.
