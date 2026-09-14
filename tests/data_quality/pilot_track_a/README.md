# Track A pilot — hand-rolled Tier 1 registry

Part of the Soda pilot (`SODA_PILOT_SPEC.md`, tracked in Notion Data Quality
Test Plan §9a and Asana task 1218450818305928). See `tests/data_quality/README.md`
for the existing production check suite this pilot is comparing against.

## What this is

A YAML config (`registry.yaml`) listing, per table: not-null columns, an
optional duplicate/uniqueness key, and hand-written custom-SQL checks — plus
a thin runner (`run_registry.py`) that reads the config and executes checks
against `workspace.genealogy` via `databricks-sql-connector`, same connection
pattern as `tests/data_quality/run_checks.py`.

**Format chosen: YAML file**, not a `ref_data_quality_registry` Delta table.
Rationale: faster to iterate during the pilot itself (no DDL, no warehouse
round-trip to change a check) — see the comment at the top of `registry.yaml`.
If this track wins, the production version should move to a Delta table to
match the project's `ref_*` config precedent (e.g. `ref_signal_weights`);
that migration is out of scope for the pilot per its own instructions.

This is deliberately minimal — no Asana-dedup filing, no
`data_quality_results` write-back. It only reports `known_failing` status
inline, enough to prove out the semantics in spec §3.

## Results (live run, 2026-09-14)

Validated against `workspace.genealogy` via the Databricks MCP `execute_sql`
tool (same SQL the runner generates — confirmed by loading `registry.yaml`
and diffing the generated SQL strings against what was actually executed).
Literal `python run_registry.py` execution needs `DATABRICKS_HOST` /
`DATABRICKS_TOKEN` / `DATABRICKS_WAREHOUSE`, which this sandbox does not
have (same constraint as the production `run_checks.py` — it runs via CI
secrets or Ed's own local credentials, not directly by Claude in this
environment). The SQL is identical either way, so the counts below are the
real result of running this registry, not an estimate.

| Check | Table | Violations | Status |
|---|---|---|---|
| not_null(branch) | gold_dna_coverage | 0 | pass |
| not_null(dna_matches_total) | gold_dna_coverage | 0 | pass |
| duplicate(branch) | gold_dna_coverage | 0 | pass |
| PILOT-A-DNA-RECON | gold_dna_coverage | 7 | fail (known_failing) |
| not_null(event_id) | gold_source_coverage | 17172 | fail (known_failing) |
| not_null(person_gedcom_id) | gold_source_coverage | 0 | pass |
| not_null(person_id) | gold_ancestral_proximity | 0 | pass |
| not_null(ancestral_proximity) | gold_ancestral_proximity | 0 | pass |
| duplicate(person_id) | gold_ancestral_proximity | 0 | pass |
| PILOT-A-BLOOD-RELATIVE | gold_ancestral_proximity | 0 | pass |

**PILOT-A-DNA-RECON matches DQ-008 exactly**: 7 of 8 branches mismatch
(Mattock, Cuthbertson, Pearson, Easter, McLean, Clifford, Ball — all
under-reporting vs a raw recount), same branches, same direction, same
counts as `dq008_dna_matches_total_reconciliation.sql`.

**PILOT-A-BLOOD-RELATIVE matches DQ-006 exactly**: 0 violations, same as
`dq006_ancestral_proximity_blood_relative_exclusivity.sql`.

10 checks total, not 12 — spec §2 enumerates not-null on 2 columns x 3
tables (6), duplicate on 2 tables (`gold_source_coverage` deliberately
excluded, no confirmed single-column natural key per spec §1), and one
custom check each on `gold_dna_coverage` and `gold_ancestral_proximity`
(2). Forcing a duplicate or reconciliation check onto a table where it
doesn't apply would be exactly the kind of pilot bug the spec warns against
(§1) — not counted toward a "12" that doesn't match the actual check list.

## Effort (§7 scoring input)

- **Setup**: `pip install databricks-sql-connector pyyaml` (2 packages,
  already a subset of what the production suite needs) — no service to
  provision, no account to create.
- **Boilerplate check (not-null)**: 2 lines of YAML per column
  (`column:` + `known_failing:`).
- **Boilerplate check (duplicate)**: 1 line of YAML per table
  (`duplicate_key:`).
- **Custom check (reconciliation/business-rule)**: paste the exact
  `.sql` body from the existing DQ-006/DQ-008 check files under a `sql:`
  block — zero translation, because the registry's custom-check contract
  is identical to the existing suite's (a SELECT returning violating rows).
- **Known-failing semantics**: native — `known_failing` + `existing_asana_task`
  fields, same shape as the production runner's check-file header.
