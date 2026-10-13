# Data-quality suite

Applies when working in `tests/data_quality/`, the DQ workflow, or any failing check.
Full plan, taxonomy and rollout history: Notion `DATA_QUALITY_TEST_PLAN`.

## Architecture

- **Tier 2**: hand-written custom `.sql` checks, one file per check (header + query).
  Zero rows means pass. Used for business rules, aggregate reconciliation and regressions.
- **Tier 1**: generic, config-driven checks (not-null, not-blank, uniqueness, FK integrity,
  freshness, row-count) generated from the YAML registry in
  `tests/data_quality/registry/tier1_*.yaml`. The registry is also materialised into
  `genealogy.ref_data_quality_registry`.
- Runnable unchanged from four surfaces: Databricks MCP, `data_quality_suite.ipynb`,
  GitHub Actions (`.github/workflows/data-quality.yml`: push to `main` plus manual dispatch,
  no nightly schedule), and locally via `run_checks.py`.
- All runs write to `genealogy.data_quality_results`.

## Behaviour to remember

- A failing non-`info` check files an Asana task in the Genealogy project's Data section,
  unless it is `known_failing` with a linked `existing_asana_task` (then it only comments on
  that task each run). `info` severity never files.
- Only a newly failing `critical` check turns CI red.
- Selective execution: `run_checks.py --layer <bronze|silver|gold|ref>` or `--object <table>`,
  composable with `--only` and `--severity`. These narrow what runs, never what is synced to
  `ref_data_quality_registry`: `sync_registry_table` does a full DELETE + reinsert, so it must
  always receive the complete object list.
- Each foreign-key check runs as its own query (see the Spark SQL gotchas in the root `CLAUDE.md`).
- Tier 2 check headers are parsed by line prefix, not YAML; IDs are stable and never renumbered
  (DQ-003 and DQ-007 are intentionally absent). Tier 1 check types: `row_count_not_zero`,
  `freshness_vs_source` (tables only; `depends_on` never names a view), `not_null`, `not_blank`,
  `uniqueness`, `fk_integrity`. One `tier1_<layer>_registry.yaml` per layer (bronze, silver, gold, ref).
- `data_quality_suite.ipynb` does not file Asana tasks; only CI / `run_checks.py` does.
- Validate a new check's baseline row count against live data before shipping it.
- A check and the algorithm it guards share the same tolerance window; change them together.
- A static per-row check is the wrong shape when most rows fail by design (e.g. DNA `active`
  NULLs). Use a drift-monitoring check instead (Test Plan section 4, backlog item 8).
- Soda Core v4 was evaluated and rejected for Tier 1 (hand-rolled won a live pilot).
  Comparison is in Test Plan section 1.4.

## Triage

Use the `failing-check-triage` skill before fixing anything. Separate: a genuine regression, a
check surfacing a different upstream bug, a check-design flaw, and external drift (Fivetran or
Drive) that no code fix can touch. For app-facing values that look wrong while checks still pass,
use `tree-signal-troubleshooting`.

## Not yet built

- Dedicated silver-QC and gold-QC tasks in the Databricks job DAG (Phase 4).
- The rest of Phase 5: batching same-table checks, consolidating scans, re-examining CI cadence.
- A confidence/exception check on `gold_person_scores.story_ready` (the story-publish gate has none).

Open failures and known issues are tracked in Asana, not here.
