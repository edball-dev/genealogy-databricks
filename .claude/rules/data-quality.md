# Data-quality suite

Applies when working in `tests/data_quality/`, the DQ workflow, or any failing check.
Check-file format, Tier 1 check types, run surfaces, flags and Asana behaviour:
`tests/data_quality/README.md`. Full plan, taxonomy and rollout history: Notion
`DATA_QUALITY_TEST_PLAN`.

## Behaviour to remember

- `ref_data_quality_registry` is synced by full DELETE + reinsert, so `--layer`/`--object`
  filters must never narrow it (`sync_registry_table` always gets the complete object list).
- `data_quality_suite.ipynb` never files Asana tasks; only CI / `run_checks.py` does.
- Only a newly failing `critical` check turns CI red. CI runs on manual dispatch only (no
  push trigger, no nightly schedule), since the checks query live Databricks and a merge does not
  deploy anything there.
- Tier 2 IDs are stable and never renumbered (DQ-003 and DQ-007 are intentionally absent).
- Validate a new check's baseline row count against live data before shipping it.
- A check and the algorithm it guards share the same tolerance window; change them together.
- A static per-row check is the wrong shape when most rows fail by design (e.g. DNA `active`
  NULLs). Use a drift-monitoring check instead (Test Plan section 4, backlog item 8).
- Soda Core v4 was evaluated and rejected for Tier 1 (`SODA_PILOT_OUTCOME.md`).

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
