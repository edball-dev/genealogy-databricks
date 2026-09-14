# Soda pilot outcome

Tracks the Data Quality Test Plan (Notion §9a) and Asana task
1218450818305928. Full pilot spec: see the task description this pilot
worked from (`SODA_PILOT_SPEC.md`, not committed — it's the planning
document, not a deliverable). Pilot code: `pilot_track_a/` (hand-rolled
registry) and `pilot_track_b/` (Soda Core v4 data contracts), each with its
own README covering effort, findings, and live results in detail.

All 10 checks that spec §2 actually enumerates (not a literal 4×3=12 — see
`pilot_track_a/README.md` for why duplicate/reconciliation/business-rule
checks don't apply to every table) were implemented on both tracks and
validated live against `workspace.genealogy` via the Databricks MCP
`execute_sql` tool on 2026-09-14. Both tracks' reconciliation and
business-rule checks match the existing DQ-008 and DQ-006 `.sql` checks'
live violation counts exactly (7 of 8 branches mismatching, 0 violations,
respectively) — confirmed, not assumed, per the spec's own bar for the
pilot.

## Scoring

| Dimension | Track A (hand-rolled) | Track B (Soda Core v4) |
|---|---|---|
| Setup effort | `pip install databricks-sql-connector pyyaml` (2 packages, one already required by the existing suite). No scaffolding needed — the registry format was designed from scratch to match the existing check-file header shape. | `pip install soda-core soda-databricks` (pulls in `databricks-sql-connector`, `databricks-sdk`, pydantic, opentelemetry, ~10+ transitive deps). Had to hand-derive the Databricks connection YAML from the connector's source — `soda data-source create -t databricks` only scaffolds postgres. Also had to discover the correct v4 package name (`soda-databricks`, not `soda-core-databricks`) by probing PyPI, since the old v3 naming pattern still exists for other connectors and looks plausible. |
| Boilerplate check effort (§2 checks 1–2) | 2 lines of YAML per not-null column (`column:` + `known_failing:`); 1 line per duplicate key. | 2–3 lines of YAML per check (`missing:` / `duplicate:` + `must_be: 0`). Same order of magnitude as Track A — this is where Soda's contract model is a genuinely good fit. |
| Custom check effort (§2 checks 3–4) | Paste the exact existing `.sql` body verbatim under a `sql:` key — zero translation, because the registry's custom-check contract (a SELECT returning violating rows) is identical to the existing suite's. | Same SQL, wrapped in `failed_rows: {query: \|, must_be: 0}` — one extra layer of YAML, no added expressiveness. The built-in `aggregate` check type was ruled out first (single-column, single-dataset only — can't do the DQ-008 JOIN or the DQ-006 GROUP BY/HAVING). Both tracks reproduce DQ-008 (7/8 branches, same counts, same direction) and DQ-006 (0 violations) exactly. |
| Known-failing semantics (§3) | Native: `known_failing` + `existing_asana_task` fields, same shape as the production runner's check-file header — zero glue code. | No native equivalent. DQ-004 (event_id 100% NULL) had to be encoded as `must_be_less_than_or_equal: 17172` — a hard-coded current-count ceiling that (a) requires knowing the live count at authoring time, (b) silently stops meaning anything once the bug is fixed (the check just always passes, no signal), and (c) has no path to the existing Asana-dedup logic without separate glue code. |
| Ad hoc-from-chat execution (§6) | Works directly via the Databricks MCP `execute_sql` tool — this is literally how every baseline in this pilot was captured. | Does not work. A Soda contract is a YAML document interpreted by the `soda` library/CLI, which issues its own SQL and evaluates thresholds client-side — there's no plain-SQL form to hand to `execute_sql`. Requires Claude Code (a real shell) or a not-yet-built Soda MCP integration. This is a real, tested behavioural gap, not a guess. |
| CI integration effort | 1 more `pip install` line beyond what's already in `requirements.txt`; no new secrets (reuses `DATABRICKS_HOST`/`TOKEN`/`WAREHOUSE`). | 1 more `pip install` line (larger dependency tree); no new secrets; but needs `SODA_CORE_TELEMETRY_ENABLED=false` set to stop the CLI phoning home to `collect.soda.io` on every run (confirmed live — every invocation attempted this by default). |
| Databricks Job task integration effort (§6) | None beyond what the existing suite already needs — runs as a plain Python script with `databricks-sql-connector` already on the path. | Needs an explicit library-install step; `soda-core`/`soda-databricks` are not preinstalled on any Databricks runtime. Two viable invocation shapes once installed: shell out to the `soda` CLI, or call `soda_core.contracts.api.verify_api.verify_contracts_locally()` directly (a genuine programmatic API — the cleaner fit for a Job Python task). |
| Output parsability | Each check result is a plain `(count, status, known_failing)` tuple — trivial to map into the existing `data_quality_results` schema and Asana-dedup logic, because it's the same shape the production runner already uses. | `soda contract verify` returns a structured `ContractVerificationSessionResult` (Python API) or a CLI exit code + console/JSON output — would need a translation layer into `data_quality_results` and the Asana-dedup logic; non-trivial but not exotic. |
| Maintenance risk | None by construction — plain SQL and YAML, no external library semantics to track. | Real and current: hit a genuine, undocumented v4 syntax trap (check type is the YAML key itself, not a `type:` field) that reading the library's own source code did *not* catch — only running `soda contract test` did. The scaffolding CLI is incomplete (postgres-only). Default telemetry-on-by-default is its own minor governance concern. The package is ~7 weeks old at time of writing per the pilot spec; this pilot's own experience backs up treating that as a real risk, not a hypothetical one. |

## Recommendation

**Hand-rolled.** Soda's contract model is a reasonable fit for the
boilerplate Tier 1 checks (missing/duplicate) — roughly at parity with the
hand-rolled registry there — but this project's Tier 2 custom checks
(exactly the checks DQ-004 through DQ-008 exist to cover) get no benefit
from it: the "custom" `failed_rows` check is the identical hand-written SQL
with an extra YAML wrapper, not a more expressive or safer way to write it.
Against that limited upside, Soda costs a broken execution surface (no ad
hoc `execute_sql` path), a real gap in known-failing/Asana-dedup semantics
that would need to be rebuilt as glue code, a heavier dependency footprint,
and demonstrated, current syntax instability in a ~7-week-old major
version. A hybrid (Soda for Tier 1 boilerplate, hand-rolled `.sql` for
Tier 2) isn't worth the added dependency, a second execution model, and
fragmented tooling for a project whose actual DQ backlog (DQ-004, DQ-006,
DQ-008) is dominated by Tier 2 logic — recommend the hand-rolled registry
(Track A, `pilot_track_a/`) as the single track Test Plan Phase 1 builds
on, migrating its YAML config to a `ref_data_quality_registry` Delta table
per the project's existing `ref_*` precedent as a Phase 1 task, not a pilot
one.
