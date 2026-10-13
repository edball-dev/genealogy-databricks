# Scoring and signals

Applies when working on person signals, scores, branch scores or the weekly snapshot.

## Pipeline

`gold_research_person_signals` (Delta table) -> `gold_person_signal_scores` ->
`gold_person_scores` -> `gold_branch_scores`.

## Formulas

Score formulas and the proximity/depth multiplier tables are in the `notebook_D_person_scores`
header; read them there. Not covered there:

- Weighted scores multiply by proximity/depth multipliers for ORDER BY display only.
- `person_priority` = proximity_priority x depth_priority, used only for action-list ordering.
- Action list: `priority_order = base_score x person_priority`, then `weighted_overall_score ASC`.
  `estimated_delta = (base_score / applicable_max) x 100 x dimension_weight`. One row per
  person (their best signal), top 15.
- Priority and display multipliers run in opposite directions (priority boosts direct ancestors).
- Signal filter defaults by mode: completeness -> tree_building, evidence -> housekeeping,
  dna -> dna, all -> story_writing. Narrative signals are always excluded.

## Adding a new signal

Steps are in the `notebook_D_person_scores` header. Also set `outcome_label` on the
`ref_signal_weights` row, and `requires_blood_relative` if appropriate. No action mapping is
needed. Query live `ref_signal_weights` first; the signal list lives there, not in this repo.

## Rules that have bitten before

- `requires_blood_relative` is evaluated before `applies_always` in notebook D (explained in the
  code). `is_blood_relative` is set at build time in notebook 08 (a `NOT EXISTS` guard on spouse
  inserts makes it mutually exclusive by construction).
- Use `MIN_BY(p.is_blood_relative, p.ancestral_proximity)` for boolean context columns,
  not `MIN()`: it picks the value from the closest-proximity row.
- Score-0 signals (`SIGNAL_UNCONFIRMED_MILITARY`, `POSSIBLE_*`) fire as hints but add
  nothing to scored denominators.
- `SIGNAL_COMMON_ANCESTOR_UNVERIFIED`: the `common_ancestor_verify` CTE must check both
  the CDA's own ID and their spouse's ID in match paths.
- `SIGNAL_UNCOVERED_SOURCES` is a plain count of UNCOVERED rows (fires at 2 or more),
  with no check that COVERED rows exist. A person can have solid covered sources and
  still have real gaps. Read the notebook before calling a signal wrong.
- `SIGNAL_FACT_CONFLICT` counts only open conflicts (`gold_fact_comparison_reviewed.is_open_conflict`).
  `SIGNAL_TRANSCRIPT_ONLY_FACTS` counts occupation rows only when `is_tree_gap`.
- A materialised table (e.g. `gold_research_person_signals`) keeps serving old data after an
  upstream view fix until its notebook is re-run. Check `DESCRIBE HISTORY`.

## Scoring notebooks

| Notebook | Purpose |
|---|---|
| `08_generate_ancestral_depth_and_proximity` | `gold_generation_depth`, `gold_ancestral_proximity` (incl. `is_blood_relative`) |
| `notebook_A_ref_actions_and_signals` | ref tables: signal weights, outcome labels |
| `notebook_B_person_signals` | `gold_research_person_signals` |
| `notebook_D_person_scores` | Cells 1-3: signal scores, person scores, branch scores; Cell 4: `gold_branch_structural_metrics`. All four outputs are **views** (live, no rebuild). |
| `notebook_scores_snapshot` | Weekly snapshot job (Sunday 3am UTC); Cell 4 joins `gold_branch_structural_metrics` |
| `notebook_ancestor_coverage_view` | `gold_ancestor_coverage` view |
| `notebook_week_plan` | `ref_week_plan` (pending replacement by the App Reorg work) |
