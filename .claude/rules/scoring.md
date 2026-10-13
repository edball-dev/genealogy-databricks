# Scoring and signals

Applies when working on person signals, scores, branch scores or the weekly snapshot.

## Pipeline

`gold_research_person_signals` (Delta table) -> `gold_person_signal_scores` ->
`gold_person_scores` -> `gold_branch_scores`.

## Formulas

- Completeness and evidence: absolute 0-100 against the person's own applicable maximum.
- Story potential: `PERCENT_RANK()` over `narr_fired`.
- Overall: completeness x 0.40 + evidence x 0.60.
- Weighted scores multiply by proximity/depth multipliers for ORDER BY display only.
- `person_priority` = proximity_priority x depth_priority, used only for action-list ordering.
- Action list: `priority_order = base_score x person_priority`, then `weighted_overall_score ASC`.
  `estimated_delta = (base_score / applicable_max) x 100 x dimension_weight`. One row per
  person (their best signal), top 15.
- Priority and score use opposite multiplier directions. Priority boosts direct ancestors
  (proximity 0 -> 3.00, 1 -> 2.00, 2 -> 1.25, 3+ -> 1.00; depth <=4 -> 1.40, 5-6 -> 1.20,
  7-8 -> 1.00, 9+ -> 0.80). Display multipliers do the inverse (0 -> 1.00, 1 -> 1.25,
  2 -> 1.50, 3+ -> 2.00).
- Signal filter defaults by mode: completeness -> tree_building, evidence -> housekeeping,
  dna -> dna, all -> story_writing. Narrative signals are always excluded.

## Adding a new signal

1. Add the boolean to `gold_research_person_signals` (notebook B).
2. Add it to the `UNPIVOT IN(...)` in notebook D Cell 1.
3. Insert a row in `ref_signal_weights` (include `outcome_label`; set
   `requires_blood_relative` if appropriate).
4. No action mapping is needed.

Query live `ref_signal_weights` first. The signal list (completeness, evidence, narrative
categories) lives there, not in this file.

## Rules that have bitten before

- `requires_blood_relative` must be evaluated **before** `applies_always` in notebook D.
  Some signals carry both flags. `is_blood_relative` is set at build time in notebook 08
  (a `NOT EXISTS` guard on spouse inserts makes it mutually exclusive by construction).
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

## Legacy scoring files (do not extend)

`gold_game_scores` also defines `gold_person_scores` / `gold_branch_scores` as views, from the
retired integrity/narrative priority views and a 35/40/25 weighting. Notebook D defines the same
names with the formulas above; whichever ran last wins, so run only notebook D.
`gold_priority_views` and `signal_detection` define the retired views, and `score_calibration`
still reads them.

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
