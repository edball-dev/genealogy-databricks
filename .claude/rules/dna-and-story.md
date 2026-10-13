# DNA tables and the story pipeline

Applies when working on `notebook_dna_gold`, `notebook_dna_load`, `gold_person_dna`,
`gold_dna_coverage`, the story-plan views, or `silver_person_story_status`.

## DNA

- Corroboration: `confirmed` = `dna_role = 'Common DNA Ancestor'` (CDA); `partial` = in a
  confirmed CDA's `gold_ancestral_proximity.path_to_ancestor`; `none` = no DNA evidence.
- `gold_person_dna.active` is NULL by design for `Common DNA Ancestor` and `DNA Connection`
  rows (untested tree persons that never join to `silver_dna_match`). NULL is also normal on
  many `DNA Match` rows: it means that citation's `(kit_id, match_name)` is not in
  `silver_dna_match`, which is a point-in-time scrape. **Never filter `WHERE active = true`
  when checking corroboration.**
- Several people on one lineage can each be CDA-tagged (e.g. a woman and her daughter).
  `gold_ancestral_proximity.nearest_direct_ancestor_id` returns the closest tagged ancestor, not
  necessarily the right ancestral couple for a match cluster. Check whether the nearest tagged
  ancestor's spouse is also tagged before treating it as the answer.
- Two scheduling problems: `notebook_dna_load` (Ancestry extraction) has no repeatable method
  and stays manual. `notebook_dna_gold` rebuilds gold DNA tables from the tree plus
  `silver_dna_match` and needs re-running whenever the **tree** changes, so it goes stale on every
  GEDCOM refresh unless it is in the main job.
- Match threshold in `notebook_dna_gold` is 20 cM.

## Pipeline orchestration

One main job, triggered by a GEDCOM download from Ancestry syncing via Fivetran, covers
notebooks 01-08 and beyond. A separate, manual OCR workflow runs `ocr_pipeline` then notebooks
01-03. The weekly score snapshot is the only scheduled job.

## Story pipeline

- Stages in `silver_person_story_status`: `not_started` -> `timeline` -> `context` -> `draft` -> `publish`.
  Doc pointers: `timeline_doc_id`, `context_doc_id`, `draft_doc_id`, `story_doc_id`.
- `story_ready` = `story_potential_score >= 90 AND overall_score >= 70 AND story_written = FALSE`.
- The Sixty Stories plan (60 chapters) is mapped chapter -> `gedcom_id` in
  `silver_story_plan_chapters`, manually curated and re-run only when the plan doc changes
  (`story_plan_chapters_load_v1.ipynb`). Plan subjects are the primary driver; data-driven
  candidates are secondary.
- Views: `gold_story_plan_people`, `gold_story_pipeline`, `gold_story_candidates_by_branch`
  (defined in `views_story_plan`).
- Open gap: nothing checks the confidence of the document matches behind `story_ready`.
