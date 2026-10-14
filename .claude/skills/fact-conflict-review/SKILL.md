---
name: fact-conflict-review
description: Review and resolve open fact conflicts between document transcripts and the tree (gold_fact_comparison_reviewed). Use when Ed asks to review conflicts, "what's open for [person]", mark conflicts as reviewed, or when a conflict looks like a wrong-person match, a tree error or a genuine discrepancy in the record. Covers triage (which fix applies), the review-mark vocabulary, and the single-line INSERT pattern. Not for failing data-quality checks (use failing-check-triage) or OCR text corrections (use ocr-transcription-correction).
---

# Fact conflict review

Ancestry is the source of truth for the tree. A conflict is not "resolved" by hiding it: first decide
*why* the transcript and the tree disagree, fix the cause at source where there is one, and only then mark.

## 1. List what is open

Work from `workspace.genealogy.gold_fact_comparison_reviewed`, never raw `status = 'CONFLICT'`:

```sql
SELECT person_gedcom_id, display_name, fact_type, tree_value, transcript_value,
       file_id, file_name, source_doc_type, conflict_severity, notes
FROM workspace.genealogy.gold_fact_comparison_reviewed
WHERE is_open_conflict = TRUE
ORDER BY display_name, file_name, fact_type
```

Group by person and then by file. Present them to Ed that way (one file at a time reads best on a phone).
Check `refreshed_at` first: if notebook 03 has not run since the last match or extraction change, the list is stale.

## 2. Triage each conflict: which fix applies?

Check the pattern across the *whole file* before deciding on a single row.

| Pattern | Likely cause | Fix |
|---|---|---|
| Several conflicts on one file for one person, birth/death years 8+ years out, or the document's age fits a different relative | Wrong-person primary match | Pin the right person with a filename id token (`[Ixxxx]`, or `[NONE]`); see `.claude/rules/ocr-and-matching.md`. Re-run notebook 01, forced notebook 02 for that file, then source coverage, 03, B. Do not mark rows. |
| Only the household member rows conflict, the pinned primary is fine | Wrong household link (a same-named relative) | Mention-level `override_person_gedcom_id` on `silver_transcript_person_mention`, then re-run. |
| Transcript value is wrong against the image | OCR/transcription error | Use `ocr-transcription-correction`. Mark `TRANSCRIPTION_ERROR` only if the transcript cannot be corrected. |
| Tree value is wrong or missing and the document is accurate | Tree error | Fix the citation in Ancestry, re-download the GEDCOM, re-run the pipeline. Mark `TREE_CORRECTED` only after the tree changes. Do not paper over it with logic. |
| Same place, different granularity (parish vs town; Heigham vs Norwich) | Not a real conflict | `FALSE_POSITIVE` with `PLACE_GRANULARITY`. |
| Same place, different spelling (Dimsdale vs Dymsdale) | Not a real conflict | `FALSE_POSITIVE` with `PLACE_SPELLING`. |
| Age/year drifts by 1-3 years on a census or certificate | Rounded or stated age | `FALSE_POSITIVE` with `MINOR_AGE_DRIFT` (adult census ages are rounded; the Census age gate is 5 years). |
| Occupation the tree lacks but is real | Tree gap | `ACCEPTED`/`RESOLVED` is wrong here. Use reason `TREE_GAP`: it feeds `gold_housekeeping_tree_gap_occupations` and `SIGNAL_TRANSCRIPT_ONLY_FACTS`, and clears when the tree gains the occupation. |
| Student/pupil status, club or association role recorded as an occupation | Not an occupation | `NOT_AN_OCCUPATION` (appears nowhere). |
| Life-stage occupation (child "scholar", "at home") | Not a working occupation | `NOT_AN_OCCUPATION`. Do not use the deprecated `LIFE_STAGE_OCCUPATION`. |
| The record itself is inconsistent (two sources disagree and a GPS assessment has resolved it) | Genuine discrepancy | `ACCEPTED`, with the assessment linked in `notes`. `ACCEPTED` is *not* a general dismissal. |
| Cannot tell | | Ask Ed; show the image-based question, do not guess. |

Always verify what a signal or column means from its source notebook before telling Ed a conflict is wrong.

## 3. Review statuses and reason codes

`review_status`: `FALSE_POSITIVE` (not a real conflict), `RESOLVED` (real, fixed at source: tree, transcript or match corrected),
`ACCEPTED` (genuine discrepancy in the record, GPS-assessed).

`reason_code` (DQ-022 is **critical**, so a typo here turns CI red): `PLACE_GRANULARITY`, `PLACE_SPELLING`,
`TRANSCRIPTION_ERROR`, `WRONG_PERSON_MATCH`, `TREE_CORRECTED`, `MINOR_AGE_DRIFT`, `TREE_GAP`,
`NOT_AN_OCCUPATION`, `OTHER`. `TREE_GAP` and `NOT_AN_OCCUPATION` apply to `fact_type = 'occupation'` only.

`PLACE_GRANULARITY` and `PLACE_SPELLING` are both `FALSE_POSITIVE` reasons. **`LIFE_STAGE_OCCUPATION` has been removed** from DQ-022's allowed list (no existing rows used it).
Never write it; use `NOT_AN_OCCUPATION` for life-stage cases.

Prefer fixing wrong-person matches with a pin over a `WRONG_PERSON_MATCH` mark: a mark only hides the row, the wrong match
still feeds evidence scores. If one is already there, delete it once the pinned match has replaced the conflict.

## 4. Writing marks

Table: `workspace.genealogy.silver_fact_conflict_review`, key `(file_id, person_gedcom_id, fact_type, transcript_value)`.
Writes go through the Databricks MCP `execute_write_sql`: fully qualified table names, **one statement, one line**,
no semicolon anywhere (even inside a string), and avoid plain words the naive keyword filter rejects ("granted", "merge").
Confirm the table is accepted; if the call is rejected, tell Ed rather than working around it.

Copy `tree_value` into `reviewed_tree_value` (the mark stops applying if the tree value changes, except `TREE_GAP`).
Set `reviewed_at`. Bulk-friendly single-line pattern:

```sql
INSERT INTO workspace.genealogy.silver_fact_conflict_review (file_id, person_gedcom_id, fact_type, transcript_value, reviewed_tree_value, review_status, reason_code, notes, reviewed_at) SELECT file_id, person_gedcom_id, fact_type, transcript_value, tree_value, 'FALSE_POSITIVE', 'PLACE_GRANULARITY', 'Parish vs town', CURRENT_TIMESTAMP() FROM workspace.genealogy.gold_fact_comparison_reviewed WHERE is_open_conflict = TRUE AND person_gedcom_id = '@I332513961335@' AND fact_type = 'birth_place'
```

Only mark what Ed has said to mark. Echo back the list (person, fact, tree vs transcript, status, reason) and wait for his go-ahead
before writing. Deleting a mark is also a write: say what will go and why.

## 5. After marking

- The view applies marks live, so no rebuild is needed for them. Notebook 03 rebuilds `gold_fact_comparison` (TRUNCATE) but the marks survive.
- `SIGNAL_FACT_CONFLICT` counts only open conflicts, so re-run notebook B for signals/scores to change.
- Re-query `is_open_conflict` for the person to confirm the rows closed.
- DQ-023 reports marks that are STALE (tree value changed since review: reopened by design) or ORPHANED
  (the key vanished after a rebuild); a sudden jump after a notebook 03 run means marks are being dropped.

## 6. Things that have bitten before

- A reviewed row's key includes `transcript_value`; a forced `notebook_02` re-extraction that changes the value or the file's matched person orphans the old mark.
- Re-pinning a file changes which person owns the facts, so old marks against the previous person no longer join. Delete them once the conflicts are gone.
- Two different people can share a name in the same family (David, John); confirm by GEDCOM id and birth year, not name.
- A document's printed ages and a tree's birth year routinely differ by 1-3 years; do not chase those.
- Nothing checks the confidence of the document match behind a conflict. If many conflicts cluster on one file, suspect the match first.
