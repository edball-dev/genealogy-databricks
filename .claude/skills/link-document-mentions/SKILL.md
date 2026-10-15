---
name: link-document-mentions
description: Review primary document matches that have no mention link (gold_housekeeping_unlinked_primary_mentions) and link the ones Ed confirms from the image, by setting override_person_gedcom_id on the mention. Use when Ed asks to work through unlinked documents, "which direct ancestors have unlinked documents", or to link a document's primary person to its mention. Not for fact conflicts (use fact-conflict-review) or OCR text (use ocr-transcription-correction).
---

# Link document mentions

A primary match says which tree person a document is about. A mention link says which row on the page is that person,
so the page's facts attach to them. About 217 primaries have no mention link; most have nothing to link to.
Nothing is linked automatically. Ed confirms each link from the image.

## 1. List what to review

Work from `workspace.genealogy.gold_housekeeping_unlinked_primary_mentions`. One row per candidate mention, with the
primary's counts repeated on every row. Direct ancestors first:

```sql
SELECT file_name, file_id, person_gedcom_id, display_name, birth_year, candidate_count,
       subject_role_candidates, candidate_person_index, candidate_name_raw, candidate_role,
       candidate_age, candidate_dob, candidate_override
FROM workspace.genealogy.gold_housekeeping_unlinked_primary_mentions
WHERE direct_ancestor = TRUE AND candidate_count > 0
ORDER BY candidate_count, file_name, candidate_person_index
```

Group by document. Show Ed the primary (name, birth year), then each candidate (name as written, role, age, date of birth).

## 2. Triage

| Situation | What to do |
|---|---|
| `candidate_count = 0` | Nothing to link. If the page should have the person (a census household, for example), the extraction missed them: flag it for `ocr-transcription-correction`, don't guess a link. |
| One candidate, role fits the subject (Head, Husband, Wife, Deceased) | Propose the link. Ed checks the image before anything is written. |
| One candidate, other role (witness, landowner, informant) | Propose the link only if Ed confirms from the image that it's the same person. Check age and birth year against the tree. |
| Several candidates, one subject role | Propose the subject-role mention. Ed confirms from the image. |
| Several candidates, no single subject role | Ed decides. Don't pick one from the data. |
| `candidate_override` already set | Already linked. Ignore. |

Check the candidate isn't already claimed by someone else before linking:

```sql
SELECT file_id, person_index, person_gedcom_id, match_method, match_confidence
FROM workspace.genealogy.silver_document_person
WHERE file_id = '<file_id>' AND person_index = <n>
```

If it's already linked to a different person (household inference, for example), don't set a second link. Raise it with Ed.

## 3. Write the link (only after Ed confirms)

Use `execute_write_sql`. One statement, one line, no semicolon, a WHERE clause:

```sql
UPDATE workspace.genealogy.silver_transcript_person_mention SET override_person_gedcom_id = '<@I...@>', corrected_at = CURRENT_TIMESTAMP(), correction_notes = 'Linked to primary match after image check' WHERE file_id = '<file_id>' AND person_index = <n>
```

Confirm the statement succeeded, then re-query the view and report what dropped out.

Don't use `silver_document_person_override` (deprecated). Don't change `match_confidence`.

## 4. Re-run the pipeline

Cell 5d2 of `notebook_01_document_matching` fills the primary's `page_index` and `person_index` from the override. It only
updates a primary that has no link yet, otherwise it adds a `MENTION_OVERRIDE` row. Then:

1. Run `notebook_01_document_matching` (Ed can run it in the UI, or Claude via `run_notebook`).
2. If the linked document has facts, Ed runs a forced `notebook_02_fact_extraction` in the Databricks UI (`dry_run=false`, `force_file_ids` set to those files). The MCP identity cannot read the secret scope.
3. `refresh_gold_source_coverage`, then `notebook_03_fact_comparison`, then `notebook_B_person_signals`, one serverless run at a time.
4. Re-query the view. The linked primaries should be gone.

Before running, check the git folder is on `main` (scheduled jobs run whatever is checked out).

## 5. Things that have bitten before

- A mention can be claimed by household inference for another person. Setting a second override creates a duplicate link. Check step 2 first.
- Census pages with no matching mention (1921 census fronts, for example) are an extraction gap, not a linking gap.
- Ed has the final say on each link. Claude can't see the images.
