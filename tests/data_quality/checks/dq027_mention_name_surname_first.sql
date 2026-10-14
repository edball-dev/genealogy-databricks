-- id: DQ-027
-- title: Person mention name_raw looks surname-first (should be Forename Surname)
-- severity: warning
-- guards_bug: 1219333187665850
-- known_failing:
-- existing_asana_task:
-- description: >
--   ocr_pipeline asks Gemini for name_raw in forename-first order, but registers printed
--   surname-first (1939 Register, electoral rolls, directories, passenger lists) sometimes
--   come through as written ('Ball Davidina', 'McLean Mrs J'). Matching compares the first
--   word as the forename and the last as the surname, so such a mention never links to its
--   tree person. Flags a mention whose first word is a tree surname (tree or alias) and not
--   also a tree forename, whose last word is not a tree surname, and whose second word is a
--   tree forename, a title (Mrs, Miss, Mr, Mstr, Master, Dr, Rev, Ms) or a single letter.
--   Fix: correct name_raw to Forename Surname (drop the title) with a scoped UPDATE, or
--   re-run the page after a prompt fix. Heuristic: a surname that is also a tree forename
--   (e.g. Davis, Clifford) is skipped, so it can miss rows (found by hand for the 2026-10-14
--   correction). Expected baseline 0 after that correction; a new row after an OCR run means
--   the prompt did not hold for that page.

WITH s AS (
  SELECT UPPER(surname) AS sn FROM genealogy.gold_person_life WHERE surname IS NOT NULL GROUP BY 1
  UNION
  SELECT UPPER(alias_surname) FROM genealogy.silver_person_alias WHERE alias_surname IS NOT NULL GROUP BY 1
),
fn AS (
  SELECT UPPER(REGEXP_EXTRACT(given_name, '^(\\S+)', 1)) AS f
  FROM genealogy.gold_person_life WHERE given_name IS NOT NULL GROUP BY 1
),
t AS (
  SELECT file_id, page_index, person_index, name_raw, role_in_record,
    UPPER(REGEXP_EXTRACT(name_raw, '^(\\S+)', 1)) AS w1,
    UPPER(REGEXP_EXTRACT(name_raw, '^\\S+\\s+(\\S+)', 1)) AS w2,
    UPPER(REGEXP_EXTRACT(name_raw, '(\\S+)$', 1)) AS wl
  FROM genealogy.silver_transcript_person_mention
  WHERE name_raw RLIKE '^\\S+\\s+\\S+'
)
SELECT t.file_id, t.page_index, t.person_index, t.name_raw, t.role_in_record
FROM t
JOIN s ON s.sn = t.w1
LEFT JOIN s s2 ON s2.sn = t.wl
LEFT JOIN fn f1 ON f1.f = t.w1
LEFT JOIN fn f2 ON f2.f = t.w2
WHERE s2.sn IS NULL
  AND f1.f IS NULL
  AND (f2.f IS NOT NULL OR t.w2 RLIKE '^(MRS|MISS|MR|MSTR|MASTER|DR|REV|MS)$' OR LENGTH(t.w2) = 1)
