-- id: DQ-026
-- title: Derived person aliases in sync with the tree (missing or stale)
-- severity: warning
-- guards_bug: 1219377677279273
-- known_failing:
-- existing_asana_task:
-- description: >
--   silver_person_alias is hand-maintained and never rebuilt; notebook_sync_derived_aliases
--   adds the derived MARRIED_NAME and FATHER_SURNAME rows (insert-only) after each GEDCOM
--   ingest. This check compares the table with what the tree derives today.
--   MISSING_DERIVED: the tree implies a married-name or father-surname alias that the
--   table lacks (the sync has not run since the tree changed, or it failed).
--   STALE_DERIVED: a table row that the sync wrote (notes start 'Married name via' or
--   'Father surname via') that the tree no longer implies, because a marriage or parentage
--   was edited in Ancestry. The sync never removes rows, so delete these by hand if they
--   could cause a wrong match. Manual rows (any other notes) are never reported as stale.
--   Derivation rules and the match key (person, UPPER given name, UPPER surname, type) must
--   stay identical to notebook_sync_derived_aliases; change them together.

WITH derived AS (
  SELECT DISTINCT
    sf.wife_gedcom_id AS person_gedcom_id,
    wp.given_name     AS alias_given_name,
    hp.surname        AS alias_surname,
    'MARRIED_NAME'    AS alias_type
  FROM genealogy.silver_family sf
  JOIN genealogy.gold_person_life wp ON sf.wife_gedcom_id = wp.person_gedcom_id
  JOIN genealogy.gold_person_life hp ON sf.husband_gedcom_id = hp.person_gedcom_id
  WHERE hp.surname IS NOT NULL
    AND sf.relationship_type IS NULL
    AND hp.surname != wp.surname
  UNION ALL
  SELECT DISTINCT
    sfc.child_gedcom_id,
    cp.given_name,
    fp.surname,
    'FATHER_SURNAME'
  FROM genealogy.silver_family_child sfc
  JOIN genealogy.silver_family sf    ON sfc.family_gedcom_id = sf.family_gedcom_id
  JOIN genealogy.gold_person_life cp ON sfc.child_gedcom_id  = cp.person_gedcom_id
  JOIN genealogy.gold_person_life fp ON sf.husband_gedcom_id = fp.person_gedcom_id
  WHERE cp.surname != fp.surname
    AND fp.surname IS NOT NULL
    AND cp.surname IS NOT NULL
    AND sfc.father_relationship_type = 'biological'
)
SELECT 'MISSING_DERIVED' AS problem, d.person_gedcom_id, d.alias_given_name, d.alias_surname, d.alias_type
FROM derived d
LEFT ANTI JOIN genealogy.silver_person_alias e
  ON  e.person_gedcom_id = d.person_gedcom_id
  AND UPPER(TRIM(e.alias_given_name)) <=> UPPER(TRIM(d.alias_given_name))
  AND UPPER(TRIM(e.alias_surname))    <=> UPPER(TRIM(d.alias_surname))
  AND e.alias_type = d.alias_type
UNION ALL
SELECT 'STALE_DERIVED', a.person_gedcom_id, a.alias_given_name, a.alias_surname, a.alias_type
FROM genealogy.silver_person_alias a
LEFT ANTI JOIN derived d
  ON  a.person_gedcom_id = d.person_gedcom_id
  AND UPPER(TRIM(a.alias_given_name)) <=> UPPER(TRIM(d.alias_given_name))
  AND UPPER(TRIM(a.alias_surname))    <=> UPPER(TRIM(d.alias_surname))
  AND a.alias_type = d.alias_type
WHERE a.alias_type IN ('MARRIED_NAME', 'FATHER_SURNAME')
  AND (a.notes LIKE 'Married name via %' OR a.notes LIKE 'Father surname via %')
