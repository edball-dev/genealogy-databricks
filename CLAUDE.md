# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Source control for the Databricks pipeline behind Ed Ball's "Genealogy Research Assistant" / *We Are The Balls* family-history project. There is no build system: almost everything is a **Databricks notebook** (`.ipynb`, with `%sql` cells and PySpark; a few `.sql` / `.py` notebooks) that runs in a Databricks workspace against Unity Catalog `workspace.genealogy` (code usually writes `genealogy.<table>`). Notebooks are executed on Databricks (or via the Databricks MCP tools: `run_notebook`, `execute_sql`, `repo_pull`), not locally. Only `tests/data_quality/` is ordinary Python that runs locally/CI.

## Data flow (medallion layout)

Layers are encoded in table prefixes: `bronze_*`, `control_*`, `silver_*`, `gold_*`, `ref_*` (hand-maintained reference/config tables, e.g. `ref_signal_weights`).

1. **GEDCOM tree pipeline**, numbered notebooks, run in order: `01_ingest_gedcom` (Ancestry GEDCOM from a Fivetran-synced Drive volume → `bronze_gedcom`; idempotent, raises `SKIP_PIPELINE` if the file id was already ingested) → `02_enrich` → `03a`/`03b` (events, sources) → `04_process_silver_tables` → `05a_parse_dates` / `05b_normalise_places` → `06_create_canonical_places` → `07_materialise_gold_tables` → `08_generate_ancestral_depth_and_proximity`. Each notebook's markdown header states its run-order prerequisites; read it before running.
2. **Document/OCR pipeline** (scanned records on Google Drive): `ocr_pipeline` (nightly Gemini transcription + person-mention extraction → `ocr_transcriptions`, `ocr_processing_log`) → `notebook_01_document_matching` (matches mentions/documents to tree people; many lettered cells 5, 5b…5j that must run in order) → `notebook_02_fact_extraction.py` (Gemini facts → `gold_transcript_facts`) → `notebook_03_fact_comparison.sql` (transcript facts vs tree facts; reviewed via `silver_fact_conflict_review`).
3. **Signals / scores / planning layer**: `notebook_A_ref_actions_and_signals` → `notebook_B_person_signals` (`gold_research_person_signals`, versioned in its header; needs Notebook A and 08 first) → `notebook_D_person_scores`, `gold_game_scores`, `score_calibration`, `notebook_scores_snapshot` (append-only history), plus `notebook_dna_load`/`notebook_dna_gold`, `notebook_week_plan`, `story_plan_*`, `person_story_status`.
4. **Views** live in `views_*.ipynb` and are deliberately split from table-building notebooks: recreating a view changes its ownership to whoever runs it, so run view notebooks only when definitions change.
5. `lib_*.ipynb` are shared helpers (`%run`-style): `lib_helper_utils` (e.g. `merge_place_canonicals`), `lib_gedcom_text_utils`.
6. `migration_*.ipynb` are one-off, dated schema migrations.

Cross-notebook dependencies are mostly implicit via table names — when changing a table's columns, grep for the table across all notebooks (and the DQ registries) before editing.

## Conventions worth knowing

- Notebooks carry long inline changelogs/version headers (e.g. "Notebook B v4.7") and reference Asana task GIDs for bugs. When you change a notebook, update its version/changelog header in the same style.
- Notebook `.ipynb` JSON is edited directly; keep cell structure intact and keep `%sql` cells idempotent (`CREATE OR REPLACE`, `MERGE`).
- Many notebooks end with "anomaly check" cells that should return 0 violations — preserve/extend them when changing logic.
- Gotchas documented in `tests/data_quality/README.md`: don't validate uniqueness with `COUNT(DISTINCT CONCAT(...))` (NULLs collapse), and don't batch multiple correlated `NOT EXISTS` checks in one query (intermittent Spark `INTERNAL_ERROR`).
- Genealogy tasks are tracked in Asana project `1218247114984172`; the DQ runner files into it.

## Data quality suite (`tests/data_quality/`)

Two tiers, both compiled to "a SELECT that returns violating rows; zero rows = pass":
- **Tier 2**: hand-written `checks/dqNNN_*.sql` with a strict comment header (`-- id:`, `title`, `severity`, `guards_bug`, `known_failing`, `existing_asana_task`, `description`) parsed by line prefix, not YAML. IDs are stable and never renumbered (DQ-003, DQ-007 intentionally absent).
- **Tier 1**: config-driven `registry/tier1_{gold,silver,bronze,ref}_registry.yaml` (`row_count_not_zero`, `freshness_vs_source`, `not_null`, `uniqueness`, `fk_integrity`, `not_blank`). All `tier1_*.yaml` files are globbed; add a layer by adding a new file. `freshness_vs_source` uses `DESCRIBE HISTORY`, so `depends_on` must never name a view.

Runs on four surfaces: `run_checks.py` locally, GitHub Actions (`.github/workflows/data-quality.yml`, on push to `main` / manual, secrets from the `production` environment), the `data_quality_suite.ipynb` notebook, and ad hoc SQL via the Databricks MCP. Results go to `genealogy.data_quality_results`; the registry is synced (full, unfiltered) to `genealogy.ref_data_quality_registry`. Only CI/`run_checks.py` files Asana tasks (the notebook does not, to avoid duplicates). `known_failing: true` checks don't turn the run red.

```bash
cd tests/data_quality
pip install -r requirements.txt
export DATABRICKS_HOST=... DATABRICKS_TOKEN=... DATABRICKS_WAREHOUSE=...   # all required; ASANA_PAT optional
python run_checks.py                                  # everything
python run_checks.py --only DQ-001,DQ-006             # specific check IDs (Tier 1 IDs look like T1-FRESHNESS_VS_SOURCE-GOLD_DNA_COVERAGE)
python run_checks.py --severity critical
python run_checks.py --layer silver --object genealogy.silver_person   # Tier 1 only; never affects Tier 2 or the registry sync
```

Exit is non-zero only if a `critical` check *newly* fails. When adding a Tier 2 check, validate the SQL against live data first and confirm the baseline row count; see `tests/data_quality/README.md` for the full format and `SODA_PILOT_OUTCOME.md` for why a hand-rolled registry was chosen over Soda.
