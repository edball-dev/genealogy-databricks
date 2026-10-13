# genealogy-databricks

Databricks notebooks, SQL views and data-quality checks that build the
`workspace.genealogy` warehouse (bronze / silver / gold) behind the
Genealogy Research Assistant app and the Claude research skills.

## How the data flows

- **Tree**: Ancestry GEDCOM export (manual) -> Google Drive -> Fivetran -> bronze -> silver -> gold.
- **Documents**: Drive `Documents` folder -> Fivetran -> OCR pipeline (Vertex AI Gemini) -> `ocr_transcriptions` -> document matching -> gold coverage and fact tables.
- **Scores and signals**: gold tables feed the research-priority scores the app reads.
- **DNA**: Ancestry/MyHeritage scrape -> `silver_dna_match` -> gold DNA tables.

## Platform

- Databricks Free Edition (AWS), serverless compute only, one 2X-Small SQL warehouse.
- This repo is checked out as a Databricks git folder named `genealogy-databricks`.
  Scheduled jobs run notebooks from whichever branch is checked out there.

## Contents

- Notebooks (`.ipynb`) and SQL scripts that build the tables and views.
- `tests/data_quality/` - data-quality suite (Tier 1 YAML registry, Tier 2 SQL checks).
- `.github/workflows/data-quality.yml` - runs the suite manually (workflow dispatch only).

## Running the data-quality suite

Four ways, same checks: via the Databricks MCP server, the
`data_quality_suite.ipynb` notebook, GitHub Actions, or locally with
`run_checks.py`. Results are written to `genealogy.data_quality_results`.

## More documentation

- Project goals and architecture: Notion `PROJECT_CONTEXT` (Claude Genealogy workspace).
- Data-quality plan and history: Notion `DATA_QUALITY_TEST_PLAN`.
- Task backlog: Asana, Genealogy project.
- Full schema: auto-exported schema doc in Google Drive (id `1g1u_t3QPqvGTXkz-SOCHEdBzMvqqsn2nXtDRhrK5U0c`).
- Conventions for Claude Code: `CLAUDE.md` and `.claude/rules/`.
