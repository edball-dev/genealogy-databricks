# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02: Fact Extraction via Gemini
# MAGIC
# MAGIC Reads OCR pages and the people mentioned on them (silver_transcript_person_mention), calls Gemini once per page (or per chunk of a long page) to extract structured facts for EVERY person on the page, attaches each person's facts to the tree person linked via silver_document_person (file_id, page_index, person_index), writes to gold_transcript_facts.
# MAGIC
# MAGIC Run on Databricks serverless compute. Requires GCP credentials in secret scope 'genealogy', key 'gcp_service_account_json'.
# MAGIC
# MAGIC **Change log:**
# MAGIC - v1: Initial version
# MAGIC - v1.1: Align Gemini initialisation, call pattern, and JSON extraction with ocr_pipeline v3.2:
# MAGIC         use from_service_account_info + explicit credentials object (not GOOGLE_APPLICATION_CREDENTIALS env var);
# MAGIC         add generation_config (temperature=0.1, max_output_tokens=8192) to generate_content call;
# MAGIC         add safety block detection on candidates/parts;
# MAGIC         replace fence-stripping JSON extraction with outermost-brace regex (more robust);
# MAGIC         add before_sleep retry logging;
# MAGIC         align model name with ocr_pipeline (gemini-3-pro-preview)
# MAGIC - v1.2: Add `inferred` boolean field to fact schema and prompt.
# MAGIC         Instructs Gemini to set inferred=true when a fact is derived from context
# MAGIC         rather than explicitly stated (e.g. birth_year calculated from age_at_doc).
# MAGIC         Downstream: gold_fact_comparison can weight inferred facts as weaker evidence.
# MAGIC - v1.3: Migrate from deprecated vertexai.generative_models SDK to google-genai SDK.
# MAGIC         vertexai.generative_models is deprecated as of June 2025, removed June 2026.
# MAGIC         New pattern: google-genai client (client.models.generate_content) with
# MAGIC         types.GenerateContentConfig. Credentials passed via google.auth directly.
# MAGIC - v1.4: Fix collateral-household fact-extraction gap (Asana 1218420653379642).
# MAGIC         Cell 3's NOT EXISTS guard was scoped to file_id alone, so once any
# MAGIC         person on a file had facts extracted, every other silver_document_person
# MAGIC         row for that same file (a Cell 5b spouse-inference row, or a Cell 5e
# MAGIC         household-member row) was silently skipped forever. Now scoped to
# MAGIC         (file_id, person_gedcom_id). Also stopped feeding every row the
# MAGIC         filename-parsed forename/surname regardless of which person the row is
# MAGIC         actually about — now pulls the matched person's own given_name/surname
# MAGIC         via gold_person_life, so a household member's row asks Gemini to extract
# MAGIC         facts about *that* person, not the file's primary/filename subject.
# MAGIC
# MAGIC - v2.0: Page-level extraction (Asana 1218420653302709). Unit is the page, not (file, person, page).
# MAGIC         Gemini is given the people on the page (person_index, name as written, role, age, detail) and returns facts per
# MAGIC         person_index; facts reach a tree person through silver_document_person.person_index, so the filename/matcher
# MAGIC         anchor stays and Gemini no longer hunts the page for a tree name. Unlinked mentions are extracted with a NULL
# MAGIC         person_gedcom_id and never guessed. Long pages are chunked (no 4,000-char cut-off); page_index/person_index are
# MAGIC         stored on each fact; no duplicates across chunks/pages. Age, name and (census) inferred birth_year come from the
# MAGIC         mention row, not Gemini. Prompt captures birthplace, years married, children, address, narrative detail.
# MAGIC         Cost control: dry_run widget (default true) reports page/call counts without calling Gemini; max_pages limit;
# MAGIC         per-page status persisted in silver_fact_extraction_status (errors retried only with retry_errors=true);
# MAGIC         census links whose mention age disagrees with the tree birth year by >5 years are logged to
# MAGIC         silver_document_match_exception and not stored against that tree person.
# MAGIC         Needs notebook_02_schema_and_cleanup.sql applied first.
# MAGIC - v2.1: Align Gemini config with ocr_pipeline: model gemini-3.1-pro-preview (gemini-3-pro-preview returns 404 on this
# MAGIC         project), thinking_level LOW, max_output_tokens 65536.
# MAGIC - v2.2: force_file_ids now limits the run to exactly those files (previously they were only added to the backlog, so
# MAGIC         max_pages could be spent on other files first).
# MAGIC
# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Install dependencies

# COMMAND ----------

# MAGIC %pip install google-genai tenacity



# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Imports, config and run controls
# MAGIC
# MAGIC Widgets: `dry_run` (default true: counts pages/calls, no Gemini call, no writes), `max_pages` (hard cap on pages per run),
# MAGIC `force_file_ids` (comma-separated file_ids to re-extract: their legacy facts, those with page_index NULL, are deleted after a successful extraction),
# MAGIC `retry_errors` (re-run pages whose status is ERROR; off by default so errors are not silently re-billed).

# COMMAND ----------

import base64
import json
import time
import re
from datetime import datetime, timezone

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, TimestampType
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

from google import genai
from google.genai import types
from google.oauth2 import service_account
import google.api_core.exceptions

GCP_PROJECT    = "genealogy-488213"
GCP_LOCATION   = "global"
GEMINI_MODEL   = "gemini-3.1-pro-preview"   # same model as ocr_pipeline VERTEX_MODEL; gemini-3-pro-preview 404s on this project
REQUEST_DELAY  = 4
MAX_RETRIES    = 5
MAX_CHUNK_CHARS = 30000     # a page longer than this is split on line boundaries into several calls
AGE_CHECK_DOC_TYPES = {"Census"}   # doc types where mention age is a true age at document date
AGE_CHECK_TOLERANCE = 5            # years between (doc year - mention age) and tree birth year before a link is rejected

dbutils.widgets.dropdown("dry_run", "true", ["true", "false"], "Dry run (no Gemini calls, no writes)")
dbutils.widgets.text("max_pages", "25", "Max pages to process this run")
dbutils.widgets.text("force_file_ids", "", "Comma-separated file_ids to re-extract")
dbutils.widgets.dropdown("retry_errors", "false", ["true", "false"], "Retry pages with status ERROR")

DRY_RUN        = dbutils.widgets.get("dry_run") == "true"
MAX_PAGES      = int(dbutils.widgets.get("max_pages") or 25)
FORCE_FILE_IDS = {f.strip() for f in dbutils.widgets.get("force_file_ids").split(",") if f.strip()}
RETRY_ERRORS   = dbutils.widgets.get("retry_errors") == "true"
print(f"dry_run={DRY_RUN} max_pages={MAX_PAGES} force_file_ids={len(FORCE_FILE_IDS)} retry_errors={RETRY_ERRORS}")

client = None
if not DRY_RUN:
    # Build explicit credentials and pass to the google-genai client.
    # google-genai replaces the deprecated vertexai.generative_models SDK (removed June 2026).
    creds_b64  = dbutils.secrets.get(scope="genealogy", key="gcp_service_account_json")
    sa_info    = json.loads(base64.b64decode(creds_b64).decode("utf-8"))
    credentials = service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION, credentials=credentials)
    print(f"google-genai client initialised. Model: {GEMINI_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Build the work list: pages, their people, and the tree links
# MAGIC
# MAGIC A page is in scope when it has at least one HIGH/MEDIUM `silver_document_person` row and no row in `silver_fact_extraction_status`
# MAGIC (or an ERROR row with `retry_errors`, or its file is in `force_file_ids`).
# MAGIC Mentions with NULL `page_index` (most of them) belong to the file's only page, or, on a multi-page file, are offered to every page and
# MAGIC Gemini returns only those present in the excerpt; duplicates are dropped on (person_index, fact_type, fact_value).

# COMMAND ----------

status_filter = "st.status = 'ERROR'" if RETRY_ERRORS else "FALSE"
force_list = ",".join("'" + f.replace("'", "") + "'" for f in FORCE_FILE_IDS) or "''"

# When force_file_ids is set the run is limited to exactly those files (status ignored); otherwise the normal backlog applies.
scope_filter = (f"ot.file_id IN ({force_list})" if FORCE_FILE_IDS
                else f"(st.file_id IS NULL OR {status_filter})")

pages_df = spark.sql(f"""
  SELECT ot.file_id, ot.file_name, ot.page_index, ot.year, ot.doc_type_detected,
         ot.transcribed_text, ot.personal_names, ot.locations,
         COUNT(*) OVER (PARTITION BY ot.file_id) AS n_pages_in_file
  FROM genealogy.ocr_transcriptions ot
  LEFT JOIN genealogy.silver_fact_extraction_status st
         ON st.file_id = ot.file_id AND st.page_index <=> ot.page_index
  WHERE ot.transcribed_text IS NOT NULL
    AND EXISTS (SELECT 1 FROM genealogy.silver_document_person sdp
                WHERE sdp.file_id = ot.file_id AND sdp.match_confidence IN ('HIGH', 'MEDIUM'))
    AND {scope_filter}
  ORDER BY ot.file_name, ot.page_index
""")
pages = [r.asDict() for r in pages_df.collect()]

file_ids = sorted({p["file_id"] for p in pages})
mentions_by_file, links_by_file = {}, {}
if file_ids:
    ids_sql = ",".join("'" + f + "'" for f in file_ids)
    for m in spark.sql(f"""
        SELECT file_id, page_index, person_index, name_raw, role_in_record, age_raw, age_years, dob_raw, detail
        FROM genealogy.silver_transcript_person_mention WHERE file_id IN ({ids_sql})
        ORDER BY file_id, person_index
    """).collect():
        mentions_by_file.setdefault(m["file_id"], []).append(m.asDict())
    # person_gedcom_id for each mention, via the key silver_document_person carries (null-safe on page_index)
    for l in spark.sql(f"""
        SELECT sdp.file_id, sdp.page_index, sdp.person_index, sdp.person_gedcom_id, pl.birth_year AS tree_birth_year
        FROM genealogy.silver_document_person sdp
        LEFT JOIN genealogy.gold_person_life pl ON pl.person_gedcom_id = sdp.person_gedcom_id
        WHERE sdp.file_id IN ({ids_sql}) AND sdp.match_confidence IN ('HIGH', 'MEDIUM')
          AND sdp.person_index IS NOT NULL
    """).collect():
        links_by_file.setdefault(l["file_id"], []).append(l.asDict())

# Legacy (page-level-unaware) facts: a (file, person) that already has rows is not re-written unless the file is forced.
legacy_pairs = set()
if file_ids:
    for r in spark.sql(f"""
        SELECT DISTINCT file_id, person_gedcom_id FROM genealogy.gold_transcript_facts
        WHERE file_id IN ({ids_sql}) AND person_gedcom_id IS NOT NULL
    """).collect():
        legacy_pairs.add((r["file_id"], r["person_gedcom_id"]))


def mentions_for_page(p):
    ms = mentions_by_file.get(p["file_id"], [])
    out = [m for m in ms if m["page_index"] is not None and m["page_index"] == p["page_index"]]
    out += [m for m in ms if m["page_index"] is None]   # unknown page: single-page file => this page; multi-page => offered, Gemini filters
    return out


def chunk_text(text, limit=MAX_CHUNK_CHARS):
    if len(text) <= limit:
        return [text]
    chunks, cur, size = [], [], 0
    for line in text.split("\n"):
        while len(line) > limit:                      # pathological single line
            if cur: chunks.append("\n".join(cur)); cur, size = [], 0
            chunks.append(line[:limit]); line = line[limit:]
        if size + len(line) + 1 > limit and cur:
            chunks.append("\n".join(cur)); cur, size = [], 0
        cur.append(line); size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


work = []
for p in pages:
    p["mentions"] = mentions_for_page(p)
    p["chunks"] = chunk_text(p["transcribed_text"])
    if p["mentions"]:
        work.append(p)
    else:
        p["no_people"] = True

no_people_pages = [p for p in pages if p.get("no_people")]
selected = work[:MAX_PAGES]

n_calls = sum(len(p["chunks"]) for p in selected)
n_chars = sum(len(p["transcribed_text"]) for p in selected)
print(f"Pages in scope (all runs): {len(work)}  (+{len(no_people_pages)} with no mentions, would be marked NO_PEOPLE)")
print(f"This run (max_pages={MAX_PAGES}): {len(selected)} pages, {n_calls} Gemini calls, {n_chars:,} transcript chars")
print(f"Whole backlog: {sum(len(p['chunks']) for p in work)} calls, {sum(len(p['transcribed_text']) for p in work):,} chars")

# Compare with the old per-person design on the same pages: one call per linked (file, person, page)
linked_pairs = {(l["file_id"], l["person_gedcom_id"]) for p in selected for l in links_by_file.get(p["file_id"], [])}
print(f"Old per-person design for the same files: ~{sum(len(p['chunks']) * max(1, len([l for l in links_by_file.get(p['file_id'], [])])) for p in selected)} calls ({len(linked_pairs)} file-person pairs)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Prompt and Gemini call

# COMMAND ----------

FACT_EXTRACTION_PROMPT = """You are a genealogy research assistant. Extract structured facts about EVERY person listed below from this historical document transcript.

DOCUMENT: type {doc_type}, year {doc_year}{chunk_note}

PEOPLE ON THIS PAGE (already identified; person_index is the key you must return):
{people_block}

TRANSCRIPT:
{transcript_text}

ADDITIONAL CONTEXT:
- Locations mentioned: {locations}

INSTRUCTIONS:
1. Return facts for each listed person, keyed by person_index. Read each person's own entry, not another person's: where several people share a first name or surname (father and son, two Georges), use the listed age and role to tell them apart.
2. If a listed person does not appear in this transcript excerpt, omit them. Do not invent people or facts.
3. Do NOT return forename, surname or age_at_doc: they are already recorded.
4. Extract these fact types where present:
   - birth_year (4-digit), birth_place, death_year, death_place, marriage_year, marriage_place
   - occupation: as written. Do NOT extract military ranks (Pte, Cpl, Sgt, Lt) as occupations
   - residence_place: address or place of residence at the time of the document
   - marital_status (e.g. Married, Widow, Unmarried)
   - years_married: number of years married, if stated
   - relationship_to_head: for households
   - father_name, mother_name, spouse_name
   - child_name: one fact per child named or listed for this person (also children born alive / still living counts as child_count)
   - narrative_detail: any other interesting detail about this person stated in the document (cause of death, military service, illness, disability, lodgers, employer, reasons for the event). One concise sentence per fact.
5. Assign confidence: high (clearly stated), medium (partially legible), low (uncertain).
6. Set `inferred` true if derived from context rather than stated (e.g. birth_year from an age). Directly stated facts are inferred false.

Your response must be a single JSON object and nothing else.
Do not include any text, explanation, or commentary before or after the JSON.
Do not wrap the JSON in markdown code fences.
Your entire response must be valid JSON starting with {{ and ending with }}.

{{
  "people": [
    {{"person_index": 0,
      "facts": [
        {{"fact_type": "birth_place", "fact_value": "Nottingham", "fact_year": null, "confidence": "high", "inferred": false}},
        {{"fact_type": "occupation", "fact_value": "Boot Maker", "fact_year": null, "confidence": "high", "inferred": false}}
      ]}}
  ]
}}"""


def extract_json(text: str) -> dict:
    """Outermost { ... } regardless of surrounding text or fences — consistent with ocr_pipeline v3."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)


def _call_gemini_once(prompt: str) -> dict:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=65536,   # as ocr_pipeline; thinking tokens count against this
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),   # as ocr_pipeline: keeps Gemini 3 thinking (and cost) down
            http_options=types.HttpOptions(timeout=180000),
        ),
    )
    if not response.candidates:
        return {"_safety_blocked": True}
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return {"_safety_blocked": True, "_finish_reason": str(getattr(candidate, "finish_reason", "unknown"))}
    raw_text = response.text.strip()
    try:
        return extract_json(raw_text)
    except (json.JSONDecodeError, ValueError):
        return {"_parse_error": True, "_raw": raw_text[:500]}


@retry(
    retry=retry_if_exception_type((
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.TooManyRequests,
        google.api_core.exceptions.DeadlineExceeded,
        google.api_core.exceptions.Cancelled,
    )),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(MAX_RETRIES),
    before_sleep=lambda rs: print(f"    Rate limited (429) — attempt {rs.attempt_number} failed, retrying..."),
)
def call_gemini_with_retry(prompt: str) -> dict:
    return _call_gemini_once(prompt)


def people_block(mentions):
    lines = []
    for m in mentions:
        bits = [f"person_index {m['person_index']}: {m['name_raw'] or '?'}"]
        if m["role_in_record"]: bits.append(f"role {m['role_in_record']}")
        if m["age_raw"]:        bits.append(f"age {m['age_raw']}")
        if m["dob_raw"]:        bits.append(f"born {m['dob_raw']}")
        if m["detail"]:         bits.append(f"detail: {m['detail']}")
        lines.append("- " + "; ".join(bits))
    return "\n".join(lines)


def derived_facts(m, doc_year, doc_type):
    """Facts taken from the mention row, no Gemini call. Birth year only where the age is a true age at the document date."""
    out = []
    if m["name_raw"]:
        out.append(("name_as_written", m["name_raw"], None, "high", False))
    if m["age_years"] is not None:
        out.append(("age_at_doc", str(m["age_years"]), None, "high", False))
        if doc_type in AGE_CHECK_DOC_TYPES and str(doc_year or "").isdigit():
            by = int(doc_year) - int(m["age_years"])
            out.append(("birth_year", str(by), by, "medium", True))
    return out

print("Functions defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Extraction loop (skipped entirely on a dry run)

# COMMAND ----------

fact_rows, status_rows, exception_rows, forced_cleanup = [], [], [], set()
now = lambda: datetime.now(timezone.utc)


def accepted_link(p, m, link):
    """Deterministic anchor check: a census link whose mention age is >AGE_CHECK_TOLERANCE years from the tree birth year is rejected."""
    if p["doc_type_detected"] in AGE_CHECK_DOC_TYPES and m["age_years"] is not None \
       and link["tree_birth_year"] is not None and str(p["year"] or "").isdigit():
        implied = int(p["year"]) - int(m["age_years"])
        if abs(implied - int(link["tree_birth_year"])) > AGE_CHECK_TOLERANCE:
            return False, f"mention {m['name_raw']} age {m['age_years']} implies b.{implied}; tree b.{link['tree_birth_year']}"
    return True, ""


def process_page(p):
    ms = p["mentions"]
    by_idx = {m["person_index"]: m for m in ms}
    seen, page_facts, calls = set(), [], 0
    for ci, chunk in enumerate(p["chunks"]):
        note = f", part {ci+1} of {len(p['chunks'])} of a long page" if len(p["chunks"]) > 1 else ""
        prompt = FACT_EXTRACTION_PROMPT.format(
            doc_type=p["doc_type_detected"] or "Unknown", doc_year=p["year"] or "", chunk_note=note,
            people_block=people_block(ms), transcript_text=chunk, locations=p["locations"] or "[]")
        parsed = call_gemini_with_retry(prompt)
        calls += 1
        if parsed.get("_safety_blocked"):
            raise RuntimeError(f"Safety blocked (finish_reason: {parsed.get('_finish_reason', 'unknown')})")
        if parsed.get("_parse_error"):
            raise RuntimeError(f"Unparseable response: {parsed['_raw']}")
        for person in parsed.get("people", []):
            pi = person.get("person_index")
            if pi not in by_idx:
                continue
            for f in person.get("facts", []):
                key = (pi, f.get("fact_type"), str(f.get("fact_value")).strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                page_facts.append((pi, f.get("fact_type"), f.get("fact_value"), f.get("fact_year"),
                                   f.get("confidence", "medium"), bool(f.get("inferred", False))))
        if ci < len(p["chunks"]) - 1:
            time.sleep(REQUEST_DELAY)
    # mention-derived facts (no Gemini); only for mentions Gemini saw on this page, or any mention when the file has one page
    present = {pi for pi, *_ in page_facts}
    for m in ms:
        if m["person_index"] in present or p["n_pages_in_file"] == 1:
            for t, v, y, c, inf in derived_facts(m, p["year"], p["doc_type_detected"]):
                if (m["person_index"], t, v.strip().lower()) not in seen:
                    seen.add((m["person_index"], t, v.strip().lower()))
                    page_facts.append((m["person_index"], t, v, y, c, inf))
    return page_facts, calls


if DRY_RUN:
    print("DRY RUN: no Gemini calls made and nothing written. Re-run with dry_run=false to extract.")
else:
    for i, p in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] {p['file_name']} page {p['page_index']} ({len(p['mentions'])} people, {len(p['chunks'])} chunk(s))")
        try:
            page_facts, calls = process_page(p)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            status_rows.append((p["file_id"], p["page_index"], "ERROR", str(e)[:1000], 0, 0, now()))
            time.sleep(REQUEST_DELAY)
            continue

        links = {}
        for l in links_by_file.get(p["file_id"], []):
            links.setdefault(l["person_index"], []).append(l)
        by_idx = {m["person_index"]: m for m in p["mentions"]}
        rejected = 0
        written = 0
        for pi, t, v, y, c, inf in page_facts:
            m = by_idx[pi]
            gid = None
            ls = links.get(pi, [])
            if len(ls) == 1 and (ls[0]["page_index"] is None or ls[0]["page_index"] == p["page_index"] or m["page_index"] is None):
                ok, why = accepted_link(p, m, ls[0])
                if ok:
                    gid = ls[0]["person_gedcom_id"]
                elif t == "name_as_written":     # log the rejection once per mention
                    rejected += 1
                    exception_rows.append((p["file_id"], p["file_name"], p["doc_type_detected"], p["year"], None, None, 1,
                                           f"FACT_EXTRACTION_LINK_REJECTED: {why}", ls[0]["person_gedcom_id"], now()))
            if gid is not None and (p["file_id"], gid) in legacy_pairs and p["file_id"] not in FORCE_FILE_IDS:
                continue                          # already has legacy facts; not duplicated
            fact_rows.append({"file_id": p["file_id"], "person_gedcom_id": gid, "fact_type": t, "fact_value": v,
                              "fact_year": y if isinstance(y, int) else None, "confidence": c, "inferred": inf,
                              "source_doc_type": p["doc_type_detected"], "extracted_at": now(),
                              "page_index": p["page_index"], "person_index": pi})
            written += 1
        status = "DONE" if page_facts else "NO_PEOPLE"
        if rejected and rejected == len({pi for pi, *_ in page_facts if links.get(pi)}) and rejected > 0:
            status = "REJECTED"
        status_rows.append((p["file_id"], p["page_index"], status, f"{rejected} link(s) rejected" if rejected else None, calls, written, now()))
        if p["file_id"] in FORCE_FILE_IDS:
            forced_cleanup.add(p["file_id"])
        print(f"  ✓ {len(page_facts)} facts ({written} kept, {rejected} link(s) rejected), {calls} call(s)")
        if i < len(selected) - 1:
            time.sleep(REQUEST_DELAY)
    for p in no_people_pages[:MAX_PAGES]:
        status_rows.append((p["file_id"], p["page_index"], "NO_PEOPLE", "no mention rows for this page", 0, 0, now()))

    print(f"\nFacts: {len(fact_rows)}  Page statuses: {len(status_rows)}  Link rejections: {len(exception_rows)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Write facts, statuses and exceptions to Delta

# COMMAND ----------

if DRY_RUN:
    print("Dry run: nothing written.")
else:
    # Forced re-extraction: remove the file's legacy facts (page_index NULL) only now that extraction succeeded.
    for fid in forced_cleanup:
        spark.sql(f"DELETE FROM genealogy.gold_transcript_facts WHERE file_id = '{fid}' AND page_index IS NULL")
    if forced_cleanup:
        print(f"Deleted legacy facts for {len(forced_cleanup)} forced file(s).")

    if fact_rows:
        schema = StructType([
            StructField("file_id",          StringType(),    False),
            StructField("person_gedcom_id", StringType(),    True),
            StructField("fact_type",        StringType(),    True),
            StructField("fact_value",       StringType(),    True),
            StructField("fact_year",        IntegerType(),   True),
            StructField("confidence",       StringType(),    True),
            StructField("inferred",         BooleanType(),   True),
            StructField("source_doc_type",  StringType(),    True),
            StructField("extracted_at",     TimestampType(), True),
            StructField("page_index",       IntegerType(),   True),
            StructField("person_index",     IntegerType(),   True),
        ])
        spark.createDataFrame(fact_rows, schema=schema).write.format("delta").mode("append").saveAsTable("genealogy.gold_transcript_facts")
        print(f"Written {len(fact_rows)} fact rows to gold_transcript_facts")

    if status_rows:
        s_schema = StructType([
            StructField("file_id", StringType(), False), StructField("page_index", IntegerType(), True),
            StructField("status", StringType(), True), StructField("detail", StringType(), True),
            StructField("n_calls", IntegerType(), True), StructField("n_facts", IntegerType(), True),
            StructField("attempted_at", TimestampType(), True)])
        sdf = spark.createDataFrame(status_rows, schema=s_schema)
        # replace any prior row for the same page (ERROR retries, forced re-runs) so each page has exactly one status
        sdf.createOrReplaceTempView("_new_extraction_status")
        spark.sql("""
          MERGE INTO genealogy.silver_fact_extraction_status t USING _new_extraction_status s
          ON t.file_id = s.file_id AND t.page_index <=> s.page_index
          WHEN MATCHED THEN UPDATE SET *
          WHEN NOT MATCHED THEN INSERT *
        """)
        print(f"Recorded {len(status_rows)} page status rows")

    if exception_rows:
        e_schema = StructType([
            StructField("file_id", StringType(), True), StructField("file_name", StringType(), True),
            StructField("doc_type_detected", StringType(), True), StructField("year", StringType(), True),
            StructField("surname", StringType(), True), StructField("forename", StringType(), True),
            StructField("candidate_count", IntegerType(), True), StructField("reason", StringType(), True),
            StructField("candidates_considered", StringType(), True), StructField("logged_at", TimestampType(), True)])
        spark.createDataFrame(exception_rows, schema=e_schema).write.format("delta").mode("append").saveAsTable("genealogy.silver_document_match_exception")
        print(f"Logged {len(exception_rows)} rejected link(s) to silver_document_match_exception")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Preview extracted facts

# COMMAND ----------

spark.sql("""
  SELECT
    gtf.file_id,
    gtf.page_index,
    gtf.person_index,
    m.name_raw,
    p.given_name,
    p.surname,
    gtf.fact_type,
    gtf.fact_value,
    gtf.confidence,
    gtf.inferred,
    gtf.source_doc_type
  FROM genealogy.gold_transcript_facts gtf
  LEFT JOIN genealogy.silver_transcript_person_mention m
         ON m.file_id = gtf.file_id AND m.person_index = gtf.person_index
  LEFT JOIN genealogy.gold_person_life p ON gtf.person_gedcom_id = p.person_gedcom_id
  WHERE gtf.person_index IS NOT NULL
  ORDER BY gtf.extracted_at DESC, gtf.file_id, gtf.person_index, gtf.fact_type
  LIMIT 200
""").display()
