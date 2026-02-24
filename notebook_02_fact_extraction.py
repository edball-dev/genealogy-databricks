# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02: Fact Extraction via Gemini
# MAGIC
# MAGIC Reads matched transcripts from silver_document_person + ocr_transcriptions, calls Gemini to extract structured facts, writes to gold_transcript_facts.
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Install dependencies

# COMMAND ----------

# MAGIC %pip install google-genai tenacity

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Imports and config

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
GEMINI_MODEL   = "gemini-3-pro-preview"   # swap to gemini-3-flash-preview if cost is a concern
REQUEST_DELAY  = 4
MAX_RETRIES    = 5

# Build explicit credentials and pass to the google-genai client.
# google-genai replaces the deprecated vertexai.generative_models SDK (removed June 2026).
creds_b64  = dbutils.secrets.get(scope="genealogy", key="gcp_service_account_json")
creds_json = base64.b64decode(creds_b64).decode("utf-8")
sa_info    = json.loads(creds_json)

credentials = service_account.Credentials.from_service_account_info(
    sa_info,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
    credentials=credentials,
)
print(f"google-genai client initialised. Model: {GEMINI_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Load matched transcripts not yet fact-extracted

# COMMAND ----------

transcripts_df = spark.sql("""
  SELECT
    sdp.file_id,
    sdp.person_gedcom_id,
    sdp.match_confidence,
    ot.file_name,
    ot.surname,
    ot.forename,
    ot.year,
    ot.doc_type_detected,
    ot.transcribed_text,
    ot.personal_names,
    ot.locations
  FROM genealogy.silver_document_person sdp
  JOIN genealogy.ocr_transcriptions ot ON sdp.file_id = ot.file_id
  WHERE sdp.match_confidence IN ('HIGH', 'MEDIUM')
    AND ot.transcribed_text IS NOT NULL
    AND NOT EXISTS (
      SELECT 1 FROM genealogy.gold_transcript_facts gtf
      WHERE gtf.file_id = sdp.file_id
    )
  ORDER BY sdp.match_confidence DESC, ot.file_name
""")

records = transcripts_df.collect()
print(f"Transcripts to process: {len(records)}")
for r in records:
    print(f"  {r['file_name']} -> {r['person_gedcom_id']} ({r['match_confidence']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Fact extraction prompt and Gemini call

# COMMAND ----------

FACT_EXTRACTION_PROMPT = """You are a genealogy research assistant. Extract structured facts about the NAMED SUBJECT from this historical document transcript.

NAMED SUBJECT: {forename} {surname} (document year: {doc_year}, document type: {doc_type})

TRANSCRIPT:
{transcript_text}

ADDITIONAL CONTEXT:
- Personal names mentioned: {personal_names}
- Locations mentioned: {locations}

INSTRUCTIONS:
1. Focus only on facts directly about the named subject ({forename} {surname}).
2. For census records, the subject may be head of household, a child, or a spouse.
3. Extract the following fact types where present:
   - birth_year: year of birth as a 4-digit number
   - birth_place: place of birth as written in the document
   - death_year: year of death
   - death_place: place of death
   - marriage_year: year of marriage
   - marriage_place: place of marriage
   - occupation: occupation or trade as written
   - forename: forename as written (may differ from filename)
   - surname: surname as written (may differ from filename)
   - residence_place: address or place of residence at time of document
   - age_at_doc: age as stated in the document (integer as string)
   - father_name: father's name if stated
   - mother_name: mother's name if stated
   - spouse_name: spouse's name if stated
4. Assign confidence: high (clearly stated), medium (inferred/partially legible), low (uncertain).
5. Set `inferred` to true if the fact is derived from context rather than explicitly stated.
   Examples of inferred facts:
   - birth_year calculated from an age stated in the document (e.g. age 49 in an 1891 census → birth_year 1842)
   - birth_place assumed from a residence_place with no direct statement
   - occupation assumed from a title or honorific rather than a stated trade
   Directly stated facts (e.g. "born 1842", "born in Nottingham") should have inferred: false.
6. Only include facts you can find evidence for in the transcript. Do NOT invent facts.

Your response must be a single JSON object and nothing else.
Do not include any text, explanation, or commentary before or after the JSON.
Do not wrap the JSON in markdown code fences.
Your entire response must be valid JSON starting with {{ and ending with }}.

{{
  "subject_confirmed": true,
  "subject_notes": "",
  "facts": [
    {{"fact_type": "birth_year", "fact_value": "1842", "fact_year": 1842, "confidence": "high", "inferred": false}},
    {{"fact_type": "birth_year", "fact_value": "1842", "fact_year": 1842, "confidence": "medium", "inferred": true}},
    {{"fact_type": "birth_place", "fact_value": "Nottingham", "fact_year": null, "confidence": "high", "inferred": false}}
  ]
}}"""


def extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from Gemini's response text.
    Finds the outermost { ... } regardless of any surrounding text,
    preamble sentences, or markdown fences — consistent with ocr_pipeline v3.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text)  # Let json.loads raise a clear error if no match


def _call_gemini_once(prompt: str) -> dict:
    """
    Single attempt to call Gemini for fact extraction.
    Separated from retry logic so tenacity can wrap it cleanly.
    Returns parsed dict, or {"_safety_blocked": True} if content is blocked.
    Uses google-genai SDK (replaces deprecated vertexai.generative_models).
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,
        ),
    )

    # Safety block detection
    if not response.candidates:
        return {"_safety_blocked": True}
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return {
            "_safety_blocked": True,
            "_finish_reason": str(getattr(candidate, "finish_reason", "unknown")),
        }

    raw_text = response.text.strip()

    try:
        return extract_json(raw_text)
    except (json.JSONDecodeError, ValueError):
        # Preserve the raw response rather than silently losing it
        return {
            "subject_confirmed": False,
            "subject_notes": f"Gemini response could not be parsed as JSON — raw text: {raw_text[:500]}",
            "facts": [],
        }


@retry(
    retry=retry_if_exception_type((
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.TooManyRequests,
    )),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(MAX_RETRIES),
    before_sleep=lambda rs: print(
        f"    Rate limited (429) — attempt {rs.attempt_number} failed, "
        f"retrying in {rs.next_action.sleep:.1f}s"
    ),
)
def call_gemini_with_retry(prompt: str) -> dict:
    """
    Call Gemini with tenacity exponential backoff on 429/quota errors.
    Non-429 errors are raised immediately without retry.
    """
    return _call_gemini_once(prompt)


def extract_facts(record):
    """
    Build prompt, call Gemini, parse response into fact rows.
    Returns (list_of_fact_dicts, subject_confirmed, subject_notes).
    """
    prompt = FACT_EXTRACTION_PROMPT.format(
        forename        = record["forename"] or "",
        surname         = record["surname"] or "",
        doc_year        = record["year"] or "",
        doc_type        = record["doc_type_detected"] or "Unknown",
        transcript_text = (record["transcribed_text"] or "")[:4000],
        personal_names  = record["personal_names"] or "[]",
        locations       = record["locations"] or "[]"
    )

    parsed = call_gemini_with_retry(prompt)

    if parsed.get("_safety_blocked"):
        return [], False, f"Safety blocked (finish_reason: {parsed.get('_finish_reason', 'unknown')})"

    rows = []
    for fact in parsed.get("facts", []):
        rows.append({
            "file_id":          record["file_id"],
            "person_gedcom_id": record["person_gedcom_id"],
            "fact_type":        fact.get("fact_type"),
            "fact_value":       fact.get("fact_value"),
            "fact_year":        fact.get("fact_year"),
            "confidence":       fact.get("confidence", "medium"),
            "inferred":         bool(fact.get("inferred", False)),
            "source_doc_type":  record["doc_type_detected"],
            "extracted_at":     datetime.now(timezone.utc),
        })
    return rows, parsed.get("subject_confirmed", True), parsed.get("subject_notes", "")


print("Functions defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Main extraction loop

# COMMAND ----------

all_facts = []
errors    = []
skipped   = []

for i, record in enumerate(records):
    file_name = record["file_name"]
    print(f"[{i+1}/{len(records)}] {file_name}")
    try:
        facts, subject_confirmed, subject_notes = extract_facts(record)
        if not subject_confirmed:
            print(f"  ⚠ Subject uncertain: {subject_notes} — skipping")
            skipped.append({"file_id": record["file_id"], "reason": subject_notes})
            time.sleep(REQUEST_DELAY)
            continue
        all_facts.extend(facts)
        print(f"  ✓ {len(facts)} facts extracted")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        errors.append({"file_id": record["file_id"], "error": str(e)})

    if i < len(records) - 1:
        time.sleep(REQUEST_DELAY)

print(f"\n--- Summary ---")
print(f"Processed:       {len(records) - len(errors) - len(skipped)}")
print(f"Facts extracted: {len(all_facts)}")
print(f"Skipped:         {len(skipped)}")
print(f"Errors:          {len(errors)}")
for e in errors:
    print(f"  ERROR: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Write facts to Delta

# COMMAND ----------

if all_facts:
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
    ])
    facts_df = spark.createDataFrame(all_facts, schema=schema)
    facts_df.write.format("delta").mode("append").saveAsTable("genealogy.gold_transcript_facts")
    print(f"Written {facts_df.count()} fact rows to gold_transcript_facts")
else:
    print("No facts to write.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Preview extracted facts

# COMMAND ----------

spark.sql("""
  SELECT
    sdp.file_name,
    p.given_name,
    p.surname,
    gtf.fact_type,
    gtf.fact_value,
    gtf.confidence,
    gtf.inferred,
    gtf.source_doc_type
  FROM genealogy.gold_transcript_facts gtf
  JOIN genealogy.silver_document_person sdp ON gtf.file_id = sdp.file_id
  JOIN genealogy.gold_person_life p ON gtf.person_gedcom_id = p.person_gedcom_id
  ORDER BY sdp.file_name, gtf.fact_type
  LIMIT 100
""").display()
