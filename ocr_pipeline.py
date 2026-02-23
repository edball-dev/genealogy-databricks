# Databricks notebook source
# MAGIC %md
# MAGIC # Genealogy OCR Pipeline
# MAGIC
# MAGIC Transcribes scanned genealogy documents from Google Drive using Vertex AI Gemini.
# MAGIC
# MAGIC **Parameters:**
# MAGIC - `folder_name`: The surname folder to process (e.g. `Family_CUTHBERTSON`)
# MAGIC
# MAGIC **Source table:** `workspace.staging_google_drive.documents`
# MAGIC **Output tables:** `workspace.genealogy.ocr_transcriptions`, `workspace.genealogy.ocr_processing_log`

# COMMAND ----------

# MAGIC %md ## 1. Setup & Configuration

# COMMAND ----------

# MAGIC %pip install google-cloud-aiplatform --upgrade

# COMMAND ----------

import json
import re
import base64
import traceback
from datetime import datetime, timezone
from pathlib import Path

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image
from google.oauth2 import service_account

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    BooleanType, TimestampType, LongType
)
from delta.tables import DeltaTable

# COMMAND ----------

# MAGIC %md ### Parameters
# MAGIC
# MAGIC When running as a Databricks Job, set `folder_name` as a job parameter.
# MAGIC For interactive use, set it in the widget below.

# COMMAND ----------

dbutils.widgets.text("folder_name", "Family_CUTHBERTSON", "Surname Folder Name")
FOLDER_NAME = dbutils.widgets.get("folder_name")

# COMMAND ----------

# --- Core configuration ---
GCP_PROJECT        = "genealogy-488213"
GCP_LOCATION       = "global"
VERTEX_MODEL       = "gemini-3-pro-preview"
SECRET_SCOPE       = "genealogy"
SECRET_KEY         = "gcp_service_account_json"

# --- Unity Catalog ---
CATALOG            = "workspace"
STAGING_SCHEMA     = "staging_google_drive"
OUTPUT_SCHEMA      = "genealogy"
SOURCE_TABLE       = f"{CATALOG}.{STAGING_SCHEMA}.documents"
TRANSCRIPTIONS_TBL = f"{CATALOG}.{OUTPUT_SCHEMA}.ocr_transcriptions"
PROC_LOG_TBL       = f"{CATALOG}.{OUTPUT_SCHEMA}.ocr_processing_log"

# --- Volume path ---
VOLUME_BASE        = f"/Volumes/{CATALOG}/{STAGING_SCHEMA}/documents"
FOLDER_PATH        = f"{VOLUME_BASE}/{FOLDER_NAME}"

print(f"Processing folder : {FOLDER_PATH}")
print(f"Source table      : {SOURCE_TABLE}")
print(f"Transcriptions    : {TRANSCRIPTIONS_TBL}")
print(f"Processing log    : {PROC_LOG_TBL}")

# COMMAND ----------

# MAGIC %md ### Initialise Vertex AI

# COMMAND ----------


sa_json_b64 = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY)
sa_json = base64.b64decode(sa_json_b64).decode("utf-8")
sa_info = json.loads(sa_json)

credentials = service_account.Credentials.from_service_account_info(
    sa_info,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION, credentials=credentials)
model = GenerativeModel(VERTEX_MODEL)

print(f"Vertex AI initialised. Model: {VERTEX_MODEL}")

# COMMAND ----------

# MAGIC %md ## 2. Create Delta Tables

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TRANSCRIPTIONS_TBL} (
        file_id          STRING        COMMENT 'Google Drive file ID from Fivetran',
        file_name        STRING        COMMENT 'Original filename',
        folder_path      STRING        COMMENT 'Volume path of the source folder',
        surname          STRING        COMMENT 'Parsed from filename',
        forename         STRING        COMMENT 'Parsed from filename',
        year             STRING        COMMENT 'Parsed from filename (may be YYYY or YYYY-MM-DD)',
        doc_type         STRING        COMMENT 'Parsed from filename — document description',
        page_index       INT           COMMENT 'Page number within document (1-based)',
        transcribed_text STRING        COMMENT 'Full transcription text from Gemini',
        personal_names   STRING        COMMENT 'JSON array of personal names found',
        locations        STRING        COMMENT 'JSON array of place names found',
        uncertain_entries STRING       COMMENT 'JSON array of uncertain readings',
        confidence       STRING        COMMENT 'high / medium / low — from Gemini',
        confidence_notes STRING        COMMENT 'Gemini explanation of confidence factors',
        confidence_flag  BOOLEAN       COMMENT 'True if confidence is low — needs manual review',
        model_used       STRING        COMMENT 'Vertex AI model used',
        processed_at     TIMESTAMP     COMMENT 'When this row was written'
    )
    USING DELTA
    COMMENT 'OCR transcriptions of scanned genealogy documents'
""")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {PROC_LOG_TBL} (
        file_id       STRING    COMMENT 'Google Drive file ID',
        file_name     STRING    COMMENT 'Original filename',
        folder_path   STRING    COMMENT 'Volume path of the source folder',
        status        STRING    COMMENT 'success / error / skipped',
        page_count    INT       COMMENT 'Number of pages processed',
        model_used    STRING    COMMENT 'Vertex AI model used',
        processed_at  TIMESTAMP COMMENT 'When processing completed',
        error_message STRING    COMMENT 'Populated if status = error'
    )
    USING DELTA
    COMMENT 'Processing log — one row per file, used for idempotency'
""")

print("Tables ready.")

# COMMAND ----------

# MAGIC %md ## 3. File Discovery

# COMMAND ----------

# DBTITLE 1,File Discovery
# Supported file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"}

# Load Fivetran documents table filtered to this folder
# _fivetran_file_path contains the volume path including filename
source_df = spark.sql(f"""
    SELECT
        file_id,
        '{VOLUME_BASE}/' || _fivetran_file_path AS file_path,
        size,
        modified_at
    FROM {SOURCE_TABLE}
    WHERE _fivetran_deleted = false
      AND _fivetran_file_path LIKE '{FOLDER_NAME}/%'
""")

# Load already successfully processed file IDs
processed_ids = spark.sql(f"""
    SELECT file_id
    FROM {PROC_LOG_TBL}
    WHERE status = 'success'
""")

# Anti-join to find unprocessed files
to_process_df = (
    source_df
    .join(processed_ids, on="file_id", how="left_anti")
    .filter(
        F.lower(F.regexp_extract(F.col("file_path"), r'(\.[^.]+)$', 1))
         .isin(list(SUPPORTED_EXTENSIONS))
    )
    .collect()
)

print(f"Files to process: {len(to_process_df)}")
for row in to_process_df:
    print(f"  {row.file_path}  ({row.size:,} bytes)")

# COMMAND ----------

# MAGIC %md ## 4. Filename Parser

# COMMAND ----------

# Document type → (specialist description, era hint, field guidance)
DOC_TYPE_MAP = {
    "census": (
        "census enumerator's return",
        "19th or early 20th century",
        "name, age, sex, occupation, birthplace, relationship to head of household, marital status"
    ),
    "birthcertificate": (
        "birth certificate",
        "19th or early 20th century",
        "child's name, date and place of birth, father's name and occupation, mother's name and maiden name, informant"
    ),
    "birth": (
        "birth certificate",
        "19th or early 20th century",
        "child's name, date and place of birth, father's name and occupation, mother's name and maiden name, informant"
    ),
    "marriagecertificate": (
        "marriage certificate",
        "19th or early 20th century",
        "names of bride and groom, ages, occupations, fathers' names and occupations, witnesses, officiant, date and place of marriage"
    ),
    "marriage": (
        "marriage certificate",
        "19th or early 20th century",
        "names of bride and groom, ages, occupations, fathers' names and occupations, witnesses, officiant, date and place of marriage"
    ),
    "deathcertificate": (
        "death certificate",
        "19th or early 20th century",
        "name of deceased, date and place of death, age, occupation, cause of death, informant"
    ),
    "death": (
        "death certificate",
        "19th or early 20th century",
        "name of deceased, date and place of death, age, occupation, cause of death, informant"
    ),
    "will": (
        "last will and testament",
        "19th or early 20th century",
        "testator's name, beneficiaries, executors, bequests, date of signing, witnesses"
    ),
    "probate": (
        "probate record",
        "19th or early 20th century",
        "deceased's name, estate value, executors, date of probate, court"
    ),
    "newspaper": (
        "newspaper clipping",
        "Victorian or Edwardian era",
        "all text as printed, including headline, article body, and any captions"
    ),
    "gazette": (
        "newspaper or gazette clipping",
        "Victorian or Edwardian era",
        "all text as printed, including headline, article body, and any captions"
    ),
    "baptism": (
        "baptism register entry",
        "18th or 19th century",
        "child's name, date of baptism, parents' names, father's occupation, parish"
    ),
    "burial": (
        "burial register entry",
        "18th or 19th century",
        "name of deceased, date of burial, age, abode, officiant, parish"
    ),
    "electoral": (
        "electoral register entry",
        "19th or early 20th century",
        "name, address, qualification, nature of qualification"
    ),
    "military": (
        "military service record",
        "World War I or World War II era",
        "name, rank, regiment, service number, date of enlistment, postings, medical notes"
    ),
}

def parse_filename(filename: str) -> dict:
    """
    Parse a genealogy document filename into structured metadata.
    Convention: SURNAME_Firstname_Year_Description.ext
    Year may be YYYY or YYYY-MM-DD.
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    result = {
        "surname": None,
        "forename": None,
        "year": None,
        "doc_type": None,
        "parse_ok": False
    }

    if len(parts) < 3:
        result["doc_type"] = stem
        return result

    result["surname"] = parts[0]
    result["forename"] = parts[1]

    # Year: third token, may be YYYY or YYYY-MM-DD
    year_token = parts[2]
    if re.match(r'^\d{4}(-\d{2}-\d{2})?$', year_token):
        result["year"] = year_token
        result["doc_type"] = "_".join(parts[3:]) if len(parts) > 3 else "Unknown"
    else:
        # Year not found in expected position
        result["doc_type"] = "_".join(parts[2:])

    result["parse_ok"] = bool(result["surname"] and result["forename"] and result["year"])
    return result


def get_doc_type_context(doc_type: str) -> tuple:
    """
    Map a doc_type string to specialist description, era, and field guidance.
    Falls back to generic if not recognised.
    """
    if not doc_type:
        return (
            "historical document",
            "19th or early 20th century",
            "all visible text and fields"
        )

    key = re.sub(r'[^a-z]', '', doc_type.lower())
    for pattern, context in DOC_TYPE_MAP.items():
        if pattern in key:
            return context

    return (
        "historical document",
        "19th or early 20th century",
        "all visible text and fields"
    )


# COMMAND ----------

# MAGIC %md ## 5. Prompt Builder

# COMMAND ----------

def build_prompt(metadata: dict) -> str:
    """
    Build a dynamic Gemini prompt based on parsed filename metadata.
    """
    doc_desc, era, fields = get_doc_type_context(metadata.get("doc_type"))

    subject_hint = ""
    if metadata.get("forename") and metadata.get("surname"):
        subject_hint = (
            f"The document relates to **{metadata['forename']} {metadata['surname']}**"
        )
        if metadata.get("year"):
            subject_hint += f", circa **{metadata['year']}**"
        subject_hint += "."

    prompt = f"""You are an expert historical document transcriptionist specialising in {era} {doc_desc}s and the handwriting styles of that period. I have uploaded an image of a handwritten or printed {doc_desc}.

{subject_hint}

## Task

Transcribe the document exactly as written, then return your output as a single JSON block.

## Transcription Rules

- **Preserve original spelling exactly**, including phonetic spellings of names and occupations (e.g., "labrer," "smithe," "Willm")
- Mark any word you cannot read with confidence as `[?]`
- For partially legible words, include your best reading followed by `[?]` (e.g., `"Thos[?]"`)
- Do not modernise, correct, or standardise any spelling
- Preserve abbreviations as written
- Transcribe all fields visible in the document, including: {fields}
- If this is a multi-column or tabular document, preserve the row and column structure in your transcription

## JSON Output Structure

Return a single JSON block with no additional commentary, using exactly these fields:

- `transcription`: the full text content of the document, preserving row/column structure where applicable
- `confidence`: a value of `"high"`, `"medium"`, or `"low"` reflecting your overall confidence in the transcription
- `confidence_notes`: a brief explanation of any factors affecting confidence (ink fading, ambiguous letterforms, damage, torn sections, etc.)
- `personal_names`: an array of all personal names found, spelled exactly as written
- `locations`: an array of all place names and locations found, spelled exactly as written
- `uncertain_entries`: an array of objects, each with `"entry"` (your best reading), `"field"` (which field or column), and `"reason"` (why you are uncertain)

Return only the JSON block. Do not include any text before or after it."""

    return prompt


# COMMAND ----------

# MAGIC %md ## 6. Gemini OCR Function

# COMMAND ----------

def read_file_as_base64(file_path: str) -> tuple[str, str]:
    """
    Read a file from the Databricks volume and return (base64_string, mime_type).
    """
    ext = Path(file_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".pdf": "application/pdf",
    }
    mime_type = mime_map.get(ext, "application/octet-stream")

    with open(file_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, mime_type


def call_gemini(file_path: str, prompt: str) -> dict:
    """
    Send a file to Gemini for transcription.
    Returns parsed JSON response dict.
    PDFs are handled natively by Gemini (no page splitting needed on our side).
    """
    b64_data, mime_type = read_file_as_base64(file_path)

    file_part = Part.from_data(data=base64.b64decode(b64_data), mime_type=mime_type)

    response = model.generate_content(
        [file_part, prompt],
        generation_config={
            "temperature": 0.1,   # Low temperature for deterministic transcription
            "max_output_tokens": 8192,
        }
    )

    raw_text = response.text.strip()

    # Strip markdown code fences if present
    raw_text = re.sub(r'^```json\s*', '', raw_text)
    raw_text = re.sub(r'\s*```$', '', raw_text)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Return a structured error response rather than raising
        parsed = {
            "transcription": raw_text,
            "confidence": "low",
            "confidence_notes": "Gemini response could not be parsed as JSON — raw text preserved in transcription field.",
            "personal_names": [],
            "locations": [],
            "uncertain_entries": []
        }

    return parsed


def ocr_file(file_path: str, metadata: dict) -> list[dict]:
    """
    Run OCR on a single file. Returns a list of row dicts (one per page).
    For images, always returns a single-element list.
    For PDFs, Gemini handles pagination internally — we treat as single page
    unless the response indicates multiple pages.
    """
    prompt = build_prompt(metadata)
    result = call_gemini(file_path, prompt)

    now = datetime.now(timezone.utc)

    row = {
        "file_id":           metadata["file_id"],
        "file_name":         metadata["file_name"],
        "folder_path":       FOLDER_PATH,
        "surname":           metadata.get("surname"),
        "forename":          metadata.get("forename"),
        "year":              metadata.get("year"),
        "doc_type":          metadata.get("doc_type"),
        "page_index":        1,
        "transcribed_text":  result.get("transcription", ""),
        "personal_names":    json.dumps(result.get("personal_names", [])),
        "locations":         json.dumps(result.get("locations", [])),
        "uncertain_entries": json.dumps(result.get("uncertain_entries", [])),
        "confidence":        result.get("confidence", "low"),
        "confidence_notes":  result.get("confidence_notes", ""),
        "confidence_flag":   result.get("confidence", "low") == "low",
        "model_used":        VERTEX_MODEL,
        "processed_at":      now,
    }

    return [row]


# COMMAND ----------

# MAGIC %md ## 7. Processing Loop

# COMMAND ----------

transcription_rows = []
log_rows = []

total = len(to_process_df)
success_count = 0
error_count = 0

for i, file_row in enumerate(to_process_df, 1):
    file_path = file_row.file_path
    file_name = Path(file_path).name
    file_id   = file_row.file_id
    now       = datetime.now(timezone.utc)

    print(f"[{i}/{total}] {file_name}")

    # Parse filename
    metadata = parse_filename(file_name)
    metadata["file_id"]   = file_id
    metadata["file_name"] = file_name

    if not metadata["parse_ok"]:
        print(f"  ⚠ Filename did not parse cleanly — will use generic prompt")

    try:
        rows = ocr_file(file_path, metadata)
        transcription_rows.extend(rows)

        log_rows.append({
            "file_id":       file_id,
            "file_name":     file_name,
            "folder_path":   FOLDER_PATH,
            "status":        "success",
            "page_count":    len(rows),
            "model_used":    VERTEX_MODEL,
            "processed_at":  now,
            "error_message": None,
        })

        confidence = rows[0]["confidence"] if rows else "unknown"
        print(f"  ✓ Done  confidence={confidence}  pages={len(rows)}")
        success_count += 1

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"  ✗ Error: {error_msg[:200]}")

        log_rows.append({
            "file_id":       file_id,
            "file_name":     file_name,
            "folder_path":   FOLDER_PATH,
            "status":        "error",
            "page_count":    0,
            "model_used":    VERTEX_MODEL,
            "processed_at":  now,
            "error_message": error_msg[:2000],
        })
        error_count += 1

print(f"\n{'='*50}")
print(f"Run complete: {success_count} succeeded, {error_count} failed out of {total} files.")

# COMMAND ----------

# MAGIC %md ## 8. Write to Delta Tables

# COMMAND ----------

if transcription_rows:
    transcription_df = spark.createDataFrame(transcription_rows)
    transcription_df = transcription_df.withColumn('page_index', F.col('page_index').cast(IntegerType()))
    transcription_df.write.format("delta").mode("append").saveAsTable(TRANSCRIPTIONS_TBL)
    print(f"Written {len(transcription_rows)} transcription rows to {TRANSCRIPTIONS_TBL}")
else:
    print("No transcription rows to write.")

if log_rows:
    log_df = spark.createDataFrame(log_rows)
    log_df = log_df.withColumn('page_count', F.col('page_count').cast(IntegerType()))
    log_df.write.format("delta").mode("append").saveAsTable(PROC_LOG_TBL)
    print(f"Written {len(log_rows)} log rows to {PROC_LOG_TBL}")
else:
    print("No log rows to write.")

# COMMAND ----------

# MAGIC %md ## 9. Run Summary

# COMMAND ----------

print(f"\n{'='*50}")
print(f"PIPELINE SUMMARY — {FOLDER_NAME}")
print(f"{'='*50}")
print(f"Files processed : {total}")
print(f"Succeeded       : {success_count}")
print(f"Errors          : {error_count}")
print()

# Show confidence breakdown for this run
if transcription_rows:
    from collections import Counter
    confidence_counts = Counter(r["confidence"] for r in transcription_rows)
    print("Confidence breakdown:")
    for level in ["high", "medium", "low"]:
        print(f"  {level:8s}: {confidence_counts.get(level, 0)}")
    print()

# Show files needing review
needs_review = [r for r in transcription_rows if r["confidence_flag"]]
if needs_review:
    print(f"Files needing manual review ({len(needs_review)}):")
    for r in needs_review:
        print(f"  {r['file_name']}  — {r['confidence_notes'][:100]}")
else:
    print("No files flagged for manual review.")

# Show any errors
errors = [r for r in log_rows if r["status"] == "error"]
if errors:
    print(f"\nErrors ({len(errors)}):")
    for r in errors:
        print(f"  {r['file_name']}: {r['error_message'][:150]}")

# COMMAND ----------

# MAGIC %md ## 10. Quick Validation Query

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     file_name,
# MAGIC     surname,
# MAGIC     forename,
# MAGIC     year,
# MAGIC     doc_type,
# MAGIC     confidence,
# MAGIC     confidence_flag,
# MAGIC     length(transcribed_text) AS text_length,
# MAGIC     processed_at
# MAGIC FROM workspace.genealogy.ocr_transcriptions
# MAGIC WHERE folder_path LIKE '%Family_CUTHBERTSON%'
# MAGIC ORDER BY processed_at DESC
# MAGIC LIMIT 50
