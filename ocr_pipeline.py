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
# MAGIC
# MAGIC **Change log:**
# MAGIC - v1: Initial version
# MAGIC - v2: Magic byte detection for extensionless files; PDF page splitting (one row per page);
# MAGIC       LLM doc type detection from controlled vocabulary; safety filter handling logged as
# MAGIC       distinct status; confidence flag now triggers on medium and low (not just low);
# MAGIC       gedcom_id column added ready for downstream person matching

# COMMAND ----------

# MAGIC %md ## 1. Setup & Configuration

# COMMAND ----------

# MAGIC %pip install google-cloud-aiplatform pypdf pdf2image --upgrade
# MAGIC %pip install poppler-utils

# COMMAND ----------

import json
import re
import base64
import traceback
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO

import pypdf
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

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
GCP_PROJECT        = "genealogy-488213"   # Replace with your GCP project ID
GCP_LOCATION       = "global"                # Latest Gemini models require global endpoint
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

# --- Controlled vocabulary for LLM doc type detection ---
DOC_TYPE_VOCABULARY = [
    "Census",
    "BirthCertificate",
    "MarriageCertificate",
    "DeathCertificate",
    "BaptismRegister",
    "BurialRegister",
    "Will",
    "Probate",
    "NewspaperClipping",
    "ElectoralRegister",
    "MilitaryRecord",
    "BMDIndex",
    "Other",
]

print(f"Processing folder : {FOLDER_PATH}")
print(f"Source table      : {SOURCE_TABLE}")
print(f"Transcriptions    : {TRANSCRIPTIONS_TBL}")
print(f"Processing log    : {PROC_LOG_TBL}")

# COMMAND ----------

# MAGIC %md ### Initialise Vertex AI

# COMMAND ----------

# Secret is stored as base64-encoded JSON (encoded before saving to Databricks secret)
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

# MAGIC %md ## 2. Create / Upgrade Delta Tables
# MAGIC
# MAGIC `CREATE TABLE IF NOT EXISTS` is idempotent — safe to rerun.
# MAGIC `ALTER TABLE ... ADD COLUMN` statements add v2 columns to existing tables;
# MAGIC they silently pass if the column already exists.

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TRANSCRIPTIONS_TBL} (
        file_id           STRING    COMMENT 'Google Drive file ID from Fivetran',
        file_name         STRING    COMMENT 'Original filename',
        folder_path       STRING    COMMENT 'Volume path of the source folder',
        surname           STRING    COMMENT 'Parsed from filename',
        forename          STRING    COMMENT 'Parsed from filename',
        year              STRING    COMMENT 'Parsed from filename (may be YYYY or YYYY-MM-DD)',
        doc_type          STRING    COMMENT 'Parsed from filename — raw description token',
        doc_type_detected STRING    COMMENT 'Document type classified by Gemini — controlled vocabulary',
        page_index        INT       COMMENT 'Page number within document (1-based)',
        transcribed_text  STRING    COMMENT 'Full transcription text from Gemini',
        personal_names    STRING    COMMENT 'JSON array of personal names found',
        locations         STRING    COMMENT 'JSON array of place names found',
        uncertain_entries STRING    COMMENT 'JSON array of uncertain readings',
        confidence        STRING    COMMENT 'high / medium / low — from Gemini',
        confidence_notes  STRING    COMMENT 'Gemini explanation of confidence factors',
        confidence_flag   BOOLEAN   COMMENT 'True if confidence is medium or low — needs review',
        gedcom_id         STRING    COMMENT 'Matched person_gedcom_id — populated by downstream matching',
        model_used        STRING    COMMENT 'Vertex AI model used',
        processed_at      TIMESTAMP COMMENT 'When this row was written'
    )
    USING DELTA
    COMMENT 'OCR transcriptions of scanned genealogy documents'
""")

# Add v2 columns to any existing table created by v1
for col_def in [
    ("doc_type_detected", "STRING COMMENT 'Document type classified by Gemini — controlled vocabulary'"),
    ("gedcom_id",         "STRING COMMENT 'Matched person_gedcom_id — populated by downstream matching'"),
]:
    col_name, col_spec = col_def
    try:
        spark.sql(f"ALTER TABLE {TRANSCRIPTIONS_TBL} ADD COLUMN {col_name} {col_spec}")
        print(f"Added column {col_name} to {TRANSCRIPTIONS_TBL}")
    except Exception:
        pass  # Column already exists — expected on reruns

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {PROC_LOG_TBL} (
        file_id        STRING    COMMENT 'Google Drive file ID',
        file_name      STRING    COMMENT 'Original filename',
        folder_path    STRING    COMMENT 'Volume path of the source folder',
        status         STRING    COMMENT 'success / error / safety_blocked / skipped',
        page_count     INT       COMMENT 'Number of pages successfully processed',
        model_used     STRING    COMMENT 'Vertex AI model used',
        processed_at   TIMESTAMP COMMENT 'When processing completed',
        error_message  STRING    COMMENT 'Populated if status is error or safety_blocked'
    )
    USING DELTA
    COMMENT 'Processing log — one row per file, used for idempotency'
""")

print("Tables ready.")

# COMMAND ----------

# MAGIC %md ## 3. File Discovery

# COMMAND ----------

# Magic byte signatures → (mime_type, canonical_extension)
# Used for extensionless files where we cannot rely on the filename
MAGIC_SIGNATURES = [
    (b'\xff\xd8\xff',       "image/jpeg",       ".jpg"),
    (b'\x89PNG\r\n\x1a\n',  "image/png",        ".png"),
    (b'II*\x00',            "image/tiff",       ".tif"),   # TIFF little-endian
    (b'MM\x00*',            "image/tiff",       ".tif"),   # TIFF big-endian
    (b'%PDF',               "application/pdf",  ".pdf"),
]

EXTENSION_MIME_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".tif":  "image/tiff",
    ".tiff": "image/tiff",
    ".pdf":  "application/pdf",
}


def get_mime_type(file_path: str):
    """
    Determine MIME type and canonical extension for a file.
    Uses the file extension if recognised; falls back to magic byte detection.
    Returns (mime_type, ext) or None if the file is unsupported.
    """
    ext = Path(file_path).suffix.lower()
    if ext in EXTENSION_MIME_MAP:
        return EXTENSION_MIME_MAP[ext], ext

    # No recognised extension — inspect the first 8 bytes
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
        for magic, mime, canon_ext in MAGIC_SIGNATURES:
            if header[:len(magic)] == magic:
                return mime, canon_ext
    except Exception:
        pass

    return None


# Load Fivetran documents table for this folder.
# _fivetran_file_path is relative to the volume root, so prepend VOLUME_BASE.
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

# Exclude files already successfully processed (idempotency)
processed_ids = spark.sql(f"""
    SELECT file_id FROM {PROC_LOG_TBL}
    WHERE status = 'success'
""")

# Do NOT filter on extension — get_mime_type() handles extensionless files at runtime
to_process_df = (
    source_df
    .join(processed_ids, on="file_id", how="left_anti")
    .collect()
)

print(f"Candidate files: {len(to_process_df)}")
for row in to_process_df:
    print(f"  {row.file_path}  ({row.size:,} bytes)")

# COMMAND ----------

# MAGIC %md ## 4. Filename Parser

# COMMAND ----------

# Document type keyword → (specialist description, era hint, key fields)
DOC_TYPE_MAP = {
    "census": (
        "census enumerator's return",
        "19th or early 20th century",
        "name, age, sex, occupation, birthplace, relationship to head of household, marital status"
    ),
    "birthcertificate": (
        "birth certificate", "19th or early 20th century",
        "child's name, date and place of birth, father's name and occupation, mother's name and maiden name, informant"
    ),
    "birth": (
        "birth certificate", "19th or early 20th century",
        "child's name, date and place of birth, father's name and occupation, mother's name and maiden name, informant"
    ),
    "marriagecertificate": (
        "marriage certificate", "19th or early 20th century",
        "names of bride and groom, ages, occupations, fathers' names and occupations, witnesses, officiant, date and place"
    ),
    "marriage": (
        "marriage certificate", "19th or early 20th century",
        "names of bride and groom, ages, occupations, fathers' names and occupations, witnesses, officiant, date and place"
    ),
    "deathcertificate": (
        "death certificate", "19th or early 20th century",
        "name of deceased, date and place of death, age, occupation, cause of death, informant"
    ),
    "death": (
        "death certificate", "19th or early 20th century",
        "name of deceased, date and place of death, age, occupation, cause of death, informant"
    ),
    "will": (
        "last will and testament", "19th or early 20th century",
        "testator's name, beneficiaries, executors, bequests, date of signing, witnesses"
    ),
    "probate": (
        "probate record", "19th or early 20th century",
        "deceased's name, estate value, executors, date of probate, court"
    ),
    "newspaper":  ("newspaper clipping", "Victorian or Edwardian era", "all text as printed including headline and captions"),
    "gazette":    ("newspaper or gazette clipping", "Victorian or Edwardian era", "all text as printed"),
    "advertiser": ("newspaper clipping", "Victorian or Edwardian era", "all text as printed"),
    "herald":     ("newspaper clipping", "Victorian or Edwardian era", "all text as printed"),
    "times":      ("newspaper clipping", "Victorian or Edwardian era", "all text as printed"),
    "baptism": (
        "baptism register entry", "18th or 19th century",
        "child's name, date of baptism, parents' names, father's occupation, parish"
    ),
    "bapt":   ("baptism register entry", "18th or 19th century", "child's name, date of baptism, parents' names, parish"),
    "burial": (
        "burial register entry", "18th or 19th century",
        "name of deceased, date of burial, age, abode, officiant, parish"
    ),
    "electoral": (
        "electoral register entry", "19th or early 20th century",
        "name, address, qualification, nature of qualification"
    ),
    "military": (
        "military service record", "World War I or World War II era",
        "name, rank, regiment, service number, date of enlistment, postings, medical notes"
    ),
    "bmd": (
        "civil registration BMD index entry", "19th or early 20th century",
        "name, registration district, volume, page number, year and quarter of registration"
    ),
    "gro": (
        "civil registration BMD index entry", "19th or early 20th century",
        "name, registration district, volume, page number, year and quarter of registration"
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

    result = {"surname": None, "forename": None, "year": None, "doc_type": None, "parse_ok": False}

    if len(parts) < 3:
        result["doc_type"] = stem
        return result

    result["surname"]  = parts[0]
    result["forename"] = parts[1]

    year_token = parts[2]
    if re.match(r'^\d{4}(-\d{2}(-\d{2})?)?$', year_token):
        result["year"]     = year_token
        result["doc_type"] = "_".join(parts[3:]) if len(parts) > 3 else "Unknown"
    else:
        result["doc_type"] = "_".join(parts[2:])

    result["parse_ok"] = bool(result["surname"] and result["forename"] and result["year"])
    return result


def get_doc_type_context(doc_type: str) -> tuple:
    """Map a raw doc_type token to (description, era, fields). Falls back to generic."""
    if not doc_type:
        return ("historical document", "19th or early 20th century", "all visible text and fields")
    key = re.sub(r'[^a-z]', '', doc_type.lower())
    for pattern, context in DOC_TYPE_MAP.items():
        if pattern in key:
            return context
    return ("historical document", "19th or early 20th century", "all visible text and fields")

# COMMAND ----------

# MAGIC %md ## 5. Prompt Builder

# COMMAND ----------

def build_prompt(metadata: dict) -> str:
    """
    Build a Gemini transcription prompt from parsed filename metadata.
    Instructs Gemini to classify document type from the controlled vocabulary
    and return structured JSON.
    """
    doc_desc, era, fields = get_doc_type_context(metadata.get("doc_type"))

    subject_hint = ""
    if metadata.get("forename") and metadata.get("surname"):
        subject_hint = f"The document relates to **{metadata['forename']} {metadata['surname']}**"
        if metadata.get("year"):
            subject_hint += f", circa **{metadata['year']}**"
        subject_hint += "."

    vocab_list = "\n".join(f"  - {v}" for v in DOC_TYPE_VOCABULARY)

    return f"""You are an expert historical document transcriptionist specialising in {era} {doc_desc}s and the handwriting styles of that period.

{subject_hint}

## Task

Transcribe the document exactly as written, then return your output as a single JSON block.

## Transcription Rules

- Preserve original spelling exactly, including phonetic or archaic spellings (e.g. "labrer", "Willm", "Thos")
- Mark any word you cannot read with confidence as [?]
- For partially legible words, include your best reading followed by [?] (e.g. "Thos[?]")
- Do not modernise, correct, or standardise any spelling or punctuation
- Preserve abbreviations as written
- Transcribe all fields visible in the document, paying particular attention to: {fields}
- For tabular or multi-column documents, preserve the row and column structure in your transcription

## JSON Output Structure

Return a single JSON block with no additional commentary before or after it, using exactly these fields:

- `transcription`: full text content of the document, preserving layout where applicable
- `doc_type_detected`: classify this document using exactly one value from this list:
{vocab_list}
- `confidence`: one of "high", "medium", or "low" reflecting your overall transcription confidence
- `confidence_notes`: brief explanation of any factors affecting confidence (ink fading, damage, ambiguous letterforms). If no issues, write "No significant issues."
- `personal_names`: JSON array of all personal names found, spelled exactly as written
- `locations`: JSON array of all place names found, spelled exactly as written
- `uncertain_entries`: JSON array of objects, each with "entry" (best reading), "field" (which column or field), and "reason" (why uncertain)

Return only the JSON block."""

# COMMAND ----------

# MAGIC %md ## 6. OCR Functions

# COMMAND ----------

def split_pdf_to_page_images(file_path: str) -> list[bytes]:
    """
    Split a PDF into a list of JPEG image bytes, one per page, using pdf2image.
    Falls back to sending the whole PDF as a single item if pdf2image is unavailable
    or conversion fails (e.g. encrypted PDF).
    """
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(file_path, dpi=200, fmt="jpeg")
        result = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            result.append(buf.getvalue())
        print(f"    PDF split into {len(result)} page image(s)")
        return result, "image/jpeg"
    except ImportError:
        print("    pdf2image not available — sending PDF as single document")
    except Exception as e:
        print(f"    PDF split failed ({e}) — sending as single document")

    with open(file_path, "rb") as f:
        return [f.read()], "application/pdf"


def call_gemini(file_bytes: bytes, mime_type: str, prompt: str) -> dict:
    """
    Send file bytes to Gemini and return a parsed dict.
    Returns {"_safety_blocked": True} when the response is blocked by safety filters,
    rather than raising an exception, so the caller can log it distinctly.
    """
    file_part = Part.from_data(data=file_bytes, mime_type=mime_type)

    response = model.generate_content(
        [file_part, prompt],
        generation_config={"temperature": 0.1, "max_output_tokens": 8192},
    )

    # Detect safety block — candidates empty or content has no parts
    if not response.candidates:
        return {"_safety_blocked": True}
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return {"_safety_blocked": True, "_finish_reason": str(getattr(candidate, "finish_reason", "unknown"))}

    raw_text = response.text.strip()

    # Strip markdown code fences if present
    raw_text = re.sub(r'^```json\s*', '', raw_text, flags=re.MULTILINE)
    raw_text = re.sub(r'```\s*$',     '', raw_text, flags=re.MULTILINE)
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "transcription":    raw_text,
            "doc_type_detected": "Other",
            "confidence":        "low",
            "confidence_notes":  "Gemini response could not be parsed as JSON — raw text preserved.",
            "personal_names":    [],
            "locations":         [],
            "uncertain_entries": [],
        }


def ocr_file(file_path: str, metadata: dict) -> tuple:
    """
    Run OCR on a single file.
    Returns (list_of_row_dicts, status_string).
    - Images produce one row.
    - PDFs are split into pages — one row per page.
    - Safety blocks return ([], "safety_blocked").
    """
    type_info = get_mime_type(file_path)
    if type_info is None:
        raise ValueError(f"Unsupported file type: {file_path}")

    mime_type, _ = type_info
    prompt = build_prompt(metadata)
    now = datetime.now(timezone.utc)

    if mime_type == "application/pdf":
        pages_bytes, page_mime = split_pdf_to_page_images(file_path)
    else:
        with open(file_path, "rb") as f:
            pages_bytes = [f.read()]
        page_mime = mime_type

    rows = []
    for page_index, page_bytes in enumerate(pages_bytes, 1):
        result = call_gemini(page_bytes, page_mime, prompt)

        if result.get("_safety_blocked"):
            return [], "safety_blocked"

        rows.append({
            "file_id":           metadata["file_id"],
            "file_name":         metadata["file_name"],
            "folder_path":       FOLDER_PATH,
            "surname":           metadata.get("surname"),
            "forename":          metadata.get("forename"),
            "year":              metadata.get("year"),
            "doc_type":          metadata.get("doc_type"),
            "doc_type_detected": result.get("doc_type_detected", "Other"),
            "page_index":        page_index,
            "transcribed_text":  result.get("transcription", ""),
            "personal_names":    json.dumps(result.get("personal_names", [])),
            "locations":         json.dumps(result.get("locations", [])),
            "uncertain_entries": json.dumps(result.get("uncertain_entries", [])),
            "confidence":        result.get("confidence", "low"),
            "confidence_notes":  result.get("confidence_notes", ""),
            # Flag medium and low confidence — not just low
            "confidence_flag":   result.get("confidence", "low") != "high",
            "gedcom_id":         None,   # Populated by downstream person-matching process
            "model_used":        VERTEX_MODEL,
            "processed_at":      now,
        })

    return rows, "success"

# COMMAND ----------

# MAGIC %md ## 7. Processing Loop

# COMMAND ----------

transcription_rows = []
log_rows = []

total         = len(to_process_df)
success_count = 0
error_count   = 0
blocked_count = 0
skipped_count = 0

for i, file_row in enumerate(to_process_df, 1):
    file_path = file_row.file_path
    file_name = Path(file_path).name
    file_id   = file_row.file_id
    now       = datetime.now(timezone.utc)

    print(f"[{i}/{total}] {file_name}")

    # Check file type early — avoids unnecessary Gemini calls for genuinely unsupported files
    type_info = get_mime_type(file_path)
    if type_info is None:
        print(f"  ⚠ Skipped — unrecognised file type")
        log_rows.append({
            "file_id": file_id, "file_name": file_name, "folder_path": FOLDER_PATH,
            "status": "skipped", "page_count": 0, "model_used": VERTEX_MODEL,
            "processed_at": now, "error_message": "Unrecognised file type — no extension and no matching magic bytes",
        })
        skipped_count += 1
        continue

    metadata = parse_filename(file_name)
    metadata["file_id"]   = file_id
    metadata["file_name"] = file_name

    if not metadata["parse_ok"]:
        print(f"  ⚠ Filename did not parse cleanly — using generic prompt")

    try:
        rows, status = ocr_file(file_path, metadata)

        if status == "safety_blocked":
            print(f"  ✗ Blocked by Gemini safety filters")
            log_rows.append({
                "file_id": file_id, "file_name": file_name, "folder_path": FOLDER_PATH,
                "status": "safety_blocked", "page_count": 0, "model_used": VERTEX_MODEL,
                "processed_at": now,
                "error_message": "Response blocked by Gemini safety filters. Try converting to image and reprocessing.",
            })
            blocked_count += 1
            continue

        transcription_rows.extend(rows)
        log_rows.append({
            "file_id": file_id, "file_name": file_name, "folder_path": FOLDER_PATH,
            "status": "success", "page_count": len(rows), "model_used": VERTEX_MODEL,
            "processed_at": now, "error_message": None,
        })

        flagged    = any(r["confidence_flag"] for r in rows)
        confidence = rows[0]["confidence"] if rows else "unknown"
        print(f"  ✓ Done  pages={len(rows)}  confidence={confidence}"
              + ("  ⚑ review flagged" if flagged else ""))
        success_count += 1

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"  ✗ Error: {error_msg[:200]}")
        log_rows.append({
            "file_id": file_id, "file_name": file_name, "folder_path": FOLDER_PATH,
            "status": "error", "page_count": 0, "model_used": VERTEX_MODEL,
            "processed_at": now, "error_message": error_msg[:2000],
        })
        error_count += 1

print(f"\n{'='*50}")
print(f"Run complete: {success_count} succeeded, {error_count} errors, "
      f"{blocked_count} safety blocked, {skipped_count} skipped  ({total} candidates)")

# COMMAND ----------

# MAGIC %md ## 8. Write to Delta Tables

# COMMAND ----------

# DBTITLE 1,Write to Delta Tables
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, TimestampType

transcription_schema = StructType([
    StructField("file_id", StringType(), True),
    StructField("file_name", StringType(), True),
    StructField("folder_path", StringType(), True),
    StructField("surname", StringType(), True),
    StructField("forename", StringType(), True),
    StructField("year", StringType(), True),
    StructField("doc_type", StringType(), True),
    StructField("doc_type_detected", StringType(), True),
    StructField("page_index", IntegerType(), True),
    StructField("transcribed_text", StringType(), True),
    StructField("personal_names", StringType(), True),
    StructField("locations", StringType(), True),
    StructField("uncertain_entries", StringType(), True),
    StructField("confidence", StringType(), True),
    StructField("confidence_notes", StringType(), True),
    StructField("confidence_flag", BooleanType(), True),
    StructField("gedcom_id", StringType(), True),
    StructField("model_used", StringType(), True),
    StructField("processed_at", TimestampType(), True),
])

log_schema = StructType([
    StructField("file_id", StringType(), True),
    StructField("file_name", StringType(), True),
    StructField("folder_path", StringType(), True),
    StructField("status", StringType(), True),
    StructField("page_count", IntegerType(), True),
    StructField("model_used", StringType(), True),
    StructField("processed_at", TimestampType(), True),
    StructField("error_message", StringType(), True),
])

if transcription_rows:
    transcription_df = spark.createDataFrame(transcription_rows, schema=transcription_schema)
    transcription_df.write.format("delta").mode("append").saveAsTable(TRANSCRIPTIONS_TBL)
    print(f"Written {len(transcription_rows)} transcription rows to {TRANSCRIPTIONS_TBL}")
else:
    print("No transcription rows to write.")

if log_rows:
    log_df = spark.createDataFrame(log_rows, schema=log_schema)
    log_df.write.format("delta").mode("append").saveAsTable(PROC_LOG_TBL)
    print(f"Written {len(log_rows)} log rows to {PROC_LOG_TBL}")
else:
    print("No log rows to write.")

# COMMAND ----------

# MAGIC %md ## 9. Run Summary

# COMMAND ----------

from collections import Counter

print(f"\n{'='*50}")
print(f"PIPELINE SUMMARY — {FOLDER_NAME}")
print(f"{'='*50}")
print(f"Total candidates  : {total}")
print(f"Succeeded         : {success_count}")
print(f"Errors            : {error_count}")
print(f"Safety blocked    : {blocked_count}")
print(f"Skipped           : {skipped_count}")
print()

if transcription_rows:
    confidence_counts = Counter(r["confidence"] for r in transcription_rows)
    print("Confidence breakdown (rows):")
    for level in ["high", "medium", "low"]:
        print(f"  {level:8s}: {confidence_counts.get(level, 0)}")
    print()

    doc_type_counts = Counter(r["doc_type_detected"] for r in transcription_rows)
    print("Document types detected by Gemini:")
    for dt, count in doc_type_counts.most_common():
        print(f"  {dt:30s}: {count}")
    print()

needs_review = [r for r in transcription_rows if r["confidence_flag"]]
if needs_review:
    print(f"Rows flagged for review ({len(needs_review)}):")
    for r in needs_review:
        print(f"  {r['file_name']} p{r['page_index']}  [{r['confidence']}]  {r['confidence_notes'][:100]}")
else:
    print("No rows flagged for review.")

if blocked_count:
    blocked_files = [r["file_name"] for r in log_rows if r["status"] == "safety_blocked"]
    print(f"\nSafety blocked ({blocked_count}):")
    for f in blocked_files:
        print(f"  {f}")
    print("  Tip: convert PDF to JPEG pages and reprocess")

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
# MAGIC     doc_type_detected,
# MAGIC     page_index,
# MAGIC     confidence,
# MAGIC     confidence_flag,
# MAGIC     length(transcribed_text) AS text_length,
# MAGIC     processed_at
# MAGIC FROM workspace.genealogy.ocr_transcriptions
# MAGIC WHERE folder_path LIKE '%Family_CUTHBERTSON%'
# MAGIC ORDER BY file_name, page_index
# MAGIC LIMIT 50
