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
# MAGIC - v3: GIF support added; robust JSON extraction (outermost { } rather than fence
# MAGIC       stripping); improved prompt — explicit JSON-only instruction, negative examples,
# MAGIC       output skeleton at end; PoorLawRecord added to controlled vocabulary; explicit
# MAGIC       Spark schema on write (eliminates cast hacks)
# MAGIC - v3.1: Replace manual retry loop with tenacity wait_random_exponential(multiplier=1,
# MAGIC         max=60) per Google's recommended approach for Vertex AI 429 errors; retry only
# MAGIC         on 429/resource-exhausted, raise immediately on all other errors; up to 5 attempts
# MAGIC - v3.2: Increase max_output_tokens to 65536 (fixes truncated JSON on long documents);
# MAGIC         add INTER_REQUEST_DELAY between files to prevent RPM exhaustion;
# MAGIC         convert GIFs to JPEG via PyMuPDF before sending (fixes Gemini 400 BadRequest);
# MAGIC         use retry_if_exception_type with SDK classes instead of string inspection;
# MAGIC         write each file's results to Delta immediately (crash/cancel safe — no work lost)
# MAGIC - v3.3: Migrate from deprecated vertexai.generative_models SDK to google-genai SDK.
# MAGIC         vertexai.generative_models is deprecated as of June 2025, removed June 2026.
# MAGIC         New pattern: google-genai client (client.models.generate_content) with
# MAGIC         types.GenerateContentConfig and types.Part.from_bytes for image data.
# MAGIC - v3.4: Log Gemini token usage per file to genealogy.ocr_token_usage Delta table.
# MAGIC         _call_gemini_once now returns (result_dict, usage_dict) tuple.
# MAGIC         call_gemini_with_retry and ocr_file updated to propagate usage.
# MAGIC         _write_token_usage helper writes one row per file (non-fatal on failure).

# COMMAND ----------

# MAGIC %md ## 1. Setup & Configuration

# COMMAND ----------

# MAGIC %pip install "google-genai==1.64.0" "pydantic<2.12" pymupdf tenacity

# COMMAND ----------

import json
import re
import time

from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import google.api_core.exceptions
import base64
import traceback
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO

from google import genai
from google.genai import types
from google.oauth2 import service_account

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    BooleanType, TimestampType
)

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
GCP_LOCATION       = "global"           # Latest Gemini models require global endpoint
VERTEX_MODEL       = "gemini-3-pro-preview"
SECRET_SCOPE       = "genealogy"
SECRET_KEY         = "gcp_service_account_json"

# --- Retry / rate limit ---
# Uses tenacity: random exponential backoff, waits up to 60s per attempt, max 5 attempts
# Follows Google's recommended approach for Vertex AI 429 errors:
# https://cloud.google.com/blog/products/ai-machine-learning/learn-how-to-handle-429-resource-exhaustion-errors-in-your-llms
INTER_REQUEST_DELAY = 4   # Seconds to wait between files — keeps RPM well under quota

# --- Unity Catalog ---
CATALOG            = "workspace"
STAGING_SCHEMA     = "staging_google_drive"
OUTPUT_SCHEMA      = "genealogy"
SOURCE_TABLE       = f"{CATALOG}.{STAGING_SCHEMA}.documents"
TRANSCRIPTIONS_TBL = f"{CATALOG}.{OUTPUT_SCHEMA}.ocr_transcriptions"
PROC_LOG_TBL       = f"{CATALOG}.{OUTPUT_SCHEMA}.ocr_processing_log"
TOKEN_USAGE_TBL    = f"{CATALOG}.{OUTPUT_SCHEMA}.ocr_token_usage"

# --- Volume path ---
VOLUME_BASE        = f"/Volumes/{CATALOG}/{STAGING_SCHEMA}/documents"
FOLDER_PATH        = f"{VOLUME_BASE}/Documents/{FOLDER_NAME}"

# --- Controlled vocabulary for LLM doc type detection ---
DOC_TYPE_VOCABULARY = [
    "Census",
    "BirthCertificate",
    "MarriageCertificate",
    "MarriageBanns",
    "DeathCertificate",
    "BaptismRegister",
    "BurialRegister",
    "Will",
    "Probate",
    "NewspaperClipping",
    "ElectoralRegister",
    "MilitaryRecord",
    "BMDIndex",
    "PoorLawRecord",
    "MonumentalInscription",
    "PassengerList",
    "Directory",
    "CV",
    "CriminalRegister",
    "Other",
]

print(f"Processing folder : {FOLDER_PATH}")
print(f"Source table      : {SOURCE_TABLE}")
print(f"Transcriptions    : {TRANSCRIPTIONS_TBL}")
print(f"Processing log    : {PROC_LOG_TBL}")
print(f"Token usage       : {TOKEN_USAGE_TBL}")

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

# google-genai client replaces deprecated vertexai.generative_models (removed June 2026)
client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
    credentials=credentials,
)

print(f"google-genai client initialised. Model: {VERTEX_MODEL}")

# COMMAND ----------

# MAGIC %md ## 2. Create / Upgrade Delta Tables
# MAGIC
# MAGIC `CREATE TABLE IF NOT EXISTS` is idempotent — safe to rerun.
# MAGIC `ALTER TABLE ... ADD COLUMN` statements add new columns to existing tables
# MAGIC and silently pass if the column already exists.

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

for col_name, col_spec in [
    ("doc_type_detected", "STRING COMMENT 'Document type classified by Gemini — controlled vocabulary'"),
    ("gedcom_id",         "STRING COMMENT 'Matched person_gedcom_id — populated by downstream matching'"),
]:
    try:
        spark.sql(f"ALTER TABLE {TRANSCRIPTIONS_TBL} ADD COLUMN {col_name} {col_spec}")
        print(f"Added column {col_name} to {TRANSCRIPTIONS_TBL}")
    except Exception:
        pass  # Column already exists

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

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {TOKEN_USAGE_TBL} (
        logged_at      TIMESTAMP COMMENT 'When this row was written',
        file_id        STRING    COMMENT 'Google Drive file ID',
        file_name      STRING    COMMENT 'Original filename',
        model_used     STRING    COMMENT 'Vertex AI model used',
        page_count     INT       COMMENT 'Number of pages processed',
        prompt_tokens  INT       COMMENT 'Input tokens charged by Gemini',
        output_tokens  INT       COMMENT 'Output tokens charged by Gemini',
        total_tokens   INT       COMMENT 'Total tokens charged by Gemini',
        thinking_tokens INT      COMMENT 'Thinking tokens charged by Gemini (billed as output)'
    )
    USING DELTA
    COMMENT 'Gemini token usage log — one row per file processed'
""")

print("Tables ready.")

# COMMAND ----------

# MAGIC %md ## 3. File Discovery

# COMMAND ----------

# Magic byte signatures → (mime_type, canonical_extension)
# GIFs are detected but then converted to JPEG before sending to Gemini —
# image/gif support is inconsistent across Gemini model versions (can return 400).
MAGIC_SIGNATURES = [
    (b'\xff\xd8\xff',         "image/jpeg",       ".jpg"),
    (b'\x89PNG\r\n\x1a\n',    "image/png",        ".png"),
    (b'II*\x00',              "image/tiff",       ".tif"),   # TIFF little-endian
    (b'MM\x00*',              "image/tiff",       ".tif"),   # TIFF big-endian
    (b'%PDF',                 "application/pdf",  ".pdf"),
    (b'GIF87a',               "image/gif",        ".gif"),
    (b'GIF89a',               "image/gif",        ".gif"),
]

EXTENSION_MIME_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".tif":  "image/tiff",
    ".tiff": "image/tiff",
    ".pdf":  "application/pdf",
    ".gif":  "image/gif",
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
      AND _fivetran_file_path LIKE 'Documents/{FOLDER_NAME}/%'
""")

# Exclude files already successfully processed (idempotency)
processed_ids = spark.sql(f"""
    SELECT file_id FROM {PROC_LOG_TBL}
    WHERE status = 'success'
""")

# Do NOT filter on extension here — get_mime_type() handles this at runtime
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
    "grant": (
        "grant of probate", "19th or early 20th century",
        "deceased's name, estate value, executors, date of grant, court"
    ),
    "newspaper":  ("newspaper clipping", "Victorian or Edwardian era", "all text as printed including headline and captions"),
    "gazette":    ("newspaper or gazette clipping", "Victorian or Edwardian era", "all text as printed"),
    "advertiser": ("newspaper clipping", "Victorian or Edwardian era", "all text as printed"),
    "herald":     ("newspaper clipping", "Victorian or Edwardian era", "all text as printed"),
    "observer":   ("newspaper clipping", "Victorian or Edwardian era", "all text as printed"),
    "times":      ("newspaper clipping", "Victorian or Edwardian era", "all text as printed"),
    "baptism": (
        "baptism register entry", "18th or 19th century",
        "child's name, date of baptism, parents' names, father's occupation, parish"
    ),
    "bapt": (
        "baptism register entry", "18th or 19th century",
        "child's name, date of baptism, parents' names, parish"
    ),
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
    "poorlaw": (
        "Poor Law parochial relief application", "19th century",
        "applicant name, family members, ages, address, parish, reason for application, decision"
    ),
    "poor": (
        "Poor Law parochial relief application", "19th century",
        "applicant name, family members, ages, address, parish, reason for application, decision"
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

    Key prompt design principles (v3):
    - JSON-only instruction is both positive ("return only") AND negative ("do not wrap",
      "do not add commentary") to reduce fence/preamble failures
    - Output skeleton appears at the very end of the prompt — LLMs weight end-of-prompt heavily
    - Vocabulary list appears immediately before the skeleton so doc_type_detected
      values are fresh in context when the skeleton is read
    """
    doc_desc, era, fields = get_doc_type_context(metadata.get("doc_type"))

    subject_hint = ""
    if metadata.get("forename") and metadata.get("surname"):
        subject_hint = f"The document relates to {metadata['forename']} {metadata['surname']}"
        if metadata.get("year"):
            subject_hint += f", circa {metadata['year']}"
        subject_hint += "."

    vocab_list = "\n".join(f"  - {v}" for v in DOC_TYPE_VOCABULARY)

    return f"""You are an expert historical document transcriptionist specialising in {era} {doc_desc}s and the handwriting styles of that period.

{subject_hint}

## Transcription Rules

- Preserve original spelling exactly, including phonetic or archaic spellings (e.g. "labrer", "Willm", "Thos")
- Mark any word you cannot read with confidence as [?]
- For partially legible words, include your best reading followed by [?] (e.g. "Thos[?]")
- Do not modernise, correct, or standardise any spelling or punctuation
- Preserve abbreviations as written
- Transcribe all fields visible in the document, paying particular attention to: {fields}
- For tabular or multi-column documents, preserve the row and column structure in your transcription

## Output Format

Your response must be a single JSON object and nothing else.

Do not include any text, explanation, or commentary before or after the JSON.
Do not wrap the JSON in markdown code fences.
Do not use ```json or ```.
Do not add a preamble sentence.
Your entire response must be valid JSON starting with {{ and ending with }}.

Use exactly this structure, choosing doc_type_detected from the vocabulary below:

Vocabulary for doc_type_detected:
{vocab_list}

{{
  "transcription": "full text of the document, preserving layout",
  "doc_type_detected": "exactly one value from the vocabulary above",
  "confidence": "high or medium or low",
  "confidence_notes": "brief note on factors affecting confidence, or No significant issues.",
  "personal_names": ["name as written", "name as written"],
  "locations": ["place as written", "place as written"],
  "uncertain_entries": [
    {{"entry": "best reading", "field": "column or field name", "reason": "why uncertain"}}
  ]
}}"""

# COMMAND ----------

# MAGIC %md ## 6. OCR Functions

# COMMAND ----------

def split_pdf_to_page_images(file_path: str) -> tuple:
    """
    Split a PDF into a list of JPEG image bytes using PyMuPDF.
    PyMuPDF ships MuPDF as a bundled wheel — no OS-level dependencies needed,
    so this works on Databricks serverless compute.
    Returns (list_of_page_bytes, mime_type).
    """
    import pymupdf

    doc = pymupdf.open(file_path)
    page_bytes = []
    for page in doc:
        # Render at 2x scale (~144 DPI) — good OCR quality without excessive token use
        mat = pymupdf.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        page_bytes.append(pix.tobytes("jpeg"))
    doc.close()

    print(f"    PDF split into {len(page_bytes)} page image(s) via PyMuPDF")
    return page_bytes, "image/jpeg"


def extract_json(text: str) -> str:
    """
    Robustly extract a JSON object from Gemini's response text.
    Finds the outermost { ... } regardless of any surrounding text,
    preamble sentences, or markdown fences. This is more reliable than
    trying to strip specific fence patterns.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text  # Return as-is and let json.loads raise a clear error


def _call_gemini_once(file_bytes: bytes, mime_type: str, prompt: str) -> tuple:
    """
    Single attempt to call Gemini. Separated from retry logic so tenacity
    can wrap it cleanly without capturing non-retryable exceptions.
    Uses google-genai SDK (replaces deprecated vertexai.generative_models).
    Returns (result_dict, usage_dict).
    usage_dict keys: prompt_tokens, output_tokens, total_tokens.
    """
    image_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

    response = client.models.generate_content(
        model=VERTEX_MODEL,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=65536,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"), # reduce Gemini 3 thinking to keep token use dow, but keep an eye on quality
            http_options=types.HttpOptions(timeout=120000),  # 2 minutes in milliseconds
        ),
    )

    # Capture token usage — always present even on safety blocks
    um = response.usage_metadata
    usage = {
        "prompt_tokens":  getattr(um, "prompt_token_count",     0) or 0,
        "output_tokens":  getattr(um, "candidates_token_count", 0) or 0,
        "thinking_tokens": getattr(um, "thoughts_token_count",    0) or 0,
        "total_tokens":   getattr(um, "total_token_count",      0) or 0,
    }

    # Detect safety block — candidates empty or content has no parts
    if not response.candidates:
        return {"_safety_blocked": True}, usage
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return {
            "_safety_blocked": True,
            "_finish_reason": str(getattr(candidate, "finish_reason", "unknown")),
        }, usage

    raw_text = response.text.strip()
    raw_text = extract_json(raw_text)

    try:
        return json.loads(raw_text), usage
    except json.JSONDecodeError:
        return {
            "transcription":     raw_text,
            "doc_type_detected": "Other",
            "confidence":        "low",
            "confidence_notes":  "Gemini response could not be parsed as JSON — raw text preserved.",
            "personal_names":    [],
            "locations":         [],
            "uncertain_entries": [],
        }, usage


@retry(
    # Retry specifically on Vertex AI quota/rate-limit exceptions.
    # ResourceExhausted = gRPC quota exhaustion; TooManyRequests = HTTP 429.
    # Using the SDK exception classes directly is more reliable than string inspection.
    retry=retry_if_exception_type((
        google.api_core.exceptions.ResourceExhausted,  # 429
        google.api_core.exceptions.TooManyRequests,
        google.api_core.exceptions.DeadlineExceeded,   # 504
        google.api_core.exceptions.Cancelled,          # 499
    )),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(5),
        before_sleep=lambda rs: print(
            f"    Rate limited (429) — attempt {rs.attempt_number} failed, retrying..."
        ),
)
def call_gemini_with_retry(file_bytes: bytes, mime_type: str, prompt: str) -> tuple:
    """
    Call Gemini with tenacity exponential backoff on 429 rate limit errors.

    Uses wait_random_exponential(multiplier=1, max=60) as recommended by Google:
    waits a random duration up to 2^attempt seconds, capped at 60 seconds.
    Jitter prevents retry storms when multiple files hit the limit simultaneously.
    Non-429 errors are raised immediately without retry.
    Returns (result_dict, usage_dict) from _call_gemini_once.
    """
    return _call_gemini_once(file_bytes, mime_type, prompt)


def convert_gif_to_jpeg(file_path: str) -> bytes:
    """
    Convert a GIF to JPEG bytes using PyMuPDF.
    Gemini nominally supports image/gif but returns 400 BadRequest on some model
    versions. Converting to JPEG is more reliable.
    """
    import pymupdf
    doc = pymupdf.open(file_path)
    page = doc[0]
    mat = pymupdf.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    doc.close()
    return pix.tobytes("jpeg")


def ocr_file(file_path: str, metadata: dict) -> tuple:
    """
    Run OCR on a single file.
    Returns (list_of_row_dicts, status_string).
    - Images produce one row.
    - PDFs are split into pages — one row per page.
    - GIFs are converted to JPEG before sending (avoids Gemini 400 BadRequest).
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
    elif mime_type == "image/gif":
        print(f"    Converting GIF to JPEG for Gemini compatibility")
        print(f"    Commented out GIF conversion as it seems to hang")
        # pages_bytes = [convert_gif_to_jpeg(file_path)]
        # page_mime = "image/jpeg"
    else:
        with open(file_path, "rb") as f:
            pages_bytes = [f.read()]
        page_mime = mime_type

    rows = []
    file_prompt_tokens  = 0
    file_output_tokens  = 0
    file_thinking_tokens  = 0
    file_total_tokens   = 0

    for page_index, page_bytes in enumerate(pages_bytes, 1):
        result, usage = call_gemini_with_retry(page_bytes, page_mime, prompt)

        file_prompt_tokens  += usage["prompt_tokens"]
        file_output_tokens  += usage["output_tokens"]
        file_thinking_tokens  += usage["thinking_tokens"]
        file_total_tokens   += usage["total_tokens"]

        if result.get("_safety_blocked"):
            return [], "safety_blocked", file_prompt_tokens, file_output_tokens, file_thinking_tokens, file_total_tokens

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
            # Flag medium and low — anything that is not high warrants a check
            "confidence_flag":   result.get("confidence", "low") != "high",
            "gedcom_id":         None,
            "model_used":        VERTEX_MODEL,
            "processed_at":      now,
        })

    return rows, "success", file_prompt_tokens, file_output_tokens, file_thinking_tokens, file_total_tokens

# COMMAND ----------

# MAGIC %md ## 7. Processing Loop

# COMMAND ----------

transcription_rows = []
log_rows = []

TRANSCRIPTION_SCHEMA = StructType([
    StructField("file_id",           StringType(),    True),
    StructField("file_name",         StringType(),    True),
    StructField("folder_path",       StringType(),    True),
    StructField("surname",           StringType(),    True),
    StructField("forename",          StringType(),    True),
    StructField("year",              StringType(),    True),
    StructField("doc_type",          StringType(),    True),
    StructField("doc_type_detected", StringType(),    True),
    StructField("page_index",        IntegerType(),   True),
    StructField("transcribed_text",  StringType(),    True),
    StructField("personal_names",    StringType(),    True),
    StructField("locations",         StringType(),    True),
    StructField("uncertain_entries", StringType(),    True),
    StructField("confidence",        StringType(),    True),
    StructField("confidence_notes",  StringType(),    True),
    StructField("confidence_flag",   BooleanType(),   True),
    StructField("gedcom_id",         StringType(),    True),
    StructField("model_used",        StringType(),    True),
    StructField("processed_at",      TimestampType(), True),
])

LOG_SCHEMA = StructType([
    StructField("file_id",       StringType(),    True),
    StructField("file_name",     StringType(),    True),
    StructField("folder_path",   StringType(),    True),
    StructField("status",        StringType(),    True),
    StructField("page_count",    IntegerType(),   True),
    StructField("model_used",    StringType(),    True),
    StructField("processed_at",  TimestampType(), True),
    StructField("error_message", StringType(),    True),
])

# Write results immediately after each file so that a cancelled or failed
# run does not lose work already completed. The idempotency check at the top
# of the loop (left_anti join on the log table) ensures files already logged
# as success are not reprocessed on the next run.

def write_transcription_rows(rows: list):
    """Write a list of transcription row dicts to Delta immediately."""
    if not rows:
        return
    df = spark.createDataFrame(rows, schema=TRANSCRIPTION_SCHEMA)
    df.write.format("delta").mode("append").saveAsTable(TRANSCRIPTIONS_TBL)


def write_log_row(row: dict):
    """Write a single processing log row dict to Delta immediately."""
    df = spark.createDataFrame([row], schema=LOG_SCHEMA)
    df.write.format("delta").mode("append").saveAsTable(PROC_LOG_TBL)


TOKEN_USAGE_SCHEMA = StructType([
    StructField("logged_at",      TimestampType(), True),
    StructField("file_id",        StringType(),    True),
    StructField("file_name",      StringType(),    True),
    StructField("model_used",     StringType(),    True),
    StructField("page_count",     IntegerType(),   True),
    StructField("prompt_tokens",  IntegerType(),   True),
    StructField("output_tokens",  IntegerType(),   True),
    StructField("thinking_tokens",  IntegerType(),   True),
    StructField("total_tokens",   IntegerType(),   True),
])


def _write_token_usage(file_id: str, file_name: str, page_count: int,
                       prompt_tokens: int, output_tokens: int, thinking_tokens: int,
                       total_tokens: int, logged_at):
    """
    Write token usage for one file to the usage log table.
    Non-fatal — a failure here never aborts the main pipeline.
    """
    try:
        df = spark.createDataFrame([{
            "logged_at":     logged_at,
            "file_id":       file_id,
            "file_name":     file_name,
            "model_used":    VERTEX_MODEL,
            "page_count":    page_count,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens":  total_tokens,
        }], schema=TOKEN_USAGE_SCHEMA)
        df.write.format("delta").mode("append").saveAsTable(TOKEN_USAGE_TBL)
    except Exception as e:
        print(f"  ⚠ Token usage logging failed (non-fatal): {e}")


# Keep summary-only accumulators for the end-of-run report.
# Individual rows are no longer held in memory.
all_rows_summary = []   # lightweight dicts: file_name, page_index, confidence, confidence_flag, confidence_notes

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

    # Check file type early — avoids Gemini calls for genuinely unsupported files
    type_info = get_mime_type(file_path)
    if type_info is None:
        print(f"  ⚠ Skipped — unrecognised file type")
        log_rows.append({
            "file_id": file_id, "file_name": file_name, "folder_path": FOLDER_PATH,
            "status": "skipped", "page_count": 0, "model_used": VERTEX_MODEL,
            "processed_at": now,
            "error_message": "Unrecognised file type — no extension and no matching magic bytes",
        })
        skipped_count += 1
        continue

    metadata = parse_filename(file_name)
    metadata["file_id"]   = file_id
    metadata["file_name"] = file_name

    if not metadata["parse_ok"]:
        print(f"  ⚠ Filename did not parse cleanly — using generic prompt")

    try:
        rows, status, prompt_tokens, output_tokens, thinking_tokens, total_tokens = ocr_file(file_path, metadata)

        if status == "safety_blocked":
            print(f"  ✗ Blocked by Gemini safety filters")
            write_log_row({
                "file_id": file_id, "file_name": file_name, "folder_path": FOLDER_PATH,
                "status": "safety_blocked", "page_count": 0, "model_used": VERTEX_MODEL,
                "processed_at": now,
                "error_message": "Response blocked by Gemini safety filters. Try converting to image and reprocessing.",
            })
            _write_token_usage(file_id, file_name, 0, prompt_tokens, output_tokens, thinking_tokens, total_tokens, now)
            blocked_count += 1
            continue

        # Write transcription rows and log entry to Delta immediately —
        # results are safe even if the run is cancelled after this point.
        write_transcription_rows(rows)
        write_log_row({
            "file_id": file_id, "file_name": file_name, "folder_path": FOLDER_PATH,
            "status": "success", "page_count": len(rows), "model_used": VERTEX_MODEL,
            "processed_at": now, "error_message": None,
        })
        _write_token_usage(file_id, file_name, len(rows), prompt_tokens, output_tokens, thinking_tokens, total_tokens, now)

        flagged    = any(r["confidence_flag"] for r in rows)
        confidence = rows[0]["confidence"] if rows else "unknown"
        print(f"  ✓ Done  pages={len(rows)}  confidence={confidence}"
              + ("  ⚑ review flagged" if flagged else ""))
        # Accumulate lightweight summary for end-of-run report
        all_rows_summary.extend([
            {"file_name": r["file_name"], "page_index": r["page_index"],
             "confidence": r["confidence"], "confidence_flag": r["confidence_flag"],
             "confidence_notes": r["confidence_notes"]}
            for r in rows
        ])
        success_count += 1

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"  ✗ Error: {error_msg[:200]}")
        write_log_row({
            "file_id": file_id, "file_name": file_name, "folder_path": FOLDER_PATH,
            "status": "error", "page_count": 0, "model_used": VERTEX_MODEL,
            "processed_at": now, "error_message": error_msg[:2000],
        })
        error_count += 1

    # Pace requests to stay within Vertex AI RPM quota.
    # Placed at end of loop body so it runs after every file (success, error,
    # or safety_blocked). Skipped files use 'continue' above so don't reach here,
    # but their API cost is zero so the omission is fine.
    if i < total:
        time.sleep(INTER_REQUEST_DELAY)

print(f"\n{'='*50}")
print(f"Run complete: {success_count} succeeded, {error_count} errors, "
      f"{blocked_count} safety blocked, {skipped_count} skipped  ({total} candidates)")

# Print token usage totals for this run
try:
    usage_summary = spark.sql(f"""
        SELECT
            SUM(prompt_tokens)  AS prompt_tokens,
            SUM(output_tokens)  AS output_tokens,
            SUM(thinking_tokens)  AS thinking_tokens,
            SUM(total_tokens)   AS total_tokens,
            COUNT(*)            AS files_logged
        FROM {TOKEN_USAGE_TBL}
        WHERE file_id IN (
            SELECT file_id FROM {PROC_LOG_TBL}
            WHERE folder_path LIKE '%{FOLDER_NAME}%'
              AND status IN ('success', 'safety_blocked')
        )
    """).collect()[0]
    print(f"\nToken usage (this folder, all-time):")
    print(f"  Prompt tokens : {usage_summary.prompt_tokens:,}")
    print(f"  Output tokens : {usage_summary.output_tokens:,}")
    print(f"  Thinking tokens : {usage_summary.thinking_tokens:,}")
    print(f"  Total tokens  : {usage_summary.total_tokens:,}")
except Exception as e:
    print(f"\n⚠ Could not retrieve token usage summary: {e}")

# COMMAND ----------

# MAGIC %md ## 8. Confirm Writes

# COMMAND ----------

# Rows are written to Delta immediately after each file in the processing loop.
# This cell just confirms the current state of both tables for this folder.
spark.sql(f"""
    SELECT status, COUNT(*) AS files, SUM(page_count) AS pages
    FROM {PROC_LOG_TBL}
    WHERE folder_path LIKE '%{FOLDER_NAME}%'
    GROUP BY status ORDER BY status
""").show()

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

if all_rows_summary:
    confidence_counts = Counter(r["confidence"] for r in all_rows_summary)
    print("Confidence breakdown (rows written this run):")
    for level in ["high", "medium", "low"]:
        print(f"  {level:8s}: {confidence_counts.get(level, 0)}")
    print()

needs_review = [r for r in all_rows_summary if r["confidence_flag"]]
if needs_review:
    print(f"Rows flagged for review ({len(needs_review)}):")
    for r in needs_review:
        print(f"  {r['file_name']} p{r['page_index']}  [{r['confidence']}]  {r['confidence_notes'][:100]}")
else:
    print("No rows flagged for review.")
print()
print("Note: doc_type breakdown available via the validation query in Section 10.")

if blocked_count:
    print(f"\nSafety blocked ({blocked_count}) — check log table for filenames:")
    print(f"  SELECT file_name FROM {PROC_LOG_TBL} WHERE status = 'safety_blocked' AND folder_path LIKE '%{FOLDER_NAME}%'")
    print("  Tip: convert PDF to JPEG pages and reprocess")

if error_count:
    print(f"\nErrors ({error_count}) — check log table for details:")
    print(f"  SELECT file_name, error_message FROM {PROC_LOG_TBL} WHERE status = 'error' AND folder_path LIKE '%{FOLDER_NAME}%'")

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
# MAGIC WHERE folder_path LIKE '%Family_AMBROSE%'
# MAGIC ORDER BY file_name, page_index
# MAGIC LIMIT 50
