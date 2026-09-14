#!/usr/bin/env python3
"""Run the SQL data-quality checks in checks/*.sql against workspace.genealogy.

Each check file is a SELECT that returns violating rows (zero rows = pass).
Every run writes one row per check to genealogy.data_quality_results, and a
newly-failing critical check files (or reuses) an Asana task. See
tests/data_quality/README.md for the check file format and how to run this
locally.
"""
import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import requests
from databricks import sql

CHECKS_DIR = Path(__file__).parent / "checks"
SAMPLE_CAP = 20

ASANA_API = "https://app.asana.com/api/1.0"
ASANA_PROJECT_GID = "1218247114984172"
ASANA_SECTION_GID = "1215610601060422"

HEADER_KEYS = {
    "id",
    "title",
    "severity",
    "guards_bug",
    "known_failing",
    "existing_asana_task",
    "description",
}


def parse_check_file(path):
    """Parse the '-- key: value' header block and the SQL body.

    Tolerant line-prefix parser, not a real YAML parser, per the spec —
    the header format is fixed and simple enough not to need one.
    """
    lines = path.read_text().splitlines()
    header = {k: "" for k in HEADER_KEYS}
    current_key = None
    sql_start = len(lines)

    for i, line in enumerate(lines):
        if not line.strip():
            sql_start = i + 1
            break
        if not line.startswith("--"):
            sql_start = i
            break
        content = line[2:]
        if content.startswith(" "):
            content = content[1:]
        match = re.match(r"^(\w+):\s?(.*)$", content)
        if match and match.group(1) in HEADER_KEYS:
            current_key = match.group(1)
            value = match.group(2).strip()
            header[current_key] = "" if (current_key == "description" and value == ">") else value
        elif current_key == "description":
            header["description"] = (header["description"] + " " + content.strip()).strip()

    sql_text = "\n".join(lines[sql_start:]).strip()

    return {
        "id": header["id"].strip(),
        "title": header["title"].strip(),
        "severity": header["severity"].strip().lower(),
        "guards_bug": header["guards_bug"].strip() or None,
        "known_failing": header["known_failing"].strip().lower() == "true",
        "existing_asana_task": header["existing_asana_task"].strip() or None,
        "description": header["description"].strip(),
        "sql": sql_text,
        "path": path,
    }


def load_checks(only=None, severity=None):
    checks = [parse_check_file(p) for p in sorted(CHECKS_DIR.glob("*.sql"))]
    if only:
        wanted = {c.strip().upper() for c in only.split(",")}
        checks = [c for c in checks if c["id"].upper() in wanted]
    if severity:
        checks = [c for c in checks if c["severity"] == severity.lower()]
    return checks


def get_connection():
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    warehouse = os.environ.get("DATABRICKS_WAREHOUSE")
    missing = [
        name
        for name, val in [
            ("DATABRICKS_HOST", host),
            ("DATABRICKS_TOKEN", token),
            ("DATABRICKS_WAREHOUSE", warehouse),
        ]
        if not val
    ]
    if missing:
        print(f"ERROR: missing required environment variable(s): {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    return sql.connect(server_hostname=host, http_path=warehouse, access_token=token)


def ensure_results_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS genealogy.data_quality_results (
          run_id STRING,
          check_id STRING,
          run_at TIMESTAMP,
          status STRING,
          severity STRING,
          violation_count BIGINT,
          sample_violations STRING,
          known_failing BOOLEAN,
          guards_bug STRING,
          asana_task_gid STRING
        ) USING DELTA
        """
    )


def json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def run_check(cursor, check):
    cursor.execute(check["sql"])
    columns = [c[0] for c in cursor.description]
    rows = cursor.fetchall()
    violation_count = len(rows)
    status = "pass" if violation_count == 0 else "fail"

    sample_violations = None
    if status == "fail":
        sample = [dict(zip(columns, row)) for row in rows[:SAMPLE_CAP]]
        sample_violations = json.dumps(sample, default=json_default)

    return status, violation_count, sample_violations


def write_result(cursor, run_id, check, status, violation_count, sample_violations, asana_task_gid):
    cursor.execute(
        f"""
        INSERT INTO genealogy.data_quality_results
        (run_id, check_id, run_at, status, severity, violation_count,
         sample_violations, known_failing, guards_bug, asana_task_gid)
        VALUES (
          {sql_literal(run_id)}, {sql_literal(check['id'])}, current_timestamp(),
          {sql_literal(status)}, {sql_literal(check['severity'])}, {violation_count},
          {sql_literal(sample_violations)}, {sql_literal(check['known_failing'])},
          {sql_literal(check['guards_bug'])}, {sql_literal(asana_task_gid)}
        )
        """
    )


def asana_headers():
    pat = os.environ.get("ASANA_PAT")
    if not pat:
        return None
    return {"Authorization": f"Bearer {pat}", "Content-Type": "application/json"}


def find_open_asana_task(headers, title):
    resp = requests.get(
        f"{ASANA_API}/sections/{ASANA_SECTION_GID}/tasks",
        headers=headers,
        params={"opt_fields": "name,completed"},
        timeout=30,
    )
    resp.raise_for_status()
    for task in resp.json().get("data", []):
        if not task.get("completed") and task.get("name") == title:
            return task["gid"]
    return None


def create_asana_task(headers, title, notes):
    resp = requests.post(
        f"{ASANA_API}/tasks",
        headers=headers,
        json={"data": {"name": title, "notes": notes, "projects": [ASANA_PROJECT_GID]}},
        timeout=30,
    )
    resp.raise_for_status()
    task_gid = resp.json()["data"]["gid"]
    move_resp = requests.post(
        f"{ASANA_API}/sections/{ASANA_SECTION_GID}/addTask",
        headers=headers,
        json={"data": {"task": task_gid}},
        timeout=30,
    )
    move_resp.raise_for_status()
    return task_gid


def add_asana_comment(headers, task_gid, text):
    resp = requests.post(
        f"{ASANA_API}/tasks/{task_gid}/stories",
        headers=headers,
        json={"data": {"text": text}},
        timeout=30,
    )
    resp.raise_for_status()


def build_task_notes(check, run_id, violation_count, sample_violations):
    lines = [
        check["description"] or "(no description provided)",
        "",
        f"Violation count: {violation_count}",
        f"See genealogy.data_quality_results, run_id = {run_id}",
        "",
        "Sample violations:",
    ]
    if sample_violations:
        sample = json.loads(sample_violations)
        for row in sample[:5]:
            lines.append(f"- {json.dumps(row, default=json_default)}")
    return "\n".join(lines)


def handle_asana(check, run_id, violation_count, sample_violations):
    """Returns the Asana task GID linked to this failure, or None."""
    if check["severity"] == "info":
        return None

    if check["known_failing"] and check["existing_asana_task"]:
        headers = asana_headers()
        if headers:
            try:
                add_asana_comment(
                    headers,
                    check["existing_asana_task"],
                    f"DQ run {run_id}: {violation_count} violations at "
                    f"{datetime.now(timezone.utc).isoformat()}",
                )
            except requests.RequestException as exc:
                print(f"  WARNING: failed to comment on existing Asana task: {exc}", file=sys.stderr)
        return check["existing_asana_task"]

    title = f"\U0001F534 [DQ] {check['id']}: {check['title']}"
    headers = asana_headers()
    if not headers:
        print("  WARNING: ASANA_PAT not set, skipping Asana task filing", file=sys.stderr)
        return None

    existing = find_open_asana_task(headers, title)
    if existing:
        return existing

    notes = build_task_notes(check, run_id, violation_count, sample_violations)
    return create_asana_task(headers, title, notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Comma-separated list of check IDs to run, e.g. DQ-001,DQ-006")
    parser.add_argument("--severity", help="Only run checks of this severity (critical|warning|info)")
    args = parser.parse_args()

    checks = load_checks(only=args.only, severity=args.severity)
    if not checks:
        print("No checks matched the given filters.", file=sys.stderr)
        sys.exit(1)

    run_id = str(uuid.uuid4())
    conn = get_connection()
    cursor = conn.cursor()
    ensure_results_table(cursor)

    results = []
    for check in checks:
        print(f"Running {check['id']}: {check['title']} ...")
        status, violation_count, sample_violations = run_check(cursor, check)

        asana_task_gid = None
        if status == "fail":
            try:
                asana_task_gid = handle_asana(check, run_id, violation_count, sample_violations)
            except requests.RequestException as exc:
                print(f"  WARNING: Asana call failed: {exc}", file=sys.stderr)

        write_result(cursor, run_id, check, status, violation_count, sample_violations, asana_task_gid)
        results.append({**check, "status": status, "violation_count": violation_count, "asana_task_gid": asana_task_gid})

    cursor.close()
    conn.close()

    print()
    print(f"{'ID':<8} {'STATUS':<6} {'ROWS':>6} {'SEVERITY':<10} TITLE")
    for r in results:
        print(f"{r['id']:<8} {r['status']:<6} {r['violation_count']:>6} {r['severity']:<10} {r['title']}")

    newly_failed_critical = [
        r for r in results if r["severity"] == "critical" and r["status"] == "fail" and not r["known_failing"]
    ]
    if newly_failed_critical:
        print()
        print("FAILED (new critical failures):", ", ".join(r["id"] for r in newly_failed_critical))
        sys.exit(1)

    print()
    print(f"run_id: {run_id} — no new critical failures")
    sys.exit(0)


if __name__ == "__main__":
    main()
