#!/usr/bin/env python3
"""Run the data-quality checks (Tier 2 files + Tier 1 registry) against workspace.genealogy.

Two check-authoring sources feed the same execution path:
- Tier 2: hand-written checks/*.sql files, each a SELECT that returns
  violating rows (zero rows = pass) — business rules, aggregate
  reconciliation, regression guards.
- Tier 1: registry/tier1_*.yaml (one file per layer — tier1_gold_registry.yaml
  from Phase 1, tier1_silver_registry.yaml from Phase 2, tier1_bronze_registry.yaml
  and tier1_ref_registry.yaml from Phase 3), a config-driven registry of
  generic checks generated into the same SELECT-returns-violations shape.
  Gold/bronze check types: row_count_not_zero, freshness_vs_source. Silver
  check types: not_null, uniqueness, fk_integrity. Ref check types adds
  not_blank (NULL or empty-string). See tests/data_quality/README.md for
  all formats.

Every run syncs genealogy.ref_data_quality_registry from the checked-in
YAML files (git is the source of truth; the Delta table is a queryable
materialization, same pattern as ref_signal_weights), writes one row per
check to genealogy.data_quality_results, and a newly-failing critical
check files (or reuses) an Asana task.

Selective execution (Phase 5): --only and --severity filter which checks
run/report, same as always. --layer and --object additionally narrow which
Tier 1 registry objects are even generated into checks -- for fast dev
iteration and for scoping a pipeline-embedded QC task (Notion Test Plan §9)
to one layer/table instead of the whole suite. Tier 2 checks/*.sql have no
layer metadata yet, so --layer/--object never affect them. Critically,
--layer/--object only narrow *this run's* check list -- the sync to
genealogy.ref_data_quality_registry always uses the full, unfiltered
registry, so a `--layer silver` run can never delete the bronze/gold/ref
rows out of that shared table.
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
import yaml
from databricks import sql

CHECKS_DIR = Path(__file__).parent / "checks"
REGISTRY_DIR = Path(__file__).parent / "registry"
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


def apply_filters(checks, only=None, severity=None):
    if only:
        wanted = {c.strip().upper() for c in only.split(",")}
        checks = [c for c in checks if c["id"].upper() in wanted]
    if severity:
        checks = [c for c in checks if c["severity"] == severity.lower()]
    return checks


def load_file_checks(only=None, severity=None):
    checks = [parse_check_file(p) for p in sorted(CHECKS_DIR.glob("*.sql"))]
    return apply_filters(checks, only, severity)


def known_layers():
    return sorted(
        p.stem[len("tier1_"):-len("_registry")] for p in REGISTRY_DIR.glob("tier1_*_registry.yaml")
    )


def load_registry_seed(layer=None):
    """Load and merge registry/tier1_*.yaml file(s)' objects into one list.

    layer: optional comma-separated layer name(s) (e.g. "silver" or
    "bronze,ref") to load only tier1_<layer>_registry.yaml for each,
    instead of every tier1_*.yaml file. Errors loudly on an unknown layer
    name rather than silently loading nothing.
    """
    if layer:
        wanted = [name.strip().lower() for name in layer.split(",") if name.strip()]
        paths = []
        for name in wanted:
            path = REGISTRY_DIR / f"tier1_{name}_registry.yaml"
            if not path.exists():
                raise FileNotFoundError(
                    f"--layer '{name}' has no matching file at {path}. "
                    f"Known layers: {', '.join(known_layers())}"
                )
            paths.append(path)
    else:
        paths = sorted(REGISTRY_DIR.glob("tier1_*.yaml"))

    objects = []
    for path in paths:
        objects.extend(yaml.safe_load(path.read_text())["objects"])
    return objects


def filter_registry_objects(objects, object_name=None):
    """Narrow registry objects to --object's comma-separated name(s), matching
    either the fully-qualified name (genealogy.silver_person) or the bare
    table name (silver_person)."""
    if not object_name:
        return objects
    wanted = {name.strip().lower() for name in object_name.split(",") if name.strip()}
    return [
        obj
        for obj in objects
        if obj["name"].lower() in wanted or obj["name"].split(".")[-1].lower() in wanted
    ]


def build_tier1_check_sql(check_type, object_name, check_cfg):
    if check_type == "row_count_not_zero":
        return f"SELECT 'EMPTY_TABLE' AS violation FROM (SELECT COUNT(*) AS n FROM {object_name}) t WHERE t.n = 0"

    if check_type == "freshness_vs_source":
        depends_on = check_cfg.get("depends_on", [])
        source_union = "\n    UNION ALL\n    ".join(
            f"SELECT timestamp AS ts FROM (DESCRIBE HISTORY {src})" for src in depends_on
        )
        return (
            "WITH target AS (\n"
            f"  SELECT MAX(timestamp) AS last_write FROM (DESCRIBE HISTORY {object_name})\n"
            "),\n"
            "source AS (\n"
            "  SELECT MAX(ts) AS last_write FROM (\n"
            f"    {source_union}\n"
            "  )\n"
            ")\n"
            "SELECT target.last_write AS target_last_write, source.last_write AS source_last_write\n"
            "FROM target, source\n"
            "WHERE target.last_write < source.last_write"
        )

    if check_type == "not_null":
        column = check_cfg["column"]
        return f"SELECT * FROM {object_name} WHERE {column} IS NULL"

    if check_type == "not_blank":
        column = check_cfg["column"]
        return f"SELECT * FROM {object_name} WHERE {column} IS NULL OR TRIM({column}) = ''"

    if check_type == "uniqueness":
        columns = ", ".join(check_cfg["columns"])
        return (
            f"SELECT {columns}, COUNT(*) AS dupe_count FROM {object_name} "
            f"GROUP BY {columns} HAVING COUNT(*) > 1"
        )

    if check_type == "fk_integrity":
        column = check_cfg["column"]
        ref_table = check_cfg["ref_table"]
        ref_column = check_cfg["ref_column"]
        return (
            f"SELECT DISTINCT {column} FROM {object_name} t "
            f"WHERE t.{column} IS NOT NULL AND NOT EXISTS "
            f"(SELECT 1 FROM {ref_table} r WHERE r.{ref_column} = t.{column})"
        )

    raise ValueError(f"unknown Tier 1 check_type: {check_type}")


def tier1_check_label(check_type, check_cfg):
    """Column/key description used in the derived check id and title."""
    if check_type == "not_null":
        return check_cfg["column"]
    if check_type == "not_blank":
        return check_cfg["column"]
    if check_type == "uniqueness":
        return "_".join(check_cfg["columns"])
    if check_type == "fk_integrity":
        return check_cfg["column"]
    return None


def build_tier1_checks(registry_objects, only=None, severity=None):
    """Generate Tier 1 registry checks into the same dict shape parse_check_file
    produces, so they flow through the existing run_check/write_result/handle_asana
    pipeline unchanged. Check IDs are derived (table + check type [+ column]),
    not manually assigned — the registry has no DQ-NNN-style numbering to keep
    in sync."""
    checks = []
    for obj in registry_objects:
        short_name = obj["name"].split(".")[-1].upper()
        for check in obj["checks"]:
            check_type = check["check_type"]
            label = tier1_check_label(check_type, check)
            id_suffix = f"-{label.upper()}" if label else ""
            title_suffix = f" ({label})" if label else ""
            checks.append(
                {
                    "id": f"T1-{check_type.upper()}-{short_name}{id_suffix}",
                    "title": f"{obj['name']} {check_type.replace('_', ' ')}{title_suffix}",
                    "severity": check["severity"].strip().lower(),
                    "guards_bug": None,
                    "known_failing": check.get("known_failing", False),
                    "existing_asana_task": check.get("existing_asana_task"),
                    "description": check.get("notes", "").strip()
                    or f"Tier 1 registry-driven check ({check_type}) for {obj['name']}.",
                    "sql": build_tier1_check_sql(check_type, obj["name"], check),
                    "path": None,
                }
            )
    return apply_filters(checks, only, severity)


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


def ensure_registry_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS genealogy.ref_data_quality_registry (
          object_name STRING,
          object_type STRING,
          check_type STRING,
          column_name STRING,
          columns STRING,
          depends_on STRING,
          ref_table STRING,
          ref_column STRING,
          severity STRING,
          known_failing BOOLEAN,
          existing_asana_task STRING,
          notes STRING
        ) USING DELTA
        """
    )
    # Phase 1 created this table without column_name/columns/ref_table/ref_column
    # (Phase 2 additions, for the silver not_null/uniqueness/fk_integrity check
    # shapes) — CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so
    # backfill any columns an earlier run didn't create.
    cursor.execute("DESCRIBE TABLE genealogy.ref_data_quality_registry")
    existing_columns = {row[0] for row in cursor.fetchall()}
    additions = {
        "column_name": "STRING",
        "columns": "STRING",
        "ref_table": "STRING",
        "ref_column": "STRING",
    }
    missing = {name: type_ for name, type_ in additions.items() if name not in existing_columns}
    if missing:
        cols_sql = ", ".join(f"{name} {type_}" for name, type_ in missing.items())
        cursor.execute(f"ALTER TABLE genealogy.ref_data_quality_registry ADD COLUMNS ({cols_sql})")


def sync_registry_table(cursor, registry_objects):
    """Re-sync genealogy.ref_data_quality_registry from the checked-in YAML —
    the Delta table is a queryable materialization (same pattern as
    ref_signal_weights), not something edited live; git is authoritative."""
    cursor.execute("DELETE FROM genealogy.ref_data_quality_registry")
    for obj in registry_objects:
        for check in obj["checks"]:
            columns = ",".join(check.get("columns", [])) or None
            depends_on = ",".join(check.get("depends_on", [])) or None
            cursor.execute(
                f"""
                INSERT INTO genealogy.ref_data_quality_registry
                (object_name, object_type, check_type, column_name, columns, depends_on,
                 ref_table, ref_column, severity, known_failing, existing_asana_task, notes)
                VALUES (
                  {sql_literal(obj['name'])}, {sql_literal(obj['type'])},
                  {sql_literal(check['check_type'])}, {sql_literal(check.get('column'))},
                  {sql_literal(columns)}, {sql_literal(depends_on)},
                  {sql_literal(check.get('ref_table'))}, {sql_literal(check.get('ref_column'))},
                  {sql_literal(check['severity'])}, {sql_literal(check.get('known_failing', False))},
                  {sql_literal(check.get('existing_asana_task'))}, {sql_literal(check.get('notes'))}
                )
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
    parser.add_argument(
        "--only", help="Comma-separated list of check IDs to run, e.g. DQ-001,DQ-006 or T1-ROW_COUNT_NOT_ZERO-GOLD_EVENT"
    )
    parser.add_argument("--severity", help="Only run checks of this severity (critical|warning|info)")
    parser.add_argument(
        "--layer",
        help=(
            "Comma-separated Tier 1 registry layer(s) to run, e.g. silver or bronze,ref "
            "-- matches registry/tier1_<layer>_registry.yaml. Omit to run every layer "
            "(the default). Tier 1 only: Tier 2 checks/*.sql always run regardless of "
            f"--layer (use --only/--severity to narrow those). Known layers: {', '.join(known_layers())}."
        ),
    )
    parser.add_argument(
        "--object",
        help=(
            "Comma-separated Tier 1 registry object name(s) to run every check for, "
            "e.g. genealogy.silver_person or silver_person,silver_family (bare table "
            "name also matches). Tier 1 only, same scope note as --layer."
        ),
    )
    args = parser.parse_args()

    # The registry table sync always uses the FULL, unfiltered registry --
    # --layer/--object narrow which checks THIS RUN executes, never what
    # genealogy.ref_data_quality_registry (a shared materialization other
    # surfaces may read) contains. A `--layer silver` run must not delete
    # the bronze/gold/ref rows out of that table.
    full_registry_objects = load_registry_seed()
    run_registry_objects = filter_registry_objects(
        load_registry_seed(layer=args.layer), object_name=args.object
    )

    checks = load_file_checks(only=args.only, severity=args.severity) + build_tier1_checks(
        run_registry_objects, only=args.only, severity=args.severity
    )
    if not checks:
        print(
            "No checks matched the given filters (--only/--severity/--layer/--object).",
            file=sys.stderr,
        )
        sys.exit(1)

    run_id = str(uuid.uuid4())
    conn = get_connection()
    cursor = conn.cursor()
    ensure_results_table(cursor)
    ensure_registry_table(cursor)
    sync_registry_table(cursor, full_registry_objects)

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
