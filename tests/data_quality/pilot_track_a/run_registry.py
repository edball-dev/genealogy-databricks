#!/usr/bin/env python3
"""Track A pilot runner: read registry.yaml, execute generic + custom checks.

Minimal by design (see spec §4) — this proves out check authoring and
execution for the pilot's 4 checks x 3 tables. It deliberately skips the
production run_checks.py's Asana-dedup filing and data_quality_results
write-back; it only needs to report known_failing status inline, which is
enough to test the semantics in spec §3.

Same connection pattern as tests/data_quality/run_checks.py:
databricks-sql-connector against workspace.genealogy via
DATABRICKS_HOST / DATABRICKS_TOKEN / DATABRICKS_WAREHOUSE.
"""
import os
import sys
from pathlib import Path

import yaml
from databricks import sql

REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


def get_connection():
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    warehouse = os.environ.get("DATABRICKS_WAREHOUSE")
    missing = [n for n, v in [("DATABRICKS_HOST", host), ("DATABRICKS_TOKEN", token), ("DATABRICKS_WAREHOUSE", warehouse)] if not v]
    if missing:
        print(f"ERROR: missing required environment variable(s): {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    return sql.connect(server_hostname=host, http_path=warehouse, access_token=token)


def run_check(cursor, sql_text):
    cursor.execute(sql_text)
    rows = cursor.fetchall()
    return len(rows), rows[:5]


def status_label(violation_count, known_failing):
    if violation_count == 0:
        return "pass"
    return "fail (known_failing)" if known_failing else "fail"


def main():
    registry = yaml.safe_load(REGISTRY_PATH.read_text())
    conn = get_connection()
    cursor = conn.cursor()

    results = []
    for table_cfg in registry["tables"]:
        table = table_cfg["table"]

        for col_cfg in table_cfg.get("not_null_columns", []):
            column = col_cfg["column"]
            known_failing = col_cfg.get("known_failing", False)
            sql_text = f"SELECT * FROM {table} WHERE {column} IS NULL"
            count, _ = run_check(cursor, sql_text)
            results.append({
                "check": f"not_null({table}.{column})",
                "violations": count,
                "status": status_label(count, known_failing),
                "existing_asana_task": col_cfg.get("existing_asana_task"),
            })

        duplicate_key = table_cfg.get("duplicate_key")
        if duplicate_key:
            sql_text = (
                f"SELECT {duplicate_key}, COUNT(*) AS dupe_count FROM {table} "
                f"GROUP BY {duplicate_key} HAVING COUNT(*) > 1"
            )
            count, _ = run_check(cursor, sql_text)
            results.append({
                "check": f"duplicate({table}.{duplicate_key})",
                "violations": count,
                "status": status_label(count, False),
                "existing_asana_task": None,
            })

        for custom in table_cfg.get("custom_checks", []):
            known_failing = custom.get("known_failing", False)
            count, _ = run_check(cursor, custom["sql"])
            results.append({
                "check": f"{custom['id']}: {custom['title']}",
                "violations": count,
                "status": status_label(count, known_failing),
                "existing_asana_task": custom.get("existing_asana_task"),
            })

    cursor.close()
    conn.close()

    print(f"{'CHECK':<70} {'STATUS':<22} VIOLATIONS")
    for r in results:
        note = f" (see Asana {r['existing_asana_task']})" if r["existing_asana_task"] else ""
        print(f"{r['check']:<70} {r['status']:<22} {r['violations']}{note}")

    newly_failed = [r for r in results if r["status"] == "fail"]
    if newly_failed:
        print()
        print("FAILED (not known_failing):", ", ".join(r["check"] for r in newly_failed))
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
