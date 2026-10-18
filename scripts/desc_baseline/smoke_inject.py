#!/usr/bin/env python3
"""Smoke test: confirm C3 schema serializers inline-inject descriptions correctly.

Builds a fake Schema, instantiates C3 with a column_descriptions dict,
and prints the output of both serializers so a human can eyeball the format.
Exits non-zero if the inline `-- comment` is not present where expected.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from egrefine.data.schema import Schema, Table, Column
from egrefine.phase3.c3_runner import C3Text2SQL


def main():
    schema = Schema(
        db_id="world_1",
        tables=[
            Table(
                name="country",
                columns=[
                    Column(name="Code", table="country", dtype="TEXT", is_pk=True),
                    Column(name="Name", table="country", dtype="TEXT"),
                    Column(name="gf", table="country", dtype="TEXT"),
                    Column(name="hos", table="country", dtype="TEXT"),
                ],
            ),
            Table(
                name="city",
                columns=[
                    Column(name="ID", table="city", dtype="INTEGER", is_pk=True),
                    Column(name="cntry_code", table="city", dtype="TEXT"),
                ],
            ),
        ],
        foreign_keys=[("city.cntry_code", "country.Code")],
    )

    descriptions = {
        ("world_1", "country", "gf"): "The country's form of government, e.g. Republic, Federal Republic, Monarchy.",
        ("world_1", "country", "hos"): "The full name of the country's current head of state.",
        ("world_1", "city", "cntry_code"): "Three-letter country code referencing country.Code, e.g. USA, JPN.",
    }

    fake_cfg = {
        "base_url": "http://localhost:8000/v1",
        "api_key": "dummy",
        "model_name": "Qwen3.5-27B",
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    # --- Without descriptions (control) ---
    c3_plain = C3Text2SQL(fake_cfg)
    print("=" * 70)
    print("CONTROL: no descriptions")
    print("=" * 70)
    print("--- _format_compact_schema ---")
    plain_compact = c3_plain._format_compact_schema(schema)
    print(plain_compact)
    print("--- _format_ddl_schema ---")
    plain_ddl = c3_plain._format_ddl_schema(schema)
    print(plain_ddl)

    # --- With descriptions ---
    c3_desc = C3Text2SQL(fake_cfg, column_descriptions=descriptions)
    print()
    print("=" * 70)
    print("WITH DESCRIPTIONS")
    print("=" * 70)
    print("--- _format_compact_schema ---")
    desc_compact = c3_desc._format_compact_schema(schema)
    print(desc_compact)
    print("--- _format_ddl_schema ---")
    desc_ddl = c3_desc._format_ddl_schema(schema)
    print(desc_ddl)

    # --- Sanity assertions ---
    failures = []
    for needle in [
        "-- The country's form of government",
        "-- The full name of the country's current head of state",
        "-- Three-letter country code",
    ]:
        if needle not in desc_compact:
            failures.append(f"compact missing: {needle}")
        if needle not in desc_ddl:
            failures.append(f"ddl missing: {needle}")

    # Control should NOT have any -- comments (except FK lines maybe)
    if "-- The country" in plain_compact or "-- The country" in plain_ddl:
        failures.append("control unexpectedly contains description")

    # Columns without descriptions should NOT have -- comment
    if "-- " in desc_compact.split("country(")[1].split(")")[0]:
        # Check that "Name" line has no comment (it shouldn't)
        for line in desc_compact.split("\n"):
            if line.strip().startswith("Name") and "--" in line:
                failures.append("Name column unexpectedly got a -- comment")

    if failures:
        print()
        print("=" * 70)
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print()
        print("=" * 70)
        print("OK — all assertions passed.")


if __name__ == "__main__":
    main()
