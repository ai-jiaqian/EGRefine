"""Phase 4: VIEW Synthesis — ALTER TABLE RENAME + CREATE VIEW strategy.

Strategy: For each table with refined columns, rename the original table to
``_orig_<table>`` and create a VIEW with the original table name that exposes
the refined column names.  This way downstream Text-to-SQL models see the
original table names with improved column names.
"""
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from egrefine.data.schema import Schema
from egrefine.phase3.scorer import SelectionResult

logger = logging.getLogger(__name__)

BACKING_PREFIX = "_orig_"


def _build_rename_map(
    results: List[SelectionResult],
    schema: Optional[Schema] = None,
) -> Dict[str, str]:
    """Build full rename map including PK→FK propagation.

    Returns dict mapping 'table.col' -> new_name for all directly changed
    columns plus FK columns whose referenced PK was renamed.
    """
    rename_map: Dict[str, str] = {}
    for r in results:
        if r.was_changed:
            rename_map[r.column.full_name] = r.selected_name

    if schema:
        pk_renames = {
            full: name for full, name in rename_map.items()
            if any(r.column.full_name == full and r.column.is_pk for r in results)
        }
        if pk_renames:
            for table in schema.tables:
                # Collect existing column names in this table (including
                # already-renamed ones) so we can detect conflicts.
                existing_names = set()
                for c in table.columns:
                    existing_names.add(
                        rename_map.get(c.full_name, c.name).lower()
                    )

                for col in table.columns:
                    if col.fk_target and col.fk_target in pk_renames:
                        new_name = pk_renames[col.fk_target]
                        # Skip propagation if the new name would collide with
                        # an existing column in the same table (case-insensitive).
                        if new_name.lower() in existing_names:
                            logger.warning(
                                "PK→FK propagation skipped: %s → '%s' "
                                "conflicts with existing column in %s",
                                col.full_name, new_name, table.name,
                            )
                            continue
                        rename_map[col.full_name] = new_name

    return rename_map


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def generate_views(
    schema: Schema,
    results: List[SelectionResult],
) -> List[str]:
    """Generate ALTER TABLE RENAME + CREATE VIEW scripts for changed tables.

    Each element is a complete SQL script for one table:

        ALTER TABLE <table> RENAME TO _orig_<table>;
        CREATE VIEW <table> AS
        SELECT
          <col_or_alias>, ...
        FROM _orig_<table>;

    Only tables that have at least one changed column are included.

    Returns:
        List of SQL script strings (one per changed table).
    """
    rename_map = _build_rename_map(results, schema)

    # Group renames by table
    changes_by_table: Dict[str, Dict[str, str]] = {}
    for full_name, new_name in rename_map.items():
        table, col = full_name.split(".", 1)
        changes_by_table.setdefault(table, {})[col] = new_name

    views: List[str] = []
    for table in schema.tables:
        if table.name not in changes_by_table:
            continue

        backing_name = f"{BACKING_PREFIX}{table.name}"
        col_map = changes_by_table[table.name]

        select_parts: List[str] = []
        for col in table.columns:
            if col.name in col_map:
                select_parts.append(f'  "{col.name}" AS "{col_map[col.name]}"')
            else:
                select_parts.append(f'  "{col.name}"')

        select_clause = ",\n".join(select_parts)

        script = (
            f"BEGIN;\n"
            f'ALTER TABLE "{table.name}" RENAME TO "{backing_name}";\n'
            f'CREATE VIEW "{table.name}" AS\n'
            f"SELECT\n"
            f"{select_clause}\n"
            f'FROM "{backing_name}";\n'
            f"COMMIT;"
        )
        views.append(script)

    return views


def generate_orig_table_map(
    results: List[SelectionResult],
    schema: Optional[Schema] = None,
) -> Dict[str, str]:
    """Return mapping from original table name to backing table name.

    Only tables that have at least one changed column (including PK→FK
    propagation) are included.

    Example: ``{"district": "_orig_district"}``
    """
    rename_map = _build_rename_map(results, schema)
    tables_with_changes = {k.split(".")[0] for k in rename_map}
    return {t: f"{BACKING_PREFIX}{t}" for t in sorted(tables_with_changes)}


def generate_refined_tables_json(
    schema: Schema,
    results: List[SelectionResult],
) -> dict:
    """Return a dict describing the full refined schema.

    All tables are included (not just changed ones).  For each column, the
    refined name is used when applicable; ``original_name`` preserves the
    original.  Foreign keys use refined column names.

    Format::

        {
            "db_id": "...",
            "tables": [
                {
                    "name": "district",
                    "columns": [
                        {"name": "city_name", "original_name": "A2",
                         "dtype": "TEXT", "is_pk": false},
                        ...
                    ]
                },
                ...
            ],
            "foreign_keys": [["account.d_id", "district.d_id"], ...]
        }
    """
    rename_map = _build_rename_map(results, schema)

    tables_out: List[dict] = []
    for table in schema.tables:
        cols_out: List[dict] = []
        for col in table.columns:
            full = col.full_name
            refined_name = rename_map.get(full, col.name)
            cols_out.append({
                "name": refined_name,
                "original_name": col.name,
                "dtype": col.dtype,
                "is_pk": col.is_pk,
            })
        tables_out.append({"name": table.name, "columns": cols_out})

    # Translate FK references to refined names
    fks_out: List[List[str]] = []
    for fk1, fk2 in schema.foreign_keys:
        refined_fk1 = _refine_fk_ref(fk1, rename_map)
        refined_fk2 = _refine_fk_ref(fk2, rename_map)
        fks_out.append([refined_fk1, refined_fk2])

    return {
        "db_id": schema.db_id,
        "tables": tables_out,
        "foreign_keys": fks_out,
    }


def generate_mapping(
    results: List[SelectionResult],
    schema: Optional[Schema] = None,
) -> Dict[str, str]:
    """Generate forward mapping: 'table.original_col' -> 'new_name'.

    Includes changed columns and FK columns affected by PK→FK propagation.
    """
    return _build_rename_map(results, schema)


# ---------------------------------------------------------------------------
# Full synthesis
# ---------------------------------------------------------------------------

def synthesize(
    schema: Schema,
    results: List[SelectionResult],
    output_dir: Optional[str] = None,
) -> dict:
    """Full Phase 4 synthesis.

    Produces:
        - views: list of ALTER+CREATE VIEW scripts
        - mapping / reverse_mapping
        - orig_table_map
        - refined_tables: full refined schema description
        - statistics

    When *output_dir* is provided the following files are written:
        views.sql, refined_tables.json, orig_table_map.json, statistics.json
    """
    views = generate_views(schema, results)
    mapping = generate_mapping(results, schema)
    reverse_mapping = {v: k.split(".")[-1] for k, v in mapping.items()}
    orig_table_map = generate_orig_table_map(results, schema)
    refined_tables = generate_refined_tables_json(schema, results)

    changed = [r for r in results if r.was_changed]
    unchanged = [r for r in results if not r.was_changed]
    tables_with_views = len({r.column.table for r in changed})

    statistics = {
        "columns_refined": len(changed),
        "columns_kept_original": len(unchanged),
        "tables_with_views": tables_with_views,
        "avg_delta": sum(r.delta for r in changed) / len(changed) if changed else 0.0,
        "fallback_count": sum(1 for r in results if r.verification_method == "llm_fallback"),
    }

    result = {
        "views": views,
        "mapping": mapping,
        "reverse_mapping": reverse_mapping,
        "orig_table_map": orig_table_map,
        "refined_tables": refined_tables,
        "statistics": statistics,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "views.sql"), "w") as f:
            f.write("\n\n".join(views))

        with open(os.path.join(output_dir, "refined_tables.json"), "w") as f:
            json.dump(refined_tables, f, indent=2)

        with open(os.path.join(output_dir, "orig_table_map.json"), "w") as f:
            json.dump(orig_table_map, f, indent=2)

        with open(os.path.join(output_dir, "statistics.json"), "w") as f:
            json.dump(statistics, f, indent=2)

        logger.info(
            "Saved views.sql, refined_tables.json, orig_table_map.json, "
            "statistics.json to %s",
            output_dir,
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _refine_fk_ref(fk_ref: str, rename_map: Dict[str, str]) -> str:
    """Translate a FK reference like 'table.col' using the rename map."""
    if fk_ref in rename_map:
        table_part = fk_ref.split(".")[0]
        return f"{table_part}.{rename_map[fk_ref]}"
    return fk_ref
