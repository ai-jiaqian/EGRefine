"""Stage 2 database setup utilities.

Provides helpers for the evaluation pipeline:
- copy_database: duplicate a SQLite file (or export MySQL → SQLite) for safe mutation
- apply_views: execute a views.sql script on a database
- remap_gold_sql: rewrite table references in gold SQL to point at backing tables
"""

from __future__ import annotations

import logging
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def copy_database(
    src_path: str | Path,
    dest_path: str | Path,
    mysql_config: Optional[Dict] = None,
) -> Path:
    """Copy a database to *dest_path* and return it (always as a SQLite file).

    - If *src_path* is a SQLite file path: bit-for-bit copy via shutil.
    - If *src_path* is a ``mysql://<db_name>`` URI: export schema + data from
      MySQL into a fresh SQLite file at *dest_path*.  Types are coarsely mapped
      (int→INTEGER, real→REAL, blob→BLOB, everything else→TEXT).

    Parent directories of *dest_path* are created automatically.
    """
    src_str = str(src_path)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if src_str.startswith("mysql://"):
        return _export_mysql_to_sqlite(src_str, dest_path, mysql_config or {})

    shutil.copy2(src_str, dest_path)
    return dest_path


def _export_mysql_to_sqlite(mysql_uri: str, dest: Path, cfg: Dict) -> Path:
    """Export a MySQL database referenced by ``mysql://<db_name>`` to SQLite.

    Uses coarse MySQL→SQLite type mapping. Foreign keys and indexes are NOT
    preserved (not needed for ExAcc evaluation, only schema+data correctness).
    """
    try:
        import mysql.connector
    except ImportError as e:
        raise ImportError(
            "mysql-connector-python is required for MySQL → SQLite export. "
            "Install with: pip install mysql-connector-python"
        ) from e

    db_name = mysql_uri.replace("mysql://", "", 1)

    if dest.exists():
        dest.unlink()

    my_conn = mysql.connector.connect(
        host=cfg.get("host", "localhost"),
        user=cfg.get("user", "root"),
        password=cfg.get("password", ""),
        port=cfg.get("port", 3306),
        database=db_name,
    )
    sq_conn = sqlite3.connect(str(dest))

    try:
        my_cur = my_conn.cursor()
        sq_cur = sq_conn.cursor()

        my_cur.execute("SHOW TABLES")
        tables = [row[0] for row in my_cur.fetchall()]
        logger.info("Exporting MySQL db=%s → SQLite %s (%d tables)",
                    db_name, dest, len(tables))

        for t in tables:
            # Describe columns
            my_cur.execute(f"DESCRIBE `{t}`")
            cols = my_cur.fetchall()  # (Field, Type, Null, Key, Default, Extra)

            col_defs = []
            col_names = []
            for c in cols:
                name = c[0]
                mysql_type = c[1]
                if isinstance(mysql_type, (bytes, bytearray)):
                    mysql_type = mysql_type.decode("utf-8", errors="ignore")
                sqlite_type = _mysql_to_sqlite_type(mysql_type)
                col_defs.append(f'"{name}" {sqlite_type}')
                col_names.append(f'"{name}"')

            create_stmt = f'CREATE TABLE "{t}" ({", ".join(col_defs)})'
            sq_cur.execute(create_stmt)

            # Bulk copy data (streaming)
            my_cur.execute(f"SELECT {', '.join(f'`{c[0]}`' for c in cols)} FROM `{t}`")
            placeholders = ",".join("?" * len(cols))
            insert_stmt = f'INSERT INTO "{t}" ({", ".join(col_names)}) VALUES ({placeholders})'
            batch = []
            for row in my_cur:
                # Normalize: bytes → str, datetime/date → iso string (SQLite TEXT)
                batch.append(tuple(_normalize_value(v) for v in row))
                if len(batch) >= 1000:
                    sq_cur.executemany(insert_stmt, batch)
                    batch.clear()
            if batch:
                sq_cur.executemany(insert_stmt, batch)

        sq_conn.commit()
    finally:
        sq_conn.close()
        my_conn.close()

    return dest


def _mysql_to_sqlite_type(mysql_type: str) -> str:
    """Coarse MySQL → SQLite type affinity mapping (sufficient for ExAcc eval)."""
    mt = mysql_type.lower()
    # Integer affinity
    if any(kw in mt for kw in ("tinyint", "smallint", "mediumint", "bigint", "int")):
        return "INTEGER"
    # Real affinity
    if any(kw in mt for kw in ("float", "double", "decimal", "numeric", "real")):
        return "REAL"
    # Blob
    if any(kw in mt for kw in ("blob", "binary", "varbinary")):
        return "BLOB"
    # Everything else (varchar, char, text, datetime, date, time, enum, etc.) → TEXT
    return "TEXT"


def _normalize_value(v):
    """Normalize MySQL row values for SQLite insertion."""
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            return v  # keep as bytes for BLOB
    # datetime, date, time → str
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def apply_views(db_path: str | Path, views_sql_path: str | Path) -> None:
    """Read *views_sql_path* and execute its contents on *db_path*.

    The SQL file is executed via ``executescript()`` so it may contain
    multiple statements, transactions (BEGIN / COMMIT), etc.
    """
    db_path = Path(db_path)
    views_sql_path = Path(views_sql_path)
    sql_text = views_sql_path.read_text(encoding="utf-8")
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(sql_text)
    finally:
        conn.close()


def remap_gold_sql(sql: str, table_map: Dict[str, str]) -> str:
    """Replace original table names in *sql* with their backing names.

    *table_map* maps original table names to backing names, e.g.
    ``{"district": "_orig_district"}``.

    Only replaces table references (after FROM/JOIN, before AS/ON, or as
    ``table.column`` qualifier), NOT column references.  This avoids
    incorrect substitution when a table name coincides with a column name
    (e.g. the ``gender`` table has a ``gender`` column).

    Replacements are applied longest-first to prevent substring collisions.
    """
    if not table_map:
        return sql

    sorted_items = sorted(table_map.items(), key=lambda kv: len(kv[0]), reverse=True)

    for original, backing in sorted_items:
        esc = re.escape(original)
        # 1) FROM / JOIN <table>  — bare names, quoted ("Match"), case-insensitive
        sql = re.sub(
            r'(?i)(\b(?:FROM|JOIN)\b\s+)"?' + esc + r'"?(?=\b|\s)',
            r"\g<1>" + backing, sql,
        )
        # 2) table.column  →  _orig_table.column  (dot immediately follows)
        sql = re.sub(
            r'(?i)"?\b' + esc + r'\b"?(?=\s*\.)',
            backing, sql,
        )

    return sql
