"""Database connection abstraction — supports SQLite (file) and MySQL."""
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# MySQL connector is optional — only needed for BEAVER
try:
    import mysql.connector
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False


def get_connection(db_ref: str, mysql_config: Optional[Dict] = None):
    """Create a database connection from a db reference string.

    Args:
        db_ref: Either a file path (SQLite) or "mysql://<database>" URI.
        mysql_config: MySQL connection params (host, user, password, port).
            Only used when db_ref starts with "mysql://".

    Returns:
        A database connection object (sqlite3.Connection or mysql.connector).
    """
    if db_ref.startswith("mysql://"):
        if not HAS_MYSQL:
            raise ImportError(
                "mysql-connector-python is required for MySQL support. "
                "Install with: pip install mysql-connector-python"
            )
        db_name = db_ref.replace("mysql://", "")
        cfg = mysql_config or {}
        return mysql.connector.connect(
            host=cfg.get("host", "localhost"),
            user=cfg.get("user", "root"),
            password=cfg.get("password", ""),
            port=cfg.get("port", 3306),
            database=db_name,
        )
    else:
        return sqlite3.connect(db_ref)


def is_mysql(db_ref: str) -> bool:
    return db_ref.startswith("mysql://")


def execute_sql(
    sql: str,
    db_ref: str,
    timeout: int = 30,
    mysql_config: Optional[Dict] = None,
) -> Optional[Set[Tuple]]:
    """Execute SQL and return result set (supports SQLite and MySQL).

    Returns:
        set of tuples (rows, ignoring order).
        None if execution fails.
    """
    try:
        if is_mysql(db_ref):
            conn = get_connection(db_ref, mysql_config)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = set(cursor.fetchall())
            cursor.close()
            conn.close()
            return results
        else:
            import time as _time
            conn = sqlite3.connect(db_ref)
            conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")
            # Real query execution timeout via progress_handler
            _deadline = _time.monotonic() + timeout
            def _check_timeout():
                if _time.monotonic() > _deadline:
                    return 1  # non-zero cancels the query
                return 0
            conn.set_progress_handler(_check_timeout, 1000)
            cursor = conn.execute(sql)
            results = set(cursor.fetchall())
            conn.close()
            return results
    except Exception as e:
        logger.debug("SQL execution failed: %s\nSQL: %s", e, sql[:200])
        return None


def _normalize_row(row: tuple) -> tuple:
    """Normalize a row for order-independent column comparison."""
    return tuple(sorted(str(v) for v in row))


def compare_results(
    pred_sql: str,
    gold_sql: str,
    db_ref: str,
    mysql_config: Optional[Dict] = None,
) -> bool:
    """Compare execution results of two SQL queries (ExAcc single judgment).

    Ignores row order (set comparison) and column order within each row
    (sort values before comparing), matching standard Spider/BIRD evaluation.
    """
    pred_result = execute_sql(pred_sql, db_ref, mysql_config=mysql_config)
    gold_result = execute_sql(gold_sql, db_ref, mysql_config=mysql_config)

    if pred_result is None or gold_result is None:
        return False

    # Fast path: exact match (same column order)
    if pred_result == gold_result:
        return True

    # Slow path: normalize column order within each row
    pred_normalized = set(_normalize_row(r) for r in pred_result)
    gold_normalized = set(_normalize_row(r) for r in gold_result)
    return pred_normalized == gold_normalized


def sample_column(
    db_ref: str,
    table_name: str,
    column_name: str,
    n: int = 20,
    mysql_config: Optional[Dict] = None,
) -> List[str]:
    """Sample N distinct non-NULL values from a column (SQLite or MySQL).

    Returns:
        List of string values. Empty list on failure.
    """
    MAX_VALUE_LENGTH = 100
    try:
        if is_mysql(db_ref):
            conn = get_connection(db_ref, mysql_config)
            cursor = conn.cursor()
            # Validate column exists
            cursor.execute(
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
                (table_name,)
            )
            valid_cols = {row[0] for row in cursor.fetchall()}
            if column_name not in valid_cols:
                cursor.close()
                conn.close()
                logger.warning("Column %s not found in table %s", column_name, table_name)
                return []
            # Use backtick quoting for MySQL
            query = (
                f"SELECT DISTINCT `{column_name}` FROM `{table_name}` "
                f"WHERE `{column_name}` IS NOT NULL "
                f"LIMIT {n}"
            )
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
        else:
            conn = sqlite3.connect(db_ref)
            cursor = conn.execute(f'PRAGMA table_info("{table_name}")')
            valid_cols = {row[1] for row in cursor.fetchall()}
            if column_name not in valid_cols:
                conn.close()
                logger.warning("Column %s not found in table %s", column_name, table_name)
                return []
            query = (
                f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
                f'WHERE "{column_name}" IS NOT NULL '
                f'LIMIT {n}'
            )
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            conn.close()

    except Exception as e:
        logger.warning(
            "Failed to sample %s.%s from %s: %s",
            table_name, column_name, db_ref, e,
        )
        return []

    values = []
    for (val,) in rows:
        s = str(val)
        if len(s) > MAX_VALUE_LENGTH:
            s = s[:MAX_VALUE_LENGTH] + "..."
        values.append(s)

    return values


def load_schema_from_mysql(
    db_id: str,
    db_name: str,
    mysql_config: Optional[Dict] = None,
):
    """Load Schema from a MySQL database using INFORMATION_SCHEMA."""
    from egrefine.data.schema import Column, Table, Schema

    conn = get_connection(f"mysql://{db_name}", mysql_config)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute(
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
        "WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'",
        (db_name,)
    )
    table_names = [row[0] for row in cursor.fetchall()]

    tables = []
    all_fks: List[Tuple[str, str]] = []

    for tname in table_names:
        # Column info
        cursor.execute(
            "SELECT COLUMN_NAME, DATA_TYPE, COLUMN_KEY "
            "FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s "
            "ORDER BY ORDINAL_POSITION",
            (db_name, tname),
        )
        col_rows = cursor.fetchall()

        # Foreign keys
        cursor.execute(
            "SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME "
            "FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s "
            "AND REFERENCED_TABLE_NAME IS NOT NULL",
            (db_name, tname),
        )
        fk_rows = cursor.fetchall()
        fk_map = {}
        for fk_col, ref_table, ref_col in fk_rows:
            fk_map[fk_col] = f"{ref_table}.{ref_col}"
            all_fks.append((f"{tname}.{fk_col}", f"{ref_table}.{ref_col}"))

        columns = []
        for cname, dtype, col_key in col_rows:
            is_pk = col_key == "PRI"
            fk_target = fk_map.get(cname)
            columns.append(Column(
                name=cname,
                table=tname,
                dtype=dtype.upper(),
                is_pk=is_pk,
                fk_target=fk_target,
            ))

        tables.append(Table(name=tname, columns=columns))

    cursor.close()
    conn.close()
    return Schema(db_id=db_id, tables=tables, foreign_keys=all_fks)
