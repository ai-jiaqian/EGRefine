"""Phase 2: Data Sampler — 从数据库采样列数据（支持 SQLite 和 MySQL）。"""
import logging
from typing import Dict, List, Optional

from egrefine.data.db_connection import sample_column as _db_sample_column

logger = logging.getLogger(__name__)

# Re-export for backward compatibility with tests
MAX_VALUE_LENGTH = 100

# Module-level MySQL config, set by experiment scripts when using BEAVER.
_mysql_config: Optional[Dict] = None


def set_mysql_config(config: Dict) -> None:
    """Set MySQL connection params for this module."""
    global _mysql_config
    _mysql_config = config


def sample_column(
    db_path: str,
    table_name: str,
    column_name: str,
    n: int = 20,
) -> List[str]:
    """从数据库中采样指定列的 N 行非 NULL 数据。

    Args:
        db_path: SQLite 文件路径 或 "mysql://<database>" URI
        table_name: 表名
        column_name: 列名
        n: 采样行数（默认 20）

    Returns:
        字符串列表，NULL 被过滤，超长字符串被截断。
    """
    return _db_sample_column(
        db_path, table_name, column_name, n, mysql_config=_mysql_config,
    )


def sample_table_columns(
    db_path: str,
    table_name: str,
    column_names: List[str],
    n: int = 20,
) -> dict:
    """批量采样同一表的多个列。

    Returns:
        {column_name: [sample_values]}
    """
    return {
        col: sample_column(db_path, table_name, col, n)
        for col in column_names
    }
