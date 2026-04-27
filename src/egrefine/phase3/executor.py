"""SQL 执行与结果比对 — 支持 SQLite 和 MySQL。"""
import logging
from typing import Dict, Optional, Set, Tuple

from egrefine.data.db_connection import (
    execute_sql as _db_execute_sql,
    compare_results as _db_compare_results,
)

logger = logging.getLogger(__name__)

# Module-level MySQL config, set by experiment scripts when using BEAVER.
_mysql_config: Optional[Dict] = None


def set_mysql_config(config: Dict) -> None:
    """Set MySQL connection params for this module."""
    global _mysql_config
    _mysql_config = config


def execute_sql(sql: str, db_path: str, timeout: int = 30) -> Optional[Set[Tuple]]:
    """执�� SQL 并返回 result set。

    Args:
        sql: SQL query string.
        db_path: SQLite file path or "mysql://<database>" URI.
        timeout: Timeout in seconds (SQLite only).

    返回:
        set of tuples（行的集合，忽略顺序）
        如果执行失败（语法错误/超时）返回 None
    """
    return _db_execute_sql(sql, db_path, timeout=timeout, mysql_config=_mysql_config)


def compare_results(pred_sql: str, gold_sql: str, db_path: str) -> bool:
    """比较两条 SQL 的执行结果是否一致（ExAcc 单条判断）。"""
    return _db_compare_results(pred_sql, gold_sql, db_path, mysql_config=_mysql_config)
