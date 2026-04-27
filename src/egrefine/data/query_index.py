"""Query-Column 索引: 解析 gold SQL，构建 column → List[NLSQLPair] 的映射。"""
import re
import logging
from typing import Dict, List, Set

from egrefine.data.schema import Schema, Column, NLSQLPair

logger = logging.getLogger(__name__)


def extract_columns_from_sql(sql: str, schema: Schema) -> List[Column]:
    """从 SQL 中提取引用的列，匹配到 schema 中的 Column 对象。

    策略: 提取 SQL 中所有标识符，与 schema 列名做匹配。
    支持 BIRD 常见的 T1.col_name 别名模式。
    """
    # 收集 schema 中所有列名 -> Column 列表 (同名列可能在多表)
    name_to_cols: Dict[str, List[Column]] = {}
    for col in schema.all_columns:
        name_to_cols.setdefault(col.name, []).append(col)

    # 从 SQL 中提取标识符
    identifiers = _extract_identifiers(sql)

    matched: Dict[str, Column] = {}  # full_name -> Column, 去重
    for ident in identifiers:
        if ident in name_to_cols:
            for col in name_to_cols[ident]:
                matched[col.full_name] = col

    return list(matched.values())


def _extract_identifiers(sql: str) -> Set[str]:
    """从 SQL 中提取所有可能的列名标识符。"""
    # 先移除字符串常量，避免匹配字符串内容
    cleaned = re.sub(r"'[^']*'", "''", sql)

    identifiers = set()

    # 模式1: alias.column (如 T1.account_id, T2.frequency)
    for match in re.finditer(r'\b\w+\.(\w+)\b', cleaned):
        identifiers.add(match.group(1))

    # 模式2: 独立的标识符 (排除 SQL 关键字)
    sql_keywords = {
        'select', 'from', 'where', 'and', 'or', 'not', 'in', 'on',
        'join', 'inner', 'left', 'right', 'outer', 'cross', 'full',
        'as', 'is', 'null', 'like', 'between', 'exists', 'case',
        'when', 'then', 'else', 'end', 'group', 'by', 'order',
        'having', 'limit', 'offset', 'union', 'all', 'distinct',
        'insert', 'into', 'values', 'update', 'set', 'delete',
        'create', 'table', 'view', 'index', 'drop', 'alter',
        'asc', 'desc', 'count', 'sum', 'avg', 'min', 'max',
        'cast', 'float', 'integer', 'text', 'real', 'varchar',
        'int', 'char', 'date', 'datetime', 'boolean', 'blob',
        'primary', 'key', 'foreign', 'references', 'pragma',
        'true', 'false', 'if', 'iif', 'strftime', 'substr',
        'length', 'replace', 'trim', 'upper', 'lower', 'abs',
        'round', 'coalesce', 'nullif', 'typeof', 'instr',
        'group_concat', 'total', 'printf',
    }
    for match in re.finditer(r'\b([a-zA-Z_]\w*)\b', cleaned):
        word = match.group(1)
        if word.lower() not in sql_keywords:
            identifiers.add(word)

    return identifiers


def build_query_index(
    pairs: List[NLSQLPair],
    schema: Schema,
) -> Dict[str, List[NLSQLPair]]:
    """构建 column full_name → List[NLSQLPair] 索引，即论文中的 Q(c_i)。

    参数:
        pairs: NL-SQL pairs (应已按 db_id 过滤)
        schema: 该数据库的 Schema

    返回:
        {"table.column": [pair1, pair2, ...], ...}
    """
    index: Dict[str, List[NLSQLPair]] = {}

    for pair in pairs:
        cols = extract_columns_from_sql(pair.gold_sql, schema)
        for col in cols:
            index.setdefault(col.full_name, []).append(pair)

    # 日志统计
    all_cols = schema.all_columns
    covered = [c for c in all_cols if c.full_name in index]
    logger.info(
        "Query index for %s: %d/%d columns covered, %d total references",
        schema.db_id, len(covered), len(all_cols),
        sum(len(v) for v in index.values()),
    )

    return index
