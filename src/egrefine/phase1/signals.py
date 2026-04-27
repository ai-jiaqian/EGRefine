"""Phase 1: 四个 Heuristic Signals 用于搜索空间剪枝。"""
import re
import logging
from typing import Dict, List, Set, Tuple

from egrefine.data.schema import Column, Schema

logger = logging.getLogger(__name__)

# S4: 高频泛化词汇表
GENERIC_VOCABULARY: Set[str] = {
    "status", "type", "code", "value", "flag", "name", "num", "desc",
    "id", "date", "text", "info", "data", "level", "state", "category",
    "group", "class", "kind", "mode", "label", "title", "result",
    "count", "amount", "total", "number", "index", "key", "note",
    "comment", "remark",
}


# ========== 命名风格检测 ==========

def _detect_style(name: str) -> str:
    """检测单个列名的命名风格。

    返回: "ALLCAPS", "camelCase", "snake_case", "alllower", "other"
    """
    # 全大写 (允许下划线分隔的全大写如 FIRST_NAME)
    if re.match(r'^[A-Z][A-Z0-9_]*$', name) and len(name) > 1:
        return "ALLCAPS"

    # camelCase: 以小写开头，包含大写字母
    if re.match(r'^[a-z]', name) and re.search(r'[A-Z]', name):
        return "camelCase"

    # snake_case: 包含下划线，且有小写字母
    if '_' in name and re.search(r'[a-z]', name):
        return "snake_case"

    # 全小写无分隔符
    if re.match(r'^[a-z][a-z0-9]*$', name):
        return "alllower"

    return "other"


# ========== Signal 函数 ==========

def s1_short_name(column: Column, max_length: int = 3) -> bool:
    """S1: 列名过短（长度 <= max_length）。"""
    return len(column.name) <= max_length


def s2_high_similarity(
    column: Column,
    similarity_matrix: Dict[Tuple[str, str], float],
    threshold: float = 0.85,
) -> bool:
    """S2: 与同 schema 内另一列的 embedding similarity 超过阈值。"""
    full = column.full_name
    for (a, b), sim in similarity_matrix.items():
        if sim >= threshold and (a == full or b == full):
            return True
    return False


def s3_naming_inconsistency(schema: Schema) -> List[Column]:
    """S3: 检测同一 schema 内混用多种命名风格，标记少数派风格的列。

    如果整个 schema 只有一种风格（或不足 2 种），返回空列表。

    返回: 少数派风格的列列表
    """
    columns = schema.all_columns
    if len(columns) < 2:
        return []

    # 统计每种风格的列
    style_groups: Dict[str, List[Column]] = {}
    for col in columns:
        style = _detect_style(col.name)
        if style == "other":
            continue
        style_groups.setdefault(style, []).append(col)

    # 过滤掉只有一个列的风格组（太少不算一种"风格"）
    significant_styles = {s: cols for s, cols in style_groups.items() if len(cols) >= 1}

    if len(significant_styles) < 2:
        return []

    # 找出主流风格（列数最多的）
    majority_style = max(significant_styles, key=lambda s: len(significant_styles[s]))

    # 少数派 = 非主流风格的所有列
    minority_columns = []
    for style, cols in significant_styles.items():
        if style != majority_style:
            minority_columns.extend(cols)

    return minority_columns


def s4_generic_vocabulary(column: Column) -> bool:
    """S4: 列名属于高频泛化词。

    完全匹配（忽略大小写）。
    """
    return column.name.lower() in GENERIC_VOCABULARY
