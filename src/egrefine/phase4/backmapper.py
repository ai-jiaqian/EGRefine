"""确定性 back-mapping: 将 refined 列名替换回原始列名。"""
import re
from typing import Dict


def backmap(sql: str, reverse_mapping: Dict[str, str]) -> str:
    """将引用 refined 列名的 SQL 翻译回原始列名。

    使用 regex word boundary 避免子串误替换。
    按名字长度降序替换，确保长名字优先匹配。
    """
    if not reverse_mapping:
        return sql

    # 按长度降序排列，避免短名字先替换破坏长名字
    sorted_mapping = sorted(
        reverse_mapping.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    )

    result = sql
    for refined_name, original_name in sorted_mapping:
        pattern = r'\b' + re.escape(refined_name) + r'\b'
        result = re.sub(pattern, original_name, result)

    return result
