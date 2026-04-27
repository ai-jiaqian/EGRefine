"""Text-to-SQL 系统封装"""
import re
import logging
from abc import ABC, abstractmethod
from typing import Optional

from egrefine.data.schema import Schema
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.sampler import sample_column

logger = logging.getLogger(__name__)


def strip_thinking(text: str) -> str:
    """Strip reasoning/thinking prefix from model output.

    Reasoning models (MiniMax M2.7, GLM-5.1, DeepSeek R1, etc.) wrap their
    internal reasoning in a `...</think>` tag and put the final answer after
    it. The CoT portion often contains draft code blocks (e.g. exploring
    "Alternative: using IN subquery...") that are NOT the final answer.
    Regex matchers like r"```sql ...```" grab the FIRST block by default,
    which would pick up the draft instead of the final answer.

    This helper returns text with everything up to and including the last
    </think> tag removed, so downstream extractors only see the final answer.
    If no </think> tag is present (non-reasoning models), the text is
    returned unchanged.
    """
    if not text:
        return text
    idx = text.rfind("</think>")
    if idx < 0:
        return text
    return text[idx + len("</think>"):]


class Text2SQLModel(ABC):
    @abstractmethod
    def generate(
        self,
        nl: str,
        schema: Schema,
        db_path: Optional[str] = None,
        column_mapping: Optional[dict] = None,
        evidence: str = "",
    ) -> str:
        """输入自然语言问题和 schema，返回预测的 SQL 查询。

        Args:
            nl: 自然语言问题
            schema: 数据库 schema
            db_path: SQLite 数据库路径（用于采样数据，可选）
            column_mapping: {new_name: original_name} 映射，用于采样时查找原始列名
            evidence: BIRD-style 业务背景/定义提示，空字符串时忽略
        """
        pass


class SimpleLLMText2SQL(Text2SQLModel):
    """Text-to-SQL: 把 schema + 样本数据 + question 发给 LLM。"""

    def __init__(self, config: dict, sample_rows: int = 3):
        self.llm = LLMClient(config)
        self.sample_rows = sample_rows

    def generate(
        self,
        nl: str,
        schema: Schema,
        db_path: Optional[str] = None,
        column_mapping: Optional[dict] = None,
        evidence: str = "",
    ) -> str:
        schema_str = self._format_schema(schema)

        # 采样数据辅助理解 schema
        sample_str = ""
        if db_path and self.sample_rows > 0:
            sample_str = self._format_samples(schema, db_path, column_mapping)

        prompt = (
            f"Given the following database schema:\n"
            f"{schema_str}\n"
        )
        if sample_str:
            prompt += f"\nSample data from each table:\n{sample_str}\n"
        if evidence:
            prompt += f"\nAdditional context (evidence): {evidence}\n"
        prompt += (
            f"\nGenerate a SQL query to answer: {nl}\n\n"
            f"Return only the SQL query, no explanation."
        )
        response = self.llm.chat([{"role": "user", "content": prompt}])
        return self._extract_sql(response)

    def _format_schema(self, schema: Schema) -> str:
        """将 Schema 格式化为 CREATE TABLE DDL 语句。"""
        parts = []
        # 收集每个表的 FK 信息
        fk_by_table = {}
        for src, tgt in schema.foreign_keys:
            src_table, src_col = src.split(".", 1)
            tgt_table, tgt_col = tgt.split(".", 1)
            fk_by_table.setdefault(src_table, []).append((src_col, tgt_table, tgt_col))

        for table in schema.tables:
            col_defs = []
            for col in table.columns:
                col_def = f"  {col.name} {col.dtype}"
                if col.is_pk:
                    col_def += " PRIMARY KEY"
                col_defs.append(col_def)

            # 添加 FK 约束
            for src_col, tgt_table, tgt_col in fk_by_table.get(table.name, []):
                col_defs.append(
                    f"  FOREIGN KEY ({src_col}) REFERENCES {tgt_table}({tgt_col})"
                )

            parts.append(
                f"CREATE TABLE {table.name} (\n"
                + ",\n".join(col_defs)
                + "\n);"
            )

        return "\n\n".join(parts)

    def _format_samples(
        self, schema: Schema, db_path: str, column_mapping: Optional[dict] = None,
    ) -> str:
        """采样每个表的前几行数据，格式化为可读文本。

        column_mapping: {new_col_name: original_col_name}，用于从数据库中
        按原始列名采样，但在 prompt 中展示新列名。
        """
        reverse = column_mapping or {}
        parts = []
        for table in schema.tables:
            col_names = [c.name for c in table.columns]
            # 采样每列（用原始列名查数据库）
            col_samples = {}
            for col in table.columns:
                db_col_name = reverse.get(col.name, col.name)
                col_samples[col.name] = sample_column(
                    db_path, table.name, db_col_name, n=self.sample_rows
                )
            # 转成行格式
            n_rows = max((len(v) for v in col_samples.values()), default=0)
            if n_rows == 0:
                continue
            header = " | ".join(col_names)
            separator = "-+-".join("-" * max(len(c), 8) for c in col_names)
            row_lines = []
            for i in range(min(n_rows, self.sample_rows)):
                vals = []
                for c in col_names:
                    samples = col_samples.get(c, [])
                    vals.append(samples[i] if i < len(samples) else "NULL")
                row_lines.append(" | ".join(vals))
            parts.append(
                f"/* {table.name} */\n{header}\n{separator}\n"
                + "\n".join(row_lines)
            )
        return "\n\n".join(parts)

    @staticmethod
    def _extract_sql(text: str) -> str:
        """从 LLM 回复中提取 SQL，处理 markdown 代码块 + reasoning CoT。"""
        text = strip_thinking(text)
        match = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
