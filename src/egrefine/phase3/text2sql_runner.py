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

    def __init__(
        self,
        config: dict,
        sample_rows: int = 3,
        column_descriptions: Optional[dict] = None,
    ):
        self.llm = LLMClient(config)
        self.sample_rows = sample_rows
        # Optional {(db_id, table, column): description} for the
        # description-baseline arms; injected as inline ``-- comment``s in the
        # DDL (same format as C3/DIN). None/empty leaves the prompt unchanged.
        self.column_descriptions = column_descriptions or {}

    def generate(
        self,
        nl: str,
        schema: Schema,
        db_path: Optional[str] = None,
        column_mapping: Optional[dict] = None,
        evidence: str = "",
    ) -> str:
        # RAW-log context so per-query calls can be attributed (db, question).
        self.llm.log_context.record = {"db_id": schema.db_id, "nl": nl,
                                       "method": "simple", "stage": "generation"}
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
            entries = []
            for col in table.columns:
                col_def = f"  {col.name} {col.dtype}"
                if col.is_pk:
                    col_def += " PRIMARY KEY"
                desc = self.column_descriptions.get((schema.db_id, table.name, col.name))
                entries.append((col_def, desc))

            # 添加 FK 约束
            for src_col, tgt_table, tgt_col in fk_by_table.get(table.name, []):
                entries.append((
                    f"  FOREIGN KEY ({src_col}) REFERENCES {tgt_table}({tgt_col})",
                    None,
                ))

            # Comma goes BEFORE the inline comment so it is not swallowed by
            # SQL's end-of-line comment syntax. Without descriptions this is
            # byte-identical to the previous ",\n".join(...) output.
            lines = []
            for i, (def_, desc) in enumerate(entries):
                sep = "," if i < len(entries) - 1 else ""
                lines.append(f"{def_}{sep}  -- {desc}" if desc else f"{def_}{sep}")

            parts.append(
                f"CREATE TABLE {table.name} (\n"
                + "\n".join(lines)
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
        if not str(db_path).startswith("mysql://"):
            return self._format_sample_rows(schema, db_path, reverse)
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

    def _format_sample_rows(
        self, schema: Schema, db_path: str, reverse: dict,
    ) -> str:
        """SQLite: show the first ``sample_rows`` REAL rows of each table.

        The per-column path above samples DISTINCT values of each column
        independently, so a printed "row" mixes values from different records
        (e.g. ``BUL | Eskimo``). Here every printed row is one actual record,
        as in the standard "CREATE TABLE + 3 rows" zero-shot prompt.
        Columns are selected by their database-side name (refined name if the
        object exposes it, else the mapped original name) and printed under
        the schema-facing name.
        """
        import sqlite3
        max_len = 100
        parts = []
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            for table in schema.tables:
                try:
                    existing = {r[1] for r in conn.execute(f'PRAGMA table_info("{table.name}")')}
                except sqlite3.Error:
                    continue
                cols, exprs = [], []
                for c in table.columns:
                    db_name = c.name if c.name in existing else reverse.get(c.name, c.name)
                    if db_name not in existing:
                        continue
                    cols.append(c.name)
                    exprs.append('"' + db_name.replace('"', '""') + '"')
                if not cols:
                    continue
                try:
                    rows = conn.execute(
                        f'SELECT {", ".join(exprs)} FROM "{table.name}" LIMIT {int(self.sample_rows)}'
                    ).fetchall()
                except sqlite3.Error as e:
                    logger.warning("Sample rows failed for %s: %s", table.name, e)
                    continue
                if not rows:
                    continue

                def fmt(v):
                    if v is None:
                        return "NULL"
                    s = v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
                    s = s.replace("\r", " ").replace("\n", " ")
                    return s[:max_len]

                header = " | ".join(cols)
                separator = "-+-".join("-" * max(len(c), 8) for c in cols)
                row_lines = [" | ".join(fmt(v) for v in r) for r in rows]
                parts.append(
                    f"/* {table.name} */\n{header}\n{separator}\n" + "\n".join(row_lines)
                )
        finally:
            conn.close()
        return "\n\n".join(parts)

    @staticmethod
    def _extract_sql(text: str) -> str:
        """从 LLM 回复中提取 SQL，处理 markdown 代码块 + reasoning CoT。"""
        text = strip_thinking(text)
        match = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
