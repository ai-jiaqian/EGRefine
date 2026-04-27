"""MAC-SQL: Multi-Agent Collaborative Framework for Text-to-SQL.

Implements Wang et al. 2024 as a Text2SQLModel with three agents:
  Selector  — schema linking, prune irrelevant tables/columns
  Decomposer — sub-question decomposition + SQL generation
  Refiner   — iterative SQL correction with execution feedback
"""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from egrefine.data.schema import Schema, Table, Column
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.sampler import sample_column
from egrefine.phase3.executor import execute_sql
from egrefine.phase3.text2sql_runner import Text2SQLModel, strip_thinking

logger = logging.getLogger(__name__)

# Maximum refinement iterations
DEFAULT_MAX_REFINE_ROUNDS = 2


class MACSQLText2SQL(Text2SQLModel):
    """MAC-SQL (Multi-Agent Collaborative) Text-to-SQL.

    Three-agent pipeline:
    1. Selector: prune schema to relevant tables/columns
    2. Decomposer: decompose question + generate SQL
    3. Refiner: iterative correction with execution feedback
    """

    def __init__(
        self,
        config: dict,
        sample_rows: int = 3,
        max_refine_rounds: int = DEFAULT_MAX_REFINE_ROUNDS,
    ):
        """
        Args:
            config: LLM config (base_url, api_key, model_name, etc.).
            sample_rows: Number of sample rows per column for schema context.
            max_refine_rounds: Maximum Refiner iterations (0 to disable).
        """
        self.llm = LLMClient(config)
        self.sample_rows = sample_rows
        self.max_refine_rounds = max_refine_rounds

    def generate(
        self,
        nl: str,
        schema: Schema,
        db_path: Optional[str] = None,
        column_mapping: Optional[dict] = None,
        evidence: str = "",
    ) -> str:
        """Generate SQL using 3-agent MAC-SQL pipeline."""
        schema_ddl = self._format_ddl(schema, db_path, column_mapping)

        # Agent 1: Selector — prune schema
        pruned_schema_ddl = self._select(nl, schema_ddl, evidence, schema=schema)

        # Agent 2: Decomposer — generate SQL
        sql = self._decompose(nl, pruned_schema_ddl, evidence)

        # Agent 3: Refiner — iterative correction with execution feedback
        if self.max_refine_rounds > 0 and db_path:
            sql = self._refine(nl, pruned_schema_ddl, sql, db_path, evidence)

        return sql

    # ------------------------------------------------------------------ #
    # Agent 1: Selector
    # ------------------------------------------------------------------ #

    SELECTOR_PROMPT = """\
You are a database schema analyst. Your ONLY job is to select relevant \
tables and columns from the schema — do NOT write any SQL queries.

Given a database schema and a question, output a JSON object with:
- "tables": list of relevant table names
- "columns": list of relevant columns as "table_name.column_name"

Rules:
1. Include tables needed for JOINs (even if not directly referenced)
2. Include primary key and foreign key columns for JOIN paths
3. Only include columns that are needed for the query
4. Return ONLY the JSON object, nothing else"""

    def _select(
        self, nl: str, schema_ddl: str, hint: str,
        schema: Optional[Schema] = None,
    ) -> str:
        """Selector agent: prune schema to relevant tables/columns.

        Returns pruned schema DDL. Falls back to full schema on parse failure.
        """
        user_content = f"Database Schema:\n{schema_ddl}\n\n"
        if hint:
            user_content += f"Hint: {hint}\n\n"
        user_content += f"Question: {nl}"

        messages = [
            {"role": "system", "content": self.SELECTOR_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = self.llm.chat(messages)

        # Try to parse JSON response and build filtered DDL
        if schema is not None:
            filtered = self._parse_selector_json(response, schema)
            if filtered:
                return filtered

        # Fallback: check if response is valid DDL
        ddl = self._extract_ddl(response)
        if "CREATE TABLE" in ddl.upper():
            return ddl

        logger.warning("Selector parse failed, using full schema")
        return schema_ddl

    def _parse_selector_json(
        self, response: str, schema: Schema,
    ) -> Optional[str]:
        """Parse Selector JSON response and build filtered DDL."""
        try:
            text = response.strip()
            # Strip code fences
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

            data = json.loads(text)
            tables = set(data.get("tables", []))
            columns = set(data.get("columns", []))

            if not tables:
                return None

            # Build filtered DDL from schema object
            fk_by_table: Dict[str, List[Tuple[str, str, str]]] = {}
            for src, tgt in schema.foreign_keys:
                st, sc = src.split(".", 1)
                tt, tc = tgt.split(".", 1)
                fk_by_table.setdefault(st, []).append((sc, tt, tc))

            parts = []
            for table in schema.tables:
                if table.name not in tables:
                    continue

                # Filter columns if specified, otherwise keep all
                table_cols = {
                    c.split(".", 1)[1] for c in columns
                    if c.startswith(f"{table.name}.")
                }

                col_defs = []
                for col in table.columns:
                    if table_cols and col.name not in table_cols and not col.is_pk:
                        continue
                    col_def = f"  {col.name} {col.dtype}"
                    if col.is_pk:
                        col_def += " PRIMARY KEY"
                    col_defs.append(col_def)

                for sc, tt, tc in fk_by_table.get(table.name, []):
                    if tt in tables:
                        col_defs.append(
                            f"  FOREIGN KEY ({sc}) REFERENCES {tt}({tc})"
                        )

                if col_defs:
                    parts.append(
                        f"CREATE TABLE {table.name} (\n"
                        + ",\n".join(col_defs)
                        + "\n);"
                    )

            return "\n\n".join(parts) if parts else None

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Selector JSON parse failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # Agent 2: Decomposer
    # ------------------------------------------------------------------ #

    DECOMPOSER_PROMPT = """\
You are an expert SQL writer. Given a database schema and a natural \
language question, generate the correct SQLite SQL query.

Instructions:
1. Analyze the question complexity
2. If simple (single table, no subquery): generate SQL directly
3. If complex (multiple tables, subqueries, set operations):
   - Break the question into sub-questions
   - Solve each sub-question step by step
   - Compose the final SQL from the sub-results
4. Use table aliases (T1, T2, ...) for multi-table queries
5. Use INNER JOIN with explicit ON conditions from foreign keys

Output format:
- For simple questions: output the SQL directly
- For complex questions: show sub-questions and their SQL, then the final SQL

Return the final SQL query at the end, prefixed with "Final SQL:"."""

    def _decompose(self, nl: str, schema_ddl: str, hint: str) -> str:
        """Decomposer agent: break down question and generate SQL."""
        user_content = f"Database Schema:\n{schema_ddl}\n\n"
        if hint:
            user_content += f"Hint: {hint}\n\n"
        user_content += f"Question: {nl}"

        messages = [
            {"role": "system", "content": self.DECOMPOSER_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = self.llm.chat(messages)
        return self._extract_sql_from_decomposer(response)

    # ------------------------------------------------------------------ #
    # Agent 3: Refiner (with execution feedback)
    # ------------------------------------------------------------------ #

    REFINER_PROMPT = """\
You are a SQL debugger. Given a database schema, a question, a SQL query, \
and its execution feedback, fix the SQL if there are errors.

Execution feedback types:
- "Error: <message>" — the SQL has a syntax or runtime error
- "Empty result" — the SQL ran but returned no rows (may indicate wrong logic)
- "Success: <N> rows" — the SQL ran and returned results (likely correct)

Common issues to fix:
1. Column does not exist — check column names against schema
2. Wrong JOIN condition — use foreign keys from schema
3. Missing GROUP BY for aggregation
4. Wrong table alias references
5. Type mismatch in comparisons (use CAST if needed)
6. Logic error (wrong WHERE condition for the question)

If the SQL is correct, return it unchanged.
Return only the corrected SQL query, no explanation."""

    def _refine(
        self, nl: str, schema_ddl: str, sql: str, db_path: str, hint: str,
    ) -> str:
        """Refiner agent: iterative correction with execution feedback."""
        for round_num in range(self.max_refine_rounds):
            # Execute current SQL
            feedback = self._get_execution_feedback(sql, db_path)

            # If successful with results, stop
            if feedback.startswith("Success:"):
                logger.debug("Refiner: SQL OK at round %d", round_num)
                return sql

            logger.debug(
                "Refiner round %d: %s", round_num + 1, feedback[:100],
            )

            # Ask LLM to fix
            user_content = f"Database Schema:\n{schema_ddl}\n\n"
            if hint:
                user_content += f"Hint: {hint}\n\n"
            user_content += (
                f"Question: {nl}\n\n"
                f"SQL Query:\n{sql}\n\n"
                f"Execution Feedback: {feedback}"
            )

            messages = [
                {"role": "system", "content": self.REFINER_PROMPT},
                {"role": "user", "content": user_content},
            ]

            response = self.llm.chat(messages)
            new_sql = self._extract_sql(response)

            # If LLM returned the same SQL, stop (no progress)
            if new_sql.strip() == sql.strip():
                logger.debug("Refiner: no change, stopping")
                return sql

            sql = new_sql

        return sql

    def _get_execution_feedback(self, sql: str, db_path: str) -> str:
        """Execute SQL and return structured feedback string."""
        try:
            result = execute_sql(sql, db_path)
            if result is None:
                return "Error: execution returned None (possible timeout or crash)"
            if len(result) == 0:
                return "Empty result: query returned 0 rows"
            return f"Success: {len(result)} rows returned"
        except Exception as e:
            return f"Error: {str(e)}"

    # ------------------------------------------------------------------ #
    # Schema Formatting
    # ------------------------------------------------------------------ #

    def _format_ddl(
        self, schema: Schema, db_path: Optional[str] = None,
        column_mapping: Optional[dict] = None,
    ) -> str:
        """Format schema as CREATE TABLE DDL with sample values."""
        reverse = column_mapping or {}
        parts = []

        fk_by_table: Dict[str, List[Tuple[str, str, str]]] = {}
        all_fks = []
        for src, tgt in schema.foreign_keys:
            src_table, src_col = src.split(".", 1)
            tgt_table, tgt_col = tgt.split(".", 1)
            fk_by_table.setdefault(src_table, []).append((src_col, tgt_table, tgt_col))
            all_fks.append(f"{src} = {tgt}")

        for table in schema.tables:
            col_defs = []
            for col in table.columns:
                col_def = f"  {col.name} {col.dtype}"
                if col.is_pk:
                    col_def += " PRIMARY KEY"
                col_defs.append(col_def)

            for src_col, tgt_table, tgt_col in fk_by_table.get(table.name, []):
                col_defs.append(
                    f"  FOREIGN KEY ({src_col}) REFERENCES {tgt_table}({tgt_col})"
                )

            ddl = (
                f"CREATE TABLE {table.name} (\n"
                + ",\n".join(col_defs)
                + "\n);"
            )

            # Add sample values if db_path available
            if db_path and self.sample_rows > 0:
                samples = self._format_table_samples(
                    table, db_path, reverse,
                )
                if samples:
                    ddl += f"\n/* Sample rows:\n{samples}\n*/"

            parts.append(ddl)

        result = "\n\n".join(parts)
        if all_fks:
            result += f"\n\nForeign_keys = [{', '.join(all_fks)}]"
        return result

    def _format_table_samples(
        self, table: Table, db_path: str, reverse: dict,
    ) -> str:
        """Format sample rows for a table."""
        col_names = [c.name for c in table.columns]
        col_samples = {}
        for col in table.columns:
            db_col_name = reverse.get(col.name, col.name)
            col_samples[col.name] = sample_column(
                db_path, table.name, db_col_name, n=self.sample_rows,
            )

        n_rows = max((len(v) for v in col_samples.values()), default=0)
        if n_rows == 0:
            return ""

        header = " | ".join(col_names)
        rows = []
        for i in range(min(n_rows, self.sample_rows)):
            vals = []
            for c in col_names:
                samples = col_samples.get(c, [])
                vals.append(samples[i] if i < len(samples) else "NULL")
            rows.append(" | ".join(vals))

        return f"{header}\n" + "\n".join(rows)

    # ------------------------------------------------------------------ #
    # SQL Extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_sql(text: str) -> str:
        """Extract SQL from LLM response (reasoning-CoT-safe)."""
        text = strip_thinking(text)
        match = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _extract_ddl(text: str) -> str:
        """Extract DDL from Selector response (reasoning-CoT-safe)."""
        text = strip_thinking(text)
        match = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _extract_sql_from_decomposer(text: str) -> str:
        """Extract final SQL from Decomposer response.

        Looks for "Final SQL:" pattern, falls back to last SELECT statement.
        Reasoning CoT is stripped first (MiniMax/GLM wrap in </think>).
        """
        text = strip_thinking(text)
        # Try code fence first
        match = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Look for "Final SQL:" pattern
        match = re.search(
            r'Final\s+SQL\s*:\s*(SELECT\b[^\n]*)', text, re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Look for last "SQL:" pattern
        matches = list(re.finditer(r'SQL\s*:\s*(SELECT\b[^\n]*)', text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()

        # Last SELECT statement
        match = re.search(r'(SELECT\b[^;]*)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return text.strip()
