"""C3: Zero-shot Text-to-SQL with ChatGPT.

Implements the C3 method (Dong et al., 2023) as a Text2SQLModel:
  Stage 1: Schema Linking — select relevant tables/columns via LLM
  Stage 2: SQL Generation — generate SQL with calibration hints
  Stage 3: Self-Consistency — execution-based majority voting
"""
import csv
import json
import logging
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from egrefine.data.schema import Schema, Table, Column
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.sampler import sample_column
from egrefine.phase3.executor import execute_sql
from egrefine.phase3.text2sql_runner import Text2SQLModel, strip_thinking

logger = logging.getLogger(__name__)


class C3Text2SQL(Text2SQLModel):
    """C3 (Clear Prompting + Calibration + Consistency) Text-to-SQL."""

    def __init__(
        self,
        config: dict,
        num_samples: int = 5,
        sample_rows: int = 3,
        desc_dir: Optional[str] = None,
        column_descriptions: Optional[Dict[Tuple[str, str, str], str]] = None,
    ):
        """
        Args:
            config: LLM config (base_url, api_key, model_name, etc.).
            num_samples: Number of SQL candidates for self-consistency voting.
            sample_rows: Number of sample rows per column for hints.
            desc_dir: Path to BIRD database_description/ root directory.
                      If provided, column descriptions are used as calibration hints.
            column_descriptions: Optional dict keyed by (db_id, table, column)
                providing one-line natural-language descriptions to inject
                inline as ``-- comment`` annotations into both schema
                serializers. Used by the description-baseline experiment.
                Keys must match the column names actually appearing in the
                Schema object passed to ``generate()`` (i.e. for refined-view
                evaluation the caller is responsible for remapping
                original→refined names before construction).
        """
        self.llm = LLMClient(config)
        self.num_samples = num_samples
        self.sample_rows = sample_rows
        self.desc_dir = desc_dir
        self.column_descriptions = column_descriptions or {}
        # Store original config for creating a high-temperature client
        self._config = config

    def _desc_for(self, db_id: str, table: str, column: str) -> Optional[str]:
        """Look up an injected description, if any, for (db_id, table, column)."""
        if not self.column_descriptions:
            return None
        return self.column_descriptions.get((db_id, table, column))

    def generate(
        self,
        nl: str,
        schema: Schema,
        db_path: Optional[str] = None,
        column_mapping: Optional[dict] = None,
        evidence: str = "",
    ) -> str:
        # Stage 1: Schema Linking
        linked_tables, linked_columns = self._schema_linking(nl, schema, evidence)

        # Build filtered schema
        filtered_schema = self._filter_schema(schema, linked_tables, linked_columns)

        # Stage 2 + 3: Generate with self-consistency
        if self.num_samples <= 1 or db_path is None:
            # No self-consistency, single deterministic generation
            return self._generate_sql(
                nl, filtered_schema, db_path, column_mapping, evidence=evidence,
            )

        # Generate multiple candidates with temperature > 0
        candidates = self._generate_candidates(
            nl, filtered_schema, db_path, column_mapping, evidence=evidence,
        )

        # Stage 3: Self-Consistency voting
        return self._self_consistency(candidates, db_path)

    # ------------------------------------------------------------------ #
    # Stage 1: Schema Linking
    # ------------------------------------------------------------------ #

    def _schema_linking(
        self, nl: str, schema: Schema, evidence: str = "",
    ) -> Tuple[List[str], List[str]]:
        """Ask LLM to select relevant tables and columns.

        Returns:
            (table_names, column_full_names) where column_full_names are "table.col".
        """
        schema_str = self._format_compact_schema(schema)

        prompt = (
            "Given the database schema and a question, select only the "
            "relevant tables and columns needed to answer the question.\n\n"
            f"Schema:\n{schema_str}\n\n"
        )
        if evidence:
            prompt += f"Additional context (evidence): {evidence}\n\n"
        prompt += (
            f"Question: {nl}\n\n"
            "Return a JSON object with two keys:\n"
            '- "tables": list of relevant table names\n'
            '- "columns": list of relevant columns as "table_name.column_name"\n\n'
            "Return only the JSON, no explanation."
        )

        response = self.llm.chat([{"role": "user", "content": prompt}])
        return self._parse_linking_response(response, schema)

    def _format_compact_schema(self, schema: Schema) -> str:
        """Format schema in C3's compact style: table_name(col1, col2, ...).

        When ``self.column_descriptions`` carries entries for any of the
        columns, the table is rendered in a multi-line form with inline
        ``-- description`` comments next to the annotated columns. This is
        used by the description-baseline experiment.
        """
        parts = []
        for table in schema.tables:
            descs = [
                self._desc_for(schema.db_id, table.name, c.name)
                for c in table.columns
            ]
            if any(descs):
                col_lines = []
                for c, d in zip(table.columns, descs):
                    if d:
                        col_lines.append(f"  {c.name},  -- {d}")
                    else:
                        col_lines.append(f"  {c.name},")
                # Strip trailing comma on the last column for cleanliness
                if col_lines:
                    col_lines[-1] = col_lines[-1].replace(",  --", "  --", 1) \
                        if "  --" in col_lines[-1] else col_lines[-1].rstrip(",")
                parts.append(
                    f"{table.name}(\n" + "\n".join(col_lines) + "\n)"
                )
            else:
                cols = ", ".join(c.name for c in table.columns)
                parts.append(f"{table.name}({cols})")

        # Add FK info
        if schema.foreign_keys:
            fk_lines = []
            for src, tgt in schema.foreign_keys:
                fk_lines.append(f"  {src} = {tgt}")
            parts.append("Foreign Keys:\n" + "\n".join(fk_lines))

        return "\n".join(parts)

    def _parse_linking_response(
        self, response: str, schema: Schema,
    ) -> Tuple[List[str], List[str]]:
        """Parse schema linking response. Falls back to full schema on failure."""
        try:
            # Strip reasoning CoT (for MiniMax/GLM-style models) then markdown fences
            text = strip_thinking(response).strip()
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict, got {type(data).__name__}")
            tables = data.get("tables", [])
            columns = data.get("columns", [])

            # Validate against actual schema
            valid_tables = {t.name for t in schema.tables}
            valid_columns = {c.full_name for c in schema.all_columns}

            tables = [t for t in tables if t in valid_tables]
            columns = [c for c in columns if c in valid_columns]

            # Ensure tables for linked columns are included
            for col in columns:
                tbl = col.split(".")[0]
                if tbl not in tables:
                    tables.append(tbl)

            if tables:
                return tables, columns

        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
            logger.warning("Schema linking parse failed: %s", e)

        # Fallback: use all tables and columns
        return (
            [t.name for t in schema.tables],
            [c.full_name for c in schema.all_columns],
        )

    def _filter_schema(
        self, schema: Schema, tables: List[str], columns: List[str],
    ) -> Schema:
        """Build a filtered schema containing only linked tables/columns."""
        table_set = set(tables)
        col_set = set(columns)

        new_tables = []
        for table in schema.tables:
            if table.name not in table_set:
                continue
            # If columns were specified for this table, filter; otherwise keep all
            table_cols = [c for c in columns if c.startswith(f"{table.name}.")]
            if table_cols:
                col_names = {c.split(".", 1)[1] for c in table_cols}
                filtered_cols = [c for c in table.columns if c.name in col_names]
                # Always keep primary keys
                for c in table.columns:
                    if c.is_pk and c.name not in col_names:
                        filtered_cols.insert(0, c)
            else:
                filtered_cols = list(table.columns)

            if filtered_cols:
                new_tables.append(Table(name=table.name, columns=filtered_cols))

        # Keep relevant foreign keys
        all_cols = {c.full_name for t in new_tables for c in t.columns}
        new_fks = [
            (src, tgt) for src, tgt in schema.foreign_keys
            if src in all_cols and tgt in all_cols
        ]

        return Schema(db_id=schema.db_id, tables=new_tables, foreign_keys=new_fks)

    # ------------------------------------------------------------------ #
    # Stage 2: SQL Generation with Calibration Hints
    # ------------------------------------------------------------------ #

    def _generate_sql(
        self,
        nl: str,
        schema: Schema,
        db_path: Optional[str] = None,
        column_mapping: Optional[dict] = None,
        temperature: Optional[float] = None,
        evidence: str = "",
    ) -> str:
        """Generate a single SQL query with calibration hints."""
        schema_str = self._format_ddl_schema(schema)
        hints = self._build_hints(schema, db_path, column_mapping)

        prompt = (
            "You are a SQLite expert. Given the database schema, "
            "generate a SQL query to answer the question.\n\n"
            f"Schema:\n{schema_str}\n"
        )
        if evidence:
            prompt += f"\nAdditional context (evidence): {evidence}\n"
        if hints:
            prompt += f"\nHints:\n{hints}\n"
        prompt += (
            f"\nQuestion: {nl}\n\n"
            "Return only the SQL query, no explanation."
        )

        # Use custom temperature if provided (for self-consistency)
        if temperature is not None:
            original_temp = self.llm.temperature
            self.llm.temperature = temperature
            try:
                response = self.llm.chat([{"role": "user", "content": prompt}])
            finally:
                self.llm.temperature = original_temp
        else:
            response = self.llm.chat([{"role": "user", "content": prompt}])

        return self._extract_sql(response)

    def _format_ddl_schema(self, schema: Schema) -> str:
        """Format schema as CREATE TABLE DDL (for generation stage).

        Description-baseline: when ``self.column_descriptions`` carries an
        entry for a column, the description is appended after the column
        definition as an inline ``-- comment``.
        """
        parts = []
        fk_by_table: Dict[str, List[Tuple[str, str, str]]] = {}
        for src, tgt in schema.foreign_keys:
            src_table, src_col = src.split(".", 1)
            tgt_table, tgt_col = tgt.split(".", 1)
            fk_by_table.setdefault(src_table, []).append((src_col, tgt_table, tgt_col))

        for table in schema.tables:
            entries: List[Tuple[str, Optional[str]]] = []
            for col in table.columns:
                col_def = f"  {col.name} {col.dtype}"
                if col.is_pk:
                    col_def += " PRIMARY KEY"
                desc = self._desc_for(schema.db_id, table.name, col.name)
                entries.append((col_def, desc))

            for src_col, tgt_table, tgt_col in fk_by_table.get(table.name, []):
                entries.append((
                    f"  FOREIGN KEY ({src_col}) REFERENCES {tgt_table}({tgt_col})",
                    None,
                ))

            # Place comma BEFORE the inline -- comment so the comma is not
            # swallowed by SQLite's end-of-line comment syntax.
            lines = []
            for i, (def_, desc) in enumerate(entries):
                sep = "," if i < len(entries) - 1 else ""
                if desc:
                    lines.append(f"{def_}{sep}  -- {desc}")
                else:
                    lines.append(f"{def_}{sep}")

            parts.append(
                f"CREATE TABLE {table.name} (\n"
                + "\n".join(lines)
                + "\n);"
            )
        return "\n\n".join(parts)

    def _build_hints(
        self,
        schema: Schema,
        db_path: Optional[str] = None,
        column_mapping: Optional[dict] = None,
    ) -> str:
        """Build calibration hints: column descriptions + sample values."""
        hints = []

        # Column descriptions from BIRD database_description CSVs
        if self.desc_dir:
            desc_hints = self._load_column_descriptions(schema)
            if desc_hints:
                hints.append(desc_hints)

        # Sample values
        if db_path and self.sample_rows > 0:
            reverse = column_mapping or {}
            for table in schema.tables:
                for col in table.columns:
                    db_col_name = reverse.get(col.name, col.name)
                    values = sample_column(
                        db_path, table.name, db_col_name, n=self.sample_rows,
                    )
                    if values:
                        val_str = ", ".join(f"'{v}'" for v in values[:5])
                        hints.append(
                            f"- {table.name}.{col.name} has values like: {val_str}"
                        )

        return "\n".join(hints)

    def _load_column_descriptions(self, schema: Schema) -> str:
        """Load column descriptions from BIRD database_description/ CSVs."""
        if not self.desc_dir:
            return ""

        db_desc_dir = os.path.join(self.desc_dir, schema.db_id, "database_description")
        if not os.path.isdir(db_desc_dir):
            return ""

        lines = []
        for table in schema.tables:
            csv_path = os.path.join(db_desc_dir, f"{table.name}.csv")
            if not os.path.isfile(csv_path):
                continue

            try:
                with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        col_name = row.get("original_column_name", "").strip()
                        description = row.get("column_description", "").strip()
                        value_desc = row.get("value_description", "").strip()

                        # Check if this column is in our filtered schema
                        if not any(c.name == col_name for c in table.columns):
                            continue

                        if description:
                            line = f"- {table.name}.{col_name}: {description}"
                            if value_desc:
                                line += f" ({value_desc})"
                            lines.append(line)
            except Exception as e:
                logger.warning("Failed to read %s: %s", csv_path, e)

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Stage 3: Self-Consistency
    # ------------------------------------------------------------------ #

    def _generate_candidates(
        self,
        nl: str,
        schema: Schema,
        db_path: Optional[str],
        column_mapping: Optional[dict],
        evidence: str = "",
    ) -> List[str]:
        """Generate N SQL candidates with temperature > 0."""
        candidates = []

        # First candidate: deterministic (temperature=0)
        sql = self._generate_sql(
            nl, schema, db_path, column_mapping, temperature=0, evidence=evidence,
        )
        candidates.append(sql)

        # Remaining candidates: higher temperature
        for _ in range(self.num_samples - 1):
            sql = self._generate_sql(
                nl, schema, db_path, column_mapping, temperature=0.7,
                evidence=evidence,
            )
            candidates.append(sql)

        return candidates

    def _self_consistency(self, candidates: List[str], db_path: str) -> str:
        """Select the most consistent SQL via execution-result majority voting."""
        if len(candidates) == 1:
            return candidates[0]

        # Execute each candidate and group by result
        result_groups: Dict[Optional[tuple], List[str]] = {}
        for sql in candidates:
            result = execute_sql(sql, db_path)
            # Convert set to sorted tuple for hashable grouping
            if result is not None:
                key = tuple(sorted(result, key=lambda x: (x is None, str(x) if x is not None else "")))
            else:
                key = None
            result_groups.setdefault(key, []).append(sql)

        # Find the largest group (excluding None/error results)
        best_group = None
        best_count = 0
        for key, sqls in result_groups.items():
            if key is not None and len(sqls) > best_count:
                best_count = len(sqls)
                best_group = sqls

        # If no valid results, fall back to first candidate
        if best_group is None:
            return candidates[0]

        # Return the first SQL from the majority group
        return best_group[0]

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_sql(text: str) -> str:
        """Extract SQL from LLM response (reasoning-CoT-safe)."""
        text = strip_thinking(text)
        match = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
