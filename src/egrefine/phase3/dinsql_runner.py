"""DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction.

Implements Pourreza & Rafiei 2023 (BIRD version) as a Text2SQLModel:
  Module 1: Schema Linking — few-shot with CoT, identify relevant tables/columns
  Module 2: Classification — EASY / NON-NESTED / NESTED with explicit rules
  Module 3: SQL Generation — difficulty-specific few-shot with reasoning
  Module 4: Self-Correction — rule-based LLM self-review

Aligned with the official DIN-SQL_BIRD.py implementation.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

from egrefine.data.schema import Schema, Table, Column
from egrefine.models.llm_client import LLMClient
from egrefine.phase3.text2sql_runner import Text2SQLModel, strip_thinking
from egrefine.phase3 import dinsql_prompts as prompts

logger = logging.getLogger(__name__)

# Valid difficulty labels
DIFFICULTY_LABELS = {"EASY", "NON-NESTED", "NESTED"}


class DINSQLText2SQL(Text2SQLModel):
    """DIN-SQL (Decomposed In-Context Learning) Text-to-SQL.

    Aligned with the official BIRD version of DIN-SQL:
    - Schema linking uses CoT reasoning ("Let's think step by step")
    - Classification uses explicit decision rules
    - Generation uses difficulty-specific reasoning strategies
    - Self-correction uses rule-based zero-shot checking
    - BIRD hints/evidence are included in all prompts
    """

    def __init__(
        self,
        config: dict,
        self_correction: bool = True,
    ):
        """
        Args:
            config: LLM config (base_url, api_key, model_name, etc.).
            self_correction: Whether to apply Module 4 self-correction.
        """
        self.llm = LLMClient(config)
        self.self_correction = self_correction

    def generate(
        self,
        nl: str,
        schema: Schema,
        db_path: Optional[str] = None,
        column_mapping: Optional[dict] = None,
        evidence: str = "",
    ) -> str:
        """Generate SQL using 4-module DIN-SQL pipeline.

        Args:
            nl: Natural language question.
            schema: Database schema.
            db_path: Database path (unused, kept for interface compatibility).
            column_mapping: Column name mapping (unused).
            evidence: BIRD hint/evidence text for calibration.
        """
        schema_ddl = self._format_ddl(schema)
        hint = evidence

        # Module 1: Schema Linking (with CoT)
        linking_result = self._schema_linking(nl, schema_ddl, hint)

        # Module 2: Classification
        difficulty = self._classify(nl, linking_result, hint)

        # Module 3: SQL Generation (difficulty-specific)
        sql = self._generate_sql(nl, schema_ddl, linking_result, difficulty, hint)

        # Module 4: Self-Correction
        if self.self_correction:
            sql = self._self_correct(nl, schema_ddl, sql, hint)

        return sql

    # ------------------------------------------------------------------ #
    # Module 1: Schema Linking (with CoT)
    # ------------------------------------------------------------------ #

    def _schema_linking(self, nl: str, schema_ddl: str, hint: str) -> str:
        """Identify relevant tables, columns, and conditions via CoT."""
        messages = [{"role": "system", "content": prompts.SCHEMA_LINKING_INSTRUCTION}]

        # Few-shot exemplars (with CoT output)
        for ex in prompts.SCHEMA_LINKING_EXEMPLARS:
            user_content = f"Schema:\n{ex['schema']}\n\nQuestion: {ex['question']}"
            if ex.get("hint"):
                user_content += f"\nHint: {ex['hint']}"
            user_content += "\nA: Let's think step by step."
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": ex["output"]})

        # Actual query
        user_content = f"Schema:\n{schema_ddl}\n\nQuestion: {nl}"
        if hint:
            user_content += f"\nHint: {hint}"
        user_content += "\nA: Let's think step by step."
        messages.append({"role": "user", "content": user_content})

        return self.llm.chat(messages)

    # ------------------------------------------------------------------ #
    # Module 2: Query Classification
    # ------------------------------------------------------------------ #

    def _classify(self, nl: str, linking_result: str, hint: str) -> str:
        """Classify query difficulty: EASY, NON-NESTED, or NESTED."""
        messages = [{"role": "system", "content": prompts.CLASSIFICATION_INSTRUCTION}]

        for ex in prompts.CLASSIFICATION_EXEMPLARS:
            user_content = f"Question: {ex['question']}\n\n"
            if ex.get("hint"):
                user_content += f"Hint: {ex['hint']}\n\n"
            user_content += f"Schema Linking:\n{ex['linking']}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": ex["label"]})

        user_content = f"Question: {nl}\n\n"
        if hint:
            user_content += f"Hint: {hint}\n\n"
        user_content += f"Schema Linking:\n{linking_result}"
        messages.append({"role": "user", "content": user_content})

        response = self.llm.chat(messages)
        label = self._parse_label(response)
        logger.debug("DIN-SQL classified '%s' as %s", nl[:50], label)
        return label

    @staticmethod
    def _parse_label(response: str) -> str:
        """Extract difficulty label from LLM response (may contain CoT reasoning)."""
        text = response.upper()
        # Check in priority order: NON-NESTED before NESTED (avoid substring match)
        if "NON-NESTED" in text or "NON_NESTED" in text:
            return "NON-NESTED"
        if "EASY" in text or "SIMPLE" in text:
            return "EASY"
        if "NESTED" in text:
            return "NESTED"
        # Default to NON-NESTED (safest middle ground)
        return "NON-NESTED"

    # ------------------------------------------------------------------ #
    # Module 3: SQL Generation
    # ------------------------------------------------------------------ #

    def _generate_sql(
        self, nl: str, schema_ddl: str, linking_result: str,
        difficulty: str, hint: str,
    ) -> str:
        """Generate SQL using difficulty-specific prompt with reasoning."""
        instruction, exemplars = self._get_generation_prompt(difficulty)

        messages = [{"role": "system", "content": instruction}]

        for ex in exemplars:
            user_content = f"Schema:\n{schema_ddl}\n\n"
            user_content += f"Schema Linking:\n{ex['linking']}\n\n"
            if ex.get("hint"):
                user_content += f"Hint: {ex['hint']}\n\n"
            user_content += f"Question: {ex['question']}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": ex["sql"]})

        user_content = f"Schema:\n{schema_ddl}\n\n"
        user_content += f"Schema Linking:\n{linking_result}\n\n"
        if hint:
            user_content += f"Hint: {hint}\n\n"
        user_content += f"Question: {nl}"
        messages.append({"role": "user", "content": user_content})

        response = self.llm.chat(messages)
        return self._extract_sql_from_cot(response)

    @staticmethod
    def _get_generation_prompt(difficulty: str):
        """Return (instruction, exemplars) for the given difficulty level."""
        if difficulty == "EASY":
            return (
                prompts.GENERATION_INSTRUCTION_EASY,
                prompts.GENERATION_EXEMPLARS_EASY,
            )
        elif difficulty == "NESTED":
            return (
                prompts.GENERATION_INSTRUCTION_NESTED,
                prompts.GENERATION_EXEMPLARS_NESTED,
            )
        else:  # NON-NESTED (default)
            return (
                prompts.GENERATION_INSTRUCTION_NON_NESTED,
                prompts.GENERATION_EXEMPLARS_NON_NESTED,
            )

    # ------------------------------------------------------------------ #
    # Module 4: Self-Correction
    # ------------------------------------------------------------------ #

    def _self_correct(self, nl: str, schema_ddl: str, sql: str, hint: str) -> str:
        """LLM self-review and fix SQL errors."""
        messages = [{"role": "system", "content": prompts.SELF_CORRECTION_INSTRUCTION}]

        for ex in prompts.SELF_CORRECTION_EXEMPLARS:
            user_content = f"Schema:\n{ex['schema']}\n\n"
            user_content += f"Question: {ex['question']}\n\n"
            if ex.get("hint"):
                user_content += f"Hint: {ex['hint']}\n\n"
            user_content += f"SQL: {ex['sql']}"
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": ex["corrected"]})

        user_content = f"Schema:\n{schema_ddl}\n\nQuestion: {nl}\n\n"
        if hint:
            user_content += f"Hint: {hint}\n\n"
        user_content += f"SQL: {sql}"
        messages.append({"role": "user", "content": user_content})

        response = self.llm.chat(messages)
        return self._extract_sql(response)

    # ------------------------------------------------------------------ #
    # Schema Formatting
    # ------------------------------------------------------------------ #

    def _format_ddl(self, schema: Schema) -> str:
        """Format schema as CREATE TABLE DDL + Foreign_keys line (official format)."""
        parts = []
        all_fks = []

        fk_by_table: Dict[str, List[Tuple[str, str, str]]] = {}
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

            parts.append(
                f"CREATE TABLE {table.name} (\n"
                + ",\n".join(col_defs)
                + "\n);"
            )

        ddl = "\n\n".join(parts)

        # Add Foreign_keys summary line (matches official format)
        if all_fks:
            ddl += f"\n\nForeign_keys = [{', '.join(all_fks)}]"

        return ddl

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
    def _extract_sql_from_cot(text: str) -> str:
        """Extract SQL from CoT response that may contain reasoning before SQL.

        Official DIN-SQL format for NON-NESTED/NESTED:
          "Let's think step by step. ... SQL: SELECT ..."
        For EASY:
          Direct SQL output.

        Reasoning models wrap their own CoT in `</think>` — strip that first
        so we don't pick up draft SQL from the reasoning stream.
        """
        text = strip_thinking(text)
        # Try code fence first
        match = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Look for "SQL: SELECT ..." pattern (official CoT format)
        # Find the LAST occurrence of "SQL:" to get the final query
        matches = list(re.finditer(r'SQL:\s*(SELECT\b[^\n]*)', text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip()

        # Look for standalone SELECT statement
        match = re.search(r'(SELECT\b[^;]*)', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return text.strip()
