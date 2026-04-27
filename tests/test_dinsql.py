"""Tests for DIN-SQL Text-to-SQL implementation (official BIRD version)."""
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase3.dinsql_runner import DINSQLText2SQL, DIFFICULTY_LABELS
from egrefine.phase3 import dinsql_prompts as prompts


def _make_schema():
    return Schema(
        db_id="test_db",
        tables=[
            Table(name="employees", columns=[
                Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
                Column(name="nm", table="employees", dtype="TEXT"),
                Column(name="sal", table="employees", dtype="REAL"),
                Column(name="dept_id", table="employees", dtype="INTEGER",
                       fk_target="departments.id"),
            ]),
            Table(name="departments", columns=[
                Column(name="id", table="departments", dtype="INTEGER", is_pk=True),
                Column(name="name", table="departments", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[("employees.dept_id", "departments.id")],
    )


def _make_test_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO departments VALUES (1, 'Engineering')")
    conn.execute(
        "CREATE TABLE employees "
        "(id INTEGER PRIMARY KEY, nm TEXT, sal REAL, dept_id INTEGER)"
    )
    conn.execute("INSERT INTO employees VALUES (1, 'Alice', 50000, 1)")
    conn.commit()
    conn.close()
    return db_path


class TestDDLFormat:
    def test_ddl_includes_tables_and_fk(self):
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        din = DINSQLText2SQL(config)
        result = din._format_ddl(_make_schema())

        assert "CREATE TABLE employees" in result
        assert "CREATE TABLE departments" in result
        assert "FOREIGN KEY (dept_id) REFERENCES departments(id)" in result

    def test_ddl_includes_fk_summary_line(self):
        """Official format includes Foreign_keys = [...] summary."""
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        din = DINSQLText2SQL(config)
        result = din._format_ddl(_make_schema())

        assert "Foreign_keys = [employees.dept_id = departments.id]" in result

    def test_ddl_no_fk_summary_when_no_fks(self):
        schema = Schema(
            db_id="test", tables=[
                Table(name="t", columns=[
                    Column(name="id", table="t", dtype="INTEGER", is_pk=True),
                ]),
            ], foreign_keys=[],
        )
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        din = DINSQLText2SQL(config)
        result = din._format_ddl(schema)

        assert "Foreign_keys" not in result


class TestClassification:
    def test_parse_label_exact(self):
        assert DINSQLText2SQL._parse_label("EASY") == "EASY"
        assert DINSQLText2SQL._parse_label("NON-NESTED") == "NON-NESTED"
        assert DINSQLText2SQL._parse_label("NESTED") == "NESTED"

    def test_parse_label_with_cot_reasoning(self):
        """Official format: CoT reasoning followed by label."""
        response = (
            "Let's think step by step. "
            "The question requires joining two tables. "
            'Classify as "NON-NESTED"'
        )
        assert DINSQLText2SQL._parse_label(response) == "NON-NESTED"

    def test_parse_label_nested_in_cot(self):
        response = (
            "Let's think step by step. "
            "This requires a subquery. "
            'Classify as "NESTED"'
        )
        assert DINSQLText2SQL._parse_label(response) == "NESTED"

    def test_parse_label_easy_in_cot(self):
        response = (
            "Let's think step by step. "
            "Single table, no JOIN. "
            'Classify as "EASY"'
        )
        assert DINSQLText2SQL._parse_label(response) == "EASY"

    def test_parse_label_simple_synonym(self):
        assert DINSQLText2SQL._parse_label("SIMPLE") == "EASY"

    def test_parse_label_default(self):
        assert DINSQLText2SQL._parse_label("I don't know") == "NON-NESTED"

    def test_parse_label_nested_not_confused_with_non_nested(self):
        assert DINSQLText2SQL._parse_label("NESTED") == "NESTED"

    def test_parse_label_non_nested_priority(self):
        assert DINSQLText2SQL._parse_label("NON-NESTED") == "NON-NESTED"


class TestSQLExtraction:
    def test_extract_plain(self):
        assert DINSQLText2SQL._extract_sql("SELECT 1") == "SELECT 1"

    def test_extract_code_fenced(self):
        assert DINSQLText2SQL._extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_extract_from_cot_with_sql_prefix(self):
        """Official NON-NESTED/NESTED format: reasoning then 'SQL: SELECT ...'"""
        text = (
            "Let's think step by step. For this question, "
            "we need to join tables = [movies, ratings].\n"
            "SQL: SELECT T1.movie_title FROM movies AS T1 "
            "INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id"
        )
        result = DINSQLText2SQL._extract_sql_from_cot(text)
        assert result.startswith("SELECT T1.movie_title")

    def test_extract_from_cot_nested_with_sub_sql(self):
        """NESTED format has Sub-SQL and then final SQL."""
        text = (
            "Let's think step by step.\n"
            "Sub-question: Find user_id.\n"
            "SQL: SELECT user_id FROM lists WHERE list_title = 'Sound4Film'\n"
            "Main query:\n"
            "SQL: SELECT COUNT(list_id) FROM lists "
            "WHERE user_id = (SELECT user_id FROM lists WHERE list_title = 'Sound4Film')"
        )
        result = DINSQLText2SQL._extract_sql_from_cot(text)
        # Should get the LAST SQL (the main query)
        assert "COUNT(list_id)" in result
        assert "Sound4Film" in result

    def test_extract_from_cot_code_fence_priority(self):
        """Code fence should take priority over SQL: prefix."""
        text = (
            "Let's think step by step.\n"
            "SQL: SELECT wrong\n"
            "```sql\nSELECT correct\n```"
        )
        result = DINSQLText2SQL._extract_sql_from_cot(text)
        assert result == "SELECT correct"

    def test_extract_from_cot_bare_select(self):
        """EASY format: just a SELECT statement."""
        text = "SELECT COUNT(movie_id) FROM movies WHERE movie_release_year = 2020"
        result = DINSQLText2SQL._extract_sql_from_cot(text)
        assert result == text

    def test_extract_from_cot_no_sql(self):
        """Fallback when no SQL pattern found."""
        text = "I cannot generate SQL for this question."
        result = DINSQLText2SQL._extract_sql_from_cot(text)
        assert result == text


class TestGenerationPromptSelection:
    def test_easy_prompt(self):
        instr, exs = DINSQLText2SQL._get_generation_prompt("EASY")
        assert instr == prompts.GENERATION_INSTRUCTION_EASY
        assert exs == prompts.GENERATION_EXEMPLARS_EASY

    def test_non_nested_prompt(self):
        instr, exs = DINSQLText2SQL._get_generation_prompt("NON-NESTED")
        assert instr == prompts.GENERATION_INSTRUCTION_NON_NESTED
        assert exs == prompts.GENERATION_EXEMPLARS_NON_NESTED

    def test_nested_prompt(self):
        instr, exs = DINSQLText2SQL._get_generation_prompt("NESTED")
        assert instr == prompts.GENERATION_INSTRUCTION_NESTED
        assert exs == prompts.GENERATION_EXEMPLARS_NESTED

    def test_unknown_defaults_to_non_nested(self):
        instr, exs = DINSQLText2SQL._get_generation_prompt("UNKNOWN")
        assert instr == prompts.GENERATION_INSTRUCTION_NON_NESTED


class TestFullPipeline:
    """Test the full 4-module pipeline with mocked LLM."""

    def test_generate_4_modules(self):
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        din = DINSQLText2SQL(config, self_correction=True)
        schema = _make_schema()

        din.llm = MagicMock()
        din.llm.chat.side_effect = [
            # M1: Schema linking (CoT output)
            (
                "Let's think step by step. \"employee names\" refers to nm in employees. "
                "Schema_links: [employees.nm]"
            ),
            # M2: Classification (CoT + label)
            'Let\'s think step by step. Single table. Classify as "EASY"',
            # M3: SQL generation
            "SELECT nm FROM employees",
            # M4: Self-correction
            "SELECT nm FROM employees",
        ]

        result = din.generate("List all employee names", schema)
        assert result == "SELECT nm FROM employees"
        assert din.llm.chat.call_count == 4

    def test_generate_without_correction(self):
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        din = DINSQLText2SQL(config, self_correction=False)
        schema = _make_schema()

        din.llm = MagicMock()
        din.llm.chat.side_effect = [
            "Schema_links: [employees.nm]",
            "EASY",
            "SELECT nm FROM employees",
        ]

        result = din.generate("List all employee names", schema)
        assert result == "SELECT nm FROM employees"
        assert din.llm.chat.call_count == 3

    def test_generate_with_evidence(self):
        """Evidence/hint should be passed to all modules."""
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        din = DINSQLText2SQL(config, self_correction=True)
        schema = _make_schema()

        din.llm = MagicMock()
        din.llm.chat.side_effect = [
            "Schema_links: [employees.sal]",
            "EASY",
            "SELECT sal FROM employees WHERE sal > 50000",
            "SELECT sal FROM employees WHERE sal > 50000",
        ]

        result = din.generate(
            "Find high salary employees", schema,
            evidence="high salary refers to sal > 50000",
        )

        # Check that hint was included in M1 (schema linking) prompt
        m1_messages = din.llm.chat.call_args_list[0][0][0]
        m1_user_msg = m1_messages[-1]["content"]
        assert "Hint: high salary refers to sal > 50000" in m1_user_msg

        # Check that hint was included in M2 (classification) prompt
        m2_messages = din.llm.chat.call_args_list[1][0][0]
        m2_user_msg = m2_messages[-1]["content"]
        assert "Hint: high salary" in m2_user_msg

    def test_generate_nested_with_cot_extraction(self):
        """NON-NESTED/NESTED responses have CoT reasoning before SQL."""
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        din = DINSQLText2SQL(config, self_correction=False)
        schema = _make_schema()

        din.llm = MagicMock()
        din.llm.chat.side_effect = [
            "Schema_links: [employees.nm, employees.sal]",
            "NESTED",
            # CoT response with sub-question decomposition
            (
                "Let's think step by step. "
                "Sub-question: What is the average salary?\n"
                "SQL: SELECT AVG(sal) FROM employees\n"
                "Main query:\n"
                "SQL: SELECT nm FROM employees WHERE sal > (SELECT AVG(sal) FROM employees)"
            ),
        ]

        result = din.generate("Find employees above average salary", schema)
        assert "AVG(sal)" in result
        assert result.startswith("SELECT nm")

    def test_cot_in_schema_linking_messages(self):
        """Schema linking should include 'Let's think step by step' in user messages."""
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        din = DINSQLText2SQL(config, self_correction=False)
        schema = _make_schema()

        din.llm = MagicMock()
        din.llm.chat.side_effect = [
            "Schema_links: [employees.nm]",
            "EASY",
            "SELECT nm FROM employees",
        ]

        din.generate("test", schema)

        m1_messages = din.llm.chat.call_args_list[0][0][0]
        # Last user message should end with CoT trigger
        last_user = m1_messages[-1]["content"]
        assert "Let's think step by step" in last_user

    def test_few_shot_messages_structure(self):
        """Verify few-shot exemplars are correctly formatted."""
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        din = DINSQLText2SQL(config, self_correction=False)
        schema = _make_schema()

        din.llm = MagicMock()
        din.llm.chat.side_effect = [
            "Schema_links: [employees.nm]",
            "EASY",
            "SELECT 1",
        ]

        din.generate("test", schema)

        # M1: system + (user+assistant)*2 exemplars + user
        m1_messages = din.llm.chat.call_args_list[0][0][0]
        expected = 1 + 2 * len(prompts.SCHEMA_LINKING_EXEMPLARS) + 1
        assert len(m1_messages) == expected
        assert m1_messages[0]["role"] == "system"

        # M2: system + (user+assistant)*6 exemplars + user
        m2_messages = din.llm.chat.call_args_list[1][0][0]
        expected = 1 + 2 * len(prompts.CLASSIFICATION_EXEMPLARS) + 1
        assert len(m2_messages) == expected


class TestPromptTemplates:
    """Verify prompt templates match official DIN-SQL BIRD format."""

    def test_linking_exemplars_have_cot(self):
        """Schema linking exemplars should use CoT reasoning."""
        for ex in prompts.SCHEMA_LINKING_EXEMPLARS:
            assert "schema" in ex
            assert "question" in ex
            assert "hint" in ex
            assert "output" in ex
            assert "Let's think step by step" in ex["output"]
            assert "Schema_links:" in ex["output"]

    def test_classification_exemplars_have_cot_and_label(self):
        """Classification exemplars should have CoT reasoning with label."""
        for ex in prompts.CLASSIFICATION_EXEMPLARS:
            assert "question" in ex
            assert "linking" in ex
            assert "label" in ex
            # Label should contain a valid difficulty label somewhere
            label_text = ex["label"].upper()
            has_label = any(l in label_text for l in DIFFICULTY_LABELS)
            assert has_label, f"No valid label found in: {ex['label']}"

    def test_generation_exemplars_have_required_keys(self):
        for exs in [
            prompts.GENERATION_EXEMPLARS_EASY,
            prompts.GENERATION_EXEMPLARS_NON_NESTED,
            prompts.GENERATION_EXEMPLARS_NESTED,
        ]:
            for ex in exs:
                assert "sql" in ex
                assert "question" in ex
                assert "linking" in ex

    def test_non_nested_exemplars_have_reasoning(self):
        """NON-NESTED exemplars should include step-by-step reasoning."""
        for ex in prompts.GENERATION_EXEMPLARS_NON_NESTED:
            assert "Let's think step by step" in ex["sql"]
            assert "SQL:" in ex["sql"]

    def test_nested_exemplars_have_subquestion(self):
        """NESTED exemplars should include sub-question decomposition."""
        for ex in prompts.GENERATION_EXEMPLARS_NESTED:
            assert "sub" in ex["sql"].lower() or "Sub" in ex["sql"]

    def test_correction_has_rules(self):
        """Self-correction instruction should list explicit rules."""
        assert "column names exist" in prompts.SELF_CORRECTION_INSTRUCTION
        assert "JOIN" in prompts.SELF_CORRECTION_INSTRUCTION
        assert "GROUP BY" in prompts.SELF_CORRECTION_INSTRUCTION
