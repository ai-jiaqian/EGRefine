"""Tests for MAC-SQL Text-to-SQL implementation."""
import json
import sqlite3
from unittest.mock import MagicMock, patch, call

import pytest

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase3.macsql_runner import MACSQLText2SQL


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
    conn.execute("INSERT INTO departments VALUES (2, 'Sales')")
    conn.execute(
        "CREATE TABLE employees "
        "(id INTEGER PRIMARY KEY, nm TEXT, sal REAL, dept_id INTEGER)"
    )
    conn.execute("INSERT INTO employees VALUES (1, 'Alice', 50000, 1)")
    conn.execute("INSERT INTO employees VALUES (2, 'Bob', 60000, 2)")
    conn.commit()
    conn.close()
    return db_path


class TestDDLFormat:
    def test_ddl_includes_all_tables(self):
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0)
        result = mac._format_ddl(_make_schema())

        assert "CREATE TABLE employees" in result
        assert "CREATE TABLE departments" in result
        assert "FOREIGN KEY (dept_id) REFERENCES departments(id)" in result
        assert "Foreign_keys = [employees.dept_id = departments.id]" in result

    def test_ddl_with_samples(self, tmp_path):
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=2)
        result = mac._format_ddl(_make_schema(), db_path=db_path)

        assert "Sample rows:" in result
        assert "Alice" in result or "Bob" in result

    def test_ddl_no_fk_line_when_empty(self):
        schema = Schema(
            db_id="test", tables=[
                Table(name="t", columns=[
                    Column(name="id", table="t", dtype="INTEGER", is_pk=True),
                ]),
            ], foreign_keys=[],
        )
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0)
        result = mac._format_ddl(schema)

        assert "Foreign_keys" not in result


class TestSelector:
    def test_selector_parses_json(self):
        """Selector should parse JSON response and build filtered DDL."""
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0)
        schema = _make_schema()

        mac.llm = MagicMock()
        mac.llm.chat.return_value = json.dumps({
            "tables": ["employees"],
            "columns": ["employees.nm", "employees.sal"],
        })

        result = mac._select("List names", "full ddl", "", schema=schema)
        assert "CREATE TABLE employees" in result
        assert "nm" in result
        # id should be kept (PK)
        assert "id" in result

    def test_selector_json_with_code_fence(self):
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0)
        schema = _make_schema()

        mac.llm = MagicMock()
        mac.llm.chat.return_value = (
            '```json\n{"tables": ["employees"], "columns": ["employees.nm"]}\n```'
        )

        result = mac._select("test", "ddl", "", schema=schema)
        assert "CREATE TABLE employees" in result

    def test_selector_fallback_on_garbage(self):
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0)

        mac.llm = MagicMock()
        mac.llm.chat.return_value = "I don't understand the question."

        result = mac._select("test", "CREATE TABLE t (id INT);", "")
        # Should fallback to full schema
        assert "CREATE TABLE t" in result

    def test_selector_fallback_ddl_response(self):
        """If LLM returns DDL instead of JSON, accept it."""
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0)

        mac.llm = MagicMock()
        mac.llm.chat.return_value = "CREATE TABLE employees (id INT, nm TEXT);"

        result = mac._select("test", "full schema", "")
        assert "CREATE TABLE employees" in result

    def test_selector_passes_hint(self):
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0)

        mac.llm = MagicMock()
        mac.llm.chat.return_value = '{"tables": ["t"], "columns": []}'

        mac._select("test", "schema", "salary > 50000 means high salary")

        call_messages = mac.llm.chat.call_args[0][0]
        user_msg = call_messages[-1]["content"]
        assert "Hint: salary > 50000" in user_msg

    def test_selector_keeps_fk_between_selected_tables(self):
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0)
        schema = _make_schema()

        mac.llm = MagicMock()
        mac.llm.chat.return_value = json.dumps({
            "tables": ["employees", "departments"],
            "columns": ["employees.nm", "employees.dept_id", "departments.name"],
        })

        result = mac._select("test", "ddl", "", schema=schema)
        assert "FOREIGN KEY (dept_id) REFERENCES departments(id)" in result


class TestDecomposer:
    def test_decomposer_simple_query(self):
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0)

        mac.llm = MagicMock()
        mac.llm.chat.return_value = "SELECT nm FROM employees"

        result = mac._decompose("List names", "schema", "")
        assert result == "SELECT nm FROM employees"

    def test_decomposer_with_final_sql_prefix(self):
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0)

        mac.llm = MagicMock()
        mac.llm.chat.return_value = (
            "Sub-question 1: Find avg salary\n"
            "SQL: SELECT AVG(sal) FROM employees\n\n"
            "Final SQL: SELECT nm FROM employees WHERE sal > (SELECT AVG(sal) FROM employees)"
        )

        result = mac._decompose("Above average salary", "schema", "")
        assert result.startswith("SELECT nm")
        assert "AVG(sal)" in result

    def test_decomposer_code_fence(self):
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0)

        mac.llm = MagicMock()
        mac.llm.chat.return_value = "```sql\nSELECT 1\n```"

        result = mac._decompose("test", "schema", "")
        assert result == "SELECT 1"


class TestRefiner:
    def test_refiner_skips_on_success(self, tmp_path):
        """If SQL executes successfully, Refiner should not call LLM."""
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, max_refine_rounds=2)

        mac.llm = MagicMock()
        # LLM should NOT be called since SQL is valid
        result = mac._refine(
            "test", "schema", "SELECT nm FROM employees", db_path, "",
        )

        assert result == "SELECT nm FROM employees"
        mac.llm.chat.assert_not_called()

    def test_refiner_fixes_error(self, tmp_path):
        """Refiner should fix SQL when execution fails."""
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, max_refine_rounds=2)

        mac.llm = MagicMock()
        mac.llm.chat.return_value = "SELECT nm FROM employees"

        result = mac._refine(
            "test", "schema",
            "SELECT nonexistent FROM employees",  # broken SQL
            db_path, "",
        )

        assert result == "SELECT nm FROM employees"
        assert mac.llm.chat.call_count == 1

        # Check that error feedback was included
        call_msg = mac.llm.chat.call_args[0][0][-1]["content"]
        assert "Error:" in call_msg or "execution returned None" in call_msg

    def test_refiner_stops_on_no_change(self, tmp_path):
        """If LLM returns same SQL, stop iterating."""
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, max_refine_rounds=3)

        mac.llm = MagicMock()
        # LLM keeps returning the same broken SQL
        mac.llm.chat.return_value = "SELECT bad FROM employees"

        result = mac._refine(
            "test", "schema",
            "SELECT bad FROM employees",
            db_path, "",
        )

        # Should stop after 1 round (no change)
        assert mac.llm.chat.call_count == 1

    def test_refiner_max_rounds(self, tmp_path):
        """Refiner should stop after max_refine_rounds."""
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, max_refine_rounds=2)

        mac.llm = MagicMock()
        # Each call returns a different but still broken SQL
        mac.llm.chat.side_effect = [
            "SELECT bad1 FROM employees",
            "SELECT bad2 FROM employees",
            "SELECT bad3 FROM employees",  # should not reach here
        ]

        mac._refine("test", "schema", "SELECT bad0 FROM employees", db_path, "")

        assert mac.llm.chat.call_count == 2  # max 2 rounds

    def test_refiner_disabled(self, tmp_path):
        """max_refine_rounds=0 should skip Refiner entirely."""
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, max_refine_rounds=0)
        schema = _make_schema()

        mac.llm = MagicMock()
        mac.llm.chat.side_effect = [
            # Selector
            "CREATE TABLE employees (id INT, nm TEXT);",
            # Decomposer
            "SELECT nm FROM employees",
        ]

        result = mac.generate("test", schema, db_path=db_path)
        assert result == "SELECT nm FROM employees"
        assert mac.llm.chat.call_count == 2  # No Refiner call

    def test_refiner_empty_result_triggers_fix(self, tmp_path):
        """Empty result should trigger a refinement attempt."""
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, max_refine_rounds=2)

        mac.llm = MagicMock()
        mac.llm.chat.return_value = "SELECT nm FROM employees WHERE id = 1"

        # SQL that returns empty result
        result = mac._refine(
            "test", "schema",
            "SELECT nm FROM employees WHERE id = 999",  # no such id
            db_path, "",
        )

        assert mac.llm.chat.call_count >= 1
        call_msg = mac.llm.chat.call_args[0][0][-1]["content"]
        assert "Empty result" in call_msg


class TestExecutionFeedback:
    def test_success_feedback(self, tmp_path):
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config)

        feedback = mac._get_execution_feedback("SELECT nm FROM employees", db_path)
        assert feedback.startswith("Success:")
        assert "2 rows" in feedback

    def test_error_feedback(self, tmp_path):
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config)

        feedback = mac._get_execution_feedback("SELECT nonexistent FROM employees", db_path)
        assert "Error:" in feedback or "None" in feedback

    def test_empty_result_feedback(self, tmp_path):
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config)

        feedback = mac._get_execution_feedback(
            "SELECT nm FROM employees WHERE id = 999", db_path,
        )
        assert "Empty result" in feedback


class TestSQLExtraction:
    def test_extract_plain(self):
        assert MACSQLText2SQL._extract_sql("SELECT 1") == "SELECT 1"

    def test_extract_code_fence(self):
        assert MACSQLText2SQL._extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_extract_from_decomposer_final_sql(self):
        text = (
            "Sub-question 1: ...\nSQL: SELECT AVG(sal) FROM e\n\n"
            "Final SQL: SELECT nm FROM employees WHERE sal > 50000"
        )
        result = MACSQLText2SQL._extract_sql_from_decomposer(text)
        assert result.startswith("SELECT nm")

    def test_extract_from_decomposer_last_sql(self):
        text = (
            "SQL: SELECT 1\n"
            "SQL: SELECT nm FROM employees"
        )
        result = MACSQLText2SQL._extract_sql_from_decomposer(text)
        assert "nm" in result

    def test_extract_from_decomposer_bare_select(self):
        text = "SELECT COUNT(*) FROM employees"
        result = MACSQLText2SQL._extract_sql_from_decomposer(text)
        assert result == text

    def test_extract_from_decomposer_code_fence_priority(self):
        text = "Final SQL: SELECT wrong\n```sql\nSELECT correct\n```"
        result = MACSQLText2SQL._extract_sql_from_decomposer(text)
        assert result == "SELECT correct"


class TestFullPipeline:
    def test_full_3_agents(self, tmp_path):
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0, max_refine_rounds=2)
        schema = _make_schema()

        mac.llm = MagicMock()
        mac.llm.chat.side_effect = [
            # Selector: JSON response
            json.dumps({"tables": ["employees"], "columns": ["employees.nm"]}),
            # Decomposer: SQL
            "SELECT nm FROM employees",
            # Refiner not called (SQL is valid)
        ]

        result = mac.generate("List all names", schema, db_path=db_path)
        assert result == "SELECT nm FROM employees"
        # 2 calls: Selector + Decomposer (Refiner skipped since SQL succeeds)
        assert mac.llm.chat.call_count == 2

    def test_full_with_refiner(self, tmp_path):
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0, max_refine_rounds=2)
        schema = _make_schema()

        mac.llm = MagicMock()
        mac.llm.chat.side_effect = [
            # Selector
            json.dumps({"tables": ["employees"], "columns": ["employees.nm"]}),
            # Decomposer: broken SQL
            "SELECT nonexistent FROM employees",
            # Refiner round 1: fix it
            "SELECT nm FROM employees",
        ]

        result = mac.generate("List names", schema, db_path=db_path)
        assert result == "SELECT nm FROM employees"
        assert mac.llm.chat.call_count == 3  # Selector + Decomposer + Refiner

    def test_full_with_evidence(self, tmp_path):
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0, max_refine_rounds=0)
        schema = _make_schema()

        mac.llm = MagicMock()
        mac.llm.chat.side_effect = [
            json.dumps({"tables": ["employees"], "columns": ["employees.sal"]}),
            "SELECT sal FROM employees WHERE sal > 50000",
        ]

        mac.generate(
            "high salary employees", schema, db_path=db_path,
            evidence="high salary means sal > 50000",
        )

        # Check hint passed to Selector
        selector_msg = mac.llm.chat.call_args_list[0][0][0][-1]["content"]
        assert "Hint: high salary means sal > 50000" in selector_msg

        # Check hint passed to Decomposer
        decomp_msg = mac.llm.chat.call_args_list[1][0][0][-1]["content"]
        assert "Hint: high salary means sal > 50000" in decomp_msg

    def test_full_no_db_path_skips_refiner(self):
        """Without db_path, Refiner should be skipped."""
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        mac = MACSQLText2SQL(config, sample_rows=0, max_refine_rounds=2)
        schema = _make_schema()

        mac.llm = MagicMock()
        mac.llm.chat.side_effect = [
            json.dumps({"tables": ["employees"], "columns": ["employees.nm"]}),
            "SELECT nm FROM employees",
        ]

        result = mac.generate("List names", schema)  # no db_path
        assert result == "SELECT nm FROM employees"
        assert mac.llm.chat.call_count == 2
