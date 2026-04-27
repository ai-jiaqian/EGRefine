"""Tests for C3 Text-to-SQL implementation."""
import json
import sqlite3
from unittest.mock import MagicMock, patch, call

import pytest

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase3.c3_runner import C3Text2SQL


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


class TestCompactSchemaFormat:
    """Test Stage 1 schema formatting."""

    def test_compact_format(self):
        schema = _make_schema()
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1)
        result = c3._format_compact_schema(schema)

        assert "employees(id, nm, sal, dept_id)" in result
        assert "departments(id, name)" in result
        assert "Foreign Keys:" in result
        assert "employees.dept_id = departments.id" in result


class TestSchemaLinking:
    """Test Stage 1 schema linking."""

    def test_parse_linking_response_valid(self):
        schema = _make_schema()
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1)

        response = json.dumps({
            "tables": ["employees"],
            "columns": ["employees.nm", "employees.sal"],
        })
        tables, columns = c3._parse_linking_response(response, schema)

        assert "employees" in tables
        assert "employees.nm" in columns
        assert "employees.sal" in columns

    def test_parse_linking_response_with_code_fence(self):
        schema = _make_schema()
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1)

        response = '```json\n{"tables": ["employees"], "columns": ["employees.nm"]}\n```'
        tables, columns = c3._parse_linking_response(response, schema)

        assert tables == ["employees"]
        assert columns == ["employees.nm"]

    def test_parse_linking_response_invalid_fallback(self):
        schema = _make_schema()
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1)

        # Invalid JSON → should fallback to full schema
        tables, columns = c3._parse_linking_response("not json", schema)

        assert len(tables) == 2  # all tables
        assert len(columns) == 6  # all columns

    def test_parse_linking_filters_invalid_names(self):
        schema = _make_schema()
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1)

        response = json.dumps({
            "tables": ["employees", "nonexistent_table"],
            "columns": ["employees.nm", "employees.fake_col"],
        })
        tables, columns = c3._parse_linking_response(response, schema)

        assert "nonexistent_table" not in tables
        assert "employees.fake_col" not in columns
        assert "employees" in tables
        assert "employees.nm" in columns

    def test_linking_adds_table_for_column(self):
        """If a column is linked but its table isn't, table should be added."""
        schema = _make_schema()
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1)

        response = json.dumps({
            "tables": [],  # no tables explicitly
            "columns": ["departments.name"],
        })
        tables, columns = c3._parse_linking_response(response, schema)

        assert "departments" in tables


class TestFilterSchema:
    """Test schema filtering after linking."""

    def test_filter_keeps_linked_tables(self):
        schema = _make_schema()
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1)

        filtered = c3._filter_schema(
            schema,
            tables=["employees"],
            columns=["employees.nm", "employees.sal"],
        )

        assert len(filtered.tables) == 1
        assert filtered.tables[0].name == "employees"
        col_names = {c.name for c in filtered.tables[0].columns}
        assert "nm" in col_names
        assert "sal" in col_names

    def test_filter_keeps_primary_keys(self):
        """PKs should always be kept even if not in linked columns."""
        schema = _make_schema()
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1)

        filtered = c3._filter_schema(
            schema,
            tables=["employees"],
            columns=["employees.nm"],  # id not included
        )

        col_names = {c.name for c in filtered.tables[0].columns}
        assert "id" in col_names  # PK kept
        assert "nm" in col_names


class TestDDLSchema:
    """Test Stage 2 DDL formatting."""

    def test_ddl_format(self):
        schema = _make_schema()
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1)
        result = c3._format_ddl_schema(schema)

        assert "CREATE TABLE employees" in result
        assert "nm TEXT" in result
        assert "FOREIGN KEY (dept_id) REFERENCES departments(id)" in result


class TestHints:
    """Test calibration hints building."""

    def test_sample_value_hints(self, tmp_path):
        schema = _make_schema()
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1, sample_rows=2)

        hints = c3._build_hints(schema, db_path)

        assert "employees.nm" in hints
        assert "Alice" in hints or "Bob" in hints

    def test_description_hints(self, tmp_path):
        """Test loading column descriptions from CSV."""
        # Create mock database_description CSV
        desc_dir = tmp_path / "test_db" / "database_description"
        desc_dir.mkdir(parents=True)
        csv_path = desc_dir / "employees.csv"
        csv_path.write_text(
            "original_column_name,column_name,column_description,"
            "data_format,value_description,\n"
            "nm,name,employee full name,text,,\n"
            "sal,salary,annual salary in USD,real,,\n"
        )

        schema = _make_schema()
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1, desc_dir=str(tmp_path))

        hints = c3._build_hints(schema)

        assert "employee full name" in hints
        assert "annual salary" in hints


class TestSelfConsistency:
    """Test Stage 3 self-consistency voting."""

    def test_majority_vote(self, tmp_path):
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=5)

        candidates = [
            "SELECT nm FROM employees WHERE id=1",  # → Alice
            "SELECT nm FROM employees WHERE id=1",  # → Alice (same result)
            "SELECT nm FROM employees WHERE id=1",  # → Alice (same result)
            "SELECT nm FROM employees WHERE id=2",  # → Bob
            "SELECT sal FROM employees WHERE id=1",  # → 50000
        ]

        result = c3._self_consistency(candidates, db_path)
        # Majority (3/5) returns Alice
        assert result == "SELECT nm FROM employees WHERE id=1"

    def test_all_errors_fallback(self, tmp_path):
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=3)

        candidates = [
            "SELECT * FROM nonexistent",
            "INVALID SQL",
            "ALSO INVALID",
        ]

        result = c3._self_consistency(candidates, db_path)
        # All errors → fallback to first candidate
        assert result == candidates[0]

    def test_single_candidate(self, tmp_path):
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1)

        result = c3._self_consistency(["SELECT 1"], db_path)
        assert result == "SELECT 1"


class TestFullGenerate:
    """Test the full generate() pipeline with mocked LLM."""

    def test_generate_no_consistency(self):
        """With num_samples=1, should skip self-consistency."""
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=1, sample_rows=0)
        schema = _make_schema()

        # Mock LLM: first call = schema linking, second = SQL generation
        c3.llm = MagicMock()
        c3.llm.chat.side_effect = [
            # Schema linking response
            json.dumps({"tables": ["employees"], "columns": ["employees.nm"]}),
            # SQL generation response
            "SELECT nm FROM employees",
        ]
        c3.llm.temperature = 0

        result = c3.generate("Get all employee names", schema)
        assert result == "SELECT nm FROM employees"
        assert c3.llm.chat.call_count == 2

    def test_generate_with_consistency(self, tmp_path):
        """With num_samples>1, should use self-consistency voting."""
        db_path = _make_test_db(tmp_path)
        config = {"base_url": "x", "api_key": "x", "model_name": "x"}
        c3 = C3Text2SQL(config, num_samples=3, sample_rows=0)
        schema = _make_schema()

        c3.llm = MagicMock()
        c3.llm.temperature = 0
        c3.llm.chat.side_effect = [
            # Schema linking
            json.dumps({"tables": ["employees"], "columns": ["employees.nm"]}),
            # 3 SQL generation calls
            "SELECT nm FROM employees WHERE id=1",
            "SELECT nm FROM employees WHERE id=1",
            "SELECT nm FROM employees WHERE id=2",
        ]

        result = c3.generate("Get Alice's name", schema, db_path=db_path)
        # Majority vote: 2 out of 3 return same result
        assert result == "SELECT nm FROM employees WHERE id=1"
        assert c3.llm.chat.call_count == 4  # 1 linking + 3 generation


class TestExtractSQL:
    """Test SQL extraction from LLM responses."""

    def test_plain_sql(self):
        assert C3Text2SQL._extract_sql("SELECT 1") == "SELECT 1"

    def test_code_fenced_sql(self):
        text = "```sql\nSELECT 1\n```"
        assert C3Text2SQL._extract_sql(text) == "SELECT 1"

    def test_code_fenced_no_lang(self):
        text = "```\nSELECT 1\n```"
        assert C3Text2SQL._extract_sql(text) == "SELECT 1"
