"""Tests for src/eval/evaluator.py — Stage 2 evaluator."""

from __future__ import annotations

import json
import sqlite3
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.eval.evaluator import (
    EvalMethodResult,
    QueryResult,
    evaluate_method,
    load_refined_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_original_db(path: Path) -> None:
    """Create a small SQLite database with one table."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, nm TEXT, sal INTEGER)")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice', 5000)")
    conn.execute("INSERT INTO employees VALUES (2, 'Bob', 6000)")
    conn.commit()
    conn.close()


def _make_refined_db(path: Path) -> None:
    """Create a refined database: backing table + VIEW with renamed columns."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE _orig_employees (id INTEGER PRIMARY KEY, nm TEXT, sal INTEGER)")
    conn.execute("INSERT INTO _orig_employees VALUES (1, 'Alice', 5000)")
    conn.execute("INSERT INTO _orig_employees VALUES (2, 'Bob', 6000)")
    conn.execute(textwrap.dedent("""\
        CREATE VIEW employees AS
        SELECT id, nm AS employee_name, sal AS salary
        FROM _orig_employees
    """))
    conn.commit()
    conn.close()


def _original_schema() -> Schema:
    return Schema(
        db_id="test",
        tables=[Table(name="employees", columns=[
            Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
            Column(name="nm", table="employees", dtype="TEXT"),
            Column(name="sal", table="employees", dtype="INTEGER"),
        ])],
        foreign_keys=[],
    )


def _refined_schema() -> Schema:
    return Schema(
        db_id="test",
        tables=[Table(name="employees", columns=[
            Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
            Column(name="employee_name", table="employees", dtype="TEXT"),
            Column(name="salary", table="employees", dtype="INTEGER"),
        ])],
        foreign_keys=[],
    )


# ---------------------------------------------------------------------------
# test_basic_evaluation
# ---------------------------------------------------------------------------

class TestBasicEvaluation:
    """Both original and refined return correct SQL -> both ExAcc = 1.0."""

    def test_both_correct(self, tmp_path: Path) -> None:
        orig_db = tmp_path / "orig.sqlite"
        ref_db = tmp_path / "refined.sqlite"
        _make_original_db(orig_db)
        _make_refined_db(ref_db)

        pairs = [
            NLSQLPair(
                nl="Who earns more than 5000?",
                gold_sql="SELECT nm FROM employees WHERE sal > 5000",
                db_id="test",
            ),
        ]

        # Model returns correct SQL for both schemas
        model = MagicMock()
        model.generate = MagicMock(side_effect=[
            # Original: correct SQL using original column names
            "SELECT nm FROM employees WHERE sal > 5000",
            # Refined: correct SQL using refined column names
            "SELECT employee_name FROM employees WHERE salary > 5000",
        ])

        table_map = {"employees": "_orig_employees"}

        result = evaluate_method(
            model=model,
            pairs=pairs,
            original_schema=_original_schema(),
            refined_schema=_refined_schema(),
            original_db_path=str(orig_db),
            refined_db_path=str(ref_db),
            table_map=table_map,
            method_name="test_model",
        )

        assert result.exacc_original == 1.0
        assert result.exacc_refined == 1.0
        assert result.delta == 0.0
        assert result.total_queries == 1
        assert len(result.original_details) == 1
        assert len(result.refined_details) == 1
        assert result.original_details[0].match is True
        assert result.refined_details[0].match is True


# ---------------------------------------------------------------------------
# test_refined_improves
# ---------------------------------------------------------------------------

class TestRefinedImproves:
    """Model wrong on original, correct on refined -> positive delta."""

    def test_delta_positive(self, tmp_path: Path) -> None:
        orig_db = tmp_path / "orig.sqlite"
        ref_db = tmp_path / "refined.sqlite"
        _make_original_db(orig_db)
        _make_refined_db(ref_db)

        pairs = [
            NLSQLPair(
                nl="Who earns more than 5000?",
                gold_sql="SELECT nm FROM employees WHERE sal > 5000",
                db_id="test",
            ),
        ]

        model = MagicMock()
        model.generate = MagicMock(side_effect=[
            # Original: wrong SQL (model confused by cryptic column name)
            "SELECT nm FROM employees WHERE nm > 5000",
            # Refined: correct SQL (clear column names help)
            "SELECT employee_name FROM employees WHERE salary > 5000",
        ])

        table_map = {"employees": "_orig_employees"}

        result = evaluate_method(
            model=model,
            pairs=pairs,
            original_schema=_original_schema(),
            refined_schema=_refined_schema(),
            original_db_path=str(orig_db),
            refined_db_path=str(ref_db),
            table_map=table_map,
            method_name="test_model",
        )

        assert result.exacc_original == 0.0
        assert result.exacc_refined == 1.0
        assert result.delta == 1.0
        assert result.original_details[0].match is False
        assert result.refined_details[0].match is True


# ---------------------------------------------------------------------------
# test_to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_round_trip(self) -> None:
        qr = QueryResult(nl="q", gold_sql="SELECT 1", pred_sql="SELECT 1", match=True)
        result = EvalMethodResult(
            method="m", db_id="db", exacc_original=0.5, exacc_refined=0.7,
            delta=0.2, total_queries=2,
            original_details=[qr], refined_details=[qr],
        )
        d = result.to_dict()
        assert d["method"] == "m"
        assert d["delta"] == 0.2
        assert len(d["original_details"]) == 1
        assert d["original_details"][0]["match"] is True


# ---------------------------------------------------------------------------
# test_empty_pairs
# ---------------------------------------------------------------------------

class TestEmptyPairs:
    def test_empty_returns_zeros(self, tmp_path: Path) -> None:
        model = MagicMock()
        result = evaluate_method(
            model=model,
            pairs=[],
            original_schema=_original_schema(),
            refined_schema=_refined_schema(),
            original_db_path="dummy",
            refined_db_path="dummy",
            table_map={},
            method_name="empty",
        )
        assert result.exacc_original == 0.0
        assert result.exacc_refined == 0.0
        assert result.total_queries == 0
        model.generate.assert_not_called()


# ---------------------------------------------------------------------------
# test_load_refined_schema
# ---------------------------------------------------------------------------

class TestLoadRefinedSchema:
    def test_load_from_json(self, tmp_path: Path) -> None:
        data = {
            "db_id": "test",
            "tables": [
                {
                    "name": "employees",
                    "columns": [
                        {"name": "id", "original_name": "id", "dtype": "INTEGER", "is_pk": True},
                        {"name": "employee_name", "original_name": "nm", "dtype": "TEXT", "is_pk": False},
                        {"name": "salary", "original_name": "sal", "dtype": "INTEGER", "is_pk": False},
                    ],
                }
            ],
            "foreign_keys": [["employees.id", "dept.emp_id"]],
        }
        json_path = tmp_path / "refined_tables.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        schema = load_refined_schema(json_path)

        assert schema.db_id == "test"
        assert len(schema.tables) == 1
        assert schema.tables[0].name == "employees"
        assert len(schema.tables[0].columns) == 3

        col_names = [c.name for c in schema.tables[0].columns]
        assert col_names == ["id", "employee_name", "salary"]

        assert schema.tables[0].columns[0].is_pk is True
        assert schema.tables[0].columns[1].is_pk is False

        assert len(schema.foreign_keys) == 1
        assert schema.foreign_keys[0] == ("employees.id", "dept.emp_id")

    def test_load_empty_tables(self, tmp_path: Path) -> None:
        data = {"db_id": "empty", "tables": [], "foreign_keys": []}
        json_path = tmp_path / "empty.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        schema = load_refined_schema(json_path)
        assert schema.db_id == "empty"
        assert schema.tables == []
        assert schema.all_columns == []
