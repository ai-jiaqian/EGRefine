"""Tests for the rewritten Phase 4 VIEW Synthesis (ALTER TABLE RENAME strategy)."""
import json
import os
import sqlite3
import tempfile

import pytest

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase4.view_synthesis import (
    BACKING_PREFIX,
    generate_mapping,
    generate_orig_table_map,
    generate_refined_tables_json,
    generate_views,
    synthesize,
)


# ========== Fixtures ==========


@pytest.fixture
def schema():
    return Schema(
        db_id="financial",
        tables=[
            Table(name="account", columns=[
                Column(name="account_id", table="account", dtype="INTEGER", is_pk=True),
                Column(name="freq", table="account", dtype="TEXT"),
                Column(name="dt", table="account", dtype="TEXT"),
            ]),
            Table(name="loan", columns=[
                Column(name="loan_id", table="loan", dtype="INTEGER", is_pk=True),
                Column(name="status", table="loan", dtype="TEXT"),
                Column(name="amt", table="loan", dtype="REAL"),
            ]),
            Table(name="district", columns=[
                Column(name="id", table="district", dtype="INTEGER", is_pk=True),
                Column(name="name", table="district", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[
            ("account.account_id", "loan.loan_id"),
        ],
    )


@pytest.fixture
def results():
    """SelectionResults where some columns in account and loan are changed."""
    return [
        SelectionResult(
            column=Column(name="freq", table="account", dtype="TEXT"),
            selected_name="transaction_frequency",
            delta=0.08, was_changed=True,
            all_scores={"freq": 0.62, "transaction_frequency": 0.70},
        ),
        SelectionResult(
            column=Column(name="dt", table="account", dtype="TEXT"),
            selected_name="account_date",
            delta=0.05, was_changed=True,
            all_scores={"dt": 0.55, "account_date": 0.60},
        ),
        SelectionResult(
            column=Column(name="status", table="loan", dtype="TEXT"),
            selected_name="loan_status",
            delta=0.03, was_changed=True,
            all_scores={"status": 0.60, "loan_status": 0.63},
        ),
        SelectionResult(
            column=Column(name="amt", table="loan", dtype="REAL"),
            selected_name="amt",
            delta=0.0, was_changed=False,
            all_scores={"amt": 0.70},
        ),
    ]


# ========== generate_views: ALTER + CREATE VIEW format ==========


class TestGenerateViews:
    def test_script_contains_alter_and_create_view(self, schema, results):
        views = generate_views(schema, results)
        for v in views:
            assert "ALTER TABLE" in v
            assert "RENAME TO" in v
            assert "CREATE VIEW" in v
            assert v.startswith("BEGIN;")
            assert v.endswith("COMMIT;")

    def test_view_uses_original_table_name(self, schema, results):
        """The VIEW should have the original table name, not a 'refined_' prefix."""
        views = generate_views(schema, results)
        account_view = next(v for v in views if "account" in v)
        # Should say CREATE VIEW "account", not CREATE VIEW "refined_account"
        assert 'CREATE VIEW "account" AS' in account_view
        assert "refined_account" not in account_view

    def test_alter_renames_to_backing(self, schema, results):
        views = generate_views(schema, results)
        account_view = next(v for v in views if "account" in v)
        assert 'ALTER TABLE "account" RENAME TO "_orig_account";' in account_view

    def test_from_clause_uses_backing_name(self, schema, results):
        views = generate_views(schema, results)
        account_view = next(v for v in views if "account" in v)
        assert 'FROM "_orig_account";' in account_view

    def test_changed_columns_get_alias(self, schema, results):
        views = generate_views(schema, results)
        account_view = next(v for v in views if "account" in v)
        assert '"freq" AS "transaction_frequency"' in account_view
        assert '"dt" AS "account_date"' in account_view

    def test_unchanged_columns_kept_as_is(self, schema, results):
        views = generate_views(schema, results)
        account_view = next(v for v in views if "account" in v)
        # account_id is unchanged -- should appear without AS
        assert '  "account_id"' in account_view
        assert '"account_id" AS' not in account_view

    def test_no_views_when_no_changes(self, schema):
        views = generate_views(schema, [])
        assert views == []

    def test_no_views_when_all_unchanged(self, schema):
        unchanged = [
            SelectionResult(
                column=Column(name="freq", table="account", dtype="TEXT"),
                selected_name="freq", delta=0.0, was_changed=False,
                all_scores={},
            ),
        ]
        views = generate_views(schema, unchanged)
        assert views == []

    def test_unchanged_table_excluded(self, schema, results):
        """district has no changes -- should not get a VIEW."""
        views = generate_views(schema, results)
        combined = "\n".join(views)
        assert "district" not in combined

    def test_only_changed_tables_get_views(self, schema, results):
        views = generate_views(schema, results)
        # account and loan have changes
        assert len(views) == 2

    def test_sql_executable_on_real_sqlite(self, tmp_path):
        """Build a real SQLite DB, apply the generated script, and query the VIEW."""
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE employees (id INTEGER PRIMARY KEY, nm TEXT, sal REAL)"
        )
        conn.execute("INSERT INTO employees VALUES (1, 'Alice', 50000)")
        conn.execute("INSERT INTO employees VALUES (2, 'Bob', 60000)")
        conn.commit()

        schema = Schema(
            db_id="test",
            tables=[
                Table(name="employees", columns=[
                    Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
                    Column(name="nm", table="employees", dtype="TEXT"),
                    Column(name="sal", table="employees", dtype="REAL"),
                ]),
            ],
            foreign_keys=[],
        )
        sel_results = [
            SelectionResult(
                column=Column(name="nm", table="employees", dtype="TEXT"),
                selected_name="employee_name",
                delta=0.1, was_changed=True,
                all_scores={"nm": 0.5, "employee_name": 0.6},
            ),
        ]

        views = generate_views(schema, sel_results)
        assert len(views) == 1

        # Execute the script (contains BEGIN/COMMIT transaction wrapper)
        conn.executescript(views[0])

        # Query through the VIEW using refined column name
        rows = conn.execute(
            "SELECT employee_name, sal FROM employees ORDER BY id"
        ).fetchall()
        assert rows == [("Alice", 50000.0), ("Bob", 60000.0)]

        # Original table is now _orig_employees
        backing_rows = conn.execute(
            "SELECT nm, sal FROM _orig_employees ORDER BY id"
        ).fetchall()
        assert backing_rows == [("Alice", 50000.0), ("Bob", 60000.0)]

        conn.close()


# ========== generate_orig_table_map ==========


class TestGenerateOrigTableMap:
    def test_includes_changed_tables(self, results):
        m = generate_orig_table_map(results)
        assert m["account"] == "_orig_account"
        assert m["loan"] == "_orig_loan"

    def test_excludes_unchanged_tables(self, results):
        """district has no results; also 'amt' was not changed so if only
        unchanged results exist for a table it should not appear."""
        # Only unchanged result for loan
        only_unchanged = [
            SelectionResult(
                column=Column(name="amt", table="loan", dtype="REAL"),
                selected_name="amt", delta=0.0, was_changed=False,
                all_scores={},
            ),
        ]
        m = generate_orig_table_map(only_unchanged)
        assert "loan" not in m

    def test_empty(self):
        assert generate_orig_table_map([]) == {}


# ========== generate_refined_tables_json ==========


class TestGenerateRefinedTablesJson:
    def test_all_tables_present(self, schema, results):
        rt = generate_refined_tables_json(schema, results)
        table_names = [t["name"] for t in rt["tables"]]
        assert "account" in table_names
        assert "loan" in table_names
        assert "district" in table_names  # unchanged table also present

    def test_db_id(self, schema, results):
        rt = generate_refined_tables_json(schema, results)
        assert rt["db_id"] == "financial"

    def test_refined_column_names(self, schema, results):
        rt = generate_refined_tables_json(schema, results)
        account = next(t for t in rt["tables"] if t["name"] == "account")
        col_names = {c["name"] for c in account["columns"]}
        assert "transaction_frequency" in col_names
        assert "account_date" in col_names
        assert "account_id" in col_names  # unchanged

    def test_original_name_preserved(self, schema, results):
        rt = generate_refined_tables_json(schema, results)
        account = next(t for t in rt["tables"] if t["name"] == "account")
        freq_col = next(c for c in account["columns"] if c["name"] == "transaction_frequency")
        assert freq_col["original_name"] == "freq"

    def test_unchanged_column_original_name(self, schema, results):
        rt = generate_refined_tables_json(schema, results)
        account = next(t for t in rt["tables"] if t["name"] == "account")
        pk_col = next(c for c in account["columns"] if c["name"] == "account_id")
        assert pk_col["original_name"] == "account_id"

    def test_column_dtype_and_is_pk(self, schema, results):
        rt = generate_refined_tables_json(schema, results)
        account = next(t for t in rt["tables"] if t["name"] == "account")
        pk_col = next(c for c in account["columns"] if c["name"] == "account_id")
        assert pk_col["dtype"] == "INTEGER"
        assert pk_col["is_pk"] is True

    def test_fk_uses_original_names_when_not_refined(self, schema, results):
        """FKs that reference non-refined columns keep their original names."""
        rt = generate_refined_tables_json(schema, results)
        # account.account_id and loan.loan_id are not refined
        assert ["account.account_id", "loan.loan_id"] in rt["foreign_keys"]

    def test_fk_translated_when_column_refined(self):
        """If a FK column is refined, the FK ref should use the new name."""
        schema = Schema(
            db_id="test",
            tables=[
                Table(name="orders", columns=[
                    Column(name="oid", table="orders", dtype="INTEGER", is_pk=True),
                    Column(name="cid", table="orders", dtype="INTEGER"),
                ]),
                Table(name="customers", columns=[
                    Column(name="cid", table="customers", dtype="INTEGER", is_pk=True),
                ]),
            ],
            foreign_keys=[("orders.cid", "customers.cid")],
        )
        sel_results = [
            SelectionResult(
                column=Column(name="cid", table="orders", dtype="INTEGER"),
                selected_name="customer_id",
                delta=0.05, was_changed=True,
                all_scores={},
            ),
        ]
        rt = generate_refined_tables_json(schema, sel_results)
        assert ["orders.customer_id", "customers.cid"] in rt["foreign_keys"]


# ========== generate_mapping (kept as-is) ==========


class TestGenerateMapping:
    def test_basic(self, results):
        mapping = generate_mapping(results)
        assert mapping["account.freq"] == "transaction_frequency"
        assert mapping["account.dt"] == "account_date"
        assert mapping["loan.status"] == "loan_status"

    def test_excludes_unchanged(self, results):
        mapping = generate_mapping(results)
        assert "loan.amt" not in mapping

    def test_empty(self):
        assert generate_mapping([]) == {}


# ========== synthesize ==========


class TestSynthesize:
    def test_returns_all_keys(self, schema, results):
        out = synthesize(schema, results)
        assert "views" in out
        assert "mapping" in out
        assert "reverse_mapping" in out
        assert "orig_table_map" in out
        assert "refined_tables" in out
        assert "statistics" in out

    def test_statistics_values(self, schema, results):
        out = synthesize(schema, results)
        stats = out["statistics"]
        assert stats["columns_refined"] == 3
        assert stats["columns_kept_original"] == 1
        assert stats["tables_with_views"] == 2

    def test_saves_four_files(self, schema, results, tmp_path):
        synthesize(schema, results, output_dir=str(tmp_path))
        assert (tmp_path / "views.sql").exists()
        assert (tmp_path / "refined_tables.json").exists()
        assert (tmp_path / "orig_table_map.json").exists()
        assert (tmp_path / "statistics.json").exists()

    def test_views_sql_content(self, schema, results, tmp_path):
        synthesize(schema, results, output_dir=str(tmp_path))
        sql = (tmp_path / "views.sql").read_text()
        assert "ALTER TABLE" in sql
        assert "CREATE VIEW" in sql

    def test_views_sql_joined_with_double_newline(self, schema, results, tmp_path):
        synthesize(schema, results, output_dir=str(tmp_path))
        sql = (tmp_path / "views.sql").read_text()
        # Two table scripts separated by double newline
        assert "\n\n" in sql

    def test_refined_tables_json_content(self, schema, results, tmp_path):
        synthesize(schema, results, output_dir=str(tmp_path))
        with open(tmp_path / "refined_tables.json") as f:
            data = json.load(f)
        assert data["db_id"] == "financial"
        assert len(data["tables"]) == 3

    def test_orig_table_map_json_content(self, schema, results, tmp_path):
        synthesize(schema, results, output_dir=str(tmp_path))
        with open(tmp_path / "orig_table_map.json") as f:
            data = json.load(f)
        assert data["account"] == "_orig_account"
        assert data["loan"] == "_orig_loan"
        assert "district" not in data

    def test_statistics_json_content(self, schema, results, tmp_path):
        synthesize(schema, results, output_dir=str(tmp_path))
        with open(tmp_path / "statistics.json") as f:
            data = json.load(f)
        assert data["columns_refined"] == 3

    def test_no_output_dir_no_files(self, schema, results):
        """When output_dir is None, no files should be written."""
        out = synthesize(schema, results)
        assert out["views"]  # data still returned
