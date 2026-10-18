"""T17: Phase 4 VIEW Synthesis tests."""
import json
import os
import tempfile
import pytest

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase4.view_synthesis import generate_views, generate_mapping, synthesize


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
        foreign_keys=[],
    )


@pytest.fixture
def refinements():
    """SelectionResults where some columns are changed."""
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


# ========== generate_views ==========

class TestGenerateViews:
    def test_basic_view_generation(self, schema, refinements):
        views = generate_views(schema, refinements)
        # Only tables with changes get VIEWs: account (2 changes) and loan (1 change)
        assert len(views) == 2

    def test_view_contains_alter_and_create(self, schema, refinements):
        views = generate_views(schema, refinements)
        for v in views:
            assert "ALTER TABLE" in v
            assert "CREATE VIEW" in v

    def test_view_renames_columns(self, schema, refinements):
        views = generate_views(schema, refinements)
        account_view = next(v for v in views if '"account"' in v)
        assert '"freq" AS "transaction_frequency"' in account_view
        assert '"dt" AS "account_date"' in account_view
        # Unchanged columns listed as-is
        assert "account_id" in account_view

    def test_view_preserves_unchanged_columns(self, schema, refinements):
        views = generate_views(schema, refinements)
        loan_view = next(v for v in views if '"loan"' in v)
        assert '"status" AS "loan_status"' in loan_view
        # amt unchanged
        assert '"amt"' in loan_view

    def test_no_view_for_unchanged_table(self, schema, refinements):
        views = generate_views(schema, refinements)
        view_text = "\n".join(views)
        assert "district" not in view_text or "_orig_district" not in view_text

    def test_backing_table_name(self, schema, refinements):
        views = generate_views(schema, refinements)
        account_view = next(v for v in views if '"account"' in v)
        assert "_orig_account" in account_view

    def test_empty_refinements(self, schema):
        views = generate_views(schema, [])
        assert views == []

    def test_all_unchanged(self, schema):
        results = [
            SelectionResult(
                column=Column(name="freq", table="account", dtype="TEXT"),
                selected_name="freq", delta=0.0, was_changed=False,
                all_scores={},
            ),
        ]
        views = generate_views(schema, results)
        assert views == []


# ========== generate_mapping ==========

class TestGenerateMapping:
    def test_basic_mapping(self, refinements):
        mapping = generate_mapping(refinements)
        assert mapping["account.freq"] == "transaction_frequency"
        assert mapping["account.dt"] == "account_date"
        assert mapping["loan.status"] == "loan_status"

    def test_excludes_unchanged(self, refinements):
        mapping = generate_mapping(refinements)
        assert "loan.amt" not in mapping

    def test_empty(self):
        assert generate_mapping([]) == {}

    def test_reverse_mapping(self, refinements):
        mapping = generate_mapping(refinements)
        reverse = {v: k.split(".")[-1] for k, v in mapping.items()}
        assert reverse["transaction_frequency"] == "freq"
        assert reverse["loan_status"] == "status"


# ========== synthesize (full output) ==========

class TestSynthesize:
    def test_synthesize_returns_all_parts(self, schema, refinements):
        result = synthesize(schema, refinements)
        assert "views" in result
        assert "mapping" in result
        assert "reverse_mapping" in result
        assert "statistics" in result

    def test_statistics(self, schema, refinements):
        result = synthesize(schema, refinements)
        stats = result["statistics"]
        assert stats["columns_refined"] == 3
        assert stats["columns_kept_original"] == 1
        assert stats["tables_with_views"] == 2

    def test_reverse_mapping_correctness(self, schema, refinements):
        result = synthesize(schema, refinements)
        rm = result["reverse_mapping"]
        assert rm["transaction_frequency"] == "freq"
        assert rm["account_date"] == "dt"
        assert rm["loan_status"] == "status"

    def test_synthesize_includes_orig_table_map(self, schema, refinements):
        result = synthesize(schema, refinements)
        assert "orig_table_map" in result
        assert result["orig_table_map"]["account"] == "_orig_account"
        assert result["orig_table_map"]["loan"] == "_orig_loan"

    def test_save_to_dir(self, schema, refinements):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = synthesize(schema, refinements, output_dir=tmpdir)
            # Check files created
            assert os.path.exists(os.path.join(tmpdir, "views.sql"))
            assert os.path.exists(os.path.join(tmpdir, "refined_tables.json"))
            assert os.path.exists(os.path.join(tmpdir, "orig_table_map.json"))
            assert os.path.exists(os.path.join(tmpdir, "statistics.json"))

            with open(os.path.join(tmpdir, "views.sql")) as f:
                sql = f.read()
            assert "CREATE VIEW" in sql
            assert "ALTER TABLE" in sql
