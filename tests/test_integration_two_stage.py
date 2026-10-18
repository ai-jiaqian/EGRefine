"""Integration test: Stage 1 (synthesis) -> Stage 2 (evaluation) end-to-end.

Creates a real SQLite database, runs Phase 4 synthesis to produce artifacts,
applies them to a copy of the database, then evaluates with a mock
Text-to-SQL model that behaves differently on original vs refined schemas.
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase4.view_synthesis import synthesize
from egrefine.eval.db_setup import copy_database, apply_views
from egrefine.eval.evaluator import evaluate_method, load_refined_schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace with a real SQLite database."""
    # -- Build the SQLite database --
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        'CREATE TABLE "district" ('
        '  "d_id" INTEGER PRIMARY KEY,'
        '  "A2" TEXT,'
        '  "A11" REAL'
        ")"
    )
    conn.execute(
        'INSERT INTO "district" VALUES (1, "Prague", 12541.0)'
    )
    conn.execute(
        'INSERT INTO "district" VALUES (2, "Beroun", 8507.0)'
    )
    conn.commit()
    conn.close()

    # -- Build Schema object --
    col_d_id = Column(name="d_id", table="district", dtype="INTEGER", is_pk=True)
    col_a2 = Column(name="A2", table="district", dtype="TEXT")
    col_a11 = Column(name="A11", table="district", dtype="REAL")
    table = Table(name="district", columns=[col_d_id, col_a2, col_a11])
    schema = Schema(db_id="test", tables=[table], foreign_keys=[])

    # -- Build SelectionResults --
    results = [
        SelectionResult(
            column=col_a2,
            selected_name="city_name",
            delta=0.15,
            was_changed=True,
            all_scores={"A2": 0.50, "city_name": 0.65},
            verification_method="execution",
        ),
        SelectionResult(
            column=col_a11,
            selected_name="average_salary",
            delta=0.10,
            was_changed=True,
            all_scores={"A11": 0.55, "average_salary": 0.65},
            verification_method="execution",
        ),
    ]

    return {
        "tmp_path": tmp_path,
        "db_path": db_path,
        "schema": schema,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStage1Artifacts:
    """Verify that synthesize() produces the expected artifact files."""

    def test_all_artifact_files_exist(self, workspace):
        output_dir = str(workspace["tmp_path"] / "stage1_output")
        synthesize(workspace["schema"], workspace["results"], output_dir=output_dir)

        expected_files = [
            "views.sql",
            "refined_tables.json",
            "orig_table_map.json",
            "statistics.json",
        ]
        for fname in expected_files:
            fpath = Path(output_dir) / fname
            assert fpath.exists(), f"Missing artifact: {fname}"

    def test_views_sql_content(self, workspace):
        output_dir = str(workspace["tmp_path"] / "stage1_output")
        synthesize(workspace["schema"], workspace["results"], output_dir=output_dir)

        views_sql = (Path(output_dir) / "views.sql").read_text()
        # Must contain the ALTER TABLE RENAME and CREATE VIEW
        assert '_orig_district' in views_sql
        assert 'city_name' in views_sql
        assert 'average_salary' in views_sql


class TestDatabaseWithViews:
    """Verify that views.sql can be applied and queries work correctly."""

    def test_refined_query_works(self, workspace):
        """SELECT city_name FROM district should work on the refined copy."""
        output_dir = str(workspace["tmp_path"] / "stage1_output")
        synthesize(workspace["schema"], workspace["results"], output_dir=output_dir)

        # Copy database and apply views
        refined_db = workspace["tmp_path"] / "refined.sqlite"
        copy_database(workspace["db_path"], refined_db)
        apply_views(refined_db, Path(output_dir) / "views.sql")

        conn = sqlite3.connect(str(refined_db))
        rows = conn.execute("SELECT city_name FROM district").fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0][0] == "Prague"
        assert rows[1][0] == "Beroun"

    def test_original_query_on_backing_table(self, workspace):
        """SELECT A2 FROM _orig_district should work on the refined copy."""
        output_dir = str(workspace["tmp_path"] / "stage1_output")
        synthesize(workspace["schema"], workspace["results"], output_dir=output_dir)

        refined_db = workspace["tmp_path"] / "refined.sqlite"
        copy_database(workspace["db_path"], refined_db)
        apply_views(refined_db, Path(output_dir) / "views.sql")

        conn = sqlite3.connect(str(refined_db))
        rows = conn.execute("SELECT A2 FROM _orig_district").fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0][0] == "Prague"

    def test_average_salary_via_view(self, workspace):
        """SELECT average_salary FROM district should return REAL values."""
        output_dir = str(workspace["tmp_path"] / "stage1_output")
        synthesize(workspace["schema"], workspace["results"], output_dir=output_dir)

        refined_db = workspace["tmp_path"] / "refined.sqlite"
        copy_database(workspace["db_path"], refined_db)
        apply_views(refined_db, Path(output_dir) / "views.sql")

        conn = sqlite3.connect(str(refined_db))
        rows = conn.execute(
            "SELECT average_salary FROM district ORDER BY average_salary"
        ).fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0][0] == pytest.approx(8507.0)
        assert rows[1][0] == pytest.approx(12541.0)


class TestEndToEndEvaluation:
    """Full Stage 1 -> Stage 2 flow with a mock Text-to-SQL model."""

    def test_evaluate_method_delta(self, workspace):
        """Mock model is confused by A2 but understands city_name.

        Expected: exacc_original=0.0, exacc_refined=1.0, delta=1.0.
        """
        output_dir = str(workspace["tmp_path"] / "stage1_output")
        synth = synthesize(
            workspace["schema"], workspace["results"], output_dir=output_dir
        )

        # Prepare refined database
        refined_db = workspace["tmp_path"] / "refined.sqlite"
        copy_database(workspace["db_path"], refined_db)
        apply_views(refined_db, Path(output_dir) / "views.sql")

        # Load refined schema from the artifact
        refined_schema = load_refined_schema(
            Path(output_dir) / "refined_tables.json"
        )

        # Build NL-SQL pair
        # Gold SQL: find the city name for district 1
        pairs = [
            NLSQLPair(
                nl="What is the city name of district 1?",
                gold_sql='SELECT A2 FROM district WHERE d_id = 1',
                db_id="test",
            ),
        ]

        # Mock Text-to-SQL model:
        # - On original schema (columns: d_id, A2, A11): generates wrong SQL
        # - On refined schema (columns: d_id, city_name, average_salary): correct
        model = MagicMock()

        def mock_generate(nl, schema, db_path, **kwargs):
            col_names = {c.name for c in schema.all_columns}
            if "city_name" in col_names:
                # Refined schema -> model understands, generates correct SQL
                return 'SELECT city_name FROM district WHERE d_id = 1'
            else:
                # Original schema -> model confused by A2, generates wrong SQL
                return 'SELECT A11 FROM district WHERE d_id = 1'

        model.generate = mock_generate

        # Table map for gold SQL remapping
        table_map = synth["orig_table_map"]

        result = evaluate_method(
            model=model,
            pairs=pairs,
            original_schema=workspace["schema"],
            refined_schema=refined_schema,
            original_db_path=str(workspace["db_path"]),
            refined_db_path=str(refined_db),
            table_map=table_map,
            method_name="mock_test",
        )

        assert result.exacc_original == pytest.approx(0.0)
        assert result.exacc_refined == pytest.approx(1.0)
        assert result.delta == pytest.approx(1.0)
        assert result.total_queries == 1
