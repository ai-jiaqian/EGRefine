"""T21: Experiment orchestration tests."""
import json
import os
import sqlite3
import tempfile
import pytest

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase3.text2sql_runner import Text2SQLModel
from egrefine.pipeline import PipelineResult
from egrefine.evaluate import EvalResult
from egrefine.experiment import (
    run_experiment,
    save_experiment,
    ExperimentResult,
    MethodResult,
    _build_comparison_table,
)


# ========== Fixtures ==========

@pytest.fixture
def schema():
    return Schema(
        db_id="test_db",
        tables=[
            Table(name="employees", columns=[
                Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
                Column(name="nm", table="employees", dtype="TEXT"),
                Column(name="sal", table="employees", dtype="REAL"),
            ]),
        ],
        foreign_keys=[],
    )


@pytest.fixture
def pairs():
    return [
        NLSQLPair(nl="List all names", gold_sql="SELECT nm FROM employees", db_id="test_db"),
        NLSQLPair(nl="Max salary", gold_sql="SELECT MAX(sal) FROM employees", db_id="test_db"),
    ]


@pytest.fixture
def sample_db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, nm TEXT, sal REAL)")
    conn.executemany(
        "INSERT INTO employees VALUES (?, ?, ?)",
        [(1, "Alice", 50000), (2, "Bob", 60000)],
    )
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


class MockModel(Text2SQLModel):
    """Model that returns correct SQL for original schema."""
    def generate(self, nl, schema, db_path=None, column_mapping=None, evidence=""):
        if "names" in nl.lower():
            return "SELECT nm FROM employees"
        return "SELECT MAX(sal) FROM employees"


def _fake_no_refinement(schema, pairs, db_path):
    return PipelineResult(
        db_id=schema.db_id, refinements=[], views=[],
        mapping={}, reverse_mapping={}, orig_table_map={},
        statistics={"method": "no_refinement", "total_columns": 3,
                    "columns_refined": 0, "columns_kept_original": 0},
    )


def _fake_method_a(schema, pairs, db_path):
    col = Column(name="nm", table="employees", dtype="TEXT")
    results = [SelectionResult(
        column=col, selected_name="employee_name", delta=0.1,
        was_changed=True, all_scores={"nm": 0.5, "employee_name": 0.6},
        verification_method="execution",
    )]
    return PipelineResult(
        db_id=schema.db_id, refinements=results,
        views=["CREATE VIEW employees AS SELECT id, nm AS employee_name, sal FROM _orig_employees;"],
        mapping={"employees.nm": "employee_name"},
        reverse_mapping={"employee_name": "nm"},
        orig_table_map={"employees": "_orig_employees"},
        statistics={"method": "method_a", "total_columns": 3,
                    "columns_refined": 1, "columns_kept_original": 0},
    )


# ========== run_experiment ==========

class TestRunExperiment:
    def test_basic_experiment(self, schema, pairs, sample_db):
        methods = {
            "no_refinement": _fake_no_refinement,
            "method_a": _fake_method_a,
        }
        result = run_experiment(
            db_ids=["test_db"],
            schemas={"test_db": schema},
            pairs=pairs,
            db_path_fn=lambda db_id: sample_db,
            methods=methods,
            eval_model=MockModel(),
        )
        assert isinstance(result, ExperimentResult)
        assert "no_refinement" in result.method_results
        assert "method_a" in result.method_results
        assert len(result.method_results["no_refinement"]) == 1
        assert len(result.method_results["method_a"]) == 1

    def test_eval_results_populated(self, schema, pairs, sample_db):
        methods = {"no_refinement": _fake_no_refinement}
        result = run_experiment(
            db_ids=["test_db"],
            schemas={"test_db": schema},
            pairs=pairs,
            db_path_fn=lambda db_id: sample_db,
            methods=methods,
            eval_model=MockModel(),
        )
        evals = result.method_evals["no_refinement"]
        assert len(evals) == 1
        assert evals[0].db_id == "test_db"
        assert evals[0].total_queries == 2

    def test_aggregates_populated(self, schema, pairs, sample_db):
        methods = {"no_refinement": _fake_no_refinement}
        result = run_experiment(
            db_ids=["test_db"],
            schemas={"test_db": schema},
            pairs=pairs,
            db_path_fn=lambda db_id: sample_db,
            methods=methods,
            eval_model=MockModel(),
        )
        agg = result.method_aggregates["no_refinement"]
        assert "avg_exacc_before" in agg
        assert "avg_delta" in agg
        assert agg["total_databases"] == 1

    def test_skips_missing_db(self, schema, pairs, sample_db):
        methods = {"no_refinement": _fake_no_refinement}
        result = run_experiment(
            db_ids=["test_db", "nonexistent_db"],
            schemas={"test_db": schema},
            pairs=pairs,
            db_path_fn=lambda db_id: sample_db,
            methods=methods,
            eval_model=MockModel(),
        )
        assert len(result.method_results["no_refinement"]) == 1

    def test_skips_db_with_no_pairs(self, schema, pairs, sample_db):
        methods = {"no_refinement": _fake_no_refinement}
        # test_db2 has a schema but no pairs
        schema2 = Schema(db_id="test_db2", tables=[], foreign_keys=[])
        result = run_experiment(
            db_ids=["test_db", "test_db2"],
            schemas={"test_db": schema, "test_db2": schema2},
            pairs=pairs,
            db_path_fn=lambda db_id: sample_db,
            methods=methods,
            eval_model=MockModel(),
        )
        assert len(result.method_results["no_refinement"]) == 1

    def test_multiple_dbs(self, schema, pairs, sample_db):
        """Same DB twice to simulate multiple databases."""
        schema2 = Schema(
            db_id="test_db2",
            tables=schema.tables,
            foreign_keys=[],
        )
        pairs2 = [
            NLSQLPair(nl="List all names", gold_sql="SELECT nm FROM employees", db_id="test_db2"),
        ]
        methods = {"no_refinement": _fake_no_refinement}
        result = run_experiment(
            db_ids=["test_db", "test_db2"],
            schemas={"test_db": schema, "test_db2": schema2},
            pairs=pairs + pairs2,
            db_path_fn=lambda db_id: sample_db,
            methods=methods,
            eval_model=MockModel(),
        )
        assert len(result.method_results["no_refinement"]) == 2
        assert result.method_aggregates["no_refinement"]["total_databases"] == 2

    def test_to_dict_serializable(self, schema, pairs, sample_db):
        methods = {
            "no_refinement": _fake_no_refinement,
            "method_a": _fake_method_a,
        }
        result = run_experiment(
            db_ids=["test_db"],
            schemas={"test_db": schema},
            pairs=pairs,
            db_path_fn=lambda db_id: sample_db,
            methods=methods,
            eval_model=MockModel(),
        )
        d = result.to_dict()
        json_str = json.dumps(d)
        assert json_str
        assert "no_refinement" in d
        assert "method_a" in d


# ========== save_experiment ==========

class TestSaveExperiment:
    def test_saves_files(self, schema, pairs, sample_db, tmp_path):
        methods = {
            "no_refinement": _fake_no_refinement,
            "method_a": _fake_method_a,
        }
        result = run_experiment(
            db_ids=["test_db"],
            schemas={"test_db": schema},
            pairs=pairs,
            db_path_fn=lambda db_id: sample_db,
            methods=methods,
            eval_model=MockModel(),
        )
        output_dir = str(tmp_path / "results")
        save_experiment(result, output_dir)

        assert os.path.exists(os.path.join(output_dir, "results.json"))
        assert os.path.exists(os.path.join(output_dir, "table_comparison.tex"))
        assert os.path.exists(os.path.join(output_dir, "table_no_refinement.tex"))
        assert os.path.exists(os.path.join(output_dir, "table_method_a.tex"))

    def test_json_loadable(self, schema, pairs, sample_db, tmp_path):
        methods = {"no_refinement": _fake_no_refinement}
        result = run_experiment(
            db_ids=["test_db"],
            schemas={"test_db": schema},
            pairs=pairs,
            db_path_fn=lambda db_id: sample_db,
            methods=methods,
            eval_model=MockModel(),
        )
        output_dir = str(tmp_path / "results")
        save_experiment(result, output_dir)

        with open(os.path.join(output_dir, "results.json")) as f:
            data = json.load(f)
        assert "no_refinement" in data


# ========== comparison table ==========

class TestComparisonTable:
    def test_has_all_methods(self):
        result = ExperimentResult(
            method_results={},
            method_evals={},
            method_aggregates={
                "no_refinement": {
                    "avg_exacc_before": 0.5, "avg_exacc_after": 0.5,
                    "avg_delta": 0.0, "avg_refinement_precision": 0.0,
                },
                "egrefine": {
                    "avg_exacc_before": 0.5, "avg_exacc_after": 0.6,
                    "avg_delta": 0.1, "avg_refinement_precision": 0.8,
                },
            },
        )
        latex = _build_comparison_table(result)
        assert "no_refinement" in latex
        assert "egrefine" in latex
        assert "\\begin{tabular}" in latex
