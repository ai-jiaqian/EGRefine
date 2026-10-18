"""T19: Evaluation module tests."""
import json
import os
import sqlite3
import tempfile
import pytest

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase3.text2sql_runner import Text2SQLModel
from egrefine.evaluate import (
    evaluate_exacc,
    evaluate_refinement,
    aggregate_results,
    to_latex_table,
    EvalResult,
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
def refined_schema():
    """Schema after refinement: nm -> employee_name."""
    return Schema(
        db_id="test_db",
        tables=[
            Table(name="employees", columns=[
                Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
                Column(name="employee_name", table="employees", dtype="TEXT"),
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
        NLSQLPair(nl="Count employees", gold_sql="SELECT COUNT(*) FROM employees", db_id="test_db"),
        NLSQLPair(nl="Get name by id", gold_sql="SELECT nm FROM employees WHERE id = 1", db_id="test_db"),
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


class CorrectModel(Text2SQLModel):
    """Model that always generates correct SQL for original schema."""
    def generate(self, nl, schema, db_path=None, column_mapping=None, evidence=""):
        if "names" in nl.lower():
            return "SELECT nm FROM employees"
        if "salary" in nl.lower():
            return "SELECT MAX(sal) FROM employees"
        if "count" in nl.lower():
            return "SELECT COUNT(*) FROM employees"
        if "by id" in nl.lower():
            return "SELECT nm FROM employees WHERE id = 1"
        return "SELECT 1"


class PartialModel(Text2SQLModel):
    """Model that gets 2 out of 4 queries right."""
    def generate(self, nl, schema, db_path=None, column_mapping=None, evidence=""):
        if "names" in nl.lower():
            return "SELECT nm FROM employees"
        if "salary" in nl.lower():
            return "SELECT MAX(sal) FROM employees"
        # Wrong for count and by-id
        return "SELECT 1"


class BrokenModel(Text2SQLModel):
    """Model that always generates wrong SQL."""
    def generate(self, nl, schema, db_path=None, column_mapping=None, evidence=""):
        return "SELECT 1"


class RefinedAwareModel(Text2SQLModel):
    """Model that works better with refined schema (employee_name)."""
    def generate(self, nl, schema, db_path=None, column_mapping=None, evidence=""):
        has_employee_name = schema.get_column("employees", "employee_name") is not None
        if has_employee_name:
            # Refined schema: gets all right
            if "names" in nl.lower():
                return "SELECT employee_name FROM employees"
            if "salary" in nl.lower():
                return "SELECT MAX(sal) FROM employees"
            if "count" in nl.lower():
                return "SELECT COUNT(*) FROM employees"
            if "by id" in nl.lower():
                return "SELECT employee_name FROM employees WHERE id = 1"
        else:
            # Original schema: gets 2/4 right
            if "salary" in nl.lower():
                return "SELECT MAX(sal) FROM employees"
            if "count" in nl.lower():
                return "SELECT COUNT(*) FROM employees"
        return "SELECT 1"


@pytest.fixture
def refinement_results():
    """SelectionResult list: nm -> employee_name (changed), sal unchanged."""
    col_nm = Column(name="nm", table="employees", dtype="TEXT")
    col_sal = Column(name="sal", table="employees", dtype="REAL")
    return [
        SelectionResult(
            column=col_nm,
            selected_name="employee_name",
            delta=0.3,
            was_changed=True,
            all_scores={"nm": 0.5, "employee_name": 0.8},
            verification_method="execution",
        ),
        SelectionResult(
            column=col_sal,
            selected_name="sal",
            delta=0.0,
            was_changed=False,
            all_scores={"sal": 0.7, "salary": 0.6},
            verification_method="execution",
        ),
    ]


# ========== evaluate_exacc ==========

class TestEvaluateExacc:
    def test_perfect_model(self, schema, pairs, sample_db):
        """Model that gets all queries right should have ExAcc=1.0."""
        exacc, _ = evaluate_exacc(
            schema=schema,
            pairs=pairs,
            model=CorrectModel(),
            db_path=sample_db,
        )
        assert exacc == 1.0

    def test_broken_model(self, schema, pairs, sample_db):
        """Model that gets nothing right should have ExAcc=0.0."""
        exacc, _ = evaluate_exacc(
            schema=schema,
            pairs=pairs,
            model=BrokenModel(),
            db_path=sample_db,
        )
        assert exacc == 0.0

    def test_partial_model(self, schema, pairs, sample_db):
        """Model gets 2/4 right -> ExAcc=0.5."""
        exacc, _ = evaluate_exacc(
            schema=schema,
            pairs=pairs,
            model=PartialModel(),
            db_path=sample_db,
        )
        assert exacc == pytest.approx(0.5)

    def test_empty_pairs(self, schema, sample_db):
        """Empty pairs returns 0.0."""
        exacc, _ = evaluate_exacc(
            schema=schema,
            pairs=[],
            model=CorrectModel(),
            db_path=sample_db,
        )
        assert exacc == 0.0

    def test_with_reverse_mapping(self, refined_schema, pairs, sample_db):
        """When reverse_mapping provided, backmap predicted SQL before execution."""
        # RefinedAwareModel generates SQL with employee_name;
        # reverse_mapping maps it back to nm for execution
        exacc, _ = evaluate_exacc(
            schema=refined_schema,
            pairs=pairs,
            model=RefinedAwareModel(),
            db_path=sample_db,
            reverse_mapping={"employee_name": "nm"},
        )
        assert exacc == 1.0


# ========== evaluate_refinement ==========

class TestEvaluateRefinement:
    def test_improvement(self, schema, pairs, sample_db, refinement_results):
        """Refinement that improves ExAcc."""
        result, _ = evaluate_refinement(
            schema=schema,
            pairs=pairs,
            model=RefinedAwareModel(),
            db_path=sample_db,
            refinement_results=refinement_results,
        )
        assert isinstance(result, EvalResult)
        assert result.exacc_before == pytest.approx(0.5)  # 2/4
        assert result.exacc_after == 1.0  # 4/4
        assert result.delta == pytest.approx(0.5)
        assert result.db_id == "test_db"

    def test_no_change(self, schema, pairs, sample_db):
        """No refinements -> exacc_before == exacc_after."""
        result, _ = evaluate_refinement(
            schema=schema,
            pairs=pairs,
            model=CorrectModel(),
            db_path=sample_db,
            refinement_results=[],
        )
        assert result.exacc_before == result.exacc_after
        assert result.delta == 0.0

    def test_refinement_precision(self, schema, pairs, sample_db, refinement_results):
        """Refinement precision = improved columns / changed columns."""
        result, _ = evaluate_refinement(
            schema=schema,
            pairs=pairs,
            model=RefinedAwareModel(),
            db_path=sample_db,
            refinement_results=refinement_results,
        )
        # Only nm was changed, and it improved -> precision = 1.0
        assert result.refinement_precision == 1.0

    def test_fields_populated(self, schema, pairs, sample_db, refinement_results):
        result, _ = evaluate_refinement(
            schema=schema,
            pairs=pairs,
            model=RefinedAwareModel(),
            db_path=sample_db,
            refinement_results=refinement_results,
        )
        assert result.db_id == "test_db"
        assert result.total_queries == 4
        assert result.columns_changed >= 0
        assert result.columns_evaluated >= 0


# ========== aggregate_results ==========

class TestAggregateResults:
    def test_aggregate_basic(self):
        """Aggregate multiple EvalResults."""
        results = [
            EvalResult(
                db_id="db1", exacc_before=0.5, exacc_after=0.7,
                delta=0.2, total_queries=10, columns_changed=3,
                columns_evaluated=5, refinement_precision=1.0,
            ),
            EvalResult(
                db_id="db2", exacc_before=0.6, exacc_after=0.8,
                delta=0.2, total_queries=20, columns_changed=4,
                columns_evaluated=6, refinement_precision=0.75,
            ),
        ]
        agg = aggregate_results(results)
        assert "avg_exacc_before" in agg
        assert "avg_exacc_after" in agg
        assert "avg_delta" in agg
        assert agg["avg_exacc_before"] == pytest.approx(0.55)
        assert agg["avg_exacc_after"] == pytest.approx(0.75)
        assert agg["avg_delta"] == pytest.approx(0.2)
        assert agg["total_databases"] == 2
        assert agg["total_queries"] == 30

    def test_aggregate_empty(self):
        agg = aggregate_results([])
        assert agg["total_databases"] == 0
        assert agg["avg_delta"] == 0.0

    def test_aggregate_precision(self):
        results = [
            EvalResult(db_id="a", exacc_before=0.4, exacc_after=0.6, delta=0.2,
                       total_queries=10, columns_changed=2, columns_evaluated=4,
                       refinement_precision=1.0),
            EvalResult(db_id="b", exacc_before=0.5, exacc_after=0.5, delta=0.0,
                       total_queries=10, columns_changed=1, columns_evaluated=3,
                       refinement_precision=0.0),
        ]
        agg = aggregate_results(results)
        assert agg["avg_refinement_precision"] == pytest.approx(0.5)


# ========== to_latex_table ==========

class TestToLatexTable:
    def test_basic_latex(self):
        results = [
            EvalResult(
                db_id="financial", exacc_before=0.62, exacc_after=0.70,
                delta=0.08, total_queries=50, columns_changed=7,
                columns_evaluated=12, refinement_precision=0.857,
            ),
        ]
        latex = to_latex_table(results)
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex
        assert "financial" in latex
        assert "0.62" in latex or "62" in latex

    def test_latex_has_header(self):
        results = [
            EvalResult(
                db_id="test", exacc_before=0.5, exacc_after=0.6,
                delta=0.1, total_queries=10, columns_changed=2,
                columns_evaluated=5, refinement_precision=1.0,
            ),
        ]
        latex = to_latex_table(results)
        assert "Database" in latex or "db" in latex.lower()
        assert "Before" in latex or "before" in latex.lower() or "ExAcc" in latex

    def test_latex_multiple_rows(self):
        results = [
            EvalResult(db_id="db1", exacc_before=0.5, exacc_after=0.6, delta=0.1,
                       total_queries=10, columns_changed=2, columns_evaluated=5,
                       refinement_precision=1.0),
            EvalResult(db_id="db2", exacc_before=0.7, exacc_after=0.8, delta=0.1,
                       total_queries=20, columns_changed=3, columns_evaluated=4,
                       refinement_precision=0.67),
        ]
        latex = to_latex_table(results)
        assert "db1" in latex
        assert "db2" in latex

    def test_empty_results(self):
        latex = to_latex_table([])
        assert "\\begin{tabular}" in latex


# ========== EvalResult ==========

class TestEvalResult:
    def test_dataclass_fields(self):
        r = EvalResult(
            db_id="test", exacc_before=0.5, exacc_after=0.7,
            delta=0.2, total_queries=10, columns_changed=3,
            columns_evaluated=5, refinement_precision=1.0,
        )
        assert r.db_id == "test"
        assert r.delta == 0.2

    def test_to_dict(self):
        r = EvalResult(
            db_id="test", exacc_before=0.5, exacc_after=0.7,
            delta=0.2, total_queries=10, columns_changed=3,
            columns_evaluated=5, refinement_precision=1.0,
        )
        d = r.to_dict()
        assert d["db_id"] == "test"
        assert d["exacc_before"] == 0.5
        assert d["delta"] == 0.2
        assert isinstance(d, dict)

    def test_serializable(self):
        r = EvalResult(
            db_id="test", exacc_before=0.5, exacc_after=0.7,
            delta=0.2, total_queries=10, columns_changed=3,
            columns_evaluated=5, refinement_precision=1.0,
        )
        json_str = json.dumps(r.to_dict())
        assert json_str
