"""T14: Phase 3 Scorer tests."""
import json
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import MagicMock, patch, call

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.phase2.prompts import CandidateName
from egrefine.phase3.scorer import (
    score_candidate,
    score_all_candidates,
    select_best,
    SelectionResult,
)
from egrefine.phase3.text2sql_runner import Text2SQLModel


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
                Column(name="dept", table="employees", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[],
    )


@pytest.fixture
def sample_db():
    """Create a test SQLite database."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, nm TEXT, sal REAL, dept TEXT)"
    )
    conn.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?)",
        [
            (1, "Alice", 50000, "Engineering"),
            (2, "Bob", 60000, "Engineering"),
            (3, "Carol", 55000, "Sales"),
        ],
    )
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture
def target_column():
    return Column(name="nm", table="employees", dtype="TEXT")


@pytest.fixture
def candidates():
    return [
        CandidateName(name="employee_name", reason="More descriptive"),
        CandidateName(name="full_name", reason="Represents full name"),
    ]


@pytest.fixture
def queries():
    """Q(c_i): queries referencing the 'nm' column."""
    return [
        NLSQLPair(nl="Who has the highest salary?", gold_sql="SELECT nm FROM employees ORDER BY sal DESC LIMIT 1", db_id="test_db"),
        NLSQLPair(nl="List all employee names", gold_sql="SELECT nm FROM employees", db_id="test_db"),
    ]


class MockText2SQL(Text2SQLModel):
    """Mock Text-to-SQL model that returns configurable SQL."""

    def __init__(self, responses: dict = None):
        # responses: {(nl, candidate_name) -> predicted_sql}
        self._responses = responses or {}
        self._default_sql = "SELECT 1"

    def generate(self, nl: str, schema: Schema, db_path=None, column_mapping=None, evidence: str = "") -> str:
        for (q, cand), sql in self._responses.items():
            if q in nl:
                return sql
        return self._default_sql


# ========== score_candidate ==========

class TestScoreCandidate:
    def test_original_name_scores(self, schema, target_column, queries, sample_db):
        """Scoring the original name with a model that generates correct SQL."""
        # Model generates correct SQL for original schema
        model = MockText2SQL({
            ("highest salary", None): "SELECT nm FROM employees ORDER BY sal DESC LIMIT 1",
            ("all employee names", None): "SELECT nm FROM employees",
        })
        score, _ = score_candidate(
            column=target_column,
            candidate_name="nm",  # original name = no change
            queries=queries,
            model=model,
            schema=schema,
            db_path=sample_db,
        )
        assert score == 1.0  # both queries correct

    def test_candidate_with_correct_backmap(self, schema, target_column, queries, sample_db):
        """Candidate name where model generates SQL using the new name, backmapped correctly."""
        # Model sees "employee_name" in schema, generates SQL with it
        model = MockText2SQL({
            ("highest salary", None): "SELECT employee_name FROM employees ORDER BY sal DESC LIMIT 1",
            ("all employee names", None): "SELECT employee_name FROM employees",
        })
        score, _ = score_candidate(
            column=target_column,
            candidate_name="employee_name",
            queries=queries,
            model=model,
            schema=schema,
            db_path=sample_db,
        )
        # After backmap: employee_name -> nm, should match gold SQL results
        assert score == 1.0

    def test_candidate_with_wrong_sql(self, schema, target_column, queries, sample_db):
        """Model generates wrong SQL for all queries."""
        model = MockText2SQL({
            ("highest salary", None): "SELECT dept FROM employees LIMIT 1",
            ("all employee names", None): "SELECT sal FROM employees",
        })
        score, _ = score_candidate(
            column=target_column,
            candidate_name="employee_name",
            queries=queries,
            model=model,
            schema=schema,
            db_path=sample_db,
        )
        assert score == 0.0

    def test_partial_score(self, schema, target_column, queries, sample_db):
        """Model gets 1 out of 2 queries right."""
        model = MockText2SQL({
            ("highest salary", None): "SELECT employee_name FROM employees ORDER BY sal DESC LIMIT 1",
            ("all employee names", None): "SELECT sal FROM employees",  # wrong
        })
        score, _ = score_candidate(
            column=target_column,
            candidate_name="employee_name",
            queries=queries,
            model=model,
            schema=schema,
            db_path=sample_db,
        )
        assert score == pytest.approx(0.5)

    def test_empty_queries_returns_none(self, schema, target_column, sample_db):
        """Empty Q(c_i) returns None (cannot evaluate)."""
        model = MockText2SQL()
        score, _ = score_candidate(
            column=target_column,
            candidate_name="employee_name",
            queries=[],
            model=model,
            schema=schema,
            db_path=sample_db,
        )
        assert score is None


# ========== score_all_candidates ==========

class TestScoreAllCandidates:
    def test_scores_original_and_candidates(self, schema, target_column, candidates, queries, sample_db):
        """Scores original + all candidates."""
        # Model always generates the same (wrong) SQL, except for original name
        model = MockText2SQL({
            ("highest salary", None): "SELECT nm FROM employees ORDER BY sal DESC LIMIT 1",
            ("all employee names", None): "SELECT nm FROM employees",
        })
        scores, _ = score_all_candidates(
            column=target_column,
            candidates=candidates,
            queries=queries,
            models=[model],
            schema=schema,
            db_path=sample_db,
        )
        # Should have original + 2 candidates
        assert "nm" in scores
        assert "employee_name" in scores
        assert "full_name" in scores
        assert len(scores) == 3

    def test_multi_model_averaging(self, schema, target_column, candidates, queries, sample_db):
        """Multi-model scores are averaged."""
        # Model A: gets everything right for original
        model_a = MockText2SQL({
            ("highest salary", None): "SELECT nm FROM employees ORDER BY sal DESC LIMIT 1",
            ("all employee names", None): "SELECT nm FROM employees",
        })
        # Model B: gets nothing right
        model_b = MockText2SQL({
            ("highest salary", None): "SELECT 1",
            ("all employee names", None): "SELECT 1",
        })
        scores, _ = score_all_candidates(
            column=target_column,
            candidates=candidates,
            queries=queries,
            models=[model_a, model_b],
            schema=schema,
            db_path=sample_db,
        )
        # Original: model_a=1.0, model_b=0.0, avg=0.5
        assert scores["nm"] == pytest.approx(0.5)

    def test_empty_queries_returns_empty(self, schema, target_column, candidates, sample_db):
        """Empty Q(c_i) returns empty scores dict."""
        model = MockText2SQL()
        scores, _ = score_all_candidates(
            column=target_column,
            candidates=candidates,
            queries=[],
            models=[model],
            schema=schema,
            db_path=sample_db,
        )
        assert scores == {}


# ========== select_best (Conservative Selection, T15) ==========

class TestSelectBest:
    def test_best_candidate_wins(self, target_column, candidates):
        """Best candidate with higher score than original is selected."""
        scores = {"nm": 0.5, "employee_name": 0.8, "full_name": 0.6}
        result = select_best(target_column, candidates, scores)
        assert result.selected_name == "employee_name"
        assert result.delta == pytest.approx(0.3)
        assert result.was_changed is True

    def test_conservative_keeps_original(self, target_column, candidates):
        """If no candidate beats original, keep original."""
        scores = {"nm": 0.8, "employee_name": 0.6, "full_name": 0.7}
        result = select_best(target_column, candidates, scores)
        assert result.selected_name == "nm"
        assert result.delta == 0.0
        assert result.was_changed is False

    def test_tie_keeps_original(self, target_column, candidates):
        """If best candidate ties with original, keep original (strict >)."""
        scores = {"nm": 0.7, "employee_name": 0.7, "full_name": 0.5}
        result = select_best(target_column, candidates, scores)
        assert result.selected_name == "nm"
        assert result.was_changed is False

    def test_empty_scores_keeps_original(self, target_column, candidates):
        """Empty scores (Q(c_i) was empty) -> keep original name."""
        scores = {}
        result = select_best(target_column, candidates, scores)
        assert result.selected_name == "nm"
        assert result.was_changed is False
        assert result.verification_method == "skipped_no_queries"

    def test_empty_scores_no_candidates(self, target_column):
        """Empty scores + no candidates -> keep original."""
        scores = {}
        result = select_best(target_column, [], scores)
        assert result.selected_name == "nm"
        assert result.was_changed is False

    def test_result_has_all_scores(self, target_column, candidates):
        scores = {"nm": 0.5, "employee_name": 0.8, "full_name": 0.6}
        result = select_best(target_column, candidates, scores)
        assert result.all_scores == scores

    def test_result_dataclass_fields(self, target_column, candidates):
        scores = {"nm": 0.5, "employee_name": 0.8, "full_name": 0.6}
        result = select_best(target_column, candidates, scores)
        assert isinstance(result, SelectionResult)
        assert result.column == target_column
        assert result.verification_method == "execution"


# ========== Integration-style tests ==========

class TestScorerIntegration:
    def test_full_flow_candidate_beats_original(self, schema, target_column, candidates, queries, sample_db):
        """Full flow: candidate gets better ExAcc than original."""
        # Model that works better with descriptive names
        # For original "nm": gets 0/2 right
        # For "employee_name": gets 2/2 right (backmapped correctly)
        class SmartModel(Text2SQLModel):
            def generate(self, nl, schema, db_path=None, column_mapping=None, evidence=""):
                # Check if schema has employee_name (refined) or nm (original)
                col = schema.get_column("employees", "employee_name")
                if col:
                    # Refined schema -> correct SQL using new name
                    if "highest" in nl:
                        return "SELECT employee_name FROM employees ORDER BY sal DESC LIMIT 1"
                    return "SELECT employee_name FROM employees"
                else:
                    # Original schema -> wrong SQL
                    return "SELECT dept FROM employees LIMIT 1"

        scores, _ = score_all_candidates(
            column=target_column,
            candidates=candidates,
            queries=queries,
            models=[SmartModel()],
            schema=schema,
            db_path=sample_db,
        )
        result = select_best(target_column, candidates, scores)
        assert result.was_changed is True
        assert result.selected_name == "employee_name"
        assert result.delta > 0

    def test_full_flow_original_wins(self, schema, target_column, candidates, queries, sample_db):
        """Full flow: original name already optimal."""
        class OriginalBestModel(Text2SQLModel):
            def generate(self, nl, schema, db_path=None, column_mapping=None, evidence=""):
                col = schema.get_column("employees", "nm")
                if col:
                    # Works perfectly with original
                    if "highest" in nl:
                        return "SELECT nm FROM employees ORDER BY sal DESC LIMIT 1"
                    return "SELECT nm FROM employees"
                else:
                    # Fails with refined names
                    return "SELECT 1"

        scores, _ = score_all_candidates(
            column=target_column,
            candidates=candidates,
            queries=queries,
            models=[OriginalBestModel()],
            schema=schema,
            db_path=sample_db,
        )
        result = select_best(target_column, candidates, scores)
        assert result.was_changed is False
        assert result.selected_name == "nm"
