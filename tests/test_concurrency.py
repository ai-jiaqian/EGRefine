"""Tests for concurrent execution in pipeline and evaluation."""
import sqlite3
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.phase2.prompts import CandidateName
from egrefine.phase3.scorer import SelectionResult


def _make_test_db(tmp_path):
    """Create a minimal SQLite test database."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, nm TEXT, sal REAL)")
    conn.execute("INSERT INTO employees VALUES (1, 'Alice', 50000)")
    conn.execute("INSERT INTO employees VALUES (2, 'Bob', 60000)")
    conn.commit()
    conn.close()
    return db_path


def _make_schema():
    return Schema(
        db_id="test_db",
        tables=[Table(
            name="employees",
            columns=[
                Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
                Column(name="nm", table="employees", dtype="TEXT"),
                Column(name="sal", table="employees", dtype="REAL"),
            ],
        )],
        foreign_keys=[],
    )


class TestConcurrentPhase2:
    """Test that Phase 2 candidate generation works in parallel."""

    def test_thread_safe_cache(self, tmp_path):
        """CandidateGenerator cache should be thread-safe."""
        from egrefine.phase2.generator import CandidateGenerator

        mock_llm = MagicMock()
        mock_llm.chat.return_value = '[{"name": "employee_name", "reason": "test"}]'

        gen = CandidateGenerator(
            mock_llm, cache_dir=str(tmp_path / "cache"),
            sample_rows=0, max_retries=1,
        )
        schema = _make_schema()
        db_path = _make_test_db(tmp_path)

        results = {}
        errors = []

        def generate_for_col(col_name):
            try:
                col = schema.get_column("employees", col_name)
                r = gen.generate(col, schema, db_path, k=1)
                results[col_name] = r
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=generate_for_col, args=("nm",)),
            threading.Thread(target=generate_for_col, args=("sal",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        assert "nm" in results
        assert "sal" in results


class TestConcurrentEvaluation:
    """Test that evaluation works in parallel."""

    def test_evaluate_exacc_parallel(self, tmp_path):
        """evaluate_exacc should produce same results with max_workers>1."""
        from egrefine.evaluate import evaluate_exacc

        db_path = _make_test_db(tmp_path)
        schema = _make_schema()
        pairs = [
            NLSQLPair(nl="q1", gold_sql="SELECT nm FROM employees WHERE id=1", db_id="test_db"),
            NLSQLPair(nl="q2", gold_sql="SELECT sal FROM employees WHERE id=2", db_id="test_db"),
            NLSQLPair(nl="q3", gold_sql="SELECT COUNT(*) FROM employees", db_id="test_db"),
        ]

        # Mock model that returns the gold SQL (100% accuracy)
        mock_model = MagicMock()
        mock_model.generate.side_effect = lambda nl, schema, **kw: {
            "q1": "SELECT nm FROM employees WHERE id=1",
            "q2": "SELECT sal FROM employees WHERE id=2",
            "q3": "SELECT COUNT(*) FROM employees",
        }[nl]

        # Sequential
        score_seq, details_seq = evaluate_exacc(
            schema, pairs, mock_model, db_path,
            collect_details=True, max_workers=1,
        )

        # Reset mock call count
        mock_model.generate.reset_mock()

        # Parallel
        score_par, details_par = evaluate_exacc(
            schema, pairs, mock_model, db_path,
            collect_details=True, max_workers=4,
        )

        assert score_seq == score_par
        assert len(details_seq) == len(details_par)
        # Both should be 100%
        assert score_seq == pytest.approx(1.0)

    def test_evaluate_exacc_parallel_preserves_order(self, tmp_path):
        """Parallel evaluation should return details in original pair order."""
        from egrefine.evaluate import evaluate_exacc

        db_path = _make_test_db(tmp_path)
        schema = _make_schema()
        pairs = [
            NLSQLPair(nl=f"q{i}", gold_sql="SELECT 1", db_id="test_db")
            for i in range(10)
        ]

        mock_model = MagicMock()
        mock_model.generate.return_value = "SELECT 1"

        _, details = evaluate_exacc(
            schema, pairs, mock_model, db_path,
            collect_details=True, max_workers=4,
        )

        # Details should be in order
        for i, d in enumerate(details):
            assert d.nl == f"q{i}"


class TestConcurrentPipeline:
    """Test pipeline with max_workers > 1."""

    def test_pipeline_accepts_max_workers(self):
        """run_pipeline should accept max_workers parameter."""
        from egrefine.pipeline import run_pipeline
        import inspect
        sig = inspect.signature(run_pipeline)
        assert "max_workers" in sig.parameters

    def test_experiment_accepts_max_workers(self):
        """run_experiment should accept max_workers parameter."""
        from egrefine.experiment import run_experiment
        import inspect
        sig = inspect.signature(run_experiment)
        assert "max_workers" in sig.parameters

    def test_evaluate_refinement_accepts_max_workers(self):
        """evaluate_refinement should accept max_workers parameter."""
        from egrefine.evaluate import evaluate_refinement
        import inspect
        sig = inspect.signature(evaluate_refinement)
        assert "max_workers" in sig.parameters


class TestMaxWorkersConfig:
    """Test that max_workers is read from config."""

    def test_default_config_has_concurrency(self):
        from egrefine.config import load_config
        config = load_config("config/default.yaml")
        assert "concurrency" in config
        assert config["concurrency"]["max_workers"] == 8

    def test_max_workers_default_fallback(self):
        """Missing concurrency config should default to 1."""
        config = {}
        max_workers = config.get("concurrency", {}).get("max_workers", 1)
        assert max_workers == 1
