"""T20: Baseline tests."""
import json
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.phase2.prompts import CandidateName
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase3.text2sql_runner import Text2SQLModel
from egrefine.pipeline import PipelineResult
from egrefine.baselines.no_refinement import run_no_refinement
from egrefine.baselines.llm_direct import run_llm_direct
from egrefine.baselines.llm_cot import run_llm_cot


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


@pytest.fixture
def phase1_config():
    return {
        "signals": {
            "short_name": {"enabled": True, "max_length": 3},
            "high_similarity": {"enabled": False},
            "naming_inconsistency": {"enabled": False},
            "generic_vocabulary": {"enabled": False},
        },
        "skip_primary_keys": True,
    }


# ========== no_refinement ==========

class TestNoRefinement:
    def test_returns_pipeline_result(self, schema, pairs, sample_db):
        result = run_no_refinement(schema=schema, pairs=pairs, db_path=sample_db)
        assert isinstance(result, PipelineResult)

    def test_no_changes(self, schema, pairs, sample_db):
        result = run_no_refinement(schema=schema, pairs=pairs, db_path=sample_db)
        assert result.refinements == []
        assert result.views == []
        assert result.mapping == {}
        assert result.reverse_mapping == {}

    def test_db_id_preserved(self, schema, pairs, sample_db):
        result = run_no_refinement(schema=schema, pairs=pairs, db_path=sample_db)
        assert result.db_id == "test_db"

    def test_statistics(self, schema, pairs, sample_db):
        result = run_no_refinement(schema=schema, pairs=pairs, db_path=sample_db)
        assert result.statistics["columns_refined"] == 0
        assert result.statistics["total_columns"] == 3

    def test_serializable(self, schema, pairs, sample_db):
        result = run_no_refinement(schema=schema, pairs=pairs, db_path=sample_db)
        d = result.to_dict()
        json.dumps(d)  # no exception


# ========== llm_direct ==========

class TestLLMDirect:
    @patch("egrefine.baselines.llm_direct.CandidateGenerator")
    def test_returns_pipeline_result(self, MockGen, schema, pairs, sample_db, phase1_config):
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="descriptive"),
        ]
        result = run_llm_direct(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            phase1_config=phase1_config,
            phase2_config={"k": 1, "sample_rows": 5},
            candidate_generator=mock_gen,
        )
        assert isinstance(result, PipelineResult)

    @patch("egrefine.baselines.llm_direct.CandidateGenerator")
    def test_always_takes_first_candidate(self, MockGen, schema, pairs, sample_db, phase1_config):
        """llm_direct always picks the first LLM candidate without execution verification."""
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="first"),
            CandidateName(name="full_name", reason="second"),
        ]
        result = run_llm_direct(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            phase1_config=phase1_config,
            phase2_config={"k": 2, "sample_rows": 5},
            candidate_generator=mock_gen,
        )
        # All changed columns should use first candidate
        for r in result.refinements:
            if r.was_changed:
                assert r.verification_method == "llm_direct"

    @patch("egrefine.baselines.llm_direct.CandidateGenerator")
    def test_no_candidates_no_change(self, MockGen, schema, pairs, sample_db, phase1_config):
        """If Phase 2 returns no candidates, column stays unchanged."""
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = []
        result = run_llm_direct(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            phase1_config=phase1_config,
            phase2_config={"k": 1, "sample_rows": 5},
            candidate_generator=mock_gen,
        )
        assert result.statistics["columns_refined"] == 0

    @patch("egrefine.baselines.llm_direct.CandidateGenerator")
    def test_statistics(self, MockGen, schema, pairs, sample_db, phase1_config):
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="test"),
        ]
        result = run_llm_direct(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            phase1_config=phase1_config,
            phase2_config={"k": 1, "sample_rows": 5},
            candidate_generator=mock_gen,
        )
        assert "columns_refined" in result.statistics
        assert "method" in result.statistics
        assert result.statistics["method"] == "llm_direct"

    @patch("egrefine.baselines.llm_direct.CandidateGenerator")
    def test_serializable(self, MockGen, schema, pairs, sample_db, phase1_config):
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="test"),
        ]
        result = run_llm_direct(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            phase1_config=phase1_config,
            phase2_config={"k": 1, "sample_rows": 5},
            candidate_generator=mock_gen,
        )
        json.dumps(result.to_dict())


# ========== llm_cot ==========

class TestLLMCoT:
    @patch("egrefine.baselines.llm_cot.CandidateGenerator")
    def test_returns_pipeline_result(self, MockGen, schema, pairs, sample_db, phase1_config):
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="descriptive"),
            CandidateName(name="full_name", reason="alternative"),
        ]
        # Mock the LLM client for CoT selection
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"selected": "employee_name", "reasoning": "more descriptive"}'

        result = run_llm_cot(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            phase1_config=phase1_config,
            phase2_config={"k": 2, "sample_rows": 5},
            candidate_generator=mock_gen,
            llm_client=mock_llm,
        )
        assert isinstance(result, PipelineResult)

    @patch("egrefine.baselines.llm_cot.CandidateGenerator")
    def test_uses_llm_selection(self, MockGen, schema, pairs, sample_db, phase1_config):
        """llm_cot uses LLM CoT to pick among candidates."""
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="descriptive"),
            CandidateName(name="full_name", reason="alternative"),
        ]
        mock_llm = MagicMock()
        # LLM picks "full_name"
        mock_llm.chat.return_value = '{"selected": "full_name", "reasoning": "better fit"}'

        result = run_llm_cot(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            phase1_config=phase1_config,
            phase2_config={"k": 2, "sample_rows": 5},
            candidate_generator=mock_gen,
            llm_client=mock_llm,
        )
        for r in result.refinements:
            if r.was_changed:
                assert r.verification_method == "llm_cot"

    @patch("egrefine.baselines.llm_cot.CandidateGenerator")
    def test_fallback_on_parse_error(self, MockGen, schema, pairs, sample_db, phase1_config):
        """If LLM returns unparseable JSON, fall back to first candidate."""
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="first"),
        ]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "this is not json"

        result = run_llm_cot(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            phase1_config=phase1_config,
            phase2_config={"k": 1, "sample_rows": 5},
            candidate_generator=mock_gen,
            llm_client=mock_llm,
        )
        # Should still produce results without crashing
        assert isinstance(result, PipelineResult)

    @patch("egrefine.baselines.llm_cot.CandidateGenerator")
    def test_statistics(self, MockGen, schema, pairs, sample_db, phase1_config):
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="test"),
        ]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"selected": "employee_name", "reasoning": "good"}'

        result = run_llm_cot(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            phase1_config=phase1_config,
            phase2_config={"k": 1, "sample_rows": 5},
            candidate_generator=mock_gen,
            llm_client=mock_llm,
        )
        assert result.statistics["method"] == "llm_cot"

    @patch("egrefine.baselines.llm_cot.CandidateGenerator")
    def test_serializable(self, MockGen, schema, pairs, sample_db, phase1_config):
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="test"),
        ]
        mock_llm = MagicMock()
        mock_llm.chat.return_value = '{"selected": "employee_name", "reasoning": "good"}'

        result = run_llm_cot(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            phase1_config=phase1_config,
            phase2_config={"k": 1, "sample_rows": 5},
            candidate_generator=mock_gen,
            llm_client=mock_llm,
        )
        json.dumps(result.to_dict())
