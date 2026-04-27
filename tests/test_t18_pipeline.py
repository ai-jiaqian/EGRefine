"""T18: Pipeline orchestration tests."""
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
from egrefine.pipeline import run_pipeline, PipelineResult


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
    """Mock that always returns original-name SQL."""
    def generate(self, nl, schema, db_path=None, column_mapping=None, evidence=""):
        if "names" in nl.lower():
            return "SELECT nm FROM employees"
        return "SELECT MAX(sal) FROM employees"


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


# ========== Tests ==========

class TestPipelineResult:
    def test_dataclass_fields(self):
        r = PipelineResult(
            db_id="test",
            refinements=[],
            views=[],
            mapping={},
            reverse_mapping={},
            orig_table_map={},
            statistics={},
        )
        assert r.db_id == "test"
        assert r.refinements == []


class TestRunPipeline:
    @patch("egrefine.pipeline.CandidateGenerator")
    def test_basic_pipeline_runs(self, MockGen, schema, pairs, sample_db, phase1_config):
        """Pipeline runs end-to-end without errors."""
        # Mock Phase 2: return candidates for short-name columns
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="descriptive"),
        ]

        result = run_pipeline(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            models=[MockModel()],
            phase1_config=phase1_config,
            phase2_config={"k": 1, "sample_rows": 5},
            phase3_config={"conservative": True, "conflict_resolution_rounds": 2},
            phase4_config={},
            candidate_generator=mock_gen,
        )
        assert isinstance(result, PipelineResult)
        assert result.db_id == "test_db"
        assert "statistics" in result.__dict__ or hasattr(result, "statistics")

    @patch("egrefine.pipeline.CandidateGenerator")
    def test_statistics_populated(self, MockGen, schema, pairs, sample_db, phase1_config):
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="test"),
        ]

        result = run_pipeline(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            models=[MockModel()],
            phase1_config=phase1_config,
            phase2_config={"k": 1, "sample_rows": 5},
            phase3_config={"conservative": True, "conflict_resolution_rounds": 2},
            phase4_config={},
            candidate_generator=mock_gen,
        )
        stats = result.statistics
        assert "total_columns" in stats
        assert "candidates_after_pruning" in stats
        assert "columns_refined" in stats
        assert "columns_kept_original" in stats
        assert "fallback_count" in stats

    @patch("egrefine.pipeline.CandidateGenerator")
    def test_no_candidates_no_crash(self, MockGen, schema, pairs, sample_db):
        """If Phase 1 produces no candidates, pipeline still works."""
        mock_gen = MockGen.return_value
        # All signals disabled, skip PKs — nm and sal should NOT be caught
        config = {
            "signals": {
                "short_name": {"enabled": False},
                "high_similarity": {"enabled": False},
                "naming_inconsistency": {"enabled": False},
                "generic_vocabulary": {"enabled": False},
            },
            "skip_primary_keys": True,
        }
        result = run_pipeline(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            models=[MockModel()],
            phase1_config=config,
            phase2_config={"k": 1, "sample_rows": 5},
            phase3_config={"conservative": True, "conflict_resolution_rounds": 2},
            phase4_config={},
            candidate_generator=mock_gen,
        )
        assert result.statistics["candidates_after_pruning"] == 0
        assert result.statistics["columns_refined"] == 0
        assert result.views == []

    @patch("egrefine.pipeline.CandidateGenerator")
    def test_refinements_list(self, MockGen, schema, pairs, sample_db, phase1_config):
        """Refinements list has correct structure."""
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="test"),
        ]

        result = run_pipeline(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            models=[MockModel()],
            phase1_config=phase1_config,
            phase2_config={"k": 1, "sample_rows": 5},
            phase3_config={"conservative": True, "conflict_resolution_rounds": 2},
            phase4_config={},
            candidate_generator=mock_gen,
        )
        # nm and sal are short names (≤3), should be candidates
        for r in result.refinements:
            assert isinstance(r, SelectionResult)

    @patch("egrefine.pipeline.CandidateGenerator")
    def test_output_serializable(self, MockGen, schema, pairs, sample_db, phase1_config):
        """Pipeline result can be serialized to JSON."""
        mock_gen = MockGen.return_value
        mock_gen.generate.return_value = [
            CandidateName(name="employee_name", reason="test"),
        ]

        result = run_pipeline(
            schema=schema,
            pairs=pairs,
            db_path=sample_db,
            models=[MockModel()],
            phase1_config=phase1_config,
            phase2_config={"k": 1, "sample_rows": 5},
            phase3_config={"conservative": True, "conflict_resolution_rounds": 2},
            phase4_config={},
            candidate_generator=mock_gen,
        )
        # to_dict should be JSON-serializable
        d = result.to_dict()
        json_str = json.dumps(d)
        assert json_str  # no exception
