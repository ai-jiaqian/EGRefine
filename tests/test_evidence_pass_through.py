"""Tests for evidence passthrough from eval → Text-to-SQL model, AND from
Phase 3 scoring → Text-to-SQL model.

Verifies:
1. evidence="" by default (backward compat): LLM prompt does NOT contain evidence text.
2. --use-evidence forwards pair.evidence to model.generate(evidence=...).
3. SimpleLLMText2SQL + C3Text2SQL both splice evidence into the prompt when non-empty,
   and leave the prompt unchanged when evidence="".
4. Phase 3 scorer.score_candidate forwards pair.evidence only when use_evidence=True.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.eval.evaluator import evaluate_method
from egrefine.phase3.scorer import score_candidate
from egrefine.phase3.text2sql_runner import SimpleLLMText2SQL, Text2SQLModel
from egrefine.phase3.c3_runner import C3Text2SQL


# ---------------------------------------------------------------------------
# Helpers (minimal schema + db)
# ---------------------------------------------------------------------------

def _make_original_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, nm TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'Alice')")
    conn.commit()
    conn.close()


def _make_refined_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE _orig_t (id INTEGER PRIMARY KEY, nm TEXT)")
    conn.execute("INSERT INTO _orig_t VALUES (1, 'Alice')")
    conn.execute("CREATE VIEW t AS SELECT id, nm AS employee_name FROM _orig_t")
    conn.commit()
    conn.close()


def _original_schema() -> Schema:
    return Schema(
        db_id="test",
        tables=[Table(name="t", columns=[
            Column(name="id", table="t", dtype="INTEGER", is_pk=True),
            Column(name="nm", table="t", dtype="TEXT"),
        ])],
        foreign_keys=[],
    )


def _refined_schema() -> Schema:
    return Schema(
        db_id="test",
        tables=[Table(name="t", columns=[
            Column(name="id", table="t", dtype="INTEGER", is_pk=True),
            Column(name="employee_name", table="t", dtype="TEXT"),
        ])],
        foreign_keys=[],
    )


# ---------------------------------------------------------------------------
# Test 1 — Default (use_evidence=False): evidence NOT passed to model
# ---------------------------------------------------------------------------

class TestEvidenceDefaultOff:
    """When use_evidence is not set (backward compat), evidence should be ""."""

    def test_model_receives_empty_evidence_by_default(self, tmp_path: Path) -> None:
        orig_db = tmp_path / "orig.sqlite"
        ref_db = tmp_path / "refined.sqlite"
        _make_original_db(orig_db)
        _make_refined_db(ref_db)

        pairs = [
            NLSQLPair(
                nl="Who is in t?",
                gold_sql="SELECT nm FROM t",
                db_id="test",
                evidence="employee_name refers to the column nm.",
            ),
        ]

        model = MagicMock()
        model.generate = MagicMock(side_effect=[
            "SELECT nm FROM t",
            "SELECT employee_name FROM t",
        ])

        evaluate_method(
            model=model, pairs=pairs,
            original_schema=_original_schema(),
            refined_schema=_refined_schema(),
            original_db_path=str(orig_db),
            refined_db_path=str(ref_db),
            table_map={"t": "_orig_t"},
            # use_evidence defaults to False
        )

        # Both calls should have received evidence=""
        assert model.generate.call_count == 2
        for call in model.generate.call_args_list:
            assert call.kwargs.get("evidence", "") == "", \
                f"evidence should be empty by default, got {call.kwargs.get('evidence')!r}"


# ---------------------------------------------------------------------------
# Test 2 — use_evidence=True: pair.evidence flows to model
# ---------------------------------------------------------------------------

class TestEvidencePassthrough:
    """use_evidence=True forwards pair.evidence to model.generate()."""

    def test_model_receives_evidence(self, tmp_path: Path) -> None:
        orig_db = tmp_path / "orig.sqlite"
        ref_db = tmp_path / "refined.sqlite"
        _make_original_db(orig_db)
        _make_refined_db(ref_db)

        evidence_text = "employee_name refers to the column nm."
        pairs = [
            NLSQLPair(
                nl="Who is in t?",
                gold_sql="SELECT nm FROM t",
                db_id="test",
                evidence=evidence_text,
            ),
        ]

        model = MagicMock()
        model.generate = MagicMock(side_effect=[
            "SELECT nm FROM t",
            "SELECT employee_name FROM t",
        ])

        evaluate_method(
            model=model, pairs=pairs,
            original_schema=_original_schema(),
            refined_schema=_refined_schema(),
            original_db_path=str(orig_db),
            refined_db_path=str(ref_db),
            table_map={"t": "_orig_t"},
            use_evidence=True,
        )

        assert model.generate.call_count == 2
        for call in model.generate.call_args_list:
            assert call.kwargs.get("evidence") == evidence_text, \
                f"expected evidence={evidence_text!r}, got {call.kwargs.get('evidence')!r}"


# ---------------------------------------------------------------------------
# Test 3 — Empty evidence still silent even with --use-evidence
# ---------------------------------------------------------------------------

class TestEmptyEvidenceSilent:
    """use_evidence=True but pair.evidence="" should pass empty string."""

    def test_empty_pair_evidence(self, tmp_path: Path) -> None:
        orig_db = tmp_path / "orig.sqlite"
        ref_db = tmp_path / "refined.sqlite"
        _make_original_db(orig_db)
        _make_refined_db(ref_db)

        pairs = [
            NLSQLPair(
                nl="Who is in t?",
                gold_sql="SELECT nm FROM t",
                db_id="test",
                evidence="",  # e.g. Dr.Spider
            ),
        ]

        model = MagicMock()
        model.generate = MagicMock(side_effect=[
            "SELECT nm FROM t",
            "SELECT employee_name FROM t",
        ])

        evaluate_method(
            model=model, pairs=pairs,
            original_schema=_original_schema(),
            refined_schema=_refined_schema(),
            original_db_path=str(orig_db),
            refined_db_path=str(ref_db),
            table_map={"t": "_orig_t"},
            use_evidence=True,
        )

        for call in model.generate.call_args_list:
            assert call.kwargs.get("evidence", "__MISSING__") == "", \
                f"expected evidence='', got {call.kwargs.get('evidence')!r}"


# ---------------------------------------------------------------------------
# Test 4 — SimpleLLMText2SQL splices evidence into prompt when non-empty
# ---------------------------------------------------------------------------

class TestSimpleLLMText2SQLEvidencePrompt:
    """Verify SimpleLLM actually embeds evidence into its prompt."""

    def test_evidence_appears_in_prompt(self) -> None:
        config = {
            "base_url": "http://dummy",
            "api_key": "x",
            "model_name": "x",
            "temperature": 0,
            "max_tokens": 16,
        }
        model = SimpleLLMText2SQL(config, sample_rows=0)

        # Capture the prompt that LLMClient.chat is called with.
        captured = {}

        def fake_chat(messages):
            captured["prompt"] = messages[0]["content"]
            return "SELECT nm FROM t"

        model.llm.chat = fake_chat
        ev = "EMP_NAME means the employee's full display name"
        model.generate(
            nl="Who is in t?",
            schema=_original_schema(),
            db_path=None,
            evidence=ev,
        )

        assert "Additional context" in captured["prompt"]
        assert ev in captured["prompt"]

    def test_no_evidence_block_when_empty(self) -> None:
        config = {
            "base_url": "http://dummy",
            "api_key": "x",
            "model_name": "x",
            "temperature": 0,
            "max_tokens": 16,
        }
        model = SimpleLLMText2SQL(config, sample_rows=0)

        captured = {}

        def fake_chat(messages):
            captured["prompt"] = messages[0]["content"]
            return "SELECT nm FROM t"

        model.llm.chat = fake_chat
        model.generate(
            nl="Who is in t?",
            schema=_original_schema(),
            db_path=None,
            evidence="",
        )

        assert "Additional context" not in captured["prompt"], \
            "Empty evidence should not emit the 'Additional context' block"


# ---------------------------------------------------------------------------
# Test 5 — C3Text2SQL splices evidence into linking + generation prompts
# ---------------------------------------------------------------------------

class TestC3EvidencePrompt:
    """Verify C3 embeds evidence into both schema-linking and generation prompts."""

    def test_evidence_in_c3_prompts(self) -> None:
        config = {
            "base_url": "http://dummy",
            "api_key": "x",
            "model_name": "x",
            "temperature": 0,
            "max_tokens": 16,
        }
        model = C3Text2SQL(config, num_samples=1, sample_rows=0)

        # C3 first does schema linking (1 call) then generates SQL (1 call, deterministic).
        # With num_samples=1 and db_path=None, it takes the non-self-consistency branch.
        prompts_seen = []

        def fake_chat(messages):
            prompts_seen.append(messages[0]["content"])
            # Return JSON for linking, then SQL for generation
            if len(prompts_seen) == 1:
                return '{"tables": ["t"], "columns": ["t.nm"]}'
            return "SELECT nm FROM t"

        model.llm.chat = fake_chat

        ev = "Employee name is stored in column 'nm'."
        model.generate(
            nl="Who is in t?",
            schema=_original_schema(),
            db_path=None,
            evidence=ev,
        )

        assert len(prompts_seen) == 2
        linking_prompt, gen_prompt = prompts_seen
        assert ev in linking_prompt, "evidence should appear in schema-linking prompt"
        assert ev in gen_prompt, "evidence should appear in generation prompt"

    def test_no_evidence_in_c3_prompts_when_empty(self) -> None:
        config = {
            "base_url": "http://dummy",
            "api_key": "x",
            "model_name": "x",
            "temperature": 0,
            "max_tokens": 16,
        }
        model = C3Text2SQL(config, num_samples=1, sample_rows=0)

        prompts_seen = []

        def fake_chat(messages):
            prompts_seen.append(messages[0]["content"])
            if len(prompts_seen) == 1:
                return '{"tables": ["t"], "columns": ["t.nm"]}'
            return "SELECT nm FROM t"

        model.llm.chat = fake_chat
        model.generate(
            nl="Who is in t?",
            schema=_original_schema(),
            db_path=None,
            evidence="",
        )

        for p in prompts_seen:
            assert "Additional context" not in p, \
                "Empty evidence should not emit the 'Additional context' block"


# ---------------------------------------------------------------------------
# Test 6 — Phase 3 scorer.score_candidate forwards evidence (opt-in)
# ---------------------------------------------------------------------------

class _CapturingMockModel(Text2SQLModel):
    """Captures each call's evidence kwarg for assertion."""

    def __init__(self) -> None:
        self.evidence_seen: list[str] = []

    def generate(
        self, nl, schema, db_path=None, column_mapping=None, evidence: str = "",
    ) -> str:
        self.evidence_seen.append(evidence)
        # Return a SQL that back-maps correctly to the gold, so score logic exercises
        return "SELECT nm FROM t"


class TestScoreCandidateEvidenceOptIn:
    """Phase 3 scorer must obey use_evidence flag."""

    @pytest.fixture
    def tiny_schema(self) -> Schema:
        return Schema(
            db_id="scorer_test",
            tables=[Table(name="t", columns=[
                Column(name="id", table="t", dtype="INTEGER", is_pk=True),
                Column(name="nm", table="t", dtype="TEXT"),
            ])],
            foreign_keys=[],
        )

    @pytest.fixture
    def tiny_db(self, tmp_path: Path) -> str:
        p = tmp_path / "tiny.sqlite"
        conn = sqlite3.connect(str(p))
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, nm TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'A')")
        conn.commit()
        conn.close()
        return str(p)

    @pytest.fixture
    def queries(self) -> list[NLSQLPair]:
        return [
            NLSQLPair(
                nl="show names",
                gold_sql="SELECT nm FROM t",
                db_id="scorer_test",
                evidence="nm is the person's name",
            ),
            NLSQLPair(
                nl="show names again",
                gold_sql="SELECT nm FROM t",
                db_id="scorer_test",
                evidence="",  # missing evidence — should pass "" even when enabled
            ),
        ]

    def test_default_off_no_evidence(self, tiny_schema, tiny_db, queries):
        model = _CapturingMockModel()
        col = tiny_schema.tables[0].columns[1]
        score_candidate(
            column=col,
            candidate_name="nm",
            queries=queries,
            model=model,
            schema=tiny_schema,
            db_path=tiny_db,
            # use_evidence defaults to False
        )
        assert model.evidence_seen == ["", ""], \
            f"default should pass empty evidence, got {model.evidence_seen!r}"

    def test_opt_in_forwards_per_pair_evidence(self, tiny_schema, tiny_db, queries):
        model = _CapturingMockModel()
        col = tiny_schema.tables[0].columns[1]
        score_candidate(
            column=col,
            candidate_name="nm",
            queries=queries,
            model=model,
            schema=tiny_schema,
            db_path=tiny_db,
            use_evidence=True,
            query_workers=1,  # keep order deterministic for the assertion
        )
        assert model.evidence_seen == [
            "nm is the person's name",
            "",  # empty evidence stays empty even with flag on
        ], f"opt-in should forward pair.evidence verbatim, got {model.evidence_seen!r}"
