"""T13: Phase 2 Candidate Generator tests."""
import json
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from egrefine.data.schema import Column, Table, Schema
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.generator import CandidateGenerator, _extract_json
from egrefine.phase2.prompts import CandidateName


# ========== Fixtures ==========

@pytest.fixture
def schema():
    return Schema(
        db_id="test_db",
        tables=[
            Table(name="district", columns=[
                Column(name="id", table="district", dtype="INTEGER", is_pk=True),
                Column(name="A2", table="district", dtype="TEXT"),
                Column(name="A3", table="district", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[],
    )


@pytest.fixture
def target_column():
    return Column(name="A2", table="district", dtype="TEXT")


@pytest.fixture
def sample_db():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE district (id INTEGER PRIMARY KEY, A2 TEXT, A3 TEXT)")
    conn.executemany(
        "INSERT INTO district VALUES (?, ?, ?)",
        [(1, "Prague", "Bohemia"), (2, "Brno", "Moravia"), (3, "Plzen", "Bohemia")],
    )
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.fixture
def mock_llm():
    client = MagicMock(spec=LLMClient)
    client.chat.return_value = json.dumps([
        {"name": "district_name", "reason": "More descriptive"},
        {"name": "city_name", "reason": "Represents city"},
        {"name": "region_name", "reason": "Area name"},
    ])
    return client


@pytest.fixture
def tmp_cache_dir():
    d = tempfile.mkdtemp()
    yield d
    # cleanup
    for f in os.listdir(d):
        os.unlink(os.path.join(d, f))
    os.rmdir(d)


# ========== _extract_json ==========

class TestExtractJson:
    def test_pure_json(self):
        text = '[{"name": "foo", "reason": "bar"}]'
        result = _extract_json(text)
        assert result == [{"name": "foo", "reason": "bar"}]

    def test_markdown_code_block(self):
        text = '```json\n[{"name": "foo", "reason": "bar"}]\n```'
        result = _extract_json(text)
        assert result[0]["name"] == "foo"

    def test_code_block_no_lang(self):
        text = '```\n[{"name": "foo", "reason": "bar"}]\n```'
        result = _extract_json(text)
        assert result[0]["name"] == "foo"

    def test_json_with_prefix_text(self):
        text = 'Here are my suggestions:\n[{"name": "foo", "reason": "bar"}]'
        result = _extract_json(text)
        assert result[0]["name"] == "foo"

    def test_json_with_thinking_tags(self):
        text = '<think>Let me think about this...</think>\n[{"name": "foo", "reason": "bar"}]'
        result = _extract_json(text)
        assert result[0]["name"] == "foo"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not extract"):
            _extract_json("This is not JSON at all")

    def test_json_object_not_array_raises(self):
        with pytest.raises(ValueError, match="Could not extract"):
            _extract_json('{"name": "foo"}')

    def test_empty_array(self):
        result = _extract_json("[]")
        assert result == []


# ========== CandidateGenerator ==========

class TestCandidateGenerator:
    def test_basic_generation(self, mock_llm, schema, target_column, sample_db):
        gen = CandidateGenerator(mock_llm)
        result = gen.generate(target_column, schema, sample_db, k=3)
        assert len(result) == 3
        assert all(isinstance(c, CandidateName) for c in result)
        assert result[0].name == "district_name"
        mock_llm.chat.assert_called_once()

    def test_cache_hit(self, mock_llm, schema, target_column, sample_db, tmp_cache_dir):
        gen = CandidateGenerator(mock_llm, cache_dir=tmp_cache_dir)
        # First call -> LLM
        result1 = gen.generate(target_column, schema, sample_db, k=3)
        assert mock_llm.chat.call_count == 1
        # Second call -> cache
        result2 = gen.generate(target_column, schema, sample_db, k=3)
        assert mock_llm.chat.call_count == 1  # no additional call
        assert len(result2) == 3
        assert result2[0].name == result1[0].name

    def test_cache_persisted_to_disk(self, mock_llm, schema, target_column, sample_db, tmp_cache_dir):
        gen = CandidateGenerator(mock_llm, cache_dir=tmp_cache_dir)
        gen.generate(target_column, schema, sample_db, k=3)
        # Verify file exists
        cache_path = os.path.join(tmp_cache_dir, "phase2_candidates.json")
        assert os.path.exists(cache_path)
        with open(cache_path) as f:
            data = json.load(f)
        assert "test_db:district:A2" in data

    def test_cache_loaded_on_init(self, mock_llm, schema, target_column, sample_db, tmp_cache_dir):
        # Pre-populate cache file
        cache_path = os.path.join(tmp_cache_dir, "phase2_candidates.json")
        cache_data = {
            "test_db:district:A2": [
                {"name": "cached_name", "reason": "from cache"}
            ]
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        gen = CandidateGenerator(mock_llm, cache_dir=tmp_cache_dir)
        result = gen.generate(target_column, schema, sample_db, k=3)
        assert result[0].name == "cached_name"
        mock_llm.chat.assert_not_called()

    def test_retry_on_json_failure(self, schema, target_column, sample_db):
        llm = MagicMock(spec=LLMClient)
        llm.chat.side_effect = [
            "not valid json",  # 1st attempt -> parse fail
            '[{"name": "ok_name", "reason": "works"}]',  # 2nd attempt -> success
        ]
        gen = CandidateGenerator(llm, max_retries=3)
        result = gen.generate(target_column, schema, sample_db, k=1)
        assert len(result) == 1
        assert result[0].name == "ok_name"
        assert llm.chat.call_count == 2

    def test_all_retries_fail(self, schema, target_column, sample_db):
        llm = MagicMock(spec=LLMClient)
        llm.chat.return_value = "garbage"
        gen = CandidateGenerator(llm, max_retries=2)
        result = gen.generate(target_column, schema, sample_db, k=3)
        assert result == []
        assert llm.chat.call_count == 2

    def test_no_cache_dir(self, mock_llm, schema, target_column, sample_db):
        """Without cache_dir, no caching but still works."""
        gen = CandidateGenerator(mock_llm, cache_dir=None)
        result = gen.generate(target_column, schema, sample_db, k=3)
        assert len(result) == 3

    def test_sample_rows_config(self, mock_llm, schema, target_column, sample_db):
        gen = CandidateGenerator(mock_llm, sample_rows=5)
        with patch("egrefine.phase2.generator.sample_column") as mock_sample:
            mock_sample.return_value = ["Prague", "Brno"]
            gen.generate(target_column, schema, sample_db, k=3)
            mock_sample.assert_called_once_with(sample_db, "district", "A2", n=5)

    def test_llm_exception_retries(self, schema, target_column, sample_db):
        """LLM raises exception (network error etc), should retry."""
        llm = MagicMock(spec=LLMClient)
        llm.chat.side_effect = [
            Exception("network timeout"),
            '[{"name": "recovered", "reason": "after retry"}]',
        ]
        gen = CandidateGenerator(llm, max_retries=3)
        result = gen.generate(target_column, schema, sample_db, k=1)
        assert len(result) == 1
        assert result[0].name == "recovered"
