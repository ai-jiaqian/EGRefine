"""T8: Phase 1 Embedding 计算模块测试"""
import math
import pytest
from unittest.mock import MagicMock

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase1.embeddings import (
    compute_column_embeddings,
    build_similarity_matrix,
    find_high_similarity_pairs,
)
from egrefine.models.embedding_client import EmbeddingClient


# ========== Fixtures ==========

@pytest.fixture
def simple_schema():
    return Schema(
        db_id="test_db",
        tables=[
            Table(name="users", columns=[
                Column(name="id", table="users", dtype="INTEGER", is_pk=True),
                Column(name="nm", table="users", dtype="TEXT"),
                Column(name="email_address", table="users", dtype="TEXT"),
            ]),
            Table(name="orders", columns=[
                Column(name="id", table="orders", dtype="INTEGER", is_pk=True),
                Column(name="amt", table="orders", dtype="REAL"),
            ]),
        ],
        foreign_keys=[],
    )


@pytest.fixture
def mock_embedding_client():
    """Mock embedding client 返回可控的向量。"""
    client = MagicMock(spec=EmbeddingClient)
    # 为每个文本返回不同的假向量
    def fake_embed(texts):
        vectors = []
        for i, _ in enumerate(texts):
            vec = [0.0] * 8
            vec[i % 8] = 1.0  # 正交向量 -> similarity = 0
            vectors.append(vec)
        return vectors
    client.embed.side_effect = fake_embed
    return client


@pytest.fixture
def mock_embedding_client_similar():
    """Mock embedding client: 前两列高度相似。"""
    client = MagicMock(spec=EmbeddingClient)
    def fake_embed(texts):
        vectors = []
        for i, _ in enumerate(texts):
            if i <= 1:
                # 前两列几乎相同
                vectors.append([1.0, 0.1 * i, 0.0, 0.0])
            else:
                vec = [0.0] * 4
                vec[i % 4] = 1.0
                vectors.append(vec)
        return vectors
    client.embed.side_effect = fake_embed
    return client


# ========== compute_column_embeddings ==========

class TestComputeColumnEmbeddings:
    def test_returns_dict_with_full_names(self, simple_schema, mock_embedding_client):
        result = compute_column_embeddings(simple_schema, mock_embedding_client)
        assert isinstance(result, dict)
        assert "users.id" in result
        assert "users.nm" in result
        assert "users.email_address" in result
        assert "orders.id" in result
        assert "orders.amt" in result

    def test_returns_vectors(self, simple_schema, mock_embedding_client):
        result = compute_column_embeddings(simple_schema, mock_embedding_client)
        for key, vec in result.items():
            assert isinstance(vec, list)
            assert all(isinstance(v, float) for v in vec)

    def test_short_names_include_table_prefix(self, simple_schema, mock_embedding_client):
        """短列名（<=5字符）应拼接表名。"""
        compute_column_embeddings(simple_schema, mock_embedding_client)
        call_args = mock_embedding_client.embed.call_args[0][0]
        # "id" (2字符) -> "users.id", "nm" (2字符) -> "users.nm"
        assert "users.id" in call_args
        assert "users.nm" in call_args
        assert "orders.amt" in call_args
        # "email_address" (13字符) -> 直接用
        assert "email_address" in call_args

    def test_empty_schema(self, mock_embedding_client):
        schema = Schema(db_id="empty", tables=[], foreign_keys=[])
        result = compute_column_embeddings(schema, mock_embedding_client)
        assert result == {}
        mock_embedding_client.embed.assert_not_called()

    def test_calls_embed_once(self, simple_schema, mock_embedding_client):
        """所有列名应在一次调用中完成 embedding。"""
        compute_column_embeddings(simple_schema, mock_embedding_client)
        assert mock_embedding_client.embed.call_count == 1
        texts = mock_embedding_client.embed.call_args[0][0]
        assert len(texts) == 5  # 5 个列


# ========== build_similarity_matrix ==========

class TestBuildSimilarityMatrix:
    def test_returns_upper_triangle(self):
        embeddings = {
            "t.a": [1.0, 0.0],
            "t.b": [0.0, 1.0],
            "t.c": [1.0, 1.0],
        }
        matrix = build_similarity_matrix(embeddings)
        # 3 choose 2 = 3 pairs
        assert len(matrix) == 3
        # 键应是排序后的 (a < b)
        for (a, b) in matrix:
            assert a < b

    def test_orthogonal_vectors_zero_similarity(self):
        embeddings = {
            "t.a": [1.0, 0.0],
            "t.b": [0.0, 1.0],
        }
        matrix = build_similarity_matrix(embeddings)
        assert matrix[("t.a", "t.b")] == pytest.approx(0.0)

    def test_identical_vectors_full_similarity(self):
        embeddings = {
            "t.a": [1.0, 2.0, 3.0],
            "t.b": [1.0, 2.0, 3.0],
        }
        matrix = build_similarity_matrix(embeddings)
        assert matrix[("t.a", "t.b")] == pytest.approx(1.0)

    def test_single_column_empty_matrix(self):
        embeddings = {"t.a": [1.0, 0.0]}
        matrix = build_similarity_matrix(embeddings)
        assert matrix == {}

    def test_empty_embeddings(self):
        matrix = build_similarity_matrix({})
        assert matrix == {}


# ========== find_high_similarity_pairs ==========

class TestFindHighSimilarityPairs:
    def test_finds_similar_pairs(self):
        matrix = {
            ("t.a", "t.b"): 0.95,
            ("t.a", "t.c"): 0.50,
            ("t.b", "t.c"): 0.87,
        }
        pairs = find_high_similarity_pairs(matrix, threshold=0.85)
        assert len(pairs) == 2
        assert pairs[0] == ("t.a", "t.b", 0.95)
        assert pairs[1] == ("t.b", "t.c", 0.87)

    def test_no_similar_pairs(self):
        matrix = {
            ("t.a", "t.b"): 0.30,
            ("t.a", "t.c"): 0.50,
        }
        pairs = find_high_similarity_pairs(matrix, threshold=0.85)
        assert pairs == []

    def test_custom_threshold(self):
        matrix = {
            ("t.a", "t.b"): 0.70,
            ("t.a", "t.c"): 0.50,
        }
        pairs = find_high_similarity_pairs(matrix, threshold=0.60)
        assert len(pairs) == 1

    def test_sorted_descending(self):
        matrix = {
            ("t.a", "t.b"): 0.86,
            ("t.a", "t.c"): 0.99,
            ("t.b", "t.c"): 0.90,
        }
        pairs = find_high_similarity_pairs(matrix, threshold=0.85)
        sims = [p[2] for p in pairs]
        assert sims == sorted(sims, reverse=True)

    def test_empty_matrix(self):
        pairs = find_high_similarity_pairs({}, threshold=0.85)
        assert pairs == []


# ========== 集成测试: embedding -> similarity -> high pairs ==========

class TestEmbeddingPipeline:
    def test_end_to_end(self, simple_schema, mock_embedding_client_similar):
        embeddings = compute_column_embeddings(
            simple_schema, mock_embedding_client_similar
        )
        matrix = build_similarity_matrix(embeddings)
        pairs = find_high_similarity_pairs(matrix, threshold=0.8)
        # 前两列 (users.id, users.nm) 应相似度高
        assert len(pairs) >= 1
        # 至少有一对的 similarity > 0.8
        assert any(sim > 0.8 for _, _, sim in pairs)
