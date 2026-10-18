"""T10: Phase 1 Pruner 整合测试"""
import pytest
from unittest.mock import MagicMock

from egrefine.data.schema import Column, Table, Schema
from egrefine.models.embedding_client import EmbeddingClient
from egrefine.phase1.pruner import prune, PruneResult


# ========== Fixtures ==========

@pytest.fixture
def financial_like_schema():
    """模拟 financial 数据库的典型 schema: 短名、泛化词、混合风格。"""
    return Schema(
        db_id="financial",
        tables=[
            Table(name="district", columns=[
                Column(name="id", table="district", dtype="INTEGER", is_pk=True),
                Column(name="A2", table="district", dtype="TEXT"),
                Column(name="A3", table="district", dtype="TEXT"),
                Column(name="A11", table="district", dtype="REAL"),
                Column(name="district_name", table="district", dtype="TEXT"),
            ]),
            Table(name="loan", columns=[
                Column(name="loan_id", table="loan", dtype="INTEGER", is_pk=True),
                Column(name="status", table="loan", dtype="TEXT"),
                Column(name="amount", table="loan", dtype="REAL"),
                Column(name="date", table="loan", dtype="TEXT"),
            ]),
            Table(name="card", columns=[
                Column(name="card_id", table="card", dtype="INTEGER", is_pk=True),
                Column(name="type", table="card", dtype="TEXT"),
                Column(name="issued", table="card", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[("loan.loan_id", "card.card_id")],
    )


@pytest.fixture
def clean_schema():
    """命名规范的 schema，应该很少被标记。"""
    return Schema(
        db_id="clean",
        tables=[
            Table(name="employees", columns=[
                Column(name="employee_id", table="employees", dtype="INTEGER", is_pk=True),
                Column(name="first_name", table="employees", dtype="TEXT"),
                Column(name="last_name", table="employees", dtype="TEXT"),
                Column(name="hire_date", table="employees", dtype="TEXT"),
                Column(name="annual_salary", table="employees", dtype="REAL"),
            ]),
        ],
        foreign_keys=[],
    )


@pytest.fixture
def default_phase1_config():
    return {
        "signals": {
            "short_name": {"enabled": True, "max_length": 3},
            "high_similarity": {"enabled": False},  # 默认关闭 S2 方便测试
            "naming_inconsistency": {"enabled": True},
            "generic_vocabulary": {"enabled": True},
        },
        "skip_primary_keys": True,
    }


@pytest.fixture
def all_signals_config():
    """全部 signal 开启，含 S2。"""
    return {
        "signals": {
            "short_name": {"enabled": True, "max_length": 3},
            "high_similarity": {"enabled": True, "threshold": 0.85},
            "naming_inconsistency": {"enabled": True},
            "generic_vocabulary": {"enabled": True},
        },
        "skip_primary_keys": True,
    }


@pytest.fixture
def mock_embedding_client():
    client = MagicMock(spec=EmbeddingClient)
    # A2 和 A3 高度相似，其他正交
    def fake_embed(texts):
        vectors = []
        for i, t in enumerate(texts):
            if "A2" in t or "A3" in t:
                vectors.append([1.0, 0.1 * i, 0.0, 0.0])
            else:
                vec = [0.0] * 4
                vec[i % 4] = 1.0
                vectors.append(vec)
        return vectors
    client.embed.side_effect = fake_embed
    return client


# ========== PruneResult ==========

class TestPruneResult:
    def test_candidate_count(self):
        cols = [Column(name="a", table="t", dtype="TEXT")]
        r = PruneResult(candidates=cols, total_columns=10,
                        signal_hits={}, skipped_pks=[])
        assert r.candidate_count == 1

    def test_compression_ratio(self):
        cols = [Column(name="a", table="t", dtype="TEXT")] * 3
        r = PruneResult(candidates=cols, total_columns=10,
                        signal_hits={}, skipped_pks=[])
        assert r.compression_ratio == pytest.approx(0.3)

    def test_compression_ratio_zero_total(self):
        r = PruneResult(candidates=[], total_columns=0,
                        signal_hits={}, skipped_pks=[])
        assert r.compression_ratio == 0.0

    def test_summary_string(self):
        r = PruneResult(
            candidates=[],
            total_columns=50,
            signal_hits={"S1_short_name": ["t.a", "t.b"]},
            skipped_pks=["t.id"],
        )
        s = r.summary()
        assert "0/50" in s
        assert "S1_short_name: 2" in s
        assert "Skipped PKs: 1" in s


# ========== prune() 基础行为 ==========

class TestPruneBasic:
    def test_financial_like_catches_short_names(self, financial_like_schema, default_phase1_config):
        result = prune(financial_like_schema, default_phase1_config)
        candidate_names = {c.full_name for c in result.candidates}
        # A2 (2字符), A3 (2字符) 应被 S1 捕获
        assert "district.A2" in candidate_names
        assert "district.A3" in candidate_names

    def test_financial_like_catches_generic(self, financial_like_schema, default_phase1_config):
        result = prune(financial_like_schema, default_phase1_config)
        candidate_names = {c.full_name for c in result.candidates}
        # status, amount, date, type 是泛化词
        assert "loan.status" in candidate_names
        assert "loan.amount" in candidate_names
        assert "loan.date" in candidate_names
        assert "card.type" in candidate_names

    def test_pks_skipped(self, financial_like_schema, default_phase1_config):
        result = prune(financial_like_schema, default_phase1_config)
        candidate_names = {c.full_name for c in result.candidates}
        # PK 列不应出现
        assert "district.id" not in candidate_names
        assert "loan.loan_id" not in candidate_names
        assert "card.card_id" not in candidate_names
        # 但 skipped_pks 应记录
        assert "district.id" in result.skipped_pks

    def test_clean_schema_minimal_candidates(self, clean_schema, default_phase1_config):
        result = prune(clean_schema, default_phase1_config)
        # 规范命名、无短名、无泛化词 -> 候选很少
        candidate_names = {c.full_name for c in result.candidates}
        assert "employees.employee_id" not in candidate_names  # PK skipped
        # 长描述性名字不应被标记
        assert "employees.annual_salary" not in candidate_names

    def test_total_columns_correct(self, financial_like_schema, default_phase1_config):
        result = prune(financial_like_schema, default_phase1_config)
        assert result.total_columns == 12  # 5 + 4 + 3


# ========== Signal 开关 ==========

class TestPruneSignalToggle:
    def test_all_disabled_empty_result(self, financial_like_schema):
        config = {
            "signals": {
                "short_name": {"enabled": False},
                "high_similarity": {"enabled": False},
                "naming_inconsistency": {"enabled": False},
                "generic_vocabulary": {"enabled": False},
            },
            "skip_primary_keys": True,
        }
        result = prune(financial_like_schema, config)
        assert result.candidate_count == 0

    def test_only_s1_enabled(self, financial_like_schema):
        config = {
            "signals": {
                "short_name": {"enabled": True, "max_length": 3},
                "high_similarity": {"enabled": False},
                "naming_inconsistency": {"enabled": False},
                "generic_vocabulary": {"enabled": False},
            },
            "skip_primary_keys": True,
        }
        result = prune(financial_like_schema, config)
        # 只有短名被标记: A2(2), A3(2), A11(3)
        candidate_names = {c.full_name for c in result.candidates}
        assert candidate_names == {"district.A2", "district.A3", "district.A11"}

    def test_only_s4_enabled(self, financial_like_schema):
        config = {
            "signals": {
                "short_name": {"enabled": False},
                "high_similarity": {"enabled": False},
                "naming_inconsistency": {"enabled": False},
                "generic_vocabulary": {"enabled": True},
            },
            "skip_primary_keys": True,
        }
        result = prune(financial_like_schema, config)
        candidate_names = {c.full_name for c in result.candidates}
        assert "loan.status" in candidate_names
        assert "card.type" in candidate_names
        assert "loan.amount" in candidate_names
        assert "loan.date" in candidate_names


# ========== S2 with embedding ==========

class TestPruneWithEmbedding:
    def test_s2_finds_similar_columns(
        self, financial_like_schema, all_signals_config, mock_embedding_client
    ):
        result = prune(financial_like_schema, all_signals_config, mock_embedding_client)
        s2_hits = result.signal_hits["S2_high_similarity"]
        # A2 和 A3 应被 S2 标记（高相似度）
        assert "district.A2" in s2_hits or "district.A3" in s2_hits

    def test_s2_no_client_warns(self, financial_like_schema, all_signals_config):
        """S2 开启但无 embedding client，应跳过 S2 不报错。"""
        result = prune(financial_like_schema, all_signals_config, embedding_client=None)
        assert result.signal_hits["S2_high_similarity"] == []


# ========== skip_primary_keys ==========

class TestSkipPrimaryKeys:
    def test_pk_not_skipped_when_disabled(self, financial_like_schema):
        config = {
            "signals": {
                "short_name": {"enabled": True, "max_length": 3},
                "high_similarity": {"enabled": False},
                "naming_inconsistency": {"enabled": False},
                "generic_vocabulary": {"enabled": True},
            },
            "skip_primary_keys": False,
        }
        result = prune(financial_like_schema, config)
        candidate_names = {c.full_name for c in result.candidates}
        # "id" 是泛化词 + 短名 -> 应被标记
        assert "district.id" in candidate_names
        assert result.skipped_pks == []


# ========== Union 行为 ==========

class TestPruneUnion:
    def test_union_deduplicates(self, financial_like_schema, default_phase1_config):
        """同一列被多个 signal 标记，candidates 不重复。"""
        result = prune(financial_like_schema, default_phase1_config)
        full_names = [c.full_name for c in result.candidates]
        assert len(full_names) == len(set(full_names))

    def test_preserves_column_order(self, financial_like_schema, default_phase1_config):
        """候选列应保持原 schema 中的顺序。"""
        result = prune(financial_like_schema, default_phase1_config)
        all_cols = financial_like_schema.all_columns
        all_names = [c.full_name for c in all_cols]
        cand_names = [c.full_name for c in result.candidates]
        # 验证候选列的相对顺序与原 schema 一致
        indices = [all_names.index(n) for n in cand_names]
        assert indices == sorted(indices)


# ========== 边界情况 ==========

class TestPruneEdgeCases:
    def test_empty_schema(self, default_phase1_config):
        schema = Schema(db_id="empty", tables=[], foreign_keys=[])
        result = prune(schema, default_phase1_config)
        assert result.candidate_count == 0
        assert result.total_columns == 0

    def test_single_pk_column(self, default_phase1_config):
        schema = Schema(
            db_id="tiny",
            tables=[Table(name="t", columns=[
                Column(name="id", table="t", dtype="INTEGER", is_pk=True),
            ])],
            foreign_keys=[],
        )
        result = prune(schema, default_phase1_config)
        assert result.candidate_count == 0
        assert result.total_columns == 1
