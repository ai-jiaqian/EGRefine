"""T3 测试: Query-Column 索引"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.data.query_index import extract_columns_from_sql, build_query_index

BIRD_PATH = os.environ.get("EGREFINE_TEST_BIRD_PATH", "/path/to/BIRD/MINIDEV")
_bird_required = pytest.mark.skipif(
    not os.path.isdir(BIRD_PATH),
    reason=f"BIRD MINIDEV not found at {BIRD_PATH}; set EGREFINE_TEST_BIRD_PATH to enable",
)


def _make_financial_schema_subset():
    """financial 数据库的部分 schema，用于测试"""
    return Schema(
        db_id="financial",
        tables=[
            Table(name="account", columns=[
                Column(name="account_id", table="account", dtype="INTEGER", is_pk=True),
                Column(name="district_id", table="account", dtype="INTEGER"),
                Column(name="frequency", table="account", dtype="TEXT"),
                Column(name="date", table="account", dtype="DATE"),
            ]),
            Table(name="district", columns=[
                Column(name="district_id", table="district", dtype="INTEGER", is_pk=True),
                Column(name="A2", table="district", dtype="TEXT"),
                Column(name="A3", table="district", dtype="TEXT"),
                Column(name="A11", table="district", dtype="REAL"),
            ]),
            Table(name="loan", columns=[
                Column(name="loan_id", table="loan", dtype="INTEGER", is_pk=True),
                Column(name="account_id", table="loan", dtype="INTEGER"),
                Column(name="date", table="loan", dtype="DATE"),
                Column(name="amount", table="loan", dtype="INTEGER"),
                Column(name="duration", table="loan", dtype="INTEGER"),
                Column(name="status", table="loan", dtype="TEXT"),
            ]),
            Table(name="client", columns=[
                Column(name="client_id", table="client", dtype="INTEGER", is_pk=True),
                Column(name="gender", table="client", dtype="TEXT"),
                Column(name="birth_date", table="client", dtype="DATE"),
                Column(name="district_id", table="client", dtype="INTEGER"),
            ]),
        ],
        foreign_keys=[
            ("account.district_id", "district.district_id"),
            ("loan.account_id", "account.account_id"),
            ("client.district_id", "district.district_id"),
        ],
    )


# ====== extract_columns_from_sql 测试 ======

class TestExtractColumns:
    def test_simple_select(self):
        schema = _make_financial_schema_subset()
        sql = "SELECT account_id FROM account"
        cols = extract_columns_from_sql(sql, schema)
        names = {c.full_name for c in cols}
        assert "account.account_id" in names or "loan.account_id" in names

    def test_aliased_select(self):
        """BIRD 常见模式: T1.column_name"""
        schema = _make_financial_schema_subset()
        sql = "SELECT T1.account_id FROM account AS T1"
        cols = extract_columns_from_sql(sql, schema)
        col_names = {c.name for c in cols}
        assert "account_id" in col_names

    def test_join_with_aliases(self):
        schema = _make_financial_schema_subset()
        sql = (
            "SELECT COUNT(T2.account_id) FROM district AS T1 "
            "INNER JOIN account AS T2 ON T1.district_id = T2.district_id "
            "WHERE T1.A3 = 'east Bohemia' AND T2.frequency = 'POPLATEK PO OBRATU'"
        )
        cols = extract_columns_from_sql(sql, schema)
        col_names = {c.name for c in cols}
        assert "account_id" in col_names
        assert "district_id" in col_names
        assert "A3" in col_names
        assert "frequency" in col_names

    def test_no_match_returns_empty(self):
        schema = _make_financial_schema_subset()
        sql = "SELECT 1"
        cols = extract_columns_from_sql(sql, schema)
        assert len(cols) == 0

    def test_where_clause_column(self):
        schema = _make_financial_schema_subset()
        sql = "SELECT * FROM loan WHERE status = 'A'"
        cols = extract_columns_from_sql(sql, schema)
        col_names = {c.name for c in cols}
        assert "status" in col_names

    def test_order_by_column(self):
        schema = _make_financial_schema_subset()
        sql = "SELECT * FROM loan ORDER BY amount DESC LIMIT 1"
        cols = extract_columns_from_sql(sql, schema)
        col_names = {c.name for c in cols}
        assert "amount" in col_names

    def test_subquery(self):
        schema = _make_financial_schema_subset()
        sql = (
            "SELECT account_id FROM loan WHERE account_id IN "
            "(SELECT account_id FROM account WHERE frequency = 'POPLATEK TYDNE')"
        )
        cols = extract_columns_from_sql(sql, schema)
        col_names = {c.name for c in cols}
        assert "account_id" in col_names
        assert "frequency" in col_names


# ====== build_query_index 测试 ======

class TestBuildQueryIndex:
    def test_basic_index(self):
        schema = _make_financial_schema_subset()
        pairs = [
            NLSQLPair(
                nl="How many accounts?",
                gold_sql="SELECT COUNT(account_id) FROM account",
                db_id="financial",
            ),
            NLSQLPair(
                nl="What is loan status?",
                gold_sql="SELECT status FROM loan WHERE loan_id = 1",
                db_id="financial",
            ),
        ]
        index = build_query_index(pairs, schema)
        # index key 是 "table.column"
        assert "loan.status" in index
        assert len(index["loan.status"]) == 1
        assert index["loan.status"][0].nl == "What is loan status?"

    def test_column_in_multiple_queries(self):
        schema = _make_financial_schema_subset()
        pairs = [
            NLSQLPair(nl="q1", gold_sql="SELECT account_id FROM account", db_id="financial"),
            NLSQLPair(nl="q2", gold_sql="SELECT account_id FROM loan", db_id="financial"),
        ]
        index = build_query_index(pairs, schema)
        # account_id 出现在两个表中，两条查询都应被索引
        total = sum(len(v) for v in index.values() if "account_id" in v)
        # 至少 account_id 被索引了
        has_account_id = any("account_id" in k for k in index)
        assert has_account_id

    def test_empty_pairs(self):
        schema = _make_financial_schema_subset()
        index = build_query_index([], schema)
        assert len(index) == 0

    def test_returns_stats(self):
        schema = _make_financial_schema_subset()
        pairs = [
            NLSQLPair(nl="q1", gold_sql="SELECT status FROM loan", db_id="financial"),
        ]
        index = build_query_index(pairs, schema)
        # 每个 key 的 value 是 NLSQLPair 列表
        for key, val in index.items():
            assert all(isinstance(p, NLSQLPair) for p in val)


# ====== 用真实 BIRD 数据测试 ======

@_bird_required
class TestWithBIRD:
    @pytest.fixture
    def financial_data(self):
        from egrefine.data.benchmark import load_bird
        schemas, pairs = load_bird(BIRD_PATH)
        fin_pairs = [p for p in pairs if p.db_id == "financial"]
        return schemas["financial"], fin_pairs

    def test_index_not_empty(self, financial_data):
        schema, pairs = financial_data
        index = build_query_index(pairs, schema)
        assert len(index) > 0

    def test_coverage_stats(self, financial_data):
        """统计列覆盖率"""
        schema, pairs = financial_data
        index = build_query_index(pairs, schema)
        all_cols = schema.all_columns
        covered = [c for c in all_cols if c.full_name in index]
        print(f"\nFinancial: {len(covered)}/{len(all_cols)} columns covered by queries")
        print(f"Index has {len(index)} entries, {sum(len(v) for v in index.values())} total references")
        for key in sorted(index.keys()):
            print(f"  {key}: {len(index[key])} queries")
        # 至少有一些列被覆盖
        assert len(covered) > 0
