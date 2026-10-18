"""T2 测试: Benchmark 数据加载"""
import os
import sys
import sqlite3
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.data.benchmark import load_schema_from_sqlite, load_bird, BIRDLoader

BIRD_PATH = os.environ.get("EGREFINE_TEST_BIRD_PATH", "/path/to/BIRD/MINIDEV")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(BIRD_PATH),
    reason=f"BIRD MINIDEV not found at {BIRD_PATH}; set EGREFINE_TEST_BIRD_PATH to enable",
)


# ====== 从 SQLite 加载 Schema ======

class TestLoadSchemaFromSQLite:
    """用真实的 BIRD financial 数据库测试"""

    @pytest.fixture
    def financial_db(self):
        return os.path.join(BIRD_PATH, "dev_databases", "financial", "financial.sqlite")

    def test_loads_tables(self, financial_db):
        schema = load_schema_from_sqlite("financial", financial_db)
        assert isinstance(schema, Schema)
        assert schema.db_id == "financial"
        table_names = {t.name for t in schema.tables}
        assert "account" in table_names
        assert "loan" in table_names
        assert len(schema.tables) == 8

    def test_loads_columns(self, financial_db):
        schema = load_schema_from_sqlite("financial", financial_db)
        account = schema.get_table("account")
        assert account is not None
        col_names = {c.name for c in account.columns}
        assert "account_id" in col_names
        assert "frequency" in col_names

    def test_detects_pk(self, financial_db):
        schema = load_schema_from_sqlite("financial", financial_db)
        col = schema.get_column("account", "account_id")
        assert col.is_pk is True
        col2 = schema.get_column("account", "frequency")
        assert col2.is_pk is False

    def test_detects_fk(self, financial_db):
        schema = load_schema_from_sqlite("financial", financial_db)
        col = schema.get_column("account", "district_id")
        assert col.fk_target == "district.district_id"

    def test_foreign_keys_list(self, financial_db):
        schema = load_schema_from_sqlite("financial", financial_db)
        assert len(schema.foreign_keys) > 0
        # account.district_id -> district.district_id 应在列表中
        assert ("account.district_id", "district.district_id") in schema.foreign_keys

    def test_column_dtype(self, financial_db):
        schema = load_schema_from_sqlite("financial", financial_db)
        col = schema.get_column("account", "account_id")
        assert col.dtype == "INTEGER"

    def test_column_table_field(self, financial_db):
        """每个 Column 的 table 字段应正确设置"""
        schema = load_schema_from_sqlite("financial", financial_db)
        for table in schema.tables:
            for col in table.columns:
                assert col.table == table.name


# ====== 用内存 SQLite 做隔离测试 ======

class TestLoadSchemaFromSQLiteIsolated:
    """用内存数据库做不依赖外部文件的测试"""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        db_path = str(tmp_path / "test.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, FOREIGN KEY(dept_id) REFERENCES departments(id))")
        conn.execute("CREATE TABLE departments (id INTEGER PRIMARY KEY, dept_name TEXT)")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.close()
        return db_path

    def test_isolated_tables(self, tmp_db):
        schema = load_schema_from_sqlite("test", tmp_db)
        assert len(schema.tables) == 2

    def test_isolated_pk(self, tmp_db):
        schema = load_schema_from_sqlite("test", tmp_db)
        assert schema.get_column("users", "id").is_pk is True
        assert schema.get_column("users", "name").is_pk is False

    def test_isolated_fk(self, tmp_db):
        schema = load_schema_from_sqlite("test", tmp_db)
        col = schema.get_column("users", "dept_id")
        assert col.fk_target == "departments.id"
        assert ("users.dept_id", "departments.id") in schema.foreign_keys


# ====== BIRD 加载测试 ======

class TestLoadBIRD:
    def test_load_returns_schemas_and_pairs(self):
        schemas, pairs = load_bird(BIRD_PATH)
        assert isinstance(schemas, dict)
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        assert len(schemas) > 0

    def test_schemas_keyed_by_db_id(self):
        schemas, _ = load_bird(BIRD_PATH)
        assert "financial" in schemas
        assert isinstance(schemas["financial"], Schema)

    def test_pairs_are_nlsqlpair(self):
        _, pairs = load_bird(BIRD_PATH)
        p = pairs[0]
        assert isinstance(p, NLSQLPair)
        assert len(p.nl) > 0
        assert len(p.gold_sql) > 0
        assert len(p.db_id) > 0

    def test_all_pair_db_ids_have_schema(self):
        schemas, pairs = load_bird(BIRD_PATH)
        for p in pairs:
            assert p.db_id in schemas, f"db_id={p.db_id} has no schema"

    def test_minidev_has_500_pairs(self):
        _, pairs = load_bird(BIRD_PATH)
        assert len(pairs) == 500

    def test_minidev_has_11_databases(self):
        schemas, _ = load_bird(BIRD_PATH)
        assert len(schemas) == 11


# ====== BIRDLoader 便捷类 ======

class TestBIRDLoader:
    def test_get_pairs_for_db(self):
        loader = BIRDLoader(BIRD_PATH)
        pairs = loader.get_pairs_for_db("financial")
        assert len(pairs) > 0
        assert all(p.db_id == "financial" for p in pairs)

    def test_get_db_path(self):
        loader = BIRDLoader(BIRD_PATH)
        db_path = loader.get_db_path("financial")
        assert os.path.exists(db_path)
        assert db_path.endswith(".sqlite")

    def test_db_ids(self):
        loader = BIRDLoader(BIRD_PATH)
        assert "financial" in loader.db_ids
        assert "toxicology" in loader.db_ids
