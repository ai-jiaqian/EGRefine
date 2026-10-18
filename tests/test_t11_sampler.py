"""T11: Phase 2 Data Sampler 测试"""
import os
import sqlite3
import tempfile
import pytest

from egrefine.phase2.sampler import sample_column, sample_table_columns, MAX_VALUE_LENGTH


@pytest.fixture
def sample_db():
    """创建临时 SQLite 数据库用于测试。"""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute(
        'CREATE TABLE employees ('
        '  id INTEGER PRIMARY KEY,'
        '  name TEXT,'
        '  salary REAL,'
        '  bio TEXT,'
        '  status TEXT'
        ')'
    )
    rows = [
        (1, "Alice", 50000.0, "A" * 200, "active"),
        (2, "Bob", 60000.0, None, "active"),
        (3, "Charlie", 55000.0, "Short bio", "inactive"),
        (4, None, 70000.0, "Another bio", "active"),
        (5, "Eve", None, None, None),
        (6, "Frank", 45000.0, "Normal", "active"),
        (7, "Grace", 80000.0, "OK", "active"),
    ]
    conn.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?, ?)", rows
    )
    conn.commit()

    # 表名/列名含特殊字符
    conn.execute('CREATE TABLE "my table" ("my col" TEXT, value INTEGER)')
    conn.execute('INSERT INTO "my table" VALUES (?, ?)', ("test", 42))
    conn.commit()

    conn.close()
    yield path
    os.unlink(path)


# ========== sample_column ==========

class TestSampleColumn:
    def test_basic_sampling(self, sample_db):
        values = sample_column(sample_db, "employees", "name", n=20)
        assert isinstance(values, list)
        # 7 rows, 1 NULL -> at most 6 distinct non-NULL
        assert len(values) <= 6
        assert "Alice" in values
        assert "Bob" in values

    def test_null_filtered(self, sample_db):
        values = sample_column(sample_db, "employees", "name", n=20)
        assert "None" not in values

    def test_long_string_truncated(self, sample_db):
        values = sample_column(sample_db, "employees", "bio", n=20)
        for v in values:
            assert len(v) <= MAX_VALUE_LENGTH + 3  # +3 for "..."
        # "A" * 200 应该被截断
        long_vals = [v for v in values if v.endswith("...")]
        assert len(long_vals) >= 1

    def test_numeric_to_string(self, sample_db):
        values = sample_column(sample_db, "employees", "salary", n=20)
        assert all(isinstance(v, str) for v in values)
        assert "50000.0" in values

    def test_limit_n(self, sample_db):
        values = sample_column(sample_db, "employees", "name", n=2)
        assert len(values) <= 2

    def test_all_null_column(self, sample_db):
        """列全为 NULL 时返回空列表。"""
        # Eve 的 salary 是 NULL，但其他行有值
        # 创建一个全 NULL 列的场景
        conn = sqlite3.connect(sample_db)
        conn.execute("CREATE TABLE nulltest (id INTEGER, val TEXT)")
        conn.execute("INSERT INTO nulltest VALUES (1, NULL)")
        conn.execute("INSERT INTO nulltest VALUES (2, NULL)")
        conn.commit()
        conn.close()
        values = sample_column(sample_db, "nulltest", "val", n=20)
        assert values == []

    def test_empty_table(self, sample_db):
        conn = sqlite3.connect(sample_db)
        conn.execute("CREATE TABLE empty_t (id INTEGER, val TEXT)")
        conn.commit()
        conn.close()
        values = sample_column(sample_db, "empty_t", "val", n=20)
        assert values == []

    def test_nonexistent_table(self, sample_db):
        values = sample_column(sample_db, "no_such_table", "col", n=20)
        assert values == []

    def test_nonexistent_column(self, sample_db):
        values = sample_column(sample_db, "employees", "no_such_col", n=20)
        assert values == []

    def test_special_chars_in_names(self, sample_db):
        """表名/列名含空格等特殊字符。"""
        values = sample_column(sample_db, "my table", "my col", n=20)
        assert "test" in values

    def test_distinct_values(self, sample_db):
        """采样使用 DISTINCT，重复值不重复出现。"""
        values = sample_column(sample_db, "employees", "status", n=20)
        assert len(values) == len(set(values))

    def test_invalid_db_path(self):
        values = sample_column("/nonexistent/path.sqlite", "t", "c", n=5)
        assert values == []


# ========== sample_table_columns ==========

class TestSampleTableColumns:
    def test_batch_sampling(self, sample_db):
        result = sample_table_columns(
            sample_db, "employees", ["name", "salary", "status"], n=20
        )
        assert set(result.keys()) == {"name", "salary", "status"}
        assert len(result["name"]) > 0
        assert len(result["salary"]) > 0

    def test_empty_column_list(self, sample_db):
        result = sample_table_columns(sample_db, "employees", [], n=20)
        assert result == {}

    def test_mixed_valid_invalid(self, sample_db):
        result = sample_table_columns(
            sample_db, "employees", ["name", "no_col"], n=20
        )
        assert len(result["name"]) > 0
        assert result["no_col"] == []
