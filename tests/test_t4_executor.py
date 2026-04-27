"""T4 测试: SQL 执行器"""
import os
import sys
import sqlite3
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.phase3.executor import execute_sql, compare_results


@pytest.fixture
def sample_db(tmp_path):
    """创建一个内存测试数据库"""
    db_path = str(tmp_path / "test.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary INTEGER, dept TEXT)")
    conn.executemany(
        "INSERT INTO employees VALUES (?, ?, ?, ?)",
        [
            (1, "Alice", 50000, "Engineering"),
            (2, "Bob", 60000, "Engineering"),
            (3, "Charlie", 45000, "Sales"),
            (4, "Diana", 70000, "Sales"),
            (5, "Eve", 55000, "HR"),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


# ====== execute_sql 测试 ======

class TestExecuteSQL:
    def test_basic_select(self, sample_db):
        result = execute_sql("SELECT name FROM employees WHERE id = 1", sample_db)
        assert result is not None
        assert ("Alice",) in result

    def test_returns_set_of_tuples(self, sample_db):
        result = execute_sql("SELECT name FROM employees ORDER BY id", sample_db)
        assert isinstance(result, set)
        assert len(result) == 5

    def test_aggregate(self, sample_db):
        result = execute_sql("SELECT COUNT(*) FROM employees", sample_db)
        assert result == {(5,)}

    def test_syntax_error_returns_none(self, sample_db):
        result = execute_sql("SELEC name FROM employees", sample_db)
        assert result is None

    def test_nonexistent_table_returns_none(self, sample_db):
        result = execute_sql("SELECT * FROM nonexistent", sample_db)
        assert result is None

    def test_empty_result(self, sample_db):
        result = execute_sql("SELECT * FROM employees WHERE id = 999", sample_db)
        assert result is not None
        assert result == set()

    def test_multiple_columns(self, sample_db):
        result = execute_sql("SELECT name, salary FROM employees WHERE id = 1", sample_db)
        assert result == {("Alice", 50000)}

    def test_timeout_parameter(self, sample_db):
        """timeout 参数不应影响正常查询"""
        result = execute_sql("SELECT 1", sample_db, timeout=5)
        assert result == {(1,)}


# ====== compare_results 测试 ======

class TestCompareResults:
    def test_identical_queries(self, sample_db):
        sql = "SELECT name FROM employees WHERE dept = 'Engineering'"
        assert compare_results(sql, sql, sample_db) is True

    def test_equivalent_queries_different_order(self, sample_db):
        """不同 ORDER BY 但结果集相同（set 比较忽略顺序）"""
        sql1 = "SELECT name FROM employees WHERE dept = 'Engineering' ORDER BY name ASC"
        sql2 = "SELECT name FROM employees WHERE dept = 'Engineering' ORDER BY name DESC"
        assert compare_results(sql1, sql2, sample_db) is True

    def test_different_results(self, sample_db):
        sql1 = "SELECT name FROM employees WHERE dept = 'Engineering'"
        sql2 = "SELECT name FROM employees WHERE dept = 'Sales'"
        assert compare_results(sql1, sql2, sample_db) is False

    def test_pred_syntax_error(self, sample_db):
        pred = "SELEC name FROM employees"
        gold = "SELECT name FROM employees"
        assert compare_results(pred, gold, sample_db) is False

    def test_gold_syntax_error(self, sample_db):
        pred = "SELECT name FROM employees"
        gold = "SELEC name FROM employees"
        assert compare_results(pred, gold, sample_db) is False

    def test_both_empty_results(self, sample_db):
        sql1 = "SELECT * FROM employees WHERE id = 999"
        sql2 = "SELECT * FROM employees WHERE id = 888"
        assert compare_results(sql1, sql2, sample_db) is True

    def test_one_empty_one_not(self, sample_db):
        sql1 = "SELECT * FROM employees WHERE id = 999"
        sql2 = "SELECT * FROM employees WHERE id = 1"
        assert compare_results(sql1, sql2, sample_db) is False


# ====== 用真实 BIRD 数据测试 ======

_bird_path = os.environ.get("EGREFINE_TEST_BIRD_PATH", "/path/to/BIRD/MINIDEV")
BIRD_DB = os.path.join(_bird_path, "dev_databases", "financial", "financial.sqlite")


class TestWithBIRD:
    @pytest.mark.skipif(not os.path.exists(BIRD_DB), reason="BIRD not available")
    def test_real_query(self):
        result = execute_sql("SELECT COUNT(*) FROM account", BIRD_DB)
        assert result is not None
        assert len(result) == 1

    @pytest.mark.skipif(not os.path.exists(BIRD_DB), reason="BIRD not available")
    def test_real_compare(self):
        sql = "SELECT COUNT(*) FROM account"
        assert compare_results(sql, sql, BIRD_DB) is True
