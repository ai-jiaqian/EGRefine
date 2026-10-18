"""T6 测试: VIEW 构造 + Back-mapping"""
import os
import sys
import sqlite3
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase3.view_builder import build_view
from egrefine.phase4.backmapper import backmap


# ====== 测试用 Schema ======

def _make_schema():
    return Schema(
        db_id="test",
        tables=[
            Table(name="employees", columns=[
                Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
                Column(name="nm", table="employees", dtype="TEXT"),
                Column(name="sal", table="employees", dtype="INTEGER"),
                Column(name="dept", table="employees", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[],
    )


# ====== build_view 测试 ======

class TestBuildView:
    def test_single_column_rename(self):
        schema = _make_schema()
        col = schema.get_column("employees", "nm")
        sql = build_view(schema, col, "employee_name")
        assert "CREATE VIEW" in sql
        assert "nm AS employee_name" in sql
        assert "FROM employees" in sql

    def test_preserves_other_columns(self):
        schema = _make_schema()
        col = schema.get_column("employees", "nm")
        sql = build_view(schema, col, "employee_name")
        # 其他列应原样出现
        assert "id" in sql
        assert "sal" in sql
        assert "dept" in sql

    def test_view_name_prefix(self):
        schema = _make_schema()
        col = schema.get_column("employees", "nm")
        sql = build_view(schema, col, "employee_name")
        assert "refined_employees" in sql

    def test_custom_prefix(self):
        schema = _make_schema()
        col = schema.get_column("employees", "nm")
        sql = build_view(schema, col, "employee_name", view_prefix="v_")
        assert "v_employees" in sql

    def test_no_change_when_same_name(self):
        """候选名与原名相同时，不应有 AS"""
        schema = _make_schema()
        col = schema.get_column("employees", "nm")
        sql = build_view(schema, col, "nm")
        # nm 不需要 AS nm
        assert "nm AS nm" not in sql

    def test_view_is_valid_sql(self, tmp_path):
        """生成的 VIEW SQL 应能在 SQLite 中执行"""
        db_path = str(tmp_path / "test.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, nm TEXT, sal INTEGER, dept TEXT)")
        conn.execute("INSERT INTO employees VALUES (1, 'Alice', 50000, 'Eng')")

        schema = _make_schema()
        col = schema.get_column("employees", "nm")
        view_sql = build_view(schema, col, "employee_name")
        conn.execute(view_sql)

        # 通过 VIEW 查询
        rows = conn.execute("SELECT employee_name FROM refined_employees").fetchall()
        assert rows == [("Alice",)]
        conn.close()


# ====== backmap 测试 (CLAUDE.md 10.1 规定的测试) ======

class TestBackmap:
    def test_simple_replace(self):
        sql = "SELECT employee_name FROM refined_employees"
        mapping = {"employee_name": "nm"}
        assert backmap(sql, mapping) == "SELECT nm FROM refined_employees"

    def test_no_substring_collision(self):
        """长名字优先替换，避免子串误替换"""
        sql = "SELECT employee_name, name FROM t"
        mapping = {"employee_name": "emp_nm", "name": "nm"}
        result = backmap(sql, mapping)
        assert "emp_nm" in result
        # name 应被替换为 nm，但 employee_name 中的 name 不应被单独替换
        # employee_name -> emp_nm (先替换长的)
        # 剩余的 name -> nm
        assert result == "SELECT emp_nm, nm FROM t"

    def test_preserves_string_literals(self):
        """不替换 SQL 字符串常量中的内容 (regex word boundary 模式)"""
        sql = "SELECT nm FROM t WHERE desc = 'employee_name_value'"
        mapping = {"employee_name": "nm"}
        result = backmap(sql, mapping)
        # employee_name_value 不是完整 word，不应被替换
        assert "employee_name_value" in result

    def test_multiple_occurrences(self):
        sql = "SELECT employee_name FROM t WHERE employee_name > 0 ORDER BY employee_name"
        mapping = {"employee_name": "nm"}
        result = backmap(sql, mapping)
        assert result.count("nm") == 3
        assert "employee_name" not in result

    def test_empty_mapping(self):
        sql = "SELECT a FROM t"
        result = backmap(sql, {})
        assert result == sql

    def test_no_match(self):
        sql = "SELECT a FROM t"
        mapping = {"nonexistent": "x"}
        result = backmap(sql, mapping)
        assert result == sql

    def test_word_boundary(self):
        """不替换列名作为其他标识符子串的情况"""
        sql = "SELECT department_name FROM t"
        mapping = {"name": "nm"}
        result = backmap(sql, mapping)
        # department_name 中的 name 不应被替换
        assert "department_nm" not in result
        assert "department_name" in result


# ====== VIEW 等价性测试 (Theorem 1) ======

class TestViewEquivalence:
    @pytest.fixture
    def db_with_data(self, tmp_path):
        db_path = str(tmp_path / "test.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, nm TEXT, sal INTEGER, dept TEXT)")
        conn.executemany("INSERT INTO employees VALUES (?, ?, ?, ?)", [
            (1, "Alice", 50000, "Engineering"),
            (2, "Bob", 60000, "Engineering"),
            (3, "Charlie", 45000, "Sales"),
        ])
        conn.commit()
        conn.close()
        return db_path

    def test_theorem1_select(self, db_with_data):
        """Theorem 1: 原表查询 == backmap(VIEW查询) 的执行结果"""
        schema = _make_schema()
        col = schema.get_column("employees", "nm")

        conn = sqlite3.connect(db_with_data)

        # 1. 原表查询
        gold_sql = "SELECT nm FROM employees WHERE sal > 45000"
        result_a = set(conn.execute(gold_sql).fetchall())

        # 2. 创建 VIEW
        view_sql = build_view(schema, col, "employee_name")
        conn.execute(view_sql)

        # 3. 在 VIEW 上查询 (用 refined 列名)
        refined_sql = "SELECT employee_name FROM refined_employees WHERE sal > 45000"
        result_b = set(conn.execute(refined_sql).fetchall())

        # 4. back-map refined SQL 到原始列名执行
        mapping = {"employee_name": "nm"}
        backmapped_sql = backmap(refined_sql, mapping)
        # 注意: backmap 不替换 VIEW 名，只替换列名
        # 用原表执行 backmapped SQL
        original_sql = backmapped_sql.replace("refined_employees", "employees")
        result_c = set(conn.execute(original_sql).fetchall())

        conn.close()

        # result_a == result_b == result_c
        assert result_a == result_b
        assert result_a == result_c

    def test_theorem1_aggregate(self, db_with_data):
        schema = _make_schema()
        col = schema.get_column("employees", "sal")

        conn = sqlite3.connect(db_with_data)

        result_a = set(conn.execute("SELECT SUM(sal) FROM employees").fetchall())

        view_sql = build_view(schema, col, "salary")
        conn.execute(view_sql)

        result_b = set(conn.execute("SELECT SUM(salary) FROM refined_employees").fetchall())

        backmapped = backmap("SELECT SUM(salary) FROM employees", {"salary": "sal"})
        result_c = set(conn.execute(backmapped).fetchall())

        conn.close()

        assert result_a == result_b
        assert result_a == result_c

    def test_theorem1_join(self, tmp_path):
        """多表 JOIN 场景的 VIEW 等价性"""
        db_path = str(tmp_path / "join_test.sqlite")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE dept (id INTEGER PRIMARY KEY, dname TEXT)")
        conn.execute("CREATE TABLE emp (id INTEGER PRIMARY KEY, nm TEXT, did INTEGER)")
        conn.executemany("INSERT INTO dept VALUES (?, ?)", [(1, "Eng"), (2, "Sales")])
        conn.executemany("INSERT INTO emp VALUES (?, ?, ?)", [(1, "Alice", 1), (2, "Bob", 2)])

        schema = Schema(
            db_id="test",
            tables=[
                Table(name="emp", columns=[
                    Column(name="id", table="emp", dtype="INTEGER", is_pk=True),
                    Column(name="nm", table="emp", dtype="TEXT"),
                    Column(name="did", table="emp", dtype="INTEGER"),
                ]),
                Table(name="dept", columns=[
                    Column(name="id", table="dept", dtype="INTEGER", is_pk=True),
                    Column(name="dname", table="dept", dtype="TEXT"),
                ]),
            ],
            foreign_keys=[("emp.did", "dept.id")],
        )

        # 原始查询
        gold_sql = "SELECT nm, dname FROM emp JOIN dept ON emp.did = dept.id"
        result_a = set(conn.execute(gold_sql).fetchall())

        # 创建 VIEW (rename nm -> employee_name)
        col = schema.get_column("emp", "nm")
        view_sql = build_view(schema, col, "employee_name")
        conn.execute(view_sql)

        # VIEW 查询
        refined_sql = "SELECT employee_name, dname FROM refined_emp JOIN dept ON refined_emp.did = dept.id"
        result_b = set(conn.execute(refined_sql).fetchall())

        conn.close()
        assert result_a == result_b
