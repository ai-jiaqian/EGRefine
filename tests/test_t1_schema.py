"""T1 测试: 数据结构与 Schema 表示"""
import os
import sys
import copy
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.data.schema import Column, Table, Schema, NLSQLPair


# ====== Column 测试 ======

class TestColumn:
    def test_basic_fields(self):
        col = Column(name="nm", table="employees", dtype="TEXT")
        assert col.name == "nm"
        assert col.table == "employees"
        assert col.dtype == "TEXT"
        assert col.is_pk is False
        assert col.fk_target is None

    def test_pk_and_fk(self):
        col = Column(name="dept_id", table="employees", dtype="INTEGER",
                     is_pk=False, fk_target="departments.id")
        assert col.fk_target == "departments.id"

    def test_full_name(self):
        col = Column(name="nm", table="employees", dtype="TEXT")
        assert col.full_name == "employees.nm"


# ====== Table 测试 ======

class TestTable:
    def test_basic(self):
        cols = [
            Column(name="id", table="t", dtype="INTEGER", is_pk=True),
            Column(name="name", table="t", dtype="TEXT"),
        ]
        t = Table(name="t", columns=cols)
        assert t.name == "t"
        assert len(t.columns) == 2

    def test_primary_keys(self):
        cols = [
            Column(name="id", table="t", dtype="INTEGER", is_pk=True),
            Column(name="name", table="t", dtype="TEXT"),
            Column(name="code", table="t", dtype="TEXT", is_pk=True),
        ]
        t = Table(name="t", columns=cols)
        pks = t.primary_keys
        assert len(pks) == 2
        assert {c.name for c in pks} == {"id", "code"}


# ====== Schema 测试 ======

def _make_schema():
    """构造一个用于测试的 Schema:
    employees(id PK, nm, dept_id FK->departments.id)
    departments(id PK, dept_name)
    """
    emp_cols = [
        Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
        Column(name="nm", table="employees", dtype="TEXT"),
        Column(name="dept_id", table="employees", dtype="INTEGER",
               fk_target="departments.id"),
    ]
    dept_cols = [
        Column(name="id", table="departments", dtype="INTEGER", is_pk=True),
        Column(name="dept_name", table="departments", dtype="TEXT"),
    ]
    return Schema(
        db_id="test_db",
        tables=[
            Table(name="employees", columns=emp_cols),
            Table(name="departments", columns=dept_cols),
        ],
        foreign_keys=[("employees.dept_id", "departments.id")],
    )


class TestSchema:
    def test_all_columns(self):
        schema = _make_schema()
        assert len(schema.all_columns) == 5

    def test_all_columns_names(self):
        schema = _make_schema()
        names = {c.full_name for c in schema.all_columns}
        assert "employees.nm" in names
        assert "departments.dept_name" in names

    def test_get_table(self):
        schema = _make_schema()
        t = schema.get_table("employees")
        assert t is not None
        assert t.name == "employees"
        assert schema.get_table("nonexistent") is None

    def test_get_column(self):
        schema = _make_schema()
        col = schema.get_column("employees", "nm")
        assert col is not None
        assert col.dtype == "TEXT"
        assert schema.get_column("employees", "nonexistent") is None

    def test_scope_same_table(self):
        """scope 应包含同表所有列"""
        schema = _make_schema()
        col = schema.get_column("employees", "nm")
        scope = schema.scope(col)
        scope_names = {c.full_name for c in scope}
        assert "employees.id" in scope_names
        assert "employees.nm" in scope_names
        assert "employees.dept_id" in scope_names

    def test_scope_fk_related(self):
        """scope 应包含 FK 关联的列"""
        schema = _make_schema()
        col = schema.get_column("employees", "dept_id")
        scope = schema.scope(col)
        scope_names = {c.full_name for c in scope}
        # 同表列
        assert "employees.id" in scope_names
        # FK 关联列
        assert "departments.id" in scope_names

    def test_scope_reverse_fk(self):
        """从 FK 目标侧也能找到关联"""
        schema = _make_schema()
        col = schema.get_column("departments", "id")
        scope = schema.scope(col)
        scope_names = {c.full_name for c in scope}
        assert "employees.dept_id" in scope_names

    def test_scope_no_fk(self):
        """没有 FK 关联的列，scope 只包含同表列"""
        schema = _make_schema()
        col = schema.get_column("departments", "dept_name")
        scope = schema.scope(col)
        scope_names = {c.full_name for c in scope}
        assert "departments.id" in scope_names
        assert "departments.dept_name" in scope_names
        assert "employees.nm" not in scope_names


# ====== apply_refinement 测试 ======

class TestApplyRefinement:
    def test_basic_rename(self):
        schema = _make_schema()
        new_schema = schema.apply_refinement({"employees.nm": "employee_name"})
        col = new_schema.get_column("employees", "employee_name")
        assert col is not None
        assert col.dtype == "TEXT"
        # 原 schema 不受影响
        assert schema.get_column("employees", "nm") is not None
        assert schema.get_column("employees", "employee_name") is None

    def test_rename_preserves_other_columns(self):
        schema = _make_schema()
        new_schema = schema.apply_refinement({"employees.nm": "employee_name"})
        assert new_schema.get_column("employees", "id") is not None
        assert new_schema.get_column("employees", "dept_id") is not None
        assert new_schema.get_column("departments", "id") is not None

    def test_rename_updates_table_field(self):
        """rename 后列的 table 字段应不变"""
        schema = _make_schema()
        new_schema = schema.apply_refinement({"employees.nm": "employee_name"})
        col = new_schema.get_column("employees", "employee_name")
        assert col.table == "employees"

    def test_rename_nonexistent_column_ignored(self):
        """rename 不存在的列不应报错"""
        schema = _make_schema()
        new_schema = schema.apply_refinement({"employees.nonexist": "whatever"})
        assert len(new_schema.all_columns) == 5

    def test_multiple_renames(self):
        schema = _make_schema()
        new_schema = schema.apply_refinement({
            "employees.nm": "employee_name",
            "departments.dept_name": "department_name",
        })
        assert new_schema.get_column("employees", "employee_name") is not None
        assert new_schema.get_column("departments", "department_name") is not None

    def test_immutability(self):
        """apply_refinement 不应修改原 Schema"""
        schema = _make_schema()
        original_cols = [c.name for c in schema.all_columns]
        schema.apply_refinement({"employees.nm": "employee_name"})
        after_cols = [c.name for c in schema.all_columns]
        assert original_cols == after_cols


# ====== NLSQLPair 测试 ======

class TestNLSQLPair:
    def test_basic(self):
        pair = NLSQLPair(
            nl="What is the name?",
            gold_sql="SELECT nm FROM employees",
            db_id="test_db",
        )
        assert pair.nl == "What is the name?"
        assert pair.db_id == "test_db"
