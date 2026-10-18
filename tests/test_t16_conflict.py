"""T16: Phase 3 Conflict Resolution tests."""
import pytest

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase2.prompts import CandidateName
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase3.conflict import resolve_conflicts


# ========== Fixtures ==========

@pytest.fixture
def schema():
    return Schema(
        db_id="test_db",
        tables=[
            Table(name="employees", columns=[
                Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
                Column(name="nm", table="employees", dtype="TEXT"),
                Column(name="desc", table="employees", dtype="TEXT"),
            ]),
            Table(name="departments", columns=[
                Column(name="id", table="departments", dtype="INTEGER", is_pk=True),
                Column(name="nm", table="departments", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[],
    )


@pytest.fixture
def schema_with_fk():
    """Schema where employees.dept_id FK -> departments.id, creating cross-table scope overlap."""
    return Schema(
        db_id="test_db",
        tables=[
            Table(name="employees", columns=[
                Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
                Column(name="nm", table="employees", dtype="TEXT"),
                Column(name="dept_id", table="employees", dtype="INTEGER",
                       fk_target="departments.id"),
            ]),
            Table(name="departments", columns=[
                Column(name="id", table="departments", dtype="INTEGER", is_pk=True),
                Column(name="nm", table="departments", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[("employees.dept_id", "departments.id")],
    )


def _make_result(col, selected, delta, was_changed, all_scores=None):
    return SelectionResult(
        column=col,
        selected_name=selected,
        delta=delta,
        was_changed=was_changed,
        all_scores=all_scores or {},
    )


# ========== Tests ==========

class TestResolveConflicts:
    def test_no_conflicts(self, schema):
        """No conflicts: different tables, different names."""
        col_a = schema.get_column("employees", "nm")
        col_b = schema.get_column("departments", "nm")
        results = [
            _make_result(col_a, "employee_name", 0.1, True,
                         {"nm": 0.5, "employee_name": 0.6, "name": 0.55}),
            _make_result(col_b, "department_name", 0.15, True,
                         {"nm": 0.4, "department_name": 0.55}),
        ]
        resolved = resolve_conflicts(results, schema)
        assert len(resolved) == 2
        assert resolved[0].selected_name == "employee_name"
        assert resolved[1].selected_name == "department_name"

    def test_same_table_conflict_higher_delta_wins(self, schema):
        """Two columns in same table refined to same name -> higher delta wins."""
        col_nm = schema.get_column("employees", "nm")
        col_desc = schema.get_column("employees", "desc")
        results = [
            _make_result(col_nm, "name", 0.2, True,
                         {"nm": 0.5, "name": 0.7, "employee_name": 0.6}),
            _make_result(col_desc, "name", 0.1, True,
                         {"desc": 0.5, "name": 0.6, "description": 0.55}),
        ]
        resolved = resolve_conflicts(results, schema)
        # col_nm has higher delta, keeps "name"
        nm_result = next(r for r in resolved if r.column.name == "nm")
        desc_result = next(r for r in resolved if r.column.name == "desc")
        assert nm_result.selected_name == "name"
        # col_desc reverts to next best candidate
        assert desc_result.selected_name != "name"

    def test_loser_reverts_to_next_best(self, schema):
        """Loser of conflict picks next best candidate from all_scores."""
        col_nm = schema.get_column("employees", "nm")
        col_desc = schema.get_column("employees", "desc")
        results = [
            _make_result(col_nm, "name", 0.2, True,
                         {"nm": 0.5, "name": 0.7}),
            _make_result(col_desc, "name", 0.1, True,
                         {"desc": 0.5, "name": 0.6, "description": 0.58}),
        ]
        resolved = resolve_conflicts(results, schema)
        desc_result = next(r for r in resolved if r.column.name == "desc")
        # Next best for desc: "description" (0.58) > "desc" (0.5)
        assert desc_result.selected_name == "description"

    def test_loser_reverts_to_original_if_no_other(self, schema):
        """If loser has no other candidate better than original, revert to original."""
        col_nm = schema.get_column("employees", "nm")
        col_desc = schema.get_column("employees", "desc")
        results = [
            _make_result(col_nm, "name", 0.2, True,
                         {"nm": 0.5, "name": 0.7}),
            _make_result(col_desc, "name", 0.1, True,
                         {"desc": 0.6, "name": 0.7}),  # only "name" beats original
        ]
        resolved = resolve_conflicts(results, schema)
        desc_result = next(r for r in resolved if r.column.name == "desc")
        assert desc_result.selected_name == "desc"
        assert desc_result.was_changed is False

    def test_no_conflict_when_unchanged(self, schema):
        """Unchanged columns don't participate in conflicts."""
        col_nm = schema.get_column("employees", "nm")
        col_desc = schema.get_column("employees", "desc")
        results = [
            _make_result(col_nm, "nm", 0.0, False, {"nm": 0.7}),
            _make_result(col_desc, "description", 0.1, True,
                         {"desc": 0.5, "description": 0.6}),
        ]
        resolved = resolve_conflicts(results, schema)
        assert resolved[0].selected_name == "nm"
        assert resolved[1].selected_name == "description"

    def test_cross_table_no_conflict_different_scope(self, schema_with_fk):
        """Columns in different tables without direct FK link don't conflict."""
        col_emp_nm = schema_with_fk.get_column("employees", "nm")
        col_dept_nm = schema_with_fk.get_column("departments", "nm")
        # employees.nm and departments.nm are NOT in same scope
        # (FK links dept_id->id, not nm->nm)
        results = [
            _make_result(col_emp_nm, "name", 0.15, True,
                         {"nm": 0.5, "name": 0.65}),
            _make_result(col_dept_nm, "name", 0.2, True,
                         {"nm": 0.4, "name": 0.6}),
        ]
        resolved = resolve_conflicts(results, schema_with_fk)
        # No conflict — both keep "name" (different tables, no shared scope)
        assert resolved[0].selected_name == "name"
        assert resolved[1].selected_name == "name"

    def test_fk_columns_in_same_scope(self):
        """FK-linked columns ARE in same scope and can conflict."""
        schema_fk = Schema(
            db_id="test_db",
            tables=[
                Table(name="orders", columns=[
                    Column(name="id", table="orders", dtype="INTEGER", is_pk=True),
                    Column(name="cid", table="orders", dtype="INTEGER",
                           fk_target="customers.id"),
                ]),
                Table(name="customers", columns=[
                    Column(name="id", table="customers", dtype="INTEGER", is_pk=True),
                ]),
            ],
            foreign_keys=[("orders.cid", "customers.id")],
        )
        col_cid = schema_fk.get_column("orders", "cid")
        col_cust_id = schema_fk.get_column("customers", "id")
        # Both refined to "customer_id" — they share scope via FK
        results = [
            _make_result(col_cid, "customer_id", 0.2, True,
                         {"cid": 0.4, "customer_id": 0.6, "client_id": 0.5}),
            _make_result(col_cust_id, "customer_id", 0.1, True,
                         {"id": 0.5, "customer_id": 0.6, "cust_id": 0.55}),
        ]
        resolved = resolve_conflicts(results, schema_fk)
        cid_result = next(r for r in resolved if r.column.name == "cid")
        id_result = next(r for r in resolved if r.column.name == "id")
        # cid has higher delta, keeps "customer_id"
        assert cid_result.selected_name == "customer_id"
        # id reverts to next best
        assert id_result.selected_name != "customer_id"

    def test_multiple_rounds(self, schema):
        """Conflict resolution may need multiple rounds."""
        # Three columns in same table, all refined to "name"
        schema_3col = Schema(
            db_id="test_db",
            tables=[Table(name="t", columns=[
                Column(name="a", table="t", dtype="TEXT"),
                Column(name="b", table="t", dtype="TEXT"),
                Column(name="c", table="t", dtype="TEXT"),
            ])],
            foreign_keys=[],
        )
        results = [
            _make_result(schema_3col.get_column("t", "a"), "name", 0.3, True,
                         {"a": 0.4, "name": 0.7, "label": 0.5}),
            _make_result(schema_3col.get_column("t", "b"), "name", 0.2, True,
                         {"b": 0.5, "name": 0.7, "label": 0.6}),
            _make_result(schema_3col.get_column("t", "c"), "name", 0.1, True,
                         {"c": 0.5, "name": 0.6, "title": 0.55}),
        ]
        resolved = resolve_conflicts(results, schema_3col, max_rounds=3)
        names = [r.selected_name for r in resolved]
        # All should have unique names
        assert len(set(names)) == len(names), f"Duplicate names: {names}"
        # "a" has highest delta, keeps "name"
        a_result = next(r for r in resolved if r.column.name == "a")
        assert a_result.selected_name == "name"

    def test_empty_results(self, schema):
        """Empty results list returns empty."""
        resolved = resolve_conflicts([], schema)
        assert resolved == []

    def test_preserves_unchanged_results(self, schema):
        """Unchanged results pass through untouched."""
        col = schema.get_column("employees", "nm")
        results = [_make_result(col, "nm", 0.0, False, {"nm": 0.7})]
        resolved = resolve_conflicts(results, schema)
        assert len(resolved) == 1
        assert resolved[0].selected_name == "nm"
