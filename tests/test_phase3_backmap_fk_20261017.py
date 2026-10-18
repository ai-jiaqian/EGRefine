"""Regressions for the two Phase 3 integration defects (2026-10-13 audit C10 / X4)."""
import sqlite3

from egrefine.data.schema import Column, Schema, Table
from egrefine.phase3.c3_runner import C3Text2SQL


def _schema():
    parent = Table("parent", [Column("id", "parent", "INTEGER", is_pk=True),
                              Column("nm", "parent", "TEXT")])
    child = Table("child", [Column("cid", "child", "INTEGER", is_pk=True),
                            Column("parent_id", "child", "INTEGER",
                                   fk_target="parent.id")])
    return Schema("toy", [parent, child], [("child.parent_id", "parent.id")])


def test_apply_refinement_rewrites_fk_references():
    s = _schema().apply_refinement({"parent.id": "parent_key"})
    assert s.foreign_keys == [("child.parent_id", "parent.parent_key")]
    assert s.get_column("child", "parent_id").fk_target == "parent.parent_key"
    all_cols = {c.full_name for c in s.all_columns}
    for src, tgt in s.foreign_keys:
        assert src in all_cols and tgt in all_cols


def test_apply_refinement_rewrites_fk_source_side():
    s = _schema().apply_refinement({"child.parent_id": "parent_ref"})
    assert s.foreign_keys == [("child.parent_ref", "parent.id")]


def test_c3_filter_keeps_renamed_fk():
    c3 = C3Text2SQL({"base_url": "x", "api_key": "x", "model_name": "x"})
    s = _schema().apply_refinement({"parent.id": "parent_key"})
    f = c3._filter_schema(s, ["parent", "child"], ["parent.nm", "child.cid"])
    assert ("child.parent_id", "parent.parent_key") in f.foreign_keys


def _db(tmp_path):
    p = tmp_path / "t.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE t (x INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (5,)])
    con.commit()
    con.close()
    return str(p)


def test_vote_backmaps_renamed_candidates(tmp_path):
    """Audit toy case: two of three samples agree once backmapped."""
    db = _db(tmp_path)
    c3 = C3Text2SQL({"base_url": "x", "api_key": "x", "model_name": "x"}, num_samples=3)
    cands = ["SELECT new_x FROM t WHERE new_x < 3",
             "SELECT new_x FROM t",
             "SELECT new_x FROM t WHERE new_x > 0"]
    # Without the mapping every candidate fails on physical t(x) -> first wins.
    assert c3._self_consistency(cands, db) == cands[0]
    # With the mapping the 2-sample majority (all rows) wins; display names kept.
    assert c3._self_consistency(cands, db, {"new_x": "x"}) == cands[1]


def test_vote_unchanged_without_mapping(tmp_path):
    db = _db(tmp_path)
    c3 = C3Text2SQL({"base_url": "x", "api_key": "x", "model_name": "x"}, num_samples=3)
    cands = ["SELECT x FROM t WHERE x < 3", "SELECT x FROM t", "SELECT x FROM t WHERE x > 0"]
    assert c3._self_consistency(cands, db) == cands[1]
    assert c3._self_consistency(cands, db, {}) == cands[1]
