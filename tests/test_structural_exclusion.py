"""Tests for refactored structural_exclusion."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from egrefine.data.schema import Column, Table, Schema
from egrefine.phase1.llm_screener import structural_exclusion


def _make_schema():
    """Schema with PK, FK, shared-name-same-type, shared-name-diff-type columns."""
    tables = [
        Table(name="stadium", columns=[
            Column(name="stadium_id", table="stadium", dtype="INTEGER", is_pk=True),
            Column(name="name", table="stadium", dtype="TEXT"),
            Column(name="capacity", table="stadium", dtype="INTEGER"),
        ]),
        Table(name="concert", columns=[
            Column(name="concert_id", table="concert", dtype="INTEGER", is_pk=True),
            Column(name="stadium_id", table="concert", dtype="INTEGER",
                   fk_target="stadium.stadium_id"),
            Column(name="name", table="concert", dtype="TEXT"),
            Column(name="year", table="concert", dtype="INTEGER"),
        ]),
        Table(name="singer", columns=[
            Column(name="singer_id", table="singer", dtype="INTEGER", is_pk=True),
            Column(name="name", table="singer", dtype="TEXT"),
            Column(name="age", table="singer", dtype="INTEGER"),
        ]),
    ]
    fks = [("concert.stadium_id", "stadium.stadium_id")]
    return Schema(db_id="test", tables=tables, foreign_keys=fks)


def test_pk_not_excluded():
    """PK columns should NOT be excluded."""
    schema = _make_schema()
    skip_set, pks, fks, shared = structural_exclusion(schema)
    assert "stadium.stadium_id" not in skip_set
    assert "concert.concert_id" not in skip_set
    assert "singer.singer_id" not in skip_set
    assert pks == []


def test_fk_still_excluded():
    """FK columns should still be excluded."""
    schema = _make_schema()
    skip_set, pks, fks, shared = structural_exclusion(schema)
    assert "concert.stadium_id" in skip_set
    assert "concert.stadium_id" in fks


def test_pk_that_is_fk_target_not_excluded():
    """PK column that is a FK target should NOT be excluded."""
    schema = _make_schema()
    skip_set, _, _, _ = structural_exclusion(schema)
    # stadium.stadium_id is PK and also FK target — should NOT be excluded
    assert "stadium.stadium_id" not in skip_set


def test_shared_same_type_excluded():
    """Same name + same type across tables → excluded."""
    schema = _make_schema()
    skip_set, pks, fks, shared = structural_exclusion(schema)
    # "name" TEXT appears in stadium, concert, singer → same type → skip
    assert "stadium.name" in skip_set
    assert "concert.name" in skip_set
    assert "singer.name" in skip_set


def test_shared_diff_type_not_excluded():
    """Same name + different type → NOT excluded."""
    tables = [
        Table(name="player", columns=[
            Column(name="id", table="player", dtype="INTEGER", is_pk=True),
            Column(name="status", table="player", dtype="TEXT"),
        ]),
        Table(name="game", columns=[
            Column(name="id", table="game", dtype="INTEGER", is_pk=True),
            Column(name="status", table="game", dtype="INTEGER"),
        ]),
    ]
    schema = Schema(db_id="test", tables=tables, foreign_keys=[])
    skip_set, _, _, _ = structural_exclusion(schema)
    assert "player.status" not in skip_set
    assert "game.status" not in skip_set


def test_single_occurrence_not_excluded():
    """Column name appearing in only one table → not shared → not excluded."""
    schema = _make_schema()
    skip_set, _, _, _ = structural_exclusion(schema)
    assert "stadium.capacity" not in skip_set
    assert "concert.year" not in skip_set
    assert "singer.age" not in skip_set
