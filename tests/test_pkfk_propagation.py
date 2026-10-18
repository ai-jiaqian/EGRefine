"""Tests for name-preserving PK->FK propagation (VLDB revision Item 8 / R5-3).

Covers: same-name propagation, role-named preservation, the same-table
collision bug (two FKs referencing one renamed PK), the duplicate-name hard
invariant, and the legacy "all" mode used for paired A/B reproduction.
"""
import pytest

from egrefine.data.schema import Column, Schema, Table
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase4 import view_synthesis as vs
from egrefine.phase4.view_synthesis import _build_rename_map


def _schema():
    """countries(cntry_id PK) <- car_makers.cntry (role-named FK)
       airports(apt_code PK)  <- flights.srcapt, flights.destapt (role-named)
       district(d_id PK)      <- account.d_id (same-name FK)"""
    countries = Table("countries", [
        Column("cntry_id", "countries", "INTEGER", is_pk=True),
        Column("name", "countries", "TEXT"),
    ])
    car_makers = Table("car_makers", [
        Column("id", "car_makers", "INTEGER", is_pk=True),
        Column("cntry", "car_makers", "INTEGER", fk_target="countries.cntry_id"),
    ])
    airports = Table("airports", [
        Column("apt_code", "airports", "TEXT", is_pk=True),
    ])
    flights = Table("flights", [
        Column("flno", "flights", "INTEGER", is_pk=True),
        Column("srcapt", "flights", "TEXT", fk_target="airports.apt_code"),
        Column("destapt", "flights", "TEXT", fk_target="airports.apt_code"),
    ])
    district = Table("district", [
        Column("d_id", "district", "INTEGER", is_pk=True),
    ])
    account = Table("account", [
        Column("acc_id", "account", "INTEGER", is_pk=True),
        Column("d_id", "account", "INTEGER", fk_target="district.d_id"),
    ])
    return Schema(
        db_id="t",
        tables=[countries, car_makers, airports, flights, district, account],
        foreign_keys=[
            ("car_makers.cntry", "countries.cntry_id"),
            ("flights.srcapt", "airports.apt_code"),
            ("flights.destapt", "airports.apt_code"),
            ("account.d_id", "district.d_id"),
        ],
    )


def _pk_rename(schema, table, col, new):
    c = next(c for t in schema.tables for c in t.columns
             if t.name == table and c.name == col)
    return SelectionResult(column=c, selected_name=new, delta=0.1,
                           was_changed=True, all_scores={new: 0.8, col: 0.7})


def test_same_name_fk_propagates():
    s = _schema()
    m = _build_rename_map([_pk_rename(s, "district", "d_id", "district_id")], s)
    assert m["account.d_id"] == "district_id"


def test_role_named_fk_preserved():
    s = _schema()
    m = _build_rename_map([_pk_rename(s, "countries", "cntry_id", "country_code")], s)
    assert m["countries.cntry_id"] == "country_code"
    assert "car_makers.cntry" not in m  # role name kept


def test_two_fks_same_pk_no_duplicate_names():
    """The 9B flight_2 bug: srcapt+destapt must never both become iata_code."""
    s = _schema()
    vs.PROPAGATION_MODE = "all"  # bug only reachable in legacy mode now
    try:
        m = _build_rename_map([_pk_rename(s, "airports", "apt_code", "iata_code")], s)
        finals = [m.get(f"flights.{c}", c) for c in ("flno", "srcapt", "destapt")]
        assert len(finals) == len({f.lower() for f in finals}), finals
    finally:
        vs.PROPAGATION_MODE = "same_name"


def test_role_named_fks_untouched_in_same_name_mode():
    s = _schema()
    m = _build_rename_map([_pk_rename(s, "airports", "apt_code", "iata_code")], s)
    assert "flights.srcapt" not in m and "flights.destapt" not in m


def test_duplicate_direct_renames_revert_to_original():
    """Two direct renames colliding on one name: later one reverts (§3.4.3)."""
    s = _schema()
    r1 = _pk_rename(s, "flights", "srcapt", "airport")
    r2 = _pk_rename(s, "flights", "destapt", "airport")
    m = _build_rename_map([r1, r2], s)
    assert m["flights.srcapt"] == "airport"
    assert "flights.destapt" not in m  # reverted to original name
    finals = [m.get(f"flights.{c}", c) for c in ("flno", "srcapt", "destapt")]
    assert len(finals) == len({f.lower() for f in finals})


def test_mixed_scenario():
    """One PK rename drives both a same-name and a role-named FK."""
    s = _schema()
    results = [
        _pk_rename(s, "district", "d_id", "district_id"),
        _pk_rename(s, "countries", "cntry_id", "country_code"),
    ]
    m = _build_rename_map(results, s)
    assert m["account.d_id"] == "district_id"      # propagated
    assert "car_makers.cntry" not in m             # preserved
    assert len(m) == 3                             # 2 direct + 1 propagated


def test_legacy_mode_reproduces_submitted_behaviour():
    s = _schema()
    vs.PROPAGATION_MODE = "all"
    try:
        m = _build_rename_map([_pk_rename(s, "countries", "cntry_id", "country_code")], s)
        assert m.get("car_makers.cntry") == "country_code"
    finally:
        vs.PROPAGATION_MODE = "same_name"


def test_propagation_never_overwrites_direct_rename():
    """A column's own verified rename takes precedence over propagation
    (legacy rule clobbered these, e.g. world_1_4 countrylanguage.cntry_code)."""
    s = _schema()
    pk = _pk_rename(s, "district", "d_id", "district_id")
    own = _pk_rename(s, "account", "d_id", "district_ref")  # FK's own choice
    m = _build_rename_map([pk, own], s)
    assert m["account.d_id"] == "district_ref"  # not clobbered by propagation


def test_execute_sql_is_readonly(tmp_path):
    """Model-generated DDL must never mutate the benchmark database
    (2026-07-16 audit: eval wrote 42 ghost tables into 26 source DBs)."""
    import sqlite3
    from egrefine.data.db_connection import execute_sql
    db = tmp_path / "t.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE a (x INT)")
    con.execute("INSERT INTO a VALUES (1)")
    con.commit(); con.close()
    assert execute_sql("SELECT x FROM a", str(db)) == {(1,)}
    assert execute_sql("CREATE TABLE ghost (id INT)", str(db)) is None
    assert execute_sql("DROP TABLE a", str(db)) is None
    con = sqlite3.connect(db)
    tables = {n for (n,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"a"}  # untouched
