"""Regression tests for method changes introduced by the PVLDB revision."""

import json
import sqlite3

import pytest

from egrefine.data.db_connection import execute_sql
from egrefine.data.schema import Column, Schema, Table
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase4 import view_synthesis as vs
from egrefine.phase4.view_synthesis import _build_rename_map
from scripts.run_refine import _load_forced_targets, _resolve_forced_columns


def _schema():
    countries = Table("countries", [
        Column("cntry_id", "countries", "INTEGER", is_pk=True),
        Column("name", "countries", "TEXT"),
    ])
    car_makers = Table("car_makers", [
        Column("id", "car_makers", "INTEGER", is_pk=True),
        Column("cntry", "car_makers", "INTEGER",
               fk_target="countries.cntry_id"),
    ])
    airports = Table("airports", [
        Column("apt_code", "airports", "TEXT", is_pk=True),
    ])
    flights = Table("flights", [
        Column("flno", "flights", "INTEGER", is_pk=True),
        Column("srcapt", "flights", "TEXT",
               fk_target="airports.apt_code"),
        Column("destapt", "flights", "TEXT",
               fk_target="airports.apt_code"),
    ])
    district = Table("district", [
        Column("d_id", "district", "INTEGER", is_pk=True),
    ])
    account = Table("account", [
        Column("acc_id", "account", "INTEGER", is_pk=True),
        Column("d_id", "account", "INTEGER", fk_target="district.d_id"),
    ])
    return Schema(
        db_id="test",
        tables=[countries, car_makers, airports, flights, district, account],
        foreign_keys=[
            ("car_makers.cntry", "countries.cntry_id"),
            ("flights.srcapt", "airports.apt_code"),
            ("flights.destapt", "airports.apt_code"),
            ("account.d_id", "district.d_id"),
        ],
    )


def _rename(schema, table, column, new_name):
    target = next(
        item for current_table in schema.tables
        for item in current_table.columns
        if current_table.name == table and item.name == column
    )
    return SelectionResult(
        column=target,
        selected_name=new_name,
        delta=0.1,
        was_changed=True,
        all_scores={column: 0.7, new_name: 0.8},
    )


def test_same_name_fk_propagates_but_role_named_fk_is_preserved():
    schema = _schema()
    mapping = _build_rename_map([
        _rename(schema, "district", "d_id", "district_id"),
        _rename(schema, "countries", "cntry_id", "country_code"),
    ], schema)
    assert mapping["account.d_id"] == "district_id"
    assert "car_makers.cntry" not in mapping


def test_propagation_never_creates_duplicate_names():
    schema = _schema()
    vs.PROPAGATION_MODE = "all"
    try:
        mapping = _build_rename_map([
            _rename(schema, "airports", "apt_code", "iata_code")
        ], schema)
        finals = [
            mapping.get(f"flights.{name}", name)
            for name in ("flno", "srcapt", "destapt")
        ]
        assert len(finals) == len({name.lower() for name in finals})
    finally:
        vs.PROPAGATION_MODE = "same_name"


def test_duplicate_direct_rename_reverts_later_column():
    schema = _schema()
    mapping = _build_rename_map([
        _rename(schema, "flights", "srcapt", "airport"),
        _rename(schema, "flights", "destapt", "airport"),
    ], schema)
    assert mapping["flights.srcapt"] == "airport"
    assert "flights.destapt" not in mapping


def test_duplicate_direct_rename_yields_to_unchanged_original_column():
    schema = Schema(
        db_id="collision",
        tables=[Table("items", [
            Column("short", "items", "TEXT"),
            Column("description", "items", "TEXT"),
        ])],
        foreign_keys=[],
    )
    mapping = _build_rename_map([
        _rename(schema, "items", "short", "description"),
    ], schema)
    assert "items.short" not in mapping


def test_duplicate_resolution_reaches_fixed_point():
    schema = Schema(
        db_id="collision_chain",
        tables=[Table("items", [
            Column("a", "items", "TEXT"),
            Column("b", "items", "TEXT"),
            Column("c", "items", "TEXT"),
        ])],
        foreign_keys=[],
    )
    mapping = _build_rename_map([
        _rename(schema, "items", "a", "c"),
        _rename(schema, "items", "b", "a"),
    ], schema)
    finals = [mapping.get(f"items.{name}", name) for name in ("a", "b", "c")]
    assert len(finals) == len({name.lower() for name in finals})


def test_forced_targets_are_ordered_and_structurally_checked(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"targets": [
        {"db_id": "test", "full_name": "car_makers.cntry"},
        {"db_id": "test", "full_name": "countries.name"},
    ]}))
    targets = _load_forced_targets(str(path))
    assert targets["test"] == ["car_makers.cntry", "countries.name"]

    schema = _schema()
    with pytest.raises(ValueError, match="structural exclusion"):
        _resolve_forced_columns(schema, targets["test"])


def test_sqlite_execution_is_read_only(tmp_path):
    database = tmp_path / "test.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE source (value INT)")
    connection.execute("INSERT INTO source VALUES (1)")
    connection.commit()
    connection.close()

    assert execute_sql("SELECT value FROM source", str(database)) == {(1,)}
    assert execute_sql("CREATE TABLE ghost (id INT)", str(database)) is None

    connection = sqlite3.connect(database)
    tables = {
        name for (name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    connection.close()
    assert tables == {"source"}
