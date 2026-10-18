import json

import pytest

from scripts.run_refine import _load_forced_targets, _resolve_forced_columns
from egrefine.data.schema import Column, Schema, Table


def test_load_forced_targets_preserves_order(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"targets": [
        {"db_id": "db_b", "full_name": "t.c2"},
        {"db_id": "db_a", "full_name": "t.c1"},
        {"db_id": "db_b", "full_name": "t.c3"},
    ]}))

    result = _load_forced_targets(str(path))
    assert list(result) == ["db_b", "db_a"]
    assert result["db_b"] == ["t.c2", "t.c3"]


def test_load_forced_targets_rejects_duplicates(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"targets": [
        {"db_id": "db", "full_name": "t.c"},
        {"db_id": "db", "full_name": "t.c"},
    ]}))

    with pytest.raises(ValueError, match="Duplicate forced target"):
        _load_forced_targets(str(path))


def test_resolve_forced_columns_rejects_missing_and_structural_keys():
    pk = Column(name="id", table="t", dtype="INT", is_pk=True)
    value = Column(name="value", table="t", dtype="TEXT")
    schema = Schema(db_id="db", tables=[Table(name="t", columns=[pk, value])],
                    foreign_keys=[])

    assert _resolve_forced_columns(schema, ["t.value"]) == [value]
    with pytest.raises(ValueError, match="missing from schema"):
        _resolve_forced_columns(schema, ["t.nope"])
    with pytest.raises(ValueError, match="structural exclusion"):
        _resolve_forced_columns(schema, ["t.id"])


def test_frozen_main_target_replay_allows_pk_only_when_requested():
    pk = Column(name="id", table="t", dtype="INT", is_pk=True)
    fk = Column(name="t_id", table="r", dtype="INT", fk_target="t.id")
    schema = Schema("db", [Table("t", [pk]), Table("r", [fk])], [("r.t_id", "t.id")])
    with pytest.raises(ValueError, match="structural exclusion"):
        _resolve_forced_columns(schema, ["t.id"])
    assert _resolve_forced_columns(schema, ["t.id"], allow_primary_keys=True) == [pk]
    with pytest.raises(ValueError, match="structural exclusion"):
        _resolve_forced_columns(schema, ["r.t_id"], allow_primary_keys=True)
