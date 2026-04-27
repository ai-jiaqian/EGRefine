"""Tests for PK→FK name propagation in Phase 4 VIEW synthesis."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from egrefine.data.schema import Column, Table, Schema
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase4.view_synthesis import generate_views, generate_refined_tables_json


def _make_schema():
    tables = [
        Table(name="stadium", columns=[
            Column(name="stdm_id", table="stadium", dtype="INTEGER", is_pk=True),
            Column(name="name", table="stadium", dtype="TEXT"),
            Column(name="cap", table="stadium", dtype="INTEGER"),
        ]),
        Table(name="concert", columns=[
            Column(name="concert_id", table="concert", dtype="INTEGER", is_pk=True),
            Column(name="stdm_id", table="concert", dtype="INTEGER",
                   fk_target="stadium.stdm_id"),
            Column(name="theme", table="concert", dtype="TEXT"),
        ]),
    ]
    fks = [("concert.stdm_id", "stadium.stdm_id")]
    return Schema(db_id="test", tables=tables, foreign_keys=fks)


def _make_results_pk_renamed():
    """PK stdm_id renamed to stadium_id, cap renamed to capacity."""
    schema = _make_schema()
    pk_col = schema.get_column("stadium", "stdm_id")
    cap_col = schema.get_column("stadium", "cap")
    return [
        SelectionResult(
            column=pk_col,
            selected_name="stadium_id",
            delta=0.10,
            was_changed=True,
            all_scores={"stdm_id": 0.5, "stadium_id": 0.6},
            verification_method="execution",
        ),
        SelectionResult(
            column=cap_col,
            selected_name="capacity",
            delta=0.08,
            was_changed=True,
            all_scores={"cap": 0.5, "capacity": 0.58},
            verification_method="execution",
        ),
    ]


def test_pk_rename_propagates_to_fk_in_views():
    """When PK stdm_id→stadium_id, FK concert.stdm_id should also become stadium_id in VIEW."""
    schema = _make_schema()
    results = _make_results_pk_renamed()
    views = generate_views(schema, results)

    view_texts = "\n".join(views)
    # concert table should have a VIEW with FK aliased
    assert '"stdm_id" AS "stadium_id"' in view_texts
    # Should appear in concert's VIEW (not just stadium's)
    concert_views = [v for v in views if '"concert"' in v and 'RENAME' in v]
    assert len(concert_views) == 1
    assert '"stdm_id" AS "stadium_id"' in concert_views[0]


def test_pk_rename_propagates_in_refined_tables_json():
    """refined_tables.json should show FK column with propagated name."""
    schema = _make_schema()
    results = _make_results_pk_renamed()
    refined = generate_refined_tables_json(schema, results)

    concert = [t for t in refined["tables"] if t["name"] == "concert"][0]
    fk_col = [c for c in concert["columns"] if c["original_name"] == "stdm_id"][0]
    assert fk_col["name"] == "stadium_id"


def test_no_propagation_when_pk_unchanged():
    """If PK is not renamed, FK should keep original name."""
    schema = _make_schema()
    cap_col = schema.get_column("stadium", "cap")
    results = [
        SelectionResult(
            column=cap_col,
            selected_name="capacity",
            delta=0.08,
            was_changed=True,
            all_scores={"cap": 0.5, "capacity": 0.58},
            verification_method="execution",
        ),
    ]
    refined = generate_refined_tables_json(schema, results)
    concert = [t for t in refined["tables"] if t["name"] == "concert"][0]
    fk_col = [c for c in concert["columns"] if c["original_name"] == "stdm_id"][0]
    assert fk_col["name"] == "stdm_id"


def test_fk_references_updated_in_refined_tables_json():
    """foreign_keys in refined_tables.json should use new PK name."""
    schema = _make_schema()
    results = _make_results_pk_renamed()
    refined = generate_refined_tables_json(schema, results)

    # FK reference should be updated to use new name
    fk_pairs = refined["foreign_keys"]
    # Original: ("concert.stdm_id", "stadium.stdm_id")
    # After: ("concert.stadium_id", "stadium.stadium_id")
    found = False
    for fk1, fk2 in fk_pairs:
        if "stadium_id" in fk1 and "stadium_id" in fk2:
            found = True
    assert found, f"FK references not updated: {fk_pairs}"
