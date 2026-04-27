"""Tests for src/eval/db_setup.py — Stage 2 database setup utilities."""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

import pytest

from egrefine.eval.db_setup import apply_views, copy_database, remap_gold_sql

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_sample_db(path: Path) -> None:
    """Create a tiny SQLite database with one table and a few rows."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE district (d_id INTEGER PRIMARY KEY, A2 TEXT, A3 TEXT)"
    )
    conn.execute("INSERT INTO district VALUES (1, 'Prague', 'central Bohemia')")
    conn.execute("INSERT INTO district VALUES (2, 'Brno', 'south Moravia')")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# copy_database tests
# ---------------------------------------------------------------------------

class TestCopyDatabase:
    def test_creates_copy_at_dest(self, tmp_path: Path) -> None:
        src = tmp_path / "original.sqlite"
        dest = tmp_path / "subdir" / "copy.sqlite"
        _create_sample_db(src)

        result = copy_database(src, dest)

        assert result == dest
        assert dest.exists()

    def test_copy_is_independent(self, tmp_path: Path) -> None:
        src = tmp_path / "original.sqlite"
        dest = tmp_path / "copy.sqlite"
        _create_sample_db(src)
        copy_database(src, dest)

        # Mutate the copy.
        conn = sqlite3.connect(str(dest))
        conn.execute("DELETE FROM district WHERE d_id = 1")
        conn.commit()
        conn.close()

        # Original must still have both rows.
        conn = sqlite3.connect(str(src))
        rows = conn.execute("SELECT COUNT(*) FROM district").fetchone()[0]
        conn.close()
        assert rows == 2

    def test_copy_has_intact_data(self, tmp_path: Path) -> None:
        src = tmp_path / "original.sqlite"
        dest = tmp_path / "copy.sqlite"
        _create_sample_db(src)
        copy_database(src, dest)

        conn = sqlite3.connect(str(dest))
        rows = conn.execute("SELECT * FROM district ORDER BY d_id").fetchall()
        conn.close()

        assert rows == [(1, "Prague", "central Bohemia"), (2, "Brno", "south Moravia")]


# ---------------------------------------------------------------------------
# apply_views tests
# ---------------------------------------------------------------------------

class TestApplyViews:
    @staticmethod
    def _views_sql() -> str:
        return textwrap.dedent("""\
            BEGIN;
            ALTER TABLE "district" RENAME TO "_orig_district";
            CREATE VIEW "district" AS
            SELECT
              "d_id",
              "A2" AS "city_name"
            FROM "_orig_district";
            COMMIT;
        """)

    def test_views_created(self, tmp_path: Path) -> None:
        db = tmp_path / "test.sqlite"
        _create_sample_db(db)
        sql_file = tmp_path / "views.sql"
        sql_file.write_text(self._views_sql(), encoding="utf-8")

        apply_views(db, sql_file)

        conn = sqlite3.connect(str(db))
        # Query via the VIEW using the refined column name.
        rows = conn.execute("SELECT city_name FROM district ORDER BY d_id").fetchall()
        conn.close()
        assert rows == [("Prague",), ("Brno",)]

    def test_original_data_via_backing_table(self, tmp_path: Path) -> None:
        db = tmp_path / "test.sqlite"
        _create_sample_db(db)
        sql_file = tmp_path / "views.sql"
        sql_file.write_text(self._views_sql(), encoding="utf-8")

        apply_views(db, sql_file)

        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT A2 FROM _orig_district ORDER BY d_id"
        ).fetchall()
        conn.close()
        assert rows == [("Prague",), ("Brno",)]

    def test_pred_and_gold_return_same_results(self, tmp_path: Path) -> None:
        """Pred SQL hits the VIEW; gold SQL (remapped) hits the backing table.

        Both should return the same result set.
        """
        db = tmp_path / "test.sqlite"
        _create_sample_db(db)
        sql_file = tmp_path / "views.sql"
        sql_file.write_text(self._views_sql(), encoding="utf-8")
        apply_views(db, sql_file)

        conn = sqlite3.connect(str(db))
        # Pred SQL uses refined column names on the VIEW.
        pred_rows = set(
            conn.execute("SELECT city_name FROM district ORDER BY d_id").fetchall()
        )
        # Gold SQL uses original column names on the backing table.
        gold_rows = set(
            conn.execute(
                "SELECT A2 FROM _orig_district ORDER BY d_id"
            ).fetchall()
        )
        conn.close()
        assert pred_rows == gold_rows


# ---------------------------------------------------------------------------
# remap_gold_sql tests
# ---------------------------------------------------------------------------

class TestRemapGoldSql:
    def test_simple_single_table(self) -> None:
        sql = "SELECT d_id FROM district WHERE A2 = 'Prague'"
        result = remap_gold_sql(sql, {"district": "_orig_district"})
        assert result == "SELECT d_id FROM _orig_district WHERE A2 = 'Prague'"

    def test_multiple_tables(self) -> None:
        sql = "SELECT * FROM district JOIN account ON district.d_id = account.d_id"
        table_map = {
            "district": "_orig_district",
            "account": "_orig_account",
        }
        result = remap_gold_sql(sql, table_map)
        assert "_orig_district" in result
        assert "_orig_account" in result
        # Original names should be gone (as standalone words).
        assert "FROM _orig_district" in result
        assert "JOIN _orig_account" in result

    def test_no_remap_when_table_not_in_map(self) -> None:
        sql = "SELECT * FROM district"
        result = remap_gold_sql(sql, {"account": "_orig_account"})
        assert result == sql

    def test_substring_safety(self) -> None:
        """``district`` must NOT match inside ``district_info``."""
        sql = "SELECT * FROM district_info WHERE district_info.id = 1"
        result = remap_gold_sql(sql, {"district": "_orig_district"})
        # district_info must be untouched.
        assert "district_info" in result
        assert "_orig_district_info" not in result
        assert "_orig_district" not in result

    def test_empty_map_returns_unchanged(self) -> None:
        sql = "SELECT 1"
        assert remap_gold_sql(sql, {}) == sql

    def test_longer_name_replaced_first(self) -> None:
        """If the map has both ``district`` and ``district_info``, the longer
        key must be replaced first so the shorter one does not corrupt it."""
        sql = "SELECT * FROM district_info JOIN district ON 1=1"
        table_map = {
            "district": "_orig_district",
            "district_info": "_orig_district_info",
        }
        result = remap_gold_sql(sql, table_map)
        assert "FROM _orig_district_info" in result
        assert "JOIN _orig_district " in result
