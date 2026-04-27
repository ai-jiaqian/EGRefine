"""BEAVER benchmark integration tests (requires local MySQL)."""
import json
import os
import pytest

from egrefine.data.schema import Column, Table, Schema, NLSQLPair


# Skip all tests if MySQL is not available
def _mysql_available():
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host="localhost", user="root", password="", port=3306,
            database="csail_stata_nova",
        )
        conn.close()
        return True
    except Exception:
        return False


requires_mysql = pytest.mark.skipif(
    not _mysql_available(),
    reason="MySQL not available or BEAVER databases not imported",
)

BEAVER_PATH = os.environ.get("EGREFINE_TEST_BEAVER_PATH", "/path/to/BEAVER")
MYSQL_CONFIG = {"host": "localhost", "user": "root", "password": "", "port": 3306}


class TestDBConnection:
    """Test db_connection module with both SQLite and MySQL."""

    def test_is_mysql(self):
        from egrefine.data.db_connection import is_mysql
        assert is_mysql("mysql://csail_stata_nova") is True
        assert is_mysql("/path/to/db.sqlite") is False

    @requires_mysql
    def test_mysql_execute_sql(self):
        from egrefine.data.db_connection import execute_sql
        result = execute_sql(
            "SELECT COUNT(*) FROM instances",
            "mysql://csail_stata_nova",
            mysql_config=MYSQL_CONFIG,
        )
        assert result is not None
        assert len(result) == 1
        count = list(result)[0][0]
        assert count > 0

    @requires_mysql
    def test_mysql_execute_sql_error(self):
        from egrefine.data.db_connection import execute_sql
        result = execute_sql(
            "SELECT * FROM nonexistent_table_xyz",
            "mysql://csail_stata_nova",
            mysql_config=MYSQL_CONFIG,
        )
        assert result is None

    @requires_mysql
    def test_mysql_compare_results(self):
        from egrefine.data.db_connection import compare_results
        match = compare_results(
            "SELECT COUNT(*) FROM instances",
            "SELECT COUNT(*) FROM instances",
            "mysql://csail_stata_nova",
            mysql_config=MYSQL_CONFIG,
        )
        assert match is True

    @requires_mysql
    def test_mysql_sample_column(self):
        from egrefine.data.db_connection import sample_column
        values = sample_column(
            "mysql://csail_stata_nova",
            "instances",
            "hostname",
            n=5,
            mysql_config=MYSQL_CONFIG,
        )
        assert len(values) > 0
        assert all(isinstance(v, str) for v in values)

    @requires_mysql
    def test_mysql_sample_column_invalid(self):
        from egrefine.data.db_connection import sample_column
        values = sample_column(
            "mysql://csail_stata_nova",
            "instances",
            "nonexistent_column_xyz",
            n=5,
            mysql_config=MYSQL_CONFIG,
        )
        assert values == []

    def test_sqlite_still_works(self, tmp_path):
        """Verify SQLite path still works through db_connection."""
        import sqlite3
        db_path = str(tmp_path / "test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'alice')")
        conn.execute("INSERT INTO t VALUES (2, 'bob')")
        conn.commit()
        conn.close()

        from egrefine.data.db_connection import execute_sql, sample_column
        result = execute_sql("SELECT name FROM t", db_path)
        assert result == {("alice",), ("bob",)}

        values = sample_column(db_path, "t", "name", n=10)
        assert set(values) == {"alice", "bob"}


class TestLoadSchemaFromMySQL:
    """Test MySQL schema loading."""

    @requires_mysql
    def test_load_csail_stata_nova(self):
        from egrefine.data.db_connection import load_schema_from_mysql
        schema = load_schema_from_mysql(
            "csail_stata_nova", "csail_stata_nova", MYSQL_CONFIG,
        )
        assert schema.db_id == "csail_stata_nova"
        assert len(schema.tables) > 0

        # Check instances table exists
        instances = schema.get_table("instances")
        assert instances is not None
        col_names = [c.name for c in instances.columns]
        assert "hostname" in col_names

    @requires_mysql
    def test_load_all_beaver_dbs(self):
        from egrefine.data.db_connection import load_schema_from_mysql
        beaver_dbs = [
            "csail_stata_cinder", "csail_stata_glance",
            "csail_stata_neutron", "csail_stata_nova", "keystone",
        ]
        for db_id in beaver_dbs:
            schema = load_schema_from_mysql(db_id, db_id, MYSQL_CONFIG)
            assert schema.db_id == db_id
            assert len(schema.tables) > 0


class TestBEAVERLoader:
    """Test BEAVERLoader class."""

    @requires_mysql
    def test_load_nw(self):
        from egrefine.data.benchmark import BEAVERLoader
        loader = BEAVERLoader(BEAVER_PATH, split="nw", mysql_config=MYSQL_CONFIG)
        assert len(loader.pairs) == 88
        assert len(loader.db_ids) == 5
        assert "csail_stata_nova" in loader.db_ids

    @requires_mysql
    def test_get_db_path_returns_mysql_uri(self):
        from egrefine.data.benchmark import BEAVERLoader
        loader = BEAVERLoader(BEAVER_PATH, split="nw", mysql_config=MYSQL_CONFIG)
        path = loader.get_db_path("csail_stata_nova")
        assert path == "mysql://csail_stata_nova"

    @requires_mysql
    def test_get_pairs_for_db(self):
        from egrefine.data.benchmark import BEAVERLoader
        loader = BEAVERLoader(BEAVER_PATH, split="nw", mysql_config=MYSQL_CONFIG)
        nova_pairs = loader.get_pairs_for_db("csail_stata_nova")
        assert len(nova_pairs) > 0
        assert all(p.db_id == "csail_stata_nova" for p in nova_pairs)

    @requires_mysql
    def test_pairs_have_gold_sql(self):
        from egrefine.data.benchmark import BEAVERLoader
        loader = BEAVERLoader(BEAVER_PATH, split="nw", mysql_config=MYSQL_CONFIG)
        for p in loader.pairs[:5]:
            assert p.gold_sql.strip()
            assert p.nl.strip()

    @requires_mysql
    def test_gold_sql_executes(self):
        """Verify at least one gold SQL actually runs against MySQL."""
        from egrefine.data.benchmark import BEAVERLoader
        from egrefine.data.db_connection import execute_sql
        loader = BEAVERLoader(BEAVER_PATH, split="nw", mysql_config=MYSQL_CONFIG)
        # Try first pair from keystone (simpler SQL)
        ks_pairs = loader.get_pairs_for_db("keystone")
        if ks_pairs:
            result = execute_sql(
                ks_pairs[0].gold_sql,
                "mysql://keystone",
                mysql_config=MYSQL_CONFIG,
            )
            # Should execute (may return empty set but not None)
            assert result is not None


class TestBEAVERConfig:
    """Test BEAVER config parsing."""

    def test_beaver_config_structure(self):
        config = {
            "data": {
                "beaver": {
                    "path": "/path/to/beaver",
                    "split": "nw",
                    "mysql": {
                        "host": "localhost",
                        "user": "root",
                        "password": "",
                        "port": 3306,
                    },
                }
            }
        }
        beaver_cfg = config["data"]["beaver"]
        assert beaver_cfg["split"] == "nw"
        assert beaver_cfg["mysql"]["host"] == "localhost"

    def test_beaver_config_optional(self):
        config = {"data": {"bird": {"path": "/path"}}}
        beaver_cfg = config["data"].get("beaver")
        assert beaver_cfg is None
