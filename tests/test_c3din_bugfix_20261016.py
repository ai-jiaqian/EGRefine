"""Regression tests for the 2026-10-16 C3 / DIN-SQL audit bugs (B1-B3).

See experiment_records/2026-10-16-c3-din-underperformance-audit.md.
"""
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

from egrefine.data.schema import Column, Schema, Table
from egrefine.models.llm_client import LLMClient
from egrefine.phase3.c3_runner import C3Text2SQL
from egrefine.phase3.dinsql_runner import DINSQLText2SQL


def _schema():
    return Schema(
        db_id="voter",
        tables=[
            Table(name="CONTESTANTS", columns=[
                Column(name="contestant_no", table="CONTESTANTS", dtype="INTEGER", is_pk=True),
                Column(name="name", table="CONTESTANTS", dtype="TEXT"),
            ]),
            Table(name="VOTES", columns=[
                Column(name="vote_id", table="VOTES", dtype="INTEGER", is_pk=True),
                Column(name="ph", table="VOTES", dtype="TEXT"),
                Column(name="contestant_no", table="VOTES", dtype="INTEGER"),
                Column(name="state", table="VOTES", dtype="TEXT"),
            ]),
            Table(name="AREA", columns=[
                Column(name="area_code", table="AREA", dtype="INTEGER", is_pk=True),
                Column(name="state", table="AREA", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[("VOTES.contestant_no", "CONTESTANTS.contestant_no")],
    )


# ---------------------------------------------------------------- B1 ----
def test_b1_filter_keeps_fk_columns_between_retained_tables():
    c3 = C3Text2SQL.__new__(C3Text2SQL)
    # Linker picked the tables but not the join column VOTES.contestant_no.
    out = c3._filter_schema(_schema(), ["CONTESTANTS", "VOTES"],
                            ["CONTESTANTS.name", "VOTES.ph", "VOTES.state"])
    votes = next(t for t in out.tables if t.name == "VOTES")
    assert "contestant_no" in {c.name for c in votes.columns}
    assert ("VOTES.contestant_no", "CONTESTANTS.contestant_no") in out.foreign_keys
    c3.column_descriptions = {}
    assert "FOREIGN KEY (contestant_no) REFERENCES CONTESTANTS(contestant_no)" in c3._format_ddl_schema(out)


def test_b1_filter_does_not_add_tables_or_unrelated_columns():
    c3 = C3Text2SQL.__new__(C3Text2SQL)
    out = c3._filter_schema(_schema(), ["VOTES"], ["VOTES.ph"])
    assert [t.name for t in out.tables] == ["VOTES"]
    # FK target table not retained -> FK column not forced in.
    assert {c.name for c in out.tables[0].columns} == {"vote_id", "ph"}
    assert out.foreign_keys == []


# ---------------------------------------------------------------- B2 ----
def test_b2_multiline_sql_after_marker_is_kept_whole():
    text = ("Let's think step by step. Join posts with users.\n\nSQL:\n"
            "SELECT T1.Title\nFROM posts AS T1\n"
            "INNER JOIN users AS T2 ON T1.OwnerUserId = T2.Id\n"
            "WHERE T2.DisplayName = 'csgillespie' ")
    sql = DINSQLText2SQL._extract_sql_from_cot(text)
    assert sql.startswith("SELECT T1.Title") and "FROM posts AS T1" in sql
    assert sql.endswith("'csgillespie'")


def test_b2_stops_at_blank_line_and_semicolon():
    text = "SQL: SELECT a\nFROM t\nWHERE x > 1;\nThis returns a.\n"
    assert DINSQLText2SQL._extract_sql_from_cot(text) == "SELECT a\nFROM t\nWHERE x > 1"
    text2 = "SQL:\nSELECT a\nFROM t\n\nExplanation: the query ..."
    assert DINSQLText2SQL._extract_sql_from_cot(text2) == "SELECT a\nFROM t"


def test_b2_last_marker_wins_for_nested_cot():
    text = ("Sub-question: avg?\nSQL: SELECT AVG(sal) FROM e\nMain query:\n"
            "SQL: SELECT nm\nFROM e\nWHERE sal > (SELECT AVG(sal) FROM e)")
    assert DINSQLText2SQL._extract_sql_from_cot(text) == (
        "SELECT nm\nFROM e\nWHERE sal > (SELECT AVG(sal) FROM e)")


# ---------------------------------------------------------------- B3 ----
def test_b3_llmclient_per_call_temperature_does_not_mutate_shared_state():
    client = LLMClient.__new__(LLMClient)
    client.model_name, client.temperature, client.max_tokens = "m", 0.0, 16
    client.max_retries, client.extra_body, client.raw_log = 1, None, None
    client.log_context = threading.local()
    seen = []
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                           model_dump=lambda mode=None: {})
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: (seen.append(kw["temperature"]), resp)[1])))
    client.chat([{"role": "user", "content": "x"}], temperature=0.7)
    client.chat([{"role": "user", "content": "x"}])
    assert seen == [0.7, 0.0] and client.temperature == 0.0


def test_b3_c3_concurrent_sampling_never_leaks_into_linking():
    """Thread A samples at 0.7 while thread B does schema linking (T=0)."""
    in_sample, linking_done = threading.Event(), threading.Event()
    temps = {}

    class FakeLLM:
        temperature = 0.0
        log_context = threading.local()

        def chat(self, messages, temperature=None, trace=None):
            t = self.temperature if temperature is None else temperature
            content = messages[0]["content"]
            if "select only the relevant" in content:
                assert in_sample.wait(2)
                temps["linking"] = t
                linking_done.set()
                return '{"tables": ["VOTES"], "columns": ["VOTES.ph"]}'
            if t == 0.7:
                in_sample.set()
                assert linking_done.wait(2)
            return "SELECT 1"

    c3 = C3Text2SQL.__new__(C3Text2SQL)
    c3.llm, c3.num_samples, c3.sample_rows = FakeLLM(), 1, 0
    c3.desc_dir, c3.column_descriptions = None, {}
    schema = _schema()
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(c3._generate_sql, "q", schema, None, None, 0.7)
        b = pool.submit(c3._schema_linking, "q", schema)
        a.result(), b.result()
    assert temps["linking"] == 0.0
    assert c3.llm.temperature == 0.0
