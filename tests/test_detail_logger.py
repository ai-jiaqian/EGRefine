"""Test detail logger — file output verification."""
import json
import os
import pytest

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase1.pruner import PruneResult
from egrefine.phase2.prompts import CandidateName
from egrefine.phase3.scorer import SelectionResult, ScoringDetail, QueryDetail
from egrefine.evaluate import EvalDetails
from egrefine.detail_logger import DetailLogger


@pytest.fixture
def schema():
    return Schema(
        db_id="test_db",
        tables=[
            Table(name="employees", columns=[
                Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
                Column(name="nm", table="employees", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[],
    )


@pytest.fixture
def dl(tmp_path):
    return DetailLogger(str(tmp_path / "output"))


class TestDetailLogger:
    def test_save_config_snapshot(self, dl):
        dl.save_config_snapshot({"phase1": {"signals": {"short_name": {"enabled": True}}}})
        path = os.path.join(dl.output_dir, "config_snapshot.yaml")
        assert os.path.exists(path)

    def test_save_schema_info(self, dl, schema):
        dl.save_schema_info(schema)
        path = os.path.join(dl.output_dir, "per_db", "test_db", "schema_info.json")
        assert os.path.exists(path)
        data = json.load(open(path))
        assert data["db_id"] == "test_db"
        assert len(data["tables"]) == 1

    def test_save_phase1(self, dl, schema):
        prune_result = PruneResult(
            candidates=[schema.all_columns[1]],  # nm
            total_columns=2,
            signal_hits={"S1_short_name": ["employees.nm"], "S2_high_similarity": [],
                         "S3_naming_inconsistency": [], "S4_generic_vocabulary": []},
            skipped_pks=["employees.id"],
        )
        dl.save_phase1("test_db", prune_result, schema)
        path = os.path.join(dl.output_dir, "per_db", "test_db", "phase1_pruning.json")
        data = json.load(open(path))
        assert data["candidate_count"] == 1
        assert "employees.nm" in data["candidates"]
        # Check per-column detail
        nm_info = [c for c in data["all_columns"] if c["column"] == "nm"][0]
        assert "S1_short_name" in nm_info["signals"]
        assert nm_info["is_candidate"] is True

    def test_save_phase2(self, dl):
        candidates = {"employees.nm": [
            CandidateName(name="employee_name", reason="descriptive"),
        ]}
        dl.save_phase2("test_db", candidates)
        path = os.path.join(dl.output_dir, "per_db", "test_db", "phase2_candidates.json")
        data = json.load(open(path))
        assert len(data) == 1
        assert data[0]["candidates"][0]["name"] == "employee_name"

    def test_save_phase3(self, dl):
        col = Column(name="nm", table="employees", dtype="TEXT")
        results = [SelectionResult(
            column=col, selected_name="employee_name", delta=0.2,
            was_changed=True, all_scores={"nm": 0.5, "employee_name": 0.7},
        )]
        scoring_details = {
            "employees.nm": {
                "nm": ScoringDetail(
                    candidate_name="nm", exacc=0.5,
                    query_details=[QueryDetail(
                        nl="test?", gold_sql="SELECT nm FROM e",
                        pred_sql="SELECT nm FROM e", backmapped_sql=None, match=True,
                    )],
                ),
            },
        }
        dl.save_phase3("test_db", results, scoring_details)
        path = os.path.join(dl.output_dir, "per_db", "test_db", "phase3_scoring.json")
        data = json.load(open(path))
        assert data[0]["selected_name"] == "employee_name"
        assert "scoring" in data[0]
        assert "nm" in data[0]["scoring"]
        assert data[0]["scoring"]["nm"]["queries"][0]["match"] is True

    def test_save_phase4(self, dl):
        synthesis = {
            "views": ["CREATE VIEW e AS SELECT nm AS employee_name FROM _orig_e;"],
            "mapping": {"employees.nm": "employee_name"},
            "reverse_mapping": {"employee_name": "nm"},
            "orig_table_map": {"e": "_orig_e"},
            "refined_tables": {"db_id": "test_db", "tables": []},
        }
        dl.save_phase4("test_db", synthesis)
        d = os.path.join(dl.output_dir, "per_db", "test_db")
        assert os.path.exists(os.path.join(d, "phase4_views.sql"))
        assert os.path.exists(os.path.join(d, "phase4_mapping.json"))
        assert os.path.exists(os.path.join(d, "phase4_refined_tables.json"))

    def test_save_evaluation(self, dl):
        details = EvalDetails(
            method="", db_id="test_db",
            exacc_before=0.5, exacc_after=0.7, delta=0.2,
            before_details=[QueryDetail(
                nl="q", gold_sql="SELECT 1", pred_sql="SELECT 1",
                backmapped_sql=None, match=True,
            )],
            after_details=[QueryDetail(
                nl="q", gold_sql="SELECT 1", pred_sql="SELECT employee_name FROM e",
                backmapped_sql="SELECT nm FROM e", match=True,
            )],
        )
        dl.save_evaluation("test_db", "egrefine", details)
        path = os.path.join(dl.output_dir, "per_db", "test_db", "evaluation", "egrefine.json")
        data = json.load(open(path))
        assert data["method"] == "egrefine"
        assert len(data["before_details"]) == 1
        assert data["after_details"][0]["backmapped_sql"] == "SELECT nm FROM e"

    def test_file_logging(self, dl):
        import logging
        handler = dl.setup_file_logging()
        test_logger = logging.getLogger("test_detail")
        test_logger.setLevel(logging.DEBUG)
        test_logger.info("test message")
        handler.flush()
        log_path = os.path.join(dl.output_dir, "experiment.log")
        assert os.path.exists(log_path)
        content = open(log_path).read()
        assert "test message" in content
        logging.getLogger().removeHandler(handler)
