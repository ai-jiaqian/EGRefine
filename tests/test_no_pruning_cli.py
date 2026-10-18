from types import SimpleNamespace

import scripts.run_refine as cli
from egrefine.data.schema import Column, Schema, Table
from egrefine.phase1.llm_screener import structural_exclusion


def test_no_pruning_keeps_pk_and_preserves_structural_exclusions(monkeypatch, tmp_path):
    pk = Column("id", "parent", "INT", is_pk=True)
    value = Column("value", "parent", "TEXT")
    shared_a = Column("shared", "parent", "TEXT")
    fk = Column("parent_id", "child", "INT", fk_target="parent.id")
    shared_b = Column("shared", "child", "TEXT")
    schema = Schema("db", [Table("parent", [pk, value, shared_a]),
                           Table("child", [fk, shared_b])],
                    [("child.parent_id", "parent.id")])
    loader = SimpleNamespace(schemas={"db": schema}, db_ids=["db"], pairs=[],
                             get_db_path=lambda db: str(tmp_path / "db.sqlite"))
    cfg = {"data": {"drspider_abbr": {"path": str(tmp_path)}},
           "models": {"candidate_llm": {"model_name": "test"},
                      "text2sql": [{"model_name": "test"}]},
           "output": {"cache_dir": str(tmp_path / "cache")},
           "phase1": {"method": "llm"}, "phase2": {"sample_rows": 3},
           "phase3": {}, "phase4": {}}
    captured = []
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(cli, "DrSpiderLoader", lambda path: loader)
    for name in ("LLMClient", "CandidateGenerator", "C3Text2SQL"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_save_artifacts", lambda *args: None)

    def pipeline(**kwargs):
        captured.extend(kwargs["override_candidates"])
        return SimpleNamespace(statistics={})

    monkeypatch.setattr(cli, "run_pipeline", pipeline)
    monkeypatch.setattr(cli.sys, "argv", ["run_refine.py", "--benchmark", "drspider_abbr",
                                         "--no-pruning", "--output", str(tmp_path / "out")])
    cli.main()
    assert captured == [pk, value]
    excluded, *_ = structural_exclusion(schema)
    assert {c.full_name for c in captured} == {
        c.full_name for c in schema.all_columns if c.full_name not in excluded
    }
