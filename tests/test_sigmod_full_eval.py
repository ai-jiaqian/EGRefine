import json
import sqlite3
import sys
from types import SimpleNamespace

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.phase4.view_synthesis import synthesize


def test_full_eval_runs_zero_change_db_and_default_still_skips(tmp_path, monkeypatch):
    import scripts.run_eval as cli
    db = tmp_path / 'tiny.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE t (id INTEGER)')
        conn.execute('INSERT INTO t VALUES (1)')
    schema = Schema('tiny', [Table('t', [Column('id', 't', 'INTEGER')])], [])
    refine = tmp_path / 'refine'
    synthesize(schema, [], output_dir=str(refine / 'tiny'))
    pair = NLSQLPair('How many rows?', 'SELECT COUNT(*) FROM t', 'tiny')
    loader = SimpleNamespace(db_ids=['tiny'], schemas={'tiny': schema}, pairs=[pair],
                             get_pairs_for_db=lambda _: [pair], get_db_path=lambda _: str(db))
    monkeypatch.setattr(cli, 'DrSpiderLoader', lambda *a, **k: loader)
    monkeypatch.setattr(cli, 'load_config', lambda _: {
        'models': {'text2sql': [{'model_name': 'dummy'}]},
        'data': {'drspider_abbr': {'path': 'unused'}}, 'concurrency': {'max_workers': 1}})
    calls = []
    class Model:
        def __init__(self, config): pass
        def generate(self, **kwargs):
            calls.append(kwargs)
            return 'SELECT COUNT(*) FROM t'
    monkeypatch.setitem(cli.METHOD_REGISTRY, 'c3', Model)
    base = ['run_eval.py', '--benchmark', 'drspider_abbr', '--schema', 'refined',
            '--refine-dir', str(refine), '--methods', 'c3']
    monkeypatch.setattr(sys, 'argv', base + ['--output', str(tmp_path / 'default')])
    cli.main()
    assert not calls
    monkeypatch.setattr(sys, 'argv', base + ['--full-eval', '--output', str(tmp_path / 'full')])
    cli.main()
    result = json.loads((tmp_path / 'full/per_db/tiny/c3.json').read_text())
    assert len(calls) == 1
    assert result['total_queries'] == 1 and result['exacc_refined'] == 1


def test_din_description_is_optional_and_reaches_schema_prompt():
    from egrefine.phase3.dinsql_runner import DINSQLText2SQL
    config = {'base_url': 'http://localhost:1/v1', 'api_key': 'dummy', 'model_name': 'dummy'}
    schema = Schema('tiny', [Table('t', [Column('nm', 't', 'TEXT')])], [])
    plain = DINSQLText2SQL(config)._format_ddl(schema)
    annotated = DINSQLText2SQL(config, column_descriptions={('tiny', 't', 'nm'): 'Employee name'})
    assert annotated._format_ddl(schema) == plain.replace('nm TEXT', 'nm TEXT /* Employee name */')
