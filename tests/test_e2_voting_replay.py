import sqlite3
from egrefine.phase3.c3_runner import C3Text2SQL
from scripts.analysis.rev_s_e2_voting_replay import replay_c3


def test_reverse_mapping_is_applied_before_vote_and_identity_is_unchanged(tmp_path):
    db=tmp_path/'db.sqlite'
    with sqlite3.connect(db) as conn:
        conn.executescript('CREATE TABLE t(old INTEGER); INSERT INTO t VALUES(1),(2);')
    # The first candidate is wrong; two mapped candidates agree on the gold.
    sqls=['SELECT COUNT(new) FROM t','SELECT SUM(new) FROM t','SELECT SUM(new) FROM t']
    def record(candidates,mapping):
        return dict(reverse_mapping=mapping,calls=[{}]+[
            {'trace':[{'response':{'choices':[{'message':{'content':s}}]}}]} for s in candidates],
            pred_sql=candidates[0],match=False,gold_sql='SELECT SUM(old) FROM t')
    # The voting method uses no client fields and makes no API calls.
    model=C3Text2SQL.__new__(C3Text2SQL)
    result=replay_c3(record(sqls,{'new':'old'}),model,db)
    assert result['selected_index']==1 and result['match']
    assert all(x['success'] for x in result['executions'])
    original=[s.replace('new','old') for s in sqls]
    identity=replay_c3(record(original,{}),model,db)
    assert identity['pred_sql']==model._self_consistency(original,str(db))
    assert identity['match']==result['match']
