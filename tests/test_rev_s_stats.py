import json
import numpy as np
import pytest
import shutil
from scripts.analysis import rev_s_stats
from scripts.analysis.rev_s_stats import load_pairs, compute, bootstrap, mcnemar


def test_paired_alignment_fill_and_cluster(tmp_path):
    no,arm,refine=(tmp_path/x for x in ['no','arm','refine'])
    def save(root,db,values,reverse=False):
        p=root/'per_db'/db;p.mkdir(parents=True)
        rows=[dict(question_id=i,match=bool(v)) for i,v in enumerate(values)]
        if reverse:rows.reverse()
        (p/'c3.json').write_text(json.dumps({'details':rows}))
    save(no,'car_1_0',[0,0]);save(no,'car_1_1',[1,0]);save(no,'world_1_0',[1])
    save(arm,'car_1_0',[1,1],True);save(arm,'car_1_1',[0,1],True)
    p=refine/'world_1_0';p.mkdir(parents=True);(p/'pipeline_result.json').write_text('{"refinements":[]}')
    a,b,g,meta=load_pairs(no,arm,'c3','drspider_abbr',refine_dir=refine,expected=(3,5))
    assert a.tolist()==[0,0,1,0,1] and b.tolist()==[1,1,0,1,1]
    assert meta['fallback_queries']==1 and g==['car_1']*4+['world_1']
    result=compute(no,arm,'c3','drspider_abbr',refine_dir=refine,expected=(3,5))
    assert (result['fixed'],result['broken'],result['net'])==(3,1,2)
    assert result['mcnemar_p']==.625
    # Enumerate the four equal-probability two-cluster draws: 50,40,40,0 pp.
    assert result['cluster_ci']==[0,50]
    assert result['delta_pp']==40 and result['noref_exacc']==40 and result['arm_exacc']==80
    assert bootstrap(a,b,g)==result['cluster_ci']
    assert mcnemar(0,0)==1 and mcnemar(3,0)==.25
    # Existing DB with missing method is incomplete, never silently filled.
    (arm/'per_db'/'world_1_0').mkdir()
    with pytest.raises(ValueError,match='Missing c3'):
        load_pairs(no,arm,'c3','drspider_abbr',expected=(3,5))


def test_universe_and_query_mismatch(tmp_path):
    for root in ['no','arm']:
        p=tmp_path/root/'per_db'/'car_1_0';p.mkdir(parents=True)
        (p/'c3.json').write_text(json.dumps({'details':[dict(question_id=0,match=True)]}))
    with pytest.raises(ValueError,match='Universe mismatch'):
        load_pairs(tmp_path/'no',tmp_path/'arm','c3','drspider_abbr')
    (tmp_path/'arm'/'per_db'/'car_1_0'/'c3.json').write_text(json.dumps({'details':[dict(question_id=1,match=True)]}))
    with pytest.raises(ValueError,match='alignment mismatch'):
        load_pairs(tmp_path/'no',tmp_path/'arm','c3','drspider_abbr',expected=(1,1))


def test_duplicate_question_gold_backing_table_alignment(tmp_path):
    for name,values in [('no',[('SELECT x FROM t',False),('SELECT y FROM t',True)]),
                        ('arm',[('SELECT y FROM _orig_t',False),('SELECT x FROM _orig_t',True)])]:
        p=tmp_path/name/'per_db'/'one_db';p.mkdir(parents=True)
        (p/'c3.json').write_text(json.dumps({'details':[
            dict(nl='same question',gold_sql=sql,match=value) for sql,value in values]}))
    a,b,_,_=load_pairs(tmp_path/'no',tmp_path/'arm','c3','bird',expected=(1,2))
    assert a.tolist()==[0,1] and b.tolist()==[1,0]


def test_registered_discovery_excludes_archived_basename_and_partial_baseline(tmp_path,monkeypatch):
    monkeypatch.setattr(rev_s_stats,'ROOT',tmp_path)
    root=tmp_path/'results/eval/drspider_abbr'
    for name in ['rev_s_9b_noref','rev_s_9b_on27brefine']:
        for i in range(90):
            p=root/name/'per_db'/str(i);p.mkdir(parents=True)
            (p/'c3.json').write_text('{}')
    registry=tmp_path/'registry.md'
    registry.write_text('### Superseded\nresults/_archive/old/rev_s_9b_noref\nresults/_archive/old/rev_s_9b_on27brefine\n')
    assert rev_s_stats.registered_new_specs(registry)==[]
    registry.write_text('### Completed ⭐待复核\nresults/eval/drspider_abbr/rev_s_9b_noref\nresults/eval/drspider_abbr/rev_s_9b_on27brefine\n')
    assert len(rev_s_stats.registered_new_specs(registry))==1
    shutil.rmtree(root/'rev_s_9b_noref/per_db/0')
    assert rev_s_stats.registered_new_specs(registry)==[]
