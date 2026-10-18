#!/usr/bin/env python3
"""Correct E2's reproduced C3 voter backmap failure using its fresh raw candidates.

No model calls. Preserve legacy scores; execute the same three generated
candidates with the full reverse map before the unchanged C3 voting rule.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import yaml
from egrefine.phase3 import c3_runner
from egrefine.phase3.c3_runner import C3Text2SQL
from egrefine.phase3.executor import compare_results
from egrefine.phase4.backmapper import backmap
from scripts.analysis.rev_s_e2 import summarize,write_json


def replay_c3(record,model,db_path):
    candidates=[model._extract_sql(c['trace'][-1]['response']['choices'][0]['message']['content'])
                for c in record['calls'][1:]]
    assert len(candidates)==3, 'E2 C3 uses exactly three candidates'
    reverse=record['reverse_mapping']
    mapped=[backmap(s,reverse) if reverse else s for s in candidates]
    executions=[]
    execute=c3_runner.execute_sql
    # This replay is serial; collect actual voter I/O without changing its rule.
    def traced(sql,path):
        result=execute(sql,path)
        executions.append(dict(sql=sql,success=result is not None,
            rows_repr=repr(sorted(result,key=str)) if result is not None else None))
        return result
    c3_runner.execute_sql=traced
    try:
        chosen=model._self_consistency(mapped,str(db_path))
    finally:
        c3_runner.execute_sql=execute
    index=mapped.index(chosen)
    return dict(db_path=str(db_path),gold_sql=record['gold_sql'],
                candidates=candidates,execution_candidates=mapped,executions=executions,
                selected_index=index,pred_sql=candidates[index],backmapped_sql=chosen,
                match=bool(compare_results(chosen,record['gold_sql'],str(db_path))),
                legacy_pred_sql=record['pred_sql'],legacy_match=record['match'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True);ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--benchmark',choices=['drspider_abbr','bird'],required=True)
    args=ap.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    tc=cfg['models']['text2sql'];tc=tc[0] if isinstance(tc,list) else tc
    model=C3Text2SQL(tc,num_samples=3)
    source=Path(args.input);output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    dbroot=Path(cfg['data'][args.benchmark]['path'])/('database_post_perturbation' if args.benchmark=='drspider_abbr' else 'dev_databases')
    scores=defaultdict(lambda:defaultdict(list));changes=[];no_map_changes=[];n=0;t0=time.time()
    for path in sorted((source/'queries').rglob('*.json')):
        record=json.loads(path.read_text())
        if record.get('error'):raise ValueError(f'Incomplete raw source: {path}')
        db=record['db_id'];stem=db+'__'+record['column'].replace('.','__')
        method=record['method'];hit=record['match']
        if method=='c3':
            replay=replay_c3(record,model,dbroot/db/(db+'.sqlite'))
            replay.update(source=str(path),db_id=db,column=record['column'],base=record['base'],name=record['name'],method=method,query_index=record['query_index'])
            write_json(output/'queries'/path.relative_to(source/'queries'),replay)
            hit=replay['match'];n+=1
            if hit!=record['match']:
                changes.append(dict(source=str(path),db=db,base=record['base'],name=record['name'],legacy=record['match'],corrected=hit))
                if not record['reverse_mapping']:no_map_changes.append(changes[-1])
        scores[stem][(record['base'],record['name'],method)].append(int(hit))
    recs=[]
    for path in sorted((source/'cols').glob('*.json')):
        record=json.loads(path.read_text());db=record['db_id'];stem=db+'__'+record['column'].replace('.','__')
        original=record['column'].split('.',1)[1]
        for base in ['iso','comp']:
            if base=='comp' and record['trivial_single_col']:
                record[base]=dict(record['iso']);continue
            for name in [original,record['selected']]:
                model_scores=[]
                for method in ['c3','dinsql']:
                    hits=scores[stem][(base,name,method)]
                    assert len(hits)==record['n_queries'],(stem,base,name,method,len(hits))
                    model_scores.append(sum(hits)/len(hits))
                record[base][name]=sum(model_scores)/2
        record['iso_delta']=record['iso'][record['selected']]-record['iso'][original]
        record['comp_delta']=record['comp'][record['selected']]-record['comp'][original]
        record['c3_vote_full_backmap']=True
        write_json(output/'cols'/path.name,record);recs.append(record)
    summary=summarize(recs)
    summary.update(source=str(source),c3_units=n,flips=changes,
                   fixed=sum(not r['legacy'] and r['corrected'] for r in changes),
                   broken=sum(r['legacy'] and not r['corrected'] for r in changes),
                   no_mapping_outcome_changes=no_map_changes,seconds=time.time()-t0,
                   protocol='Same fresh API candidates; full reverse mapping before unchanged C3 vote. DIN unchanged.')
    write_json(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
