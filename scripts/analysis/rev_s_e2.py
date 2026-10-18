#!/usr/bin/env python3
"""E2's existing ISO/COMP protocol, bounded query concurrency and raw traces."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
from scripts.rev_composed_check import load_committed
from egrefine.data.benchmark import load_bird, load_drspider
from egrefine.data.query_index import build_query_index
from egrefine.phase3.c3_runner import C3Text2SQL
from egrefine.phase3.dinsql_runner import DINSQLText2SQL
from egrefine.phase3.executor import compare_results
from egrefine.phase4.backmapper import backmap


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def summarize(recs):
    nontrivial = [r for r in recs if not r['trivial_single_col']]
    positive = [r for r in nontrivial if r['iso_delta'] > 1e-12]
    return dict(n_scored=len(recs), n_nontrivial=len(nontrivial),
                n_trivial_single_col=len(recs)-len(nontrivial),
                iso_nonpositive=sum(r['iso_delta'] <= 1e-12 for r in nontrivial),
                iso_positive=len(positive),
                comp_positive=sum(r['comp_delta'] > 1e-12 for r in positive),
                comp_zero=sum(abs(r['comp_delta']) <= 1e-12 for r in positive),
                comp_negative=sum(r['comp_delta'] < -1e-12 for r in positive),
                iso_positive_mean_iso=sum(r['iso_delta'] for r in positive)/len(positive) if positive else None,
                iso_positive_mean_comp=sum(r['comp_delta'] for r in positive)/len(positive) if positive else None,
                mean_iso_delta=sum(r['iso_delta'] for r in recs)/len(recs),
                mean_comp_delta=sum(r['comp_delta'] for r in recs)/len(recs),
                reverts=[r for r in recs if r['comp_delta'] < -1e-12])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--refine-dir', required=True)
    ap.add_argument('--benchmark', choices=['drspider_abbr', 'bird'], required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--columns', help='JSON list of {db,column}; omit for all commits')
    ap.add_argument('--workers', type=int, default=40)
    args = ap.parse_args()
    if not 1 <= args.workers <= 48:
        ap.error('E2S concurrency must be between 1 and 48')
    config = yaml.safe_load(Path(args.config).read_text())
    tc = config['models']['text2sql']
    if isinstance(tc, list):
        assert len(tc) == 1, 'E2 uses one assigned endpoint only'
        tc = tc[0]
    path = config['data'][args.benchmark]['path']
    schemas, pairs = (load_drspider(path) if args.benchmark == 'drspider_abbr' else load_bird(path))
    db_dir = Path(path) / ('database_post_perturbation' if args.benchmark == 'drspider_abbr' else 'dev_databases')
    desc_dir = str(db_dir) if args.benchmark == 'bird' else None
    committed = load_committed(args.refine_dir)
    selected = {(r['db'], r['column']) for r in json.loads(Path(args.columns).read_text())} if args.columns else None
    grouped = defaultdict(list)
    for p in pairs:
        grouped[p.db_id].append(p)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    local = threading.local()
    jobs, columns = [], {}
    for db, renames in committed.items():
        schema = schemas[db]
        qi = build_query_index(grouped[db], schema)
        for column, selected_name in renames.items():
            if selected is not None and (db, column) not in selected:
                continue
            stem = db+'__'+column.replace('.', '__')
            if (out/'cols'/f'{stem}.json').exists():
                continue
            qs = qi[column]
            assert qs, (db, column, 'empty Q')
            orig_name = column.split('.', 1)[1]
            columns[stem] = dict(db_id=db, column=column, selected=selected_name,
                                 n_queries=len(qs), trivial_single_col=len(renames)==1)
            bases = {'iso': {}}
            if len(renames)>1:
                bases['comp'] = {k:v for k,v in renames.items() if k!=column}
            for base, other in bases.items():
                for name in [orig_name, selected_name]:
                    mapping = dict(other)
                    if name != orig_name:
                        mapping[column] = name
                    modified = schema.apply_refinement(mapping)
                    reverse = {v:k.split('.',1)[1] for k,v in mapping.items()}
                    for method in ['c3', 'dinsql']:
                        for i, pair in enumerate(qs):
                            jobs.append((stem, base, name, method, i, pair, modified, reverse, str(db_dir/db/f'{db}.sqlite')))

    def run(job):
        stem, base, name, method, i, pair, schema, reverse, db_path = job
        original_name=columns[stem]['column'].split('.',1)[1]
        arm_index=0 if name==original_name else 1
        # APFS can be case-insensitive: distinct model inputs still need
        # distinct traces for Document_Description -> document_description.
        target = out/'queries'/stem/base/f'{arm_index}__{name}'/method/f'{i:04d}.json'
        # Only completed traces from this exact output can be resumed.
        if target.exists():
            return stem, base, name, method, json.loads(target.read_text())['match']
        if not hasattr(local, 'models'):
            local.models = {'c3': C3Text2SQL(tc, num_samples=3, sample_rows=3, desc_dir=desc_dir),
                            'dinsql': DINSQLText2SQL(tc, self_correction=True)}
        model = local.models[method]
        calls = []
        chat = model.llm.chat
        def traced(messages, **kwargs):
            trace = []
            event = {'stage': getattr(model.llm.log_context, 'record', {}).get('stage'), 'trace':trace}
            calls.append(event)
            return chat(messages, trace=trace)
        model.llm.chat = traced
        start = time.time()
        rec = dict(db_id=pair.db_id, column=columns[stem]['column'], base=base, name=name,
                   method=method, query_index=i, nl=pair.nl, gold_sql=pair.gold_sql,
                   schema=asdict(schema), reverse_mapping=reverse, calls=calls)
        try:
            pred = model.generate(pair.nl, schema, db_path=db_path, column_mapping=reverse, evidence='')
            sql = backmap(pred, reverse) if reverse else pred
            rec.update(pred_sql=pred, backmapped_sql=sql, match=bool(compare_results(sql, pair.gold_sql, db_path)))
            for event in calls:
                for attempt in event['trace']:
                    if 'response' not in attempt:
                        raise RuntimeError('API error recorded in E2 trace; stop before publishing a score')
        except Exception as exc:
            rec.update(error=repr(exc), match=False)
            raise
        finally:
            model.llm.chat = chat
            rec['seconds'] = time.time()-start
            write_json(target, rec)
        return stem, base, name, method, rec['match']

    print(f'columns={len(columns)} model-query units={len(jobs)} workers={args.workers}', flush=True)
    results = defaultdict(lambda: defaultdict(list))
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            stem, base, name, method, match = future.result()
            results[stem][(base,name,method)].append(int(match))
            if count % 100 == 0:
                print(f'{count}/{len(jobs)} units in {time.time()-start:.1f}s', flush=True)
    for stem, rec in columns.items():
        scores = results[stem]
        original = rec['column'].split('.',1)[1]
        for base in ['iso','comp']:
            if base=='comp' and rec['trivial_single_col']:
                rec[base] = dict(rec['iso'])
                continue
            rec[base] = {}
            for name in [original,rec['selected']]:
                per_model=[]
                for method in ['c3','dinsql']:
                    hits=scores[(base,name,method)]
                    assert len(hits)==rec['n_queries'], (stem,base,name,method,len(hits))
                    per_model.append(sum(hits)/len(hits))
                rec[base][name] = sum(per_model)/2
        rec['iso_delta']=rec['iso'][rec['selected']]-rec['iso'][original]
        rec['comp_delta']=rec['comp'][rec['selected']]-rec['comp'][original]
        write_json(out/'cols'/f'{stem}.json', rec)
    recs=[json.loads(p.read_text()) for p in sorted((out/'cols').glob('*.json'))]
    summary=summarize(recs)
    summary.update(seconds=time.time()-start, model_query_units=len(jobs), workers=args.workers,
                   config=str(Path(args.config).resolve()), refine_dir=args.refine_dir,
                   protocol='legacy E2: directly committed columns only; single-direct-commit COMP=ISO')
    write_json(out/'summary.json',summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2),flush=True)


if __name__=='__main__':
    main()
