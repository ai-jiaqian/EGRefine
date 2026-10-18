#!/usr/bin/env python3
"""Strict query alignment, exact McNemar and base-DB paired bootstrap.

Bootstrap intervals condition on saved predictions, not repeated model seeds.
McNemar and query bootstrap are nominal query-independent comparisons.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {'drspider_abbr': (90, 2853), 'bird': (11, 1534)}
SEED = 20261007
ALIASES = {'c3':['c3'], 'dinsql':['dinsql','din'], 'macsql':['macsql','mac']}


def base_db(db, benchmark):
    return re.sub(r'_\d+$', '', db) if benchmark == 'drspider_abbr' else db


def per_db(path):
    path = Path(path)
    return path/'per_db' if (path/'per_db').is_dir() else path


def method_file(db, method):
    for alias in ALIASES[method]:
        p = db/f'{alias}.json'
        if p.is_file():
            return p
    raise ValueError(f'Missing {method} result in existing DB {db}')


def rows(path, refined, test_questions=None):
    payload = json.loads(path.read_text())
    keys = ['refined_details','original_details'] if refined else ['original_details','refined_details']
    keys += ['details','queries','results','predictions']
    if isinstance(payload,list):
        data=payload
    else:
        data=next((payload[k] for k in keys if isinstance(payload.get(k),list) and payload[k]),None)
    if not data:
        raise ValueError(f'Empty query details: {path}')
    records={}
    seen=Counter()
    nl_counts=Counter(re.sub(r'\s+',' ',str(r.get('nl',r.get('question','')))).strip() for r in data)
    for row in data:
        nl=re.sub(r'\s+',' ',str(row.get('nl',row.get('question','')))).strip()
        if test_questions is not None and nl not in test_questions:
            continue
        explicit=row.get('question_id',row.get('query_id',row.get('id')))
        gold=re.sub(r'\s+',' ',str(row.get('gold_sql',row.get('SQL','')))).strip()
        if not nl and explicit is None:
            raise ValueError(f'Query lacks stable identity: {path}')
        # Refined eval stores gold SQL rewritten to _orig_ backing tables.
        # Remove that known identifier prefix; gold disambiguates duplicate NL
        # in BIRD even if arm rows arrive in a different order.
        gold=re.sub(r'\b_orig_', '', gold)
        identity=(('id',str(explicit)) if explicit is not None else
                  ('nl_gold',nl,gold) if nl_counts[nl]>1 else ('nl',nl))
        seen[identity]+=1
        key=(*identity,seen[identity])
        val=next((row[k] for k in ['match','exec_match','execution_match','correct'] if k in row),None)
        if val not in (True,False,0,1):
            raise ValueError(f'Invalid binary outcome {val!r}: {path}')
        records[key]=int(val)
    return records


def load_pairs(noref, arm, method, benchmark, split=None, refine_dir=None, expected=None):
    no, ar = per_db(noref), per_db(arm)
    if not no.is_dir() or not ar.is_dir():
        raise ValueError('Both per_db directories must exist')
    dbs=sorted(p.name for p in no.iterdir() if p.is_dir())
    extra={p.name for p in ar.iterdir() if p.is_dir()}-set(dbs)
    if extra:
        raise ValueError(f'Arm has extra databases: {sorted(extra)}')
    splits=json.loads(Path(split).read_text())['split'] if split else None
    a,b,clusters=[],[],[]
    missing=[]
    for db in dbs:
        questions=None
        if splits is not None:
            questions={re.sub(r'\s+',' ',s).strip() for s in splits[base_db(db,benchmark)]['test']}
        original=rows(method_file(no/db,method),False,questions)
        has_result=any((ar/db/f'{alias}.json').is_file() for alias in ALIASES[method])
        if not has_result:
            if (ar/db).is_dir() and refine_dir is None:
                raise ValueError(f'Missing {method} result in existing DB {ar/db}')
            if refine_dir:
                result=Path(refine_dir)/db/'pipeline_result.json'
                if not result.is_file() and result.parent.exists():
                    raise ValueError(f'Cannot verify no-commit missing DB: {result}')
                # Registered subset-refine arms do not publish a directory for
                # DBs absent from their refinement workload (e.g. DS25 battle_death_4).
                data=json.loads(result.read_text()) if result.is_file() else {}
                if any(r.get('was_changed') and r.get('original_name') != r.get('refined_name')
                       for r in data.get('refinements',[])):
                    raise ValueError(f'Arm missing committed DB {db}')
            refined=original
            missing.append(db)
        else:
            refined=rows(method_file(ar/db,method),True,questions)
        if set(original)!=set(refined):
            raise ValueError(f'Query alignment mismatch {db}: missing={len(set(original)-set(refined))}, extra={len(set(refined)-set(original))}')
        for key,value in original.items():
            a.append(value);b.append(refined[key]);clusters.append(base_db(db,benchmark))
    if expected is None:
        expected=(90,1430) if split else EXPECTED.get(benchmark)
    if expected is not None and (len(dbs),len(a))!=tuple(expected):
        raise ValueError(f'Universe mismatch: {(len(dbs),len(a))}; required={expected}')
    if not a:
        raise ValueError('No aligned queries')
    return np.array(a),np.array(b),clusters,dict(db_count=len(dbs),fallback_dbs=missing,
        fallback_queries=sum(len(rows(method_file(no/db,method),False,
              {re.sub(r'\s+',' ',s).strip() for s in splits[base_db(db,benchmark)]['test']} if splits else None)) for db in missing))


def mcnemar(fixed,broken):
    n=fixed+broken
    return min(1.0,2*sum(math.comb(n,k) for k in range(min(fixed,broken)+1))/(2**n)) if n else 1.0


def bootstrap(a,b,clusters,B=10000,seed=SEED,legacy=False):
    groups={}
    for x,y,g in zip(a,b,clusters):
        groups.setdefault(g,[0,0])
        groups[g][0]+=int(y)-int(x);groups[g][1]+=1
    # Sorted base IDs provide stable results regardless of filesystem or row order.
    vals=[groups[g] for g in sorted(groups)]
    rng=random.Random(seed)
    draws=[]
    for _ in range(B):
        sample=rng.choices(vals,k=len(vals))
        draws.append(100*sum(v[0] for v in sample)/sum(v[1] for v in sample))
    if legacy:
        draws.sort()
        return [draws[int(.025*B)-1],draws[int(.975*B)-1]]
    return [float(x) for x in np.percentile(draws,[2.5,97.5])]


def query_bootstrap(a,b,B=10000,seed=SEED):
    # Resampling paired differences is exactly resampling the two outcome arrays together.
    diff=b-a
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(B):
        vals.append(100*diff[rng.integers(0,len(diff),size=len(diff))].mean())
    return [float(x) for x in np.percentile(vals,[2.5,97.5])]


def compute(noref,arm,method,benchmark,split=None,refine_dir=None,expected=None,B=10000,seed=SEED):
    a,b,clusters,metadata=load_pairs(noref,arm,method,benchmark,split,refine_dir,expected)
    fixed=int(((a==0)&(b==1)).sum());broken=int(((a==1)&(b==0)).sum())
    result=dict(noref=str(noref),arm=str(arm),method=method,benchmark=benchmark,n=len(a),
        noref_exacc=100*float(a.mean()),arm_exacc=100*float(b.mean()),
        delta_pp=100*float((b-a).mean()),fixed=fixed,broken=broken,net=fixed-broken,
        mcnemar_p=mcnemar(fixed,broken),cluster_count=len(set(clusters)),B=B,seed=seed,
        cluster_ci=bootstrap(a,b,clusters,B,seed) if benchmark!='beaver' else None,
        query_ci=query_bootstrap(a,b,B,seed) if benchmark!='beaver' else None,
        inference_note='BEAVER: few DBs, McNemar only' if benchmark=='beaver' else 'fixed saved outcomes; query tests nominal iid; cluster ratio percentile CI',
        split=str(split) if split else '',**metadata)
    return result


def write_summary(results,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Own aggregate files may be updated as additional registered arms complete.
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in results)
    text=['# E2S paired statistics','',
        '[1-seed, no new model calls; ⭐待复核] Stored per_db outcomes. Query-weighted ExAcc; absent no-commit DBs copied from the paired NoRef. Base databases are resampled with replacement, carrying all variants and queries. B=10000, seed=20261007. McNemar and query bootstrap assume query independence; cluster CI is the primary sensitivity check. No multiple-comparison correction.','',
        '| Arm | Method | n | NoRef | Arm | Δ pp | Fixed/broken/net | McNemar p | Base-DB 95% CI pp | Query 95% CI pp | Fallback q |',
        '|---|---|---:|---:|---:|---:|---|---:|---|---|---:|']
    fmt=lambda ci: f'[{ci[0]:.2f}, {ci[1]:.2f}]' if ci else 'BEAVER: few DBs, omitted'
    for r in results:
        text.append(f"| {Path(r['arm']).name} | {r['method']} | {r['n']} | {r['noref_exacc']:.2f} | {r['arm_exacc']:.2f} | {r['delta_pp']:+.2f} | {r['fixed']}/{r['broken']}/{r['net']:+d} | {r['mcnemar_p']:.5g} | {fmt(r['cluster_ci'])} | {fmt(r['query_ci'])} | {r['fallback_queries']} |")
    text += ['', 'Exact input directories, cluster counts, split and missing-DB lists are recorded in summary.csv. This estimates sample variation conditional on saved outcomes, not model-seed variation or generalization beyond this benchmark.','']
    if (output/'summary_notes.md').exists():
        text[4:4] = ['**Read review notes before citing numbers:** [summary_notes.md](summary_notes.md) includes unresolved current-file discrepancies and reproduction details.','']
    (output/'summary.md').write_text('\n'.join(text))


def registered_new_specs(registry):
    """Find completed, explicitly registered rev_s full-eval comparisons.

    New workers use full eval. Wait for every DB's method file; never fill a
    still-running arm from NoRef. Endpoint families have separate baselines.
    """
    raw_text=Path(registry).read_text()
    # Worker entries are explicitly marked pending/complete review. Archived
    # partial campaigns can mention the same reused basename; exclude them.
    text='\n'.join(section for section in re.split(r'(?=^#{1,6}\s)',raw_text,flags=re.M)
                   if '⭐待复核' in section or '⭐ 待复核' in section or '⭐已复核' in section)
    specs=[]
    for benchmark in ['drspider_abbr','bird','beaver']:
        root=ROOT/'results/eval'/benchmark
        for arm in sorted(root.glob('rev_s_*')):
            if f'eval/{benchmark}/{arm.name}' not in text or 'noref' in arm.name or 'e2_' in arm.name:
                continue
            for method in ALIASES:
                if benchmark=='drspider_abbr':
                    baseline=('rev_s_9b_noref' if arm.name.startswith('rev_s_9b_') else
                              'rev_s_mac_noref' if method=='macsql' else 'rev_s_noref_pool')
                elif benchmark=='bird':
                    baseline='rev_s_bird_noref_27b'
                else:
                    baseline='rev_s_beaver_noref'
                no=root/baseline
                if f'eval/{benchmark}/{baseline}' not in text or not (no/'per_db').is_dir():
                    continue
                dbs=[p.name for p in (no/'per_db').iterdir() if p.is_dir()]
                if benchmark in EXPECTED and len(dbs)!=EXPECTED[benchmark][0]:
                    continue
                target=arm
                if arm.name=='rev_s_mac_pre':
                    # A's saved derivative uses the official pre/post per-DB
                    # index to align NL, preserving untouched raw predictions.
                    target=arm/'paired_stats/aligned_pre'
                if not dbs or not all(any((r/'per_db'/db/f'{alias}.json').is_file() for alias in ALIASES[method]) for r in [no,target] for db in dbs):
                    continue
                spec=dict(noref=str(no.relative_to(ROOT)),arm=str(target.relative_to(ROOT)),method=method,benchmark=benchmark)
                if 'ds50' in arm.name:
                    spec['split']='data/splits/drspider_abbr_devsplit_seed42.json'
                specs.append(spec)
    return specs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--noref');ap.add_argument('--arm')
    ap.add_argument('--method',choices=list(ALIASES));ap.add_argument('--benchmark',choices=['drspider_abbr','bird','beaver'])
    ap.add_argument('--split');ap.add_argument('--refine-dir')
    ap.add_argument('--manifest',help='JSON list of explicit registered comparison inputs')
    ap.add_argument('--registry',help='Also append completed rev_s full-eval arms listed in this registry')
    ap.add_argument('--output',default='results/rev_analysis/rev_s_stats')
    args=ap.parse_args()
    specs=json.loads(Path(args.manifest).read_text()) if args.manifest else [dict(noref=args.noref,arm=args.arm,method=args.method,benchmark=args.benchmark,split=args.split,refine_dir=args.refine_dir)]
    if args.registry:
        known={(s['noref'],s['arm'],s['method']) for s in specs}
        specs += [s for s in registered_new_specs(args.registry) if (s['noref'],s['arm'],s['method']) not in known]
    results=[]
    for spec in specs:
        result=compute(**spec)
        results.append(result)
        print(f"{Path(result['arm']).name} {result['method']}: Δ={result['delta_pp']:+.4f} cluster={result['cluster_ci']} p={result['mcnemar_p']:.5g}",flush=True)
    write_summary(results,args.output)


if __name__=='__main__':
    try:
        main()
    except (ValueError,KeyError,FileNotFoundError) as e:
        sys.exit(f'ERROR: {e}')
