#!/usr/bin/env python3
"""E2-only run_eval entry point: isolate C3 mutable temperature per thread."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from egrefine.phase3.c3_runner import C3Text2SQL
from egrefine.data.benchmark import DrSpiderLoader


def install_threadlocal_c3(cls=C3Text2SQL):
    original=cls.generate
    local=threading.local()
    def generate(self,*args,**kwargs):
        if not hasattr(local,'model'):
            local.model=cls(self._config,num_samples=self.num_samples,
                sample_rows=self.sample_rows,desc_dir=self.desc_dir,
                column_descriptions=self.column_descriptions)
        return original(local.model,*args,**kwargs)
    cls.generate=generate


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--smoke-questions')
    args,remaining=parser.parse_known_args()
    if args.smoke_questions:
        selected=json.loads(Path(args.smoke_questions).read_text())
        get_pairs=DrSpiderLoader.get_pairs_for_db
        def filtered(self,db_id):
            rows=[p for p in get_pairs(self,db_id) if p.nl in selected[db_id]]
            assert len(rows)==len(selected[db_id]),(db_id,'smoke questions mismatch')
            return rows
        DrSpiderLoader.get_pairs_for_db=filtered
    install_threadlocal_c3()
    sys.argv=['scripts/run_eval.py',*remaining]
    runpy.run_path('scripts/run_eval.py',run_name='__main__')


if __name__=='__main__':main()
