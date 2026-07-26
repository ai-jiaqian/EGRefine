#!/usr/bin/env python3
"""
SNAILS-style baseline (faithful, full re-run) for the EGRefine vs SNAILS comparison.

Pipeline (no shortcuts):
  1. SCREEN  : SNAILS released CANINE naturalness classifier flags unnatural (N2/N3) columns.
               (precomputed per-column in --screened JSON; ALL schema columns considered.)
  2. GENERATE: for each unnatural column, schema-refiner's Phase-2 CandidateGenerator
               produces names via the SAME Qwen refiner; take candidates[0] (as LLM-Direct does).
               FRESH generation forced via a dedicated cache dir.
  3. ACCEPT  : take the generated name, NO execution verification (SNAILS accepts by naturalness).
  4. Synthesize VIEWs and write a native refine-dir (pipeline_result.json + views) per DB,
     so the existing eval pipeline can consume it like any other refinement.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/run_snails_style.py \
      --config config/revision_qwen35_27b.yaml \
      --screened /path/to/snails_screened.json \
      --output results/refine/drspider_abbr_snails27b \
      --cache-dir ./cache_snails27b --max-workers 24
"""
import argparse, json, logging, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.config import load_config
from egrefine.data.benchmark import DrSpiderLoader
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.generator import CandidateGenerator
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase4.view_synthesis import synthesize
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("snails_style")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/revision_qwen35_27b.yaml")
    ap.add_argument("--screened", required=True, help="snails_screened.json (CANINE labels per column)")
    ap.add_argument("--output", default="results/refine/drspider_abbr_snails27b")
    ap.add_argument("--cache-dir", default="./cache_snails27b", help="FRESH cache dir (force real generation)")
    ap.add_argument("--dbs", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="limit #DBs (smoke)")
    ap.add_argument("--max-workers", type=int, default=24)
    args = ap.parse_args()

    config = load_config(args.config)
    screened = json.load(open(args.screened))
    # unnatural set per db: {db: set((table,col))}
    unnat = {db: {(r["table"], r["column"]) for r in rows if r.get("unnatural")}
             for db, rows in screened.items()}

    loader = DrSpiderLoader(config["data"]["drspider_abbr"]["path"])
    db_ids = args.dbs or [d for d in loader.db_ids if d in unnat]
    if args.limit: db_ids = db_ids[:args.limit]

    llm_config = config["models"]["candidate_llm"]
    gen = CandidateGenerator(
        LLMClient(llm_config), cache_dir=args.cache_dir,
        sample_rows=config["phase2"]["sample_rows"],
        max_retries=llm_config.get("max_retries", 3),
    )
    k = config["phase2"].get("k", 3)
    os.makedirs(args.output, exist_ok=True)

    tot_unnat = tot_changed = 0
    t0 = time.time()
    for i, db in enumerate(db_ids, 1):
        schema = loader.schemas[db]
        db_path = loader.get_db_path(db)
        targets = [c for c in schema.all_columns if (c.table, c.name) in unnat.get(db, set())]

        def _gen(col):
            cands = gen.generate(col, schema, db_path, k=k)
            if not cands:
                return None
            name = cands[0].name
            return SelectionResult(column=col, selected_name=name, delta=0.0,
                                   was_changed=(name != col.name), all_scores={},
                                   verification_method="snails_style")
        results = []
        if args.max_workers > 1 and len(targets) > 1:
            with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                futs = {pool.submit(_gen, c): c for c in targets}
                for f in as_completed(futs):
                    r = f.result()
                    if r: results.append(r)
        else:
            for c in targets:
                r = _gen(c)
                if r: results.append(r)

        changed = [r for r in results if r.was_changed]
        tot_unnat += len(targets); tot_changed += len(changed)
        out_db = os.path.join(args.output, db)
        os.makedirs(out_db, exist_ok=True)
        synth = synthesize(schema, results, output_dir=out_db)
        pr = {
            "db_id": db,
            "refinements": [{
                "table": r.column.table, "column": r.column.name,
                "original_name": r.column.name, "refined_name": r.selected_name,
                "delta": r.delta, "was_changed": r.was_changed,
                "verification_method": r.verification_method, "all_scores": r.all_scores,
            } for r in results],
            "view_definitions": synth["views"], "mapping": synth["mapping"],
            "reverse_mapping": synth["reverse_mapping"], "orig_table_map": synth["orig_table_map"],
            "statistics": {"method": "snails_style", "columns_scored": len(results),
                           "columns_refined": len(changed),
                           "columns_kept_original": len(results) - len(changed)},
        }
        json.dump(pr, open(os.path.join(out_db, "pipeline_result.json"), "w"), indent=2)
        logger.info("[%d/%d] %s: unnatural=%d generated_changed=%d", i, len(db_ids), db, len(targets), len(changed))

    print(f"\nDONE dbs={len(db_ids)} canine_unnatural={tot_unnat} snails_changed={tot_changed} "
          f"time={time.time()-t0:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
