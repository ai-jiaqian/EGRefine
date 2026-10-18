#!/usr/bin/env python3
"""Full experiment: run EGRefine + baselines on all BIRD databases.

Usage:
    PYTHONPATH=. python3 scripts/run_full.py
    PYTHONPATH=. python3 scripts/run_full.py --dbs financial california_schools
    PYTHONPATH=. python3 scripts/run_full.py --max-columns 5 --output results/full
"""
import argparse
import json
import logging
import os
import sys
import time
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.config import load_config
from egrefine.data.benchmark import BIRDLoader
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.generator import CandidateGenerator
from egrefine.phase3.text2sql_runner import SimpleLLMText2SQL
from egrefine.pipeline import run_pipeline
from egrefine.baselines.no_refinement import run_no_refinement
from egrefine.baselines.llm_direct import run_llm_direct
from egrefine.baselines.llm_cot import run_llm_cot
from egrefine.experiment import run_experiment, save_experiment
from egrefine.detail_logger import DetailLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="EGRefine Full Experiment")
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--dbs", nargs="*", default=None,
                        help="Specific databases to run (default: all)")
    parser.add_argument("--max-columns", type=int, default=None,
                        help="Limit columns per DB (to save API cost)")
    parser.add_argument("--output", default="results/full",
                        help="Output directory")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Only run EGRefine, skip baselines")
    args = parser.parse_args()

    config = load_config(args.config)

    # ===== Load data =====
    bird = BIRDLoader(config["data"]["bird"]["path"])
    db_ids = args.dbs or bird.db_ids
    print(f"\n{'='*70}")
    print(f"EGRefine FULL EXPERIMENT")
    print(f"{'='*70}")
    print(f"Databases: {len(db_ids)} ({', '.join(db_ids)})")
    print(f"Total pairs: {len(bird.pairs)}")
    print(f"Max columns: {args.max_columns or 'unlimited'}")
    print(f"Output: {args.output}")

    # ===== Setup shared components =====
    phase1_config = config["phase1"]
    phase1_config["signals"]["high_similarity"]["enabled"] = False  # skip embedding for now

    llm_config = config["models"]["candidate_llm"]
    llm_client = LLMClient(llm_config)
    cache_dir = config["output"].get("cache_dir", "./cache")

    candidate_generator = CandidateGenerator(
        llm_client, cache_dir=cache_dir,
        sample_rows=config["phase2"]["sample_rows"],
        max_retries=llm_config.get("max_retries", 3),
    )

    text2sql_configs = config["models"]["text2sql"]
    models = [SimpleLLMText2SQL(tc) for tc in text2sql_configs]
    eval_model = models[0]  # Use first model for evaluation
    max_workers = config.get("concurrency", {}).get("max_workers", 1)

    print(f"Text-to-SQL models: {[tc['name'] for tc in text2sql_configs]}")
    print(f"Concurrency: max_workers={max_workers}")

    # ===== Detail logger =====
    dl = DetailLogger(args.output)
    dl.save_config_snapshot(config)
    log_handler = dl.setup_file_logging()

    # ===== Limit pruner if needed =====
    if args.max_columns:
        _apply_column_limit(args.max_columns)

    # ===== Define methods =====
    def method_egrefine(schema, pairs, db_path):
        return run_pipeline(
            schema=schema, pairs=pairs, db_path=db_path,
            models=models,
            phase1_config=phase1_config,
            phase2_config=config["phase2"],
            phase3_config=config["phase3"],
            phase4_config=config["phase4"],
            candidate_generator=candidate_generator,
            detail_logger=dl,
            max_workers=max_workers,
        )

    def method_no_refinement(schema, pairs, db_path):
        return run_no_refinement(schema=schema, pairs=pairs, db_path=db_path)

    def method_llm_direct(schema, pairs, db_path):
        return run_llm_direct(
            schema=schema, pairs=pairs, db_path=db_path,
            phase1_config=phase1_config,
            phase2_config=config["phase2"],
            candidate_generator=candidate_generator,
        )

    def method_llm_cot(schema, pairs, db_path):
        return run_llm_cot(
            schema=schema, pairs=pairs, db_path=db_path,
            phase1_config=phase1_config,
            phase2_config=config["phase2"],
            candidate_generator=candidate_generator,
            llm_client=llm_client,
        )

    methods = {"no_refinement": method_no_refinement}
    if not args.skip_baselines:
        methods["llm_direct"] = method_llm_direct
        methods["llm_cot"] = method_llm_cot
    methods["egrefine"] = method_egrefine

    # ===== Run experiment =====
    print(f"\nMethods: {list(methods.keys())}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = run_experiment(
        db_ids=db_ids,
        schemas=bird.schemas,
        pairs=bird.pairs,
        db_path_fn=bird.get_db_path,
        methods=methods,
        eval_model=eval_model,
        detail_logger=dl,
        max_workers=max_workers,
    )
    total_time = time.time() - t0

    # ===== Report =====
    print(f"\n{'='*70}")
    print(f"RESULTS (total time: {total_time:.1f}s)")
    print(f"{'='*70}")

    for method_name, agg in result.method_aggregates.items():
        print(f"\n--- {method_name} ---")
        print(f"  Databases: {agg['total_databases']}")
        print(f"  Avg ExAcc before: {agg['avg_exacc_before']:.4f}")
        print(f"  Avg ExAcc after:  {agg['avg_exacc_after']:.4f}")
        print(f"  Avg Delta:        {agg['avg_delta']:+.4f}")
        print(f"  Avg Precision:    {agg['avg_refinement_precision']:.4f}")

    # ===== Save =====
    save_experiment(result, args.output)
    print(f"\nResults saved to {args.output}/")

    # Print comparison table
    from egrefine.experiment import _build_comparison_table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE (LaTeX)")
    print(f"{'='*70}")
    print(_build_comparison_table(result))


def _apply_column_limit(max_columns: int):
    """Monkey-patch pruner to limit candidate count."""
    from egrefine.phase1.pruner import prune as _original_prune
    import egrefine.pipeline
    import egrefine.baselines.llm_direct
    import egrefine.baselines.llm_cot

    def limited_prune(schema, config, **kwargs):
        result = _original_prune(schema, config, **kwargs)
        if len(result.candidates) > max_columns:
            result.candidates = result.candidates[:max_columns]
        return result

    src.pipeline.prune = limited_prune
    src.baselines.llm_direct.prune = limited_prune
    src.baselines.llm_cot.prune = limited_prune


if __name__ == "__main__":
    main()
