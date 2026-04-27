#!/usr/bin/env python3
"""Stage 1 entry point: run EGRefine Phase 1-4 pipeline on BIRD databases.

Saves static artifacts per database: views.sql, refined_tables.json,
orig_table_map.json, statistics.json, pipeline_result.json.

Usage:
    PYTHONPATH=. python3 scripts/run_refine.py
    PYTHONPATH=. python3 scripts/run_refine.py --config config/local.yaml
    PYTHONPATH=. python3 scripts/run_refine.py --dbs financial california_schools
    PYTHONPATH=. python3 scripts/run_refine.py --max-columns 5 --output results/refine
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.config import load_config
from egrefine.data.benchmark import BIRDLoader, BEAVERLoader, DrSpiderLoader
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.generator import CandidateGenerator
from egrefine.phase3.c3_runner import C3Text2SQL
from egrefine.phase3.dinsql_runner import DINSQLText2SQL
from egrefine.phase3.macsql_runner import MACSQLText2SQL
from egrefine.phase3.text2sql_runner import SimpleLLMText2SQL
from egrefine.phase4.view_synthesis import generate_refined_tables_json
from egrefine.pipeline import run_pipeline
from egrefine.baselines.llm_direct import run_llm_direct
from egrefine.baselines.llm_cot import run_llm_cot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _apply_column_limit(max_columns: int):
    """Monkey-patch pruner (rule + LLM variants) to limit candidate count."""
    from egrefine.phase1.pruner import prune as _original_prune
    from egrefine.phase1.pruner import prune_llm as _original_prune_llm
    import egrefine.pipeline

    def limited_prune(schema, config, **kwargs):
        result = _original_prune(schema, config, **kwargs)
        if len(result.candidates) > max_columns:
            result.candidates = result.candidates[:max_columns]
        return result

    def limited_prune_llm(schema, config, *args, **kwargs):
        result = _original_prune_llm(schema, config, *args, **kwargs)
        if len(result.candidates) > max_columns:
            result.candidates = result.candidates[:max_columns]
        return result

    src.pipeline.prune = limited_prune
    src.pipeline.prune_llm = limited_prune_llm


def _save_artifacts(result, schema, output_dir: str):
    """Save pipeline artifacts for one database to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # pipeline_result.json — full pipeline result
    with open(os.path.join(output_dir, "pipeline_result.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # views.sql — ALTER TABLE + CREATE VIEW statements
    with open(os.path.join(output_dir, "views.sql"), "w") as f:
        f.write("\n\n".join(result.views))

    # statistics.json
    with open(os.path.join(output_dir, "statistics.json"), "w") as f:
        json.dump(result.statistics, f, indent=2)

    # orig_table_map.json — mapping from original table names to backing names
    with open(os.path.join(output_dir, "orig_table_map.json"), "w") as f:
        json.dump(result.orig_table_map, f, indent=2)

    # refined_tables.json — full refined schema description
    # Format compatible with load_refined_schema() in src/eval/evaluator.py
    refined_tables = generate_refined_tables_json(schema, result.refinements)
    with open(os.path.join(output_dir, "refined_tables.json"), "w") as f:
        json.dump(refined_tables, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="EGRefine Stage 1: Run Phase 1-4 pipeline on BIRD databases"
    )
    parser.add_argument("--config", default="config/local.yaml",
                        help="Path to config YAML (default: config/local.yaml)")
    parser.add_argument("--benchmark", default="bird",
                        choices=["bird", "beaver", "drspider_abbr", "drspider_syn"],
                        help="Benchmark dataset (default: bird)")
    parser.add_argument("--dbs", nargs="*", default=None,
                        help="Specific database IDs to run (default: all)")
    parser.add_argument("--method", default="egrefine",
                        choices=["egrefine", "llm-direct", "llm-cot"],
                        help="Refinement method (default: egrefine)")
    parser.add_argument("--no-conservative", action="store_true",
                        help="Disable conservative selection rule (ablation)")
    parser.add_argument("--no-pruning", action="store_true",
                        help="Skip Phase 1 filtering; pass all non-FK non-shared columns "
                             "as candidates (ablation: w/o Phase 1)")
    parser.add_argument("--verify-algorithms", default="c3",
                        choices=["c3", "dinsql", "macsql",
                                 "c3+dinsql", "c3+macsql", "dinsql+macsql"],
                        help="Phase 3 verification algorithm(s) (default: c3). "
                             "Use '+' to combine algorithms for joint scoring.")
    parser.add_argument("--max-columns", type=int, default=None,
                        help="Limit Phase 1 candidates per DB (saves API cost)")
    parser.add_argument("--output", default="results/refine",
                        help="Output directory (default: results/refine)")
    parser.add_argument("--use-evidence", action="store_true", default=False,
                        help="Forward BIRD per-query evidence/hint to the "
                             "Text-to-SQL models during Phase 3 scoring. "
                             "Default: off (enterprise-scenario framing). "
                             "For BIRD evidence ablation experiments.")
    args = parser.parse_args()

    config = load_config(args.config)

    # ===== Load benchmark dataset =====
    if args.benchmark == "bird":
        loader = BIRDLoader(config["data"]["bird"]["path"])
    elif args.benchmark == "beaver":
        beaver_cfg = config["data"]["beaver"]
        loader = BEAVERLoader(
            beaver_cfg["path"],
            split=beaver_cfg.get("split", "nw"),
            mysql_config=beaver_cfg.get("mysql"),
        )
    elif args.benchmark in ("drspider_abbr", "drspider_syn"):
        ds_cfg = config["data"][args.benchmark]
        loader = DrSpiderLoader(ds_cfg["path"])
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    db_ids = args.dbs or loader.db_ids

    # ===== Apply --no-conservative =====
    if args.no_conservative:
        config["phase3"]["conservative"] = False

    print(f"\n{'='*60}")
    print("EGRefine Stage 1 — Schema Refinement Pipeline")
    print(f"{'='*60}")
    print(f"  Method:      {args.method}")
    print(f"  Databases:   {len(db_ids)} ({', '.join(db_ids)})")
    print(f"  Total pairs: {len(loader.pairs)}")
    print(f"  Max columns: {args.max_columns or 'unlimited'}")
    print(f"  Output:      {args.output}")

    # ===== Setup shared components =====
    phase1_config = config["phase1"]

    # Phase 1 LLM client (for method="llm")
    phase1_llm_client = None
    if phase1_config.get("method") == "llm":
        p1_model_cfg = phase1_config.get("model", {})
        if p1_model_cfg.get("base_url"):
            phase1_llm_client = LLMClient(p1_model_cfg)
            print(f"  Phase 1:     LLM mode ({p1_model_cfg.get('model_name', '?')})")
        else:
            logger.warning("phase1.method=llm but no model config, falling back to rule-based")
            phase1_config["method"] = "rule"

    # Disable embedding similarity if no embedding model configured (rule mode only)
    if phase1_config.get("method", "rule") == "rule":
        if not config.get("models", {}).get("embedding"):
            phase1_config.setdefault("signals", {}).setdefault("high_similarity", {})["enabled"] = False

    llm_config = config["models"]["candidate_llm"]
    llm_client = LLMClient(llm_config)
    cache_dir = config["output"].get("cache_dir", "./cache")

    # Multi-algorithm verification uses a separate cache to avoid mixing
    # with single-algorithm cached scores (different scoring = different results)
    if args.verify_algorithms != "c3":
        cache_dir = cache_dir + f"_{args.verify_algorithms.replace('+', '_')}"
        logger.info("Multi-algorithm mode: cache_dir=%s", cache_dir)

    # Evidence mode uses a separate cache: per-query BIRD evidence changes the
    # prompts seen by the Phase 3 scoring model, so scores differ from the
    # non-evidence cache and must not be reused across the boundary.
    if args.use_evidence:
        cache_dir = cache_dir + "_evidence"
        logger.info("Evidence mode: cache_dir=%s", cache_dir)

    # Path for loading database_description CSVs in Phase 2 (BIRD only)
    desc_base_path = None
    if args.benchmark == "bird":
        desc_base_path = config["data"]["bird"]["path"]
    # Dr.Spider and BEAVER have no database_description CSVs

    candidate_generator = CandidateGenerator(
        llm_client,
        cache_dir=cache_dir,
        sample_rows=config["phase2"]["sample_rows"],
        max_retries=llm_config.get("max_retries", 3),
        desc_base_path=desc_base_path,
    )

    text2sql_configs = config["models"]["text2sql"]
    if not isinstance(text2sql_configs, list):
        text2sql_configs = [text2sql_configs]

    # Build Phase 3 verification models based on --verify-algorithms
    desc_dir = None
    if args.benchmark == "bird":
        desc_dir = os.path.join(config["data"]["bird"]["path"], "dev_databases")

    verify_algos = args.verify_algorithms.split("+")
    models = []
    for algo in verify_algos:
        if algo == "c3":
            # C3: one instance per text2sql config (typically one LLM backend)
            for tc in text2sql_configs:
                models.append(C3Text2SQL(tc, num_samples=3, sample_rows=3, desc_dir=desc_dir))
        elif algo == "dinsql":
            # DIN-SQL: one instance per text2sql config
            for tc in text2sql_configs:
                models.append(DINSQLText2SQL(tc, self_correction=True))
        elif algo == "macsql":
            # MAC-SQL: one instance per text2sql config
            for tc in text2sql_configs:
                models.append(MACSQLText2SQL(tc))
        else:
            raise ValueError(f"Unknown verification algorithm: {algo}")

    algo_names = [type(m).__name__ for m in models]

    max_workers = config.get("concurrency", {}).get("max_workers", 1)

    if args.method == "egrefine":
        print(f"  Text-to-SQL: {[tc.get('name', tc['model_name']) for tc in text2sql_configs]}")
        print(f"  Verify:      {args.verify_algorithms} → {len(models)} model(s): {algo_names}")
    print(f"  Concurrency: max_workers={max_workers}")
    print(f"{'='*60}\n")

    # ===== Apply column limit if requested =====
    if args.max_columns:
        _apply_column_limit(args.max_columns)

    # ===== Run pipeline for each database =====
    total_start = time.time()
    summary = []

    for i, db_id in enumerate(db_ids, 1):
        if db_id not in loader.schemas:
            logger.warning("Schema not found for %s, skipping", db_id)
            continue

        schema = loader.schemas[db_id]
        pairs = loader.pairs
        db_path = loader.get_db_path(db_id)

        print(f"[{i}/{len(db_ids)}] Processing {db_id} ...")
        t0 = time.time()

        try:
            if args.method == "llm-direct":
                result = run_llm_direct(
                    schema=schema,
                    pairs=pairs,
                    db_path=db_path,
                    phase1_config=phase1_config,
                    phase2_config=config["phase2"],
                    candidate_generator=candidate_generator,
                    max_workers=max_workers,
                    phase1_llm_client=phase1_llm_client,
                )
            elif args.method == "llm-cot":
                result = run_llm_cot(
                    schema=schema,
                    pairs=pairs,
                    db_path=db_path,
                    phase1_config=phase1_config,
                    phase2_config=config["phase2"],
                    candidate_generator=candidate_generator,
                    llm_client=llm_client,
                    max_workers=max_workers,
                    phase1_llm_client=phase1_llm_client,
                )
            else:
                override_candidates = None
                if args.no_pruning:
                    # Ablation: skip Phase 1 filtering entirely. Still apply
                    # Structural Exclusion (FK / same-name-same-type) for
                    # correctness — otherwise FK rename without PK driver
                    # breaks VIEW semantics.
                    from egrefine.phase1.llm_screener import structural_exclusion
                    skip_set, _, _, _ = structural_exclusion(schema)
                    override_candidates = [
                        c for c in schema.all_columns
                        if c.full_name not in skip_set and not c.is_pk
                    ]
                    logger.info(
                        "[%s] --no-pruning: %d candidates (all non-PK non-FK)",
                        db_id, len(override_candidates),
                    )
                result = run_pipeline(
                    schema=schema,
                    pairs=pairs,
                    db_path=db_path,
                    models=models,
                    phase1_config=phase1_config,
                    phase2_config=config["phase2"],
                    phase3_config=config["phase3"],
                    phase4_config=config["phase4"],
                    candidate_generator=candidate_generator,
                    override_candidates=override_candidates,
                    max_workers=max_workers,
                    cache_dir=cache_dir,
                    phase1_llm_client=phase1_llm_client,
                    use_evidence=args.use_evidence,
                )

            elapsed = time.time() - t0
            db_output_dir = os.path.join(args.output, db_id)
            _save_artifacts(result, schema, db_output_dir)

            n_refined = result.statistics.get("columns_refined", 0)
            n_total = result.statistics.get("total_columns", 0)
            avg_delta = result.statistics.get("avg_delta", 0.0)

            summary.append({
                "db_id": db_id,
                "columns_refined": n_refined,
                "total_columns": n_total,
                "avg_delta": avg_delta,
                "elapsed": round(elapsed, 1),
            })

            print(f"  -> {n_refined}/{n_total} columns refined, "
                  f"avg delta={avg_delta:.4f}, "
                  f"time={elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            logger.error("Failed on %s after %.1fs: %s", db_id, elapsed, e)
            summary.append({
                "db_id": db_id,
                "error": str(e),
                "elapsed": round(elapsed, 1),
            })
            print(f"  -> ERROR: {e}")

    total_elapsed = time.time() - total_start

    # ===== Print summary =====
    print(f"\n{'='*60}")
    print(f"SUMMARY  (total time: {total_elapsed:.1f}s)")
    print(f"{'='*60}")

    for s in summary:
        if "error" in s:
            print(f"  {s['db_id']:30s}  ERROR  ({s['elapsed']}s)")
        else:
            print(f"  {s['db_id']:30s}  "
                  f"{s['columns_refined']:2d} refined / {s['total_columns']:3d} cols  "
                  f"delta={s['avg_delta']:+.4f}  ({s['elapsed']}s)")

    # Save summary
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump({"databases": summary, "total_elapsed": round(total_elapsed, 1)}, f, indent=2)

    print(f"\nArtifacts saved to {args.output}/")


if __name__ == "__main__":
    main()
