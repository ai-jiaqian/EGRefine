#!/usr/bin/env python3
"""T23: Cross-model generalization experiment.

Tests whether refinements found by model A also improve model B's ExAcc.
Reuses existing pipeline results — only runs evaluation with the second model.

Usage:
    # Use refinement from previous run_full, evaluate with deepseek
    PYTHONPATH=. python3 scripts/run_cross_model.py --dbs financial --max-columns 2

    # Specify custom output
    PYTHONPATH=. python3 scripts/run_cross_model.py --output results/cross_model
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.config import load_config
from egrefine.data.benchmark import BIRDLoader
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.generator import CandidateGenerator
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase3.text2sql_runner import SimpleLLMText2SQL
from egrefine.pipeline import run_pipeline
from egrefine.evaluate import evaluate_refinement, aggregate_results
from egrefine.detail_logger import DetailLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_cross_model(
    db_ids, schemas, pairs, db_path_fn,
    refinement_models, eval_models,
    phase1_config, phase2_config, phase3_config, phase4_config,
    candidate_generator, detail_logger=None, max_columns=None,
    max_workers=1,
):
    """Run cross-model generalization experiment.

    1. Run EGRefine pipeline with refinement_models (Phase 3 scoring).
    2. Evaluate the resulting refinements with each eval_model.

    Args:
        refinement_models: Models used for Phase 3 scoring.
        eval_models: Dict of {name: Text2SQLModel} for evaluation.

    Returns:
        Dict of {eval_model_name: {db_id: eval_result}}.
    """
    results = {}

    for db_id in db_ids:
        if db_id not in schemas:
            continue
        schema = schemas[db_id]
        db_path = db_path_fn(db_id)
        db_pairs = [p for p in pairs if p.db_id == db_id]
        if not db_pairs:
            continue

        # Step 1: Run pipeline with refinement models
        logger.info("[%s] Running EGRefine pipeline (refinement phase)...", db_id)
        t0 = time.time()
        pipeline_result = run_pipeline(
            schema=schema, pairs=db_pairs, db_path=db_path,
            models=refinement_models,
            phase1_config=phase1_config,
            phase2_config=phase2_config,
            phase3_config=phase3_config,
            phase4_config=phase4_config,
            candidate_generator=candidate_generator,
            detail_logger=detail_logger,
            max_workers=max_workers,
        )
        pipeline_time = time.time() - t0
        logger.info("[%s] Pipeline done in %.1fs", db_id, pipeline_time)

        # Step 2: Evaluate with each eval model
        for eval_name, eval_model in eval_models.items():
            logger.info("[%s] Evaluating with %s...", db_id, eval_name)
            t0 = time.time()
            collect = detail_logger is not None

            eval_result, eval_details = evaluate_refinement(
                schema=schema, pairs=db_pairs, model=eval_model,
                db_path=db_path,
                refinement_results=pipeline_result.refinements,
                collect_details=collect,
                max_workers=max_workers,
            )
            eval_time = time.time() - t0

            if detail_logger and eval_details:
                detail_logger.save_evaluation(db_id, f"cross_{eval_name}", eval_details)

            if eval_name not in results:
                results[eval_name] = []
            results[eval_name].append({
                "db_id": db_id,
                "eval_result": eval_result,
                "pipeline_stats": pipeline_result.statistics,
                "eval_time": round(eval_time, 2),
            })

            logger.info(
                "[%s] %s: before=%.3f after=%.3f delta=%+.3f (%.1fs)",
                db_id, eval_name,
                eval_result.exacc_before, eval_result.exacc_after,
                eval_result.delta, eval_time,
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="EGRefine Cross-Model Generalization")
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--dbs", nargs="*", default=None,
                        help="Databases to run (default: first DB)")
    parser.add_argument("--max-columns", type=int, default=None,
                        help="Limit columns per DB")
    parser.add_argument("--output", default="results/cross_model",
                        help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    bird = BIRDLoader(config["data"]["bird"]["path"])

    db_ids = args.dbs or bird.db_ids[:1]

    print(f"\n{'='*70}")
    print(f"CROSS-MODEL GENERALIZATION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Databases: {db_ids}")

    # Setup
    phase1_config = config["phase1"]
    phase1_config["signals"]["high_similarity"]["enabled"] = False

    llm_client = LLMClient(config["models"]["candidate_llm"])
    cache_dir = config["output"].get("cache_dir", "./cache")
    candidate_generator = CandidateGenerator(
        llm_client, cache_dir=cache_dir,
        sample_rows=config["phase2"]["sample_rows"],
        max_retries=config["models"]["candidate_llm"].get("max_retries", 3),
    )

    # Refinement models (from config text2sql list)
    text2sql_configs = config["models"]["text2sql"]
    refinement_models = [SimpleLLMText2SQL(tc) for tc in text2sql_configs]
    refinement_name = text2sql_configs[0]["name"]

    # Evaluation models: all configured text2sql models + cross-model entries
    eval_models = {}
    for tc in text2sql_configs:
        eval_models[tc["name"]] = SimpleLLMText2SQL(tc)

    cross_configs = config["models"].get("cross_eval", [])
    for tc in cross_configs:
        eval_models[tc["name"]] = SimpleLLMText2SQL(tc)

    print(f"Refinement model: {refinement_name}")
    print(f"Evaluation models: {list(eval_models.keys())}")
    print(f"Output: {args.output}")
    print(f"{'='*70}\n")

    if len(eval_models) < 2:
        print("WARNING: Only 1 evaluation model configured. "
              "Add 'cross_eval' entries in config to test generalization.")

    dl = DetailLogger(args.output)
    dl.save_config_snapshot(config)
    dl.setup_file_logging()

    if args.max_columns:
        from scripts.run_full import _apply_column_limit
        _apply_column_limit(args.max_columns)

    # Run
    t0 = time.time()
    max_workers = config.get("concurrency", {}).get("max_workers", 1)
    results = run_cross_model(
        db_ids=db_ids, schemas=bird.schemas, pairs=bird.pairs,
        db_path_fn=bird.get_db_path,
        refinement_models=refinement_models,
        eval_models=eval_models,
        phase1_config=phase1_config,
        phase2_config=config["phase2"],
        phase3_config=config["phase3"],
        phase4_config=config["phase4"],
        candidate_generator=candidate_generator,
        detail_logger=dl,
        max_workers=max_workers,
    )
    total_time = time.time() - t0

    # Report
    print(f"\n{'='*70}")
    print(f"CROSS-MODEL RESULTS (total: {total_time:.1f}s)")
    print(f"{'='*70}")
    print(f"Refinement model: {refinement_name}")

    save_data = {"refinement_model": refinement_name, "eval_results": {}}

    for eval_name, db_results in results.items():
        evals = [r["eval_result"] for r in db_results]
        agg = aggregate_results(evals)

        same_model = eval_name == refinement_name
        label = f"{eval_name} {'(same)' if same_model else '(cross)'}"

        print(f"\n--- Eval: {label} ---")
        for r in db_results:
            er = r["eval_result"]
            print(f"  [{er.db_id}] before={er.exacc_before:.4f} "
                  f"after={er.exacc_after:.4f} delta={er.delta:+.4f}")
        print(f"  Avg ExAcc before: {agg['avg_exacc_before']:.4f}")
        print(f"  Avg ExAcc after:  {agg['avg_exacc_after']:.4f}")
        print(f"  Avg Delta:        {agg['avg_delta']:+.4f}")

        save_data["eval_results"][eval_name] = {
            "is_same_model": same_model,
            "per_db": [
                {"db_id": r["db_id"], "eval": r["eval_result"].to_dict(),
                 "eval_time": r["eval_time"]}
                for r in db_results
            ],
            "aggregate": agg,
        }

    # LaTeX comparison
    print(f"\n{'='*70}")
    print("CROSS-MODEL TABLE (LaTeX)")
    print(f"{'='*70}")
    latex = _build_cross_model_table(refinement_name, results)
    print(latex)

    # Save
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "cross_model_results.json"), "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output, "table_cross_model.tex"), "w") as f:
        f.write(latex)
    print(f"\nResults saved to {args.output}/")


def _build_cross_model_table(refinement_model: str, results: dict) -> str:
    """Build LaTeX table: refinement model vs evaluation models."""
    lines = []
    lines.append("\\begin{tabular}{llrrr}")
    lines.append("\\toprule")
    lines.append(
        "Refinement & Evaluation & ExAcc$_{\\text{before}}$ & "
        "ExAcc$_{\\text{after}}$ & $\\Delta$ \\\\"
    )
    lines.append("\\midrule")

    for eval_name, db_results in results.items():
        evals = [r["eval_result"] for r in db_results]
        agg = aggregate_results(evals)
        lines.append(
            f"{refinement_model} & {eval_name} & "
            f"{agg['avg_exacc_before']:.2f} & "
            f"{agg['avg_exacc_after']:.2f} & "
            f"{agg['avg_delta']:+.2f} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
