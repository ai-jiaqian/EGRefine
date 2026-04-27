#!/usr/bin/env python3
"""T22: Ablation experiment — measure contribution of each EGRefine component.

Ablation variants:
  wo_phase3:       Skip execution feedback (= llm_direct baseline, offline)
  wo_conservative: Remove conservative selection rule (reuse Phase 3 scores)
  wo_phase1:       Skip pruning, all non-PK columns are candidates (expensive)

Cost-aware design:
  - wo_phase3 and wo_conservative reuse logged Phase 3 scores (offline analysis
    + moderate-cost evaluation). No Phase 2/3 LLM calls needed.
  - wo_phase1 requires full pipeline re-run; limited to --dbs (default 1 DB).

Usage:
    # Offline analysis only (no LLM calls, reads from previous run_full output)
    PYTHONPATH=. python3 scripts/run_ablation.py --results-dir results/full --analyze

    # Run wo_conservative with live evaluation (moderate cost)
    PYTHONPATH=. python3 scripts/run_ablation.py --ablation wo_conservative --dbs financial

    # Run wo_phase1 on a single DB (expensive)
    PYTHONPATH=. python3 scripts/run_ablation.py --ablation wo_phase1 --dbs financial --max-columns 5
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
from egrefine.data.schema import Column, Schema
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.generator import CandidateGenerator
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase3.text2sql_runner import SimpleLLMText2SQL
from egrefine.pipeline import run_pipeline
from egrefine.evaluate import evaluate_refinement, aggregate_results, EvalResult
from egrefine.experiment import save_experiment, ExperimentResult, MethodResult
from egrefine.detail_logger import DetailLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Offline analysis: read logged Phase 3 scores, recompute
# ============================================================

def load_phase3_scores(results_dir: str, db_id: str) -> list:
    """Load phase3_scoring.json from a previous run."""
    path = os.path.join(results_dir, "per_db", db_id, "phase3_scoring.json")
    if not os.path.exists(path):
        logger.warning("Phase 3 scores not found: %s", path)
        return []
    with open(path) as f:
        return json.load(f)


def load_results_json(results_dir: str) -> dict:
    """Load results.json from a previous run."""
    path = os.path.join(results_dir, "results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def analyze_wo_conservative(results_dir: str, db_ids: list) -> dict:
    """Offline analysis: what would change without conservative rule.

    Reads phase3_scoring.json and recomputes selections.
    Returns analysis dict with per-DB and aggregate metrics.
    """
    analysis = {"per_db": {}, "summary": {}}
    total_extra_changes = 0
    total_columns = 0
    deltas_conservative = []
    deltas_no_conservative = []

    for db_id in db_ids:
        entries = load_phase3_scores(results_dir, db_id)
        if not entries:
            continue

        extra_changes = []
        for entry in entries:
            scores = entry.get("all_scores", {})
            if not scores:
                continue

            original_name = entry["original_name"]
            original_score = scores.get(original_name, 0.0)
            best_name = max(scores, key=scores.get)
            best_score = scores[best_name]
            current_selected = entry["selected_name"]

            # With conservative rule: current behavior
            was_changed = entry["was_changed"]

            # Without conservative rule: always pick best
            would_change = best_name != original_name

            if would_change and not was_changed:
                extra_changes.append({
                    "column": entry["column"],
                    "original": original_name,
                    "would_select": best_name,
                    "delta": round(best_score - original_score, 4),
                    "note": "negative delta" if best_score < original_score else "tied",
                })

            # Collect deltas for aggregate
            if was_changed:
                deltas_conservative.append(entry["delta"])
            if would_change:
                deltas_no_conservative.append(round(best_score - original_score, 4))

            total_columns += 1

        total_extra_changes += len(extra_changes)
        analysis["per_db"][db_id] = {
            "total_scored": len(entries),
            "currently_changed": sum(1 for e in entries if e["was_changed"]),
            "would_change_extra": len(extra_changes),
            "extra_changes": extra_changes,
        }

    analysis["summary"] = {
        "total_columns_scored": total_columns,
        "columns_changed_with_conservative": len(deltas_conservative),
        "columns_changed_without_conservative": len(deltas_no_conservative),
        "extra_changes": total_extra_changes,
        "avg_delta_conservative": (
            sum(deltas_conservative) / len(deltas_conservative)
            if deltas_conservative else 0.0
        ),
        "avg_delta_no_conservative": (
            sum(deltas_no_conservative) / len(deltas_no_conservative)
            if deltas_no_conservative else 0.0
        ),
    }
    return analysis


def analyze_wo_phase3(results_dir: str) -> dict:
    """Offline analysis: reference llm_direct results from previous experiment."""
    results = load_results_json(results_dir)
    info = {"note": "wo_phase3 is equivalent to llm_direct baseline"}
    for method in ["llm_direct", "egrefine"]:
        if method in results:
            agg = results[method].get("aggregate", {})
            info[method] = {
                "avg_exacc_before": agg.get("avg_exacc_before"),
                "avg_exacc_after": agg.get("avg_exacc_after"),
                "avg_delta": agg.get("avg_delta"),
            }
    return info


def run_offline_analysis(results_dir: str, db_ids: list):
    """Run all offline analyses and print report."""
    print(f"\n{'='*70}")
    print("ABLATION — OFFLINE ANALYSIS")
    print(f"{'='*70}")
    print(f"Source: {results_dir}\n")

    # wo_phase3
    print("--- w/o Phase 3 (= llm_direct) ---")
    wo3 = analyze_wo_phase3(results_dir)
    print(f"  Note: {wo3['note']}")
    for method in ["egrefine", "llm_direct"]:
        if method in wo3:
            m = wo3[method]
            print(f"  {method}: ExAcc {m['avg_exacc_before']:.4f} → "
                  f"{m['avg_exacc_after']:.4f} (Δ={m['avg_delta']:+.4f})")

    # wo_conservative
    print(f"\n--- w/o Conservative Rule ---")
    wo_cons = analyze_wo_conservative(results_dir, db_ids)
    s = wo_cons["summary"]
    print(f"  Columns scored: {s['total_columns_scored']}")
    print(f"  Changed WITH conservative: {s['columns_changed_with_conservative']}")
    print(f"  Changed WITHOUT conservative: {s['columns_changed_without_conservative']}")
    print(f"  Extra changes (conservative prevented): {s['extra_changes']}")
    print(f"  Avg delta (conservative): {s['avg_delta_conservative']:+.4f}")
    print(f"  Avg delta (no conservative): {s['avg_delta_no_conservative']:+.4f}")

    if s["extra_changes"] > 0:
        print("\n  Extra changes that conservative rule prevented:")
        for db_id, db_info in wo_cons["per_db"].items():
            for ch in db_info["extra_changes"]:
                print(f"    {ch['column']}: {ch['original']} → {ch['would_select']} "
                      f"(Δ={ch['delta']:+.4f}, {ch['note']})")

    print(f"\n{'='*70}")
    return {"wo_phase3": wo3, "wo_conservative": wo_cons}


# ============================================================
# Live ablation: run pipeline variants with evaluation
# ============================================================

def run_ablation_egrefine(
    db_ids, schemas, pairs, db_path_fn, models, eval_model,
    phase1_config, phase2_config, phase3_config, phase4_config,
    candidate_generator, detail_logger=None, max_workers=1,
):
    """Run standard EGRefine as same-run baseline for fair comparison."""
    method_results = []
    method_evals = []

    for db_id in db_ids:
        if db_id not in schemas:
            continue
        schema = schemas[db_id]
        db_path = db_path_fn(db_id)
        db_pairs = [p for p in pairs if p.db_id == db_id]
        if not db_pairs:
            continue

        logger.info("[%s] Running egrefine (baseline)...", db_id)
        t0 = time.time()

        pipeline_result = run_pipeline(
            schema=schema, pairs=db_pairs, db_path=db_path,
            models=models,
            phase1_config=phase1_config,
            phase2_config=phase2_config,
            phase3_config=phase3_config,
            phase4_config=phase4_config,
            candidate_generator=candidate_generator,
            detail_logger=detail_logger,
            max_workers=max_workers,
        )
        elapsed = time.time() - t0

        eval_result, eval_details = evaluate_refinement(
            schema=schema, pairs=db_pairs, model=eval_model,
            db_path=db_path, refinement_results=pipeline_result.refinements,
            collect_details=detail_logger is not None,
            max_workers=max_workers,
        )
        if detail_logger and eval_details:
            detail_logger.save_evaluation(db_id, "egrefine", eval_details)

        method_results.append(MethodResult(
            method="egrefine", db_id=db_id,
            pipeline_result=pipeline_result, eval_result=eval_result,
            elapsed=elapsed,
        ))
        method_evals.append(eval_result)

        logger.info(
            "[%s] egrefine: before=%.3f after=%.3f delta=%+.3f (%.1fs)",
            db_id, eval_result.exacc_before, eval_result.exacc_after,
            eval_result.delta, elapsed,
        )

    return method_results, method_evals


def run_ablation_wo_conservative(
    db_ids, schemas, pairs, db_path_fn, models, eval_model,
    phase1_config, phase2_config, phase3_config, phase4_config,
    candidate_generator, detail_logger=None, max_workers=1,
):
    """Run EGRefine with conservative=False."""
    ablation_phase3 = {**phase3_config, "conservative": False}
    method_results = []
    method_evals = []

    for db_id in db_ids:
        if db_id not in schemas:
            continue
        schema = schemas[db_id]
        db_path = db_path_fn(db_id)
        db_pairs = [p for p in pairs if p.db_id == db_id]
        if not db_pairs:
            continue

        logger.info("[%s] Running wo_conservative...", db_id)
        t0 = time.time()

        pipeline_result = run_pipeline(
            schema=schema, pairs=db_pairs, db_path=db_path,
            models=models,
            phase1_config=phase1_config,
            phase2_config=phase2_config,
            phase3_config=ablation_phase3,
            phase4_config=phase4_config,
            candidate_generator=candidate_generator,
            detail_logger=detail_logger,
            max_workers=max_workers,
        )
        elapsed = time.time() - t0

        eval_result, eval_details = evaluate_refinement(
            schema=schema, pairs=db_pairs, model=eval_model,
            db_path=db_path, refinement_results=pipeline_result.refinements,
            collect_details=detail_logger is not None,
            max_workers=max_workers,
        )
        if detail_logger and eval_details:
            detail_logger.save_evaluation(db_id, "wo_conservative", eval_details)

        method_results.append(MethodResult(
            method="wo_conservative", db_id=db_id,
            pipeline_result=pipeline_result, eval_result=eval_result,
            elapsed=elapsed,
        ))
        method_evals.append(eval_result)

        logger.info(
            "[%s] wo_conservative: before=%.3f after=%.3f delta=%+.3f (%.1fs)",
            db_id, eval_result.exacc_before, eval_result.exacc_after,
            eval_result.delta, elapsed,
        )

    return method_results, method_evals


def run_ablation_wo_phase1(
    db_ids, schemas, pairs, db_path_fn, models, eval_model,
    phase1_config, phase2_config, phase3_config, phase4_config,
    candidate_generator, detail_logger=None, max_workers=1,
):
    """Run EGRefine with Phase 1 disabled (all non-PK columns as candidates)."""
    method_results = []
    method_evals = []

    for db_id in db_ids:
        if db_id not in schemas:
            continue
        schema = schemas[db_id]
        db_path = db_path_fn(db_id)
        db_pairs = [p for p in pairs if p.db_id == db_id]
        if not db_pairs:
            continue

        # All non-PK columns as candidates
        skip_pk = phase1_config.get("skip_primary_keys", True)
        all_candidates = [
            c for c in schema.all_columns
            if not (skip_pk and c.is_pk)
        ]

        logger.info(
            "[%s] Running wo_phase1: %d/%d columns as candidates",
            db_id, len(all_candidates), len(schema.all_columns),
        )
        t0 = time.time()

        pipeline_result = run_pipeline(
            schema=schema, pairs=db_pairs, db_path=db_path,
            models=models,
            phase1_config=phase1_config,
            phase2_config=phase2_config,
            phase3_config=phase3_config,
            phase4_config=phase4_config,
            candidate_generator=candidate_generator,
            detail_logger=detail_logger,
            override_candidates=all_candidates,
            max_workers=max_workers,
        )
        elapsed = time.time() - t0

        eval_result, eval_details = evaluate_refinement(
            schema=schema, pairs=db_pairs, model=eval_model,
            db_path=db_path, refinement_results=pipeline_result.refinements,
            collect_details=detail_logger is not None,
            max_workers=max_workers,
        )
        if detail_logger and eval_details:
            detail_logger.save_evaluation(db_id, "wo_phase1", eval_details)

        method_results.append(MethodResult(
            method="wo_phase1", db_id=db_id,
            pipeline_result=pipeline_result, eval_result=eval_result,
            elapsed=elapsed,
        ))
        method_evals.append(eval_result)

        logger.info(
            "[%s] wo_phase1: before=%.3f after=%.3f delta=%+.3f (%.1fs)",
            db_id, eval_result.exacc_before, eval_result.exacc_after,
            eval_result.delta, elapsed,
        )

    return method_results, method_evals


def main():
    parser = argparse.ArgumentParser(description="EGRefine Ablation Experiment")
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--results-dir", default="results/full",
                        help="Directory with previous run_full results (for offline analysis)")
    parser.add_argument("--analyze", action="store_true",
                        help="Offline analysis only, no LLM calls")
    parser.add_argument("--ablation", choices=["wo_conservative", "wo_phase1", "all"],
                        default="all",
                        help="Which ablation to run (default: all)")
    parser.add_argument("--dbs", nargs="*", default=None,
                        help="Databases to run on (default: all for analysis, first DB for live)")
    parser.add_argument("--max-columns", type=int, default=None,
                        help="Limit columns per DB (for wo_phase1 cost control)")
    parser.add_argument("--output", default="results/ablation",
                        help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    bird = BIRDLoader(config["data"]["bird"]["path"])

    # Determine DB list
    if args.dbs:
        db_ids = args.dbs
    elif args.analyze:
        db_ids = bird.db_ids
    else:
        # For live runs, default to first DB to control cost
        db_ids = bird.db_ids[:1]
        logger.info("Defaulting to 1 DB for live ablation: %s", db_ids)

    # ===== Offline analysis mode =====
    if args.analyze:
        analysis = run_offline_analysis(args.results_dir, db_ids)
        # Save analysis
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "offline_analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis saved to {args.output}/offline_analysis.json")
        return

    # ===== Live ablation mode =====
    print(f"\n{'='*70}")
    print(f"ABLATION EXPERIMENT (live)")
    print(f"{'='*70}")
    print(f"Ablation: {args.ablation}")
    print(f"Databases: {db_ids}")
    print(f"Output: {args.output}")

    # Setup components
    phase1_config = config["phase1"]
    phase1_config["signals"]["high_similarity"]["enabled"] = False

    llm_client = LLMClient(config["models"]["candidate_llm"])
    cache_dir = config["output"].get("cache_dir", "./cache")
    candidate_generator = CandidateGenerator(
        llm_client, cache_dir=cache_dir,
        sample_rows=config["phase2"]["sample_rows"],
        max_retries=config["models"]["candidate_llm"].get("max_retries", 3),
    )

    text2sql_configs = config["models"]["text2sql"]
    models = [SimpleLLMText2SQL(tc) for tc in text2sql_configs]
    eval_model = models[0]

    dl = DetailLogger(args.output)
    dl.save_config_snapshot(config)
    log_handler = dl.setup_file_logging()

    if args.max_columns:
        from scripts.run_full import _apply_column_limit
        _apply_column_limit(args.max_columns)

    # Collect all results
    all_method_results = {}
    all_method_evals = {}

    max_workers = config.get("concurrency", {}).get("max_workers", 1)
    common_args = dict(
        db_ids=db_ids, schemas=bird.schemas, pairs=bird.pairs,
        db_path_fn=bird.get_db_path, models=models, eval_model=eval_model,
        phase1_config=phase1_config, phase2_config=config["phase2"],
        phase3_config=config["phase3"], phase4_config=config["phase4"],
        candidate_generator=candidate_generator, detail_logger=dl,
        max_workers=max_workers,
    )

    # --- Always run standard EGRefine first as same-run baseline ---
    print(f"\n--- Running: egrefine (same-run baseline) ---")
    results, evals = run_ablation_egrefine(**common_args)
    all_method_results["egrefine"] = results
    all_method_evals["egrefine"] = evals

    # --- wo_conservative ---
    if args.ablation in ("wo_conservative", "all"):
        print(f"\n--- Running: wo_conservative ---")
        results, evals = run_ablation_wo_conservative(**common_args)
        all_method_results["wo_conservative"] = results
        all_method_evals["wo_conservative"] = evals

    # --- wo_phase1 ---
    if args.ablation in ("wo_phase1", "all"):
        print(f"\n--- Running: wo_phase1 ---")
        results, evals = run_ablation_wo_phase1(**common_args)
        all_method_results["wo_phase1"] = results
        all_method_evals["wo_phase1"] = evals

    # ===== Report =====
    print(f"\n{'='*70}")
    print(f"ABLATION RESULTS")
    print(f"{'='*70}")

    method_aggregates = {}
    for method_name, evals in all_method_evals.items():
        agg = aggregate_results(evals)
        method_aggregates[method_name] = agg
        print(f"\n--- {method_name} ---")
        print(f"  Databases: {agg['total_databases']}")
        print(f"  Avg ExAcc before: {agg['avg_exacc_before']:.4f}")
        print(f"  Avg ExAcc after:  {agg['avg_exacc_after']:.4f}")
        print(f"  Avg Delta:        {agg['avg_delta']:+.4f}")
        print(f"  Avg Precision:    {agg['avg_refinement_precision']:.4f}")

    # Save
    os.makedirs(args.output, exist_ok=True)
    save_data = {
        "ablation_results": {
            method: {
                "per_db": [
                    {"db_id": r.db_id, "elapsed": round(r.elapsed, 2),
                     "eval": r.eval_result.to_dict()}
                    for r in results
                ],
                "aggregate": method_aggregates.get(method, {}),
            }
            for method, results in all_method_results.items()
        },
    }
    with open(os.path.join(args.output, "ablation_results.json"), "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
