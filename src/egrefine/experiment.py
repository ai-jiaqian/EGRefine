"""Experiment orchestration: run multiple methods across multiple databases."""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from egrefine.data.schema import Schema, NLSQLPair
from egrefine.evaluate import EvalResult, evaluate_refinement, aggregate_results, to_latex_table
from egrefine.pipeline import PipelineResult
from egrefine.phase3.text2sql_runner import Text2SQLModel

logger = logging.getLogger(__name__)


@dataclass
class MethodResult:
    """Result of one method on one database."""
    method: str
    db_id: str
    pipeline_result: PipelineResult
    eval_result: EvalResult
    elapsed: float


@dataclass
class ExperimentResult:
    """Full experiment results across all databases and methods."""
    method_results: Dict[str, List[MethodResult]]  # method_name -> per-db results
    method_evals: Dict[str, List[EvalResult]]       # method_name -> per-db EvalResults
    method_aggregates: Dict[str, dict]               # method_name -> aggregated metrics

    def to_dict(self) -> dict:
        result = {}
        for method, results in self.method_results.items():
            result[method] = {
                "per_db": [
                    {
                        "db_id": r.db_id,
                        "elapsed": round(r.elapsed, 2),
                        "eval": r.eval_result.to_dict(),
                        "pipeline_stats": r.pipeline_result.statistics,
                    }
                    for r in results
                ],
                "aggregate": self.method_aggregates[method],
            }
        return result


# Type alias: a method is a callable that takes (schema, pairs, db_path) and returns PipelineResult
MethodFn = Callable[..., PipelineResult]


def run_experiment(
    db_ids: List[str],
    schemas: Dict[str, Schema],
    pairs: List[NLSQLPair],
    db_path_fn: Callable[[str], str],
    methods: Dict[str, MethodFn],
    eval_model: Text2SQLModel,
    max_columns: Optional[int] = None,
    detail_logger=None,
    max_workers: int = 1,
) -> ExperimentResult:
    """Run all methods on all databases and evaluate.

    Args:
        db_ids: List of database IDs to process.
        schemas: {db_id: Schema}.
        pairs: All NL-SQL pairs.
        db_path_fn: Function that maps db_id -> SQLite path.
        methods: {method_name: callable(schema, pairs, db_path) -> PipelineResult}.
        eval_model: Text-to-SQL model for evaluation.
        max_columns: If set, limit pruner output (for cost control).
        detail_logger: Optional DetailLogger for saving per-query evaluation details.

    Returns:
        ExperimentResult with all results aggregated.
    """
    collect_details = detail_logger is not None
    method_results: Dict[str, List[MethodResult]] = {m: [] for m in methods}
    method_evals: Dict[str, List[EvalResult]] = {m: [] for m in methods}

    for db_id in db_ids:
        if db_id not in schemas:
            logger.warning("Skipping %s: schema not found", db_id)
            continue

        schema = schemas[db_id]
        db_path = db_path_fn(db_id)
        db_pairs = [p for p in pairs if p.db_id == db_id]

        if not db_pairs:
            logger.warning("Skipping %s: no NL-SQL pairs", db_id)
            continue

        logger.info(
            "\n[%s] %d tables, %d columns, %d queries",
            db_id, len(schema.tables), len(schema.all_columns), len(db_pairs),
        )

        for method_name, method_fn in methods.items():
            logger.info("[%s] Running %s...", db_id, method_name)
            t0 = time.time()

            try:
                pipeline_result = method_fn(schema, db_pairs, db_path)
                elapsed = time.time() - t0

                eval_result, eval_details = evaluate_refinement(
                    schema=schema,
                    pairs=db_pairs,
                    model=eval_model,
                    db_path=db_path,
                    refinement_results=pipeline_result.refinements,
                    collect_details=collect_details,
                    max_workers=max_workers,
                )

                if detail_logger and eval_details:
                    detail_logger.save_evaluation(db_id, method_name, eval_details)

                mr = MethodResult(
                    method=method_name,
                    db_id=db_id,
                    pipeline_result=pipeline_result,
                    eval_result=eval_result,
                    elapsed=elapsed,
                )
                method_results[method_name].append(mr)
                method_evals[method_name].append(eval_result)

                logger.info(
                    "[%s] %s: before=%.3f after=%.3f delta=%+.3f (%.1fs)",
                    db_id, method_name,
                    eval_result.exacc_before, eval_result.exacc_after,
                    eval_result.delta, elapsed,
                )

            except Exception as e:
                logger.error("[%s] %s FAILED: %s", db_id, method_name, e)

    # Aggregate
    method_aggregates = {}
    for method_name in methods:
        method_aggregates[method_name] = aggregate_results(method_evals[method_name])

    return ExperimentResult(
        method_results=method_results,
        method_evals=method_evals,
        method_aggregates=method_aggregates,
    )


def save_experiment(result: ExperimentResult, output_dir: str):
    """Save experiment results to JSON and LaTeX files."""
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Saved results to %s", json_path)

    # Per-method LaTeX tables
    for method_name, evals in result.method_evals.items():
        if evals:
            latex = to_latex_table(evals, caption=method_name)
            latex_path = os.path.join(output_dir, f"table_{method_name}.tex")
            with open(latex_path, "w") as f:
                f.write(latex)

    # Comparison table: one row per method with aggregated metrics
    comparison = _build_comparison_table(result)
    comp_path = os.path.join(output_dir, "table_comparison.tex")
    with open(comp_path, "w") as f:
        f.write(comparison)
    logger.info("Saved comparison table to %s", comp_path)


def _build_comparison_table(result: ExperimentResult) -> str:
    """Build a LaTeX comparison table: one row per method."""
    lines = []
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Method & ExAcc$_{\\text{before}}$ & ExAcc$_{\\text{after}}$ & "
        "$\\Delta$ & Precision \\\\"
    )
    lines.append("\\midrule")

    for method_name, agg in result.method_aggregates.items():
        lines.append(
            f"{method_name} & {agg['avg_exacc_before']:.2f} & "
            f"{agg['avg_exacc_after']:.2f} & "
            f"{agg['avg_delta']:+.2f} & "
            f"{agg['avg_refinement_precision']:.2f} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)
