"""T19: Evaluation module — ExAcc before/after, Recovery Rate, Refinement Precision."""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from egrefine.data.schema import Schema, NLSQLPair
from egrefine.phase3.executor import compare_results
from egrefine.phase3.scorer import QueryDetail, SelectionResult
from egrefine.phase3.text2sql_runner import Text2SQLModel
from egrefine.phase4.backmapper import backmap

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation result for one database."""
    db_id: str
    exacc_before: float
    exacc_after: float
    delta: float
    total_queries: int
    columns_changed: int
    columns_evaluated: int
    refinement_precision: float

    def to_dict(self) -> dict:
        return {
            "db_id": self.db_id,
            "exacc_before": self.exacc_before,
            "exacc_after": self.exacc_after,
            "delta": self.delta,
            "total_queries": self.total_queries,
            "columns_changed": self.columns_changed,
            "columns_evaluated": self.columns_evaluated,
            "refinement_precision": self.refinement_precision,
        }


@dataclass
class EvalDetails:
    """Detailed evaluation record with per-query predictions."""
    method: str
    db_id: str
    exacc_before: float
    exacc_after: float
    delta: float
    before_details: List[QueryDetail]
    after_details: List[QueryDetail]

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "db_id": self.db_id,
            "exacc_before": self.exacc_before,
            "exacc_after": self.exacc_after,
            "delta": self.delta,
            "before_details": [q.to_dict() for q in self.before_details],
            "after_details": [q.to_dict() for q in self.after_details],
        }


def evaluate_exacc(
    schema: Schema,
    pairs: List[NLSQLPair],
    model: Text2SQLModel,
    db_path: str,
    reverse_mapping: Optional[Dict[str, str]] = None,
    collect_details: bool = False,
    max_workers: int = 1,
    use_evidence: bool = False,
) -> Tuple[float, List[QueryDetail]]:
    """Compute ExAcc: run Text-to-SQL on all pairs and compare with gold.

    Args:
        schema: Schema to present to the model (original or refined).
        pairs: NL-SQL pairs with gold SQL.
        model: Text-to-SQL model.
        db_path: Path to database (SQLite path or mysql:// URI).
        reverse_mapping: If provided, backmap predicted SQL before execution.
        collect_details: If True, return per-query details.
        max_workers: Number of concurrent threads for LLM calls.
        use_evidence: If True, forward per-query BIRD evidence to the model.

    Returns:
        Tuple of (ExAcc score, list of QueryDetail).
    """
    if not pairs:
        return 0.0, []

    def _eval_one(pair):
        evidence = pair.evidence if use_evidence else ""
        pred_sql = model.generate(
            pair.nl, schema, db_path=db_path,
            column_mapping=reverse_mapping,
            evidence=evidence,
        )

        backmapped_sql = None
        exec_sql = pred_sql
        if reverse_mapping:
            backmapped_sql = backmap(pred_sql, reverse_mapping)
            exec_sql = backmapped_sql

        match = compare_results(exec_sql, pair.gold_sql, db_path)

        detail = None
        if collect_details:
            detail = QueryDetail(
                nl=pair.nl,
                gold_sql=pair.gold_sql,
                pred_sql=pred_sql,
                backmapped_sql=backmapped_sql,
                match=match,
            )
        return match, detail

    correct = 0
    details: List[QueryDetail] = []

    if max_workers > 1 and len(pairs) > 1:
        # Parallel evaluation: submit all queries concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(_eval_one, pair): i
                for i, pair in enumerate(pairs)
            }
            # Collect results preserving order for details
            results_by_idx = {}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results_by_idx[idx] = fut.result()

            for i in range(len(pairs)):
                match, detail = results_by_idx[i]
                if match:
                    correct += 1
                if detail:
                    details.append(detail)
    else:
        for pair in pairs:
            match, detail = _eval_one(pair)
            if match:
                correct += 1
            if detail:
                details.append(detail)

    exacc = correct / len(pairs)
    logger.info(
        "[%s] ExAcc = %.3f (%d/%d)",
        schema.db_id, exacc, correct, len(pairs),
    )
    return exacc, details


def evaluate_refinement(
    schema: Schema,
    pairs: List[NLSQLPair],
    model: Text2SQLModel,
    db_path: str,
    refinement_results: List[SelectionResult],
    collect_details: bool = False,
    max_workers: int = 1,
) -> Tuple[EvalResult, Optional[EvalDetails]]:
    """Evaluate refinement: compute ExAcc before/after and derived metrics.

    Args:
        schema: Original database schema.
        pairs: NL-SQL pairs for this database.
        model: Text-to-SQL model.
        db_path: Path to database (SQLite path or mysql:// URI).
        refinement_results: SelectionResult list from pipeline.
        collect_details: If True, return EvalDetails with per-query info.
        max_workers: Number of concurrent threads for LLM calls.

    Returns:
        Tuple of (EvalResult, EvalDetails or None).
    """
    db_pairs = [p for p in pairs if p.db_id == schema.db_id]

    # ExAcc before: original schema
    exacc_before, before_details = evaluate_exacc(
        schema, db_pairs, model, db_path,
        collect_details=collect_details,
        max_workers=max_workers,
    )

    # Build refined schema + reverse mapping
    changed = [r for r in refinement_results if r.was_changed]
    if changed:
        forward_mapping = {r.column.full_name: r.selected_name for r in changed}
        refined_schema = schema.apply_refinement(forward_mapping)
        reverse_mapping = {r.selected_name: r.column.name for r in changed}

        # ExAcc after: refined schema with backmap
        exacc_after, after_details = evaluate_exacc(
            refined_schema, db_pairs, model, db_path,
            reverse_mapping=reverse_mapping,
            collect_details=collect_details,
            max_workers=max_workers,
        )
    else:
        exacc_after = exacc_before
        after_details = before_details  # same

    delta = exacc_after - exacc_before

    # Refinement precision
    precision_hits = sum(1 for r in changed if r.delta > 0)
    refinement_precision = precision_hits / len(changed) if changed else 0.0

    eval_result = EvalResult(
        db_id=schema.db_id,
        exacc_before=round(exacc_before, 4),
        exacc_after=round(exacc_after, 4),
        delta=round(delta, 4),
        total_queries=len(db_pairs),
        columns_changed=len(changed),
        columns_evaluated=len(refinement_results),
        refinement_precision=round(refinement_precision, 4),
    )

    eval_details = None
    if collect_details:
        eval_details = EvalDetails(
            method="",  # caller fills this in
            db_id=schema.db_id,
            exacc_before=round(exacc_before, 4),
            exacc_after=round(exacc_after, 4),
            delta=round(delta, 4),
            before_details=before_details,
            after_details=after_details,
        )

    logger.info(
        "[%s] Before=%.3f After=%.3f Delta=%+.3f Precision=%.2f (%d changed/%d evaluated)",
        schema.db_id, exacc_before, exacc_after, delta,
        refinement_precision, len(changed), len(refinement_results),
    )
    return eval_result, eval_details


def aggregate_results(results: List[EvalResult]) -> dict:
    """Aggregate evaluation results across multiple databases."""
    if not results:
        return {
            "total_databases": 0,
            "total_queries": 0,
            "avg_exacc_before": 0.0,
            "avg_exacc_after": 0.0,
            "avg_delta": 0.0,
            "avg_refinement_precision": 0.0,
            "total_columns_changed": 0,
            "total_columns_evaluated": 0,
        }

    n = len(results)
    return {
        "total_databases": n,
        "total_queries": sum(r.total_queries for r in results),
        "avg_exacc_before": sum(r.exacc_before for r in results) / n,
        "avg_exacc_after": sum(r.exacc_after for r in results) / n,
        "avg_delta": sum(r.delta for r in results) / n,
        "avg_refinement_precision": sum(r.refinement_precision for r in results) / n,
        "total_columns_changed": sum(r.columns_changed for r in results),
        "total_columns_evaluated": sum(r.columns_evaluated for r in results),
    }


def to_latex_table(results: List[EvalResult], caption: str = "") -> str:
    """Generate a LaTeX table from evaluation results."""
    lines = []
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Database & \\#Queries & \\#Changed & "
        "ExAcc$_{\\text{before}}$ & ExAcc$_{\\text{after}}$ & "
        "$\\Delta$ & Precision \\\\"
    )
    lines.append("\\midrule")

    for r in results:
        lines.append(
            f"{r.db_id} & {r.total_queries} & {r.columns_changed} & "
            f"{r.exacc_before:.2f} & {r.exacc_after:.2f} & "
            f"{r.delta:+.2f} & {r.refinement_precision:.2f} \\\\"
        )

    if results:
        agg = aggregate_results(results)
        lines.append("\\midrule")
        lines.append(
            f"\\textbf{{Average}} & {agg['total_queries']} & "
            f"{agg['total_columns_changed']} & "
            f"{agg['avg_exacc_before']:.2f} & {agg['avg_exacc_after']:.2f} & "
            f"{agg['avg_delta']:+.2f} & {agg['avg_refinement_precision']:.2f} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)
