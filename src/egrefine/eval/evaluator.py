"""Stage 2 evaluator — compute ExAcc on original vs refined schemas.

For one database + one Text-to-SQL method, runs the model on both the
original and refined schemas, compares execution results against gold
SQL, and reports the delta in execution accuracy.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.eval.db_setup import remap_gold_sql
from egrefine.phase3.executor import compare_results

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Per-query evaluation detail."""

    nl: str
    gold_sql: str
    pred_sql: str
    match: bool


@dataclass
class EvalMethodResult:
    """Aggregated result for one method on one database."""

    method: str
    db_id: str
    exacc_original: float
    exacc_refined: float
    delta: float
    total_queries: int
    original_details: List[QueryResult] = field(default_factory=list)
    refined_details: List[QueryResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "db_id": self.db_id,
            "exacc_original": self.exacc_original,
            "exacc_refined": self.exacc_refined,
            "delta": self.delta,
            "total_queries": self.total_queries,
            "original_details": [
                {"nl": d.nl, "gold_sql": d.gold_sql, "pred_sql": d.pred_sql, "match": d.match}
                for d in self.original_details
            ],
            "refined_details": [
                {"nl": d.nl, "gold_sql": d.gold_sql, "pred_sql": d.pred_sql, "match": d.match}
                for d in self.refined_details
            ],
        }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _call_generate(
    model: Any, nl: str, schema: Schema, db_path: str, evidence: str = "",
) -> str:
    """Call model.generate() with optional BIRD evidence.

    By default ``evidence=""`` (empty), which reproduces the original
    "enterprise scenario" behavior where no hints are passed. When
    ``use_evidence=True`` is set on the eval pass, BIRD's per-query evidence
    field is forwarded to the Text-to-SQL model.
    """
    return model.generate(
        nl=nl, schema=schema, db_path=db_path, evidence=evidence,
    )


def _eval_one_pair(
    model: Any,
    pair: NLSQLPair,
    schema: Schema,
    db_path: str,
    table_map: Optional[Dict[str, str]] = None,
    use_evidence: bool = False,
) -> Tuple[str, bool]:
    """Evaluate a single pair: generate SQL, compare with gold.

    If table_map is provided, gold SQL is remapped to hit backing tables.
    If use_evidence is True, ``pair.evidence`` is forwarded to the model.
    Returns (pred_sql, match).
    """
    # BIRD evidence gating: only pass when explicitly enabled. Default
    # preserves the original "enterprise scenario" framing.
    evidence = pair.evidence if use_evidence else ""
    pred_sql = _call_generate(model, pair.nl, schema, db_path, evidence=evidence)
    gold_sql = remap_gold_sql(pair.gold_sql, table_map) if table_map else pair.gold_sql
    match = compare_results(pred_sql, gold_sql, db_path)
    return pred_sql, gold_sql, match


def _run_eval_pass(
    model: Any,
    pairs: List[NLSQLPair],
    schema: Schema,
    db_path: str,
    table_map: Optional[Dict[str, str]] = None,
    max_workers: int = 1,
    use_evidence: bool = False,
) -> Tuple[float, List[QueryResult]]:
    """Run one evaluation pass (original or refined).

    Returns (exacc, details).
    """
    details: List[QueryResult] = []

    if max_workers > 1 and len(pairs) > 1:
        results_by_idx: Dict[int, Tuple[str, str, bool]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(
                    _eval_one_pair, model, pair, schema, db_path, table_map,
                    use_evidence,
                ): i
                for i, pair in enumerate(pairs)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results_by_idx[idx] = fut.result()

        for i in range(len(pairs)):
            pred_sql, gold_sql, match = results_by_idx[i]
            details.append(QueryResult(
                nl=pairs[i].nl, gold_sql=gold_sql, pred_sql=pred_sql, match=match,
            ))
    else:
        for pair in pairs:
            pred_sql, gold_sql, match = _eval_one_pair(
                model, pair, schema, db_path, table_map,
                use_evidence=use_evidence,
            )
            details.append(QueryResult(
                nl=pair.nl, gold_sql=gold_sql, pred_sql=pred_sql, match=match,
            ))

    correct = sum(1 for d in details if d.match)
    exacc = correct / len(pairs) if pairs else 0.0
    return exacc, details


def evaluate_method(
    model: Any,
    pairs: List[NLSQLPair],
    original_schema: Schema,
    refined_schema: Schema,
    original_db_path: str,
    refined_db_path: str,
    table_map: Dict[str, str],
    method_name: str = "",
    max_workers: int = 1,
    use_evidence: bool = False,
) -> EvalMethodResult:
    """Evaluate a Text-to-SQL model on both original and refined schemas.

    Args:
        model: A Text-to-SQL model with a ``generate()`` method.
        pairs: NL-SQL pairs for one database.
        original_schema: The original database schema.
        refined_schema: The refined database schema (with renamed columns).
        original_db_path: Path to the original SQLite database.
        refined_db_path: Path to the refined SQLite database (with VIEWs applied).
        table_map: Mapping from original table names to backing table names
            (e.g. ``{"district": "_orig_district"}``), used to remap gold SQL
            for execution on the refined database.
        method_name: Human-readable method identifier for logging.
        max_workers: Number of concurrent threads for LLM calls.

    Returns:
        An :class:`EvalMethodResult` with ExAcc for both schemas and delta.
    """
    total = len(pairs)
    if total == 0:
        return EvalMethodResult(
            method=method_name,
            db_id=original_schema.db_id,
            exacc_original=0.0,
            exacc_refined=0.0,
            delta=0.0,
            total_queries=0,
        )

    # Evaluate on original schema (no table remapping)
    exacc_original, original_details = _run_eval_pass(
        model, pairs, original_schema, original_db_path,
        max_workers=max_workers, use_evidence=use_evidence,
    )

    # Evaluate on refined schema (gold SQL remapped to backing tables)
    exacc_refined, refined_details = _run_eval_pass(
        model, pairs, refined_schema, refined_db_path,
        table_map=table_map, max_workers=max_workers,
        use_evidence=use_evidence,
    )

    delta = exacc_refined - exacc_original

    logger.info(
        "[%s] db=%s  original=%.3f  refined=%.3f  delta=%+.3f  n=%d",
        method_name, original_schema.db_id,
        exacc_original, exacc_refined, delta, total,
    )

    return EvalMethodResult(
        method=method_name,
        db_id=original_schema.db_id,
        exacc_original=exacc_original,
        exacc_refined=exacc_refined,
        delta=delta,
        total_queries=total,
        original_details=original_details,
        refined_details=refined_details,
    )


def evaluate_original_only(
    model: Any,
    pairs: List[NLSQLPair],
    schema: Schema,
    db_path: str,
    method_name: str = "",
    max_workers: int = 1,
    table_map: Optional[Dict[str, str]] = None,
    label: str = "original",
    use_evidence: bool = False,
) -> EvalMethodResult:
    """Evaluate a Text-to-SQL model on a single schema (original or refined).

    Returns an EvalMethodResult with only one side populated.
    """
    total = len(pairs)
    if total == 0:
        return EvalMethodResult(
            method=method_name, db_id=schema.db_id,
            exacc_original=0.0, exacc_refined=0.0, delta=0.0, total_queries=0,
        )

    exacc, details = _run_eval_pass(
        model, pairs, schema, db_path,
        table_map=table_map, max_workers=max_workers,
        use_evidence=use_evidence,
    )

    logger.info("[%s] db=%s  %s=%.3f  n=%d", method_name, schema.db_id, label, exacc, total)

    if label == "original":
        return EvalMethodResult(
            method=method_name, db_id=schema.db_id,
            exacc_original=exacc, exacc_refined=0.0, delta=0.0,
            total_queries=total, original_details=details,
        )
    else:
        return EvalMethodResult(
            method=method_name, db_id=schema.db_id,
            exacc_original=0.0, exacc_refined=exacc, delta=0.0,
            total_queries=total, refined_details=details,
        )


# ---------------------------------------------------------------------------
# Schema loading from refined JSON
# ---------------------------------------------------------------------------

def load_refined_schema(json_path: str | Path) -> Schema:
    """Load a refined schema from a ``refined_tables.json`` file.

    Expected format::

        {
          "db_id": "test",
          "tables": [
            {
              "name": "t",
              "columns": [
                {"name": "col", "original_name": "c", "dtype": "TEXT", "is_pk": false}
              ]
            }
          ],
          "foreign_keys": [["t1.c1", "t2.c2"]]
        }

    Returns:
        A :class:`Schema` object built from the JSON definition.
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    db_id = data["db_id"]
    tables: List[Table] = []
    for tdata in data.get("tables", []):
        columns: List[Column] = []
        for cdata in tdata["columns"]:
            columns.append(Column(
                name=cdata["name"],
                table=tdata["name"],
                dtype=cdata.get("dtype", "TEXT"),
                is_pk=cdata.get("is_pk", False),
                fk_target=cdata.get("fk_target"),
            ))
        tables.append(Table(name=tdata["name"], columns=columns))

    foreign_keys = [tuple(fk) for fk in data.get("foreign_keys", [])]

    return Schema(db_id=db_id, tables=tables, foreign_keys=foreign_keys)
