"""Re-derive refine artifacts at a new min_delta (τ) without re-running Phase 3.

Reads a source refine dir (with per-DB pipeline_result.json containing all_scores)
and writes a new refine dir whose views.sql / refined_tables.json / orig_table_map.json
reflect a different conservative-rule τ threshold.

Usage:
    PYTHONPATH=. python3 scripts/rethreshold_refine.py \
        --source-refine results/refine/drspider_abbr_multi \
        --benchmark drspider_abbr \
        --config config/revision_qwen35_27b.yaml \
        --tau 0.03 \
        --output results/refine/drspider_abbr_tau003
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import yaml

from egrefine.data.benchmark import load_drspider, load_bird
from egrefine.data.schema import Column, Schema
from egrefine.phase2.prompts import CandidateName
from egrefine.phase3.conflict import resolve_conflicts
from egrefine.phase3.scorer import SelectionResult, select_best
from egrefine.phase4.view_synthesis import synthesize

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _column_from_schema(schema: Schema, table: str, col_name: str) -> Column:
    """Look up the Column dataclass from the loaded Schema."""
    for t in schema.tables:
        if t.name == table:
            for c in t.columns:
                if c.name == col_name:
                    return c
    raise KeyError(f"Column {table}.{col_name} not found in schema {schema.db_id}")


def _selection_results_from_pipeline_result(
    pipeline_result: dict,
    schema: Schema,
    tau: float,
    conservative: bool = True,
) -> List[SelectionResult]:
    """Rebuild SelectionResult list by re-applying select_best at a new τ."""
    results = []
    for r in pipeline_result.get("refinements", []):
        table = r["table"]
        col_name = r["original_name"]
        all_scores = r.get("all_scores", {})

        try:
            col = _column_from_schema(schema, table, col_name)
        except KeyError:
            logger.warning("Skipping %s.%s: not in schema", table, col_name)
            continue

        # Reconstruct candidate list: all_scores keys minus the original
        cand_names = [n for n in all_scores.keys() if n != col_name]
        candidates = [CandidateName(name=n, reason="") for n in cand_names]

        if not all_scores:
            # No scores — preserve the original record's verification_method
            results.append(SelectionResult(
                column=col,
                selected_name=col_name,
                delta=0.0,
                was_changed=False,
                all_scores={},
                verification_method=r.get("verification_method", "skipped_no_queries"),
            ))
            continue

        sel = select_best(
            column=col,
            candidates=candidates,
            scores=all_scores,
            conservative=conservative,
            min_delta=tau,
        )
        results.append(sel)
    return results


def load_benchmark_schemas(benchmark: str, config: dict) -> Dict[str, Schema]:
    if benchmark == "drspider_abbr":
        path = config["data"]["drspider_abbr"]["path"]
        schemas, _ = load_drspider(path)
    elif benchmark == "drspider_syn":
        path = config["data"]["drspider_syn"]["path"]
        schemas, _ = load_drspider(path)
    elif benchmark == "bird":
        path = config["data"]["bird"]["path"]
        schemas, _ = load_bird(path)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return schemas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-refine", required=True,
                    help="Source refine dir with per-DB pipeline_result.json")
    ap.add_argument("--benchmark", required=True,
                    choices=["drspider_abbr", "drspider_syn", "bird"])
    ap.add_argument("--config", required=True, help="Config yaml (for data paths)")
    ap.add_argument("--tau", type=float, required=True,
                    help="New min_delta threshold (e.g. 0.03)")
    ap.add_argument("--output", required=True, help="Output refine dir")
    ap.add_argument("--conflict-rounds", type=int, default=2)
    ap.add_argument("--no-conservative", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    src = Path(args.source_refine)
    dst = Path(args.output)
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=False)

    logger.info("Loading %s schemas ...", args.benchmark)
    schemas = load_benchmark_schemas(args.benchmark, config)
    logger.info("Loaded %d schemas", len(schemas))

    # For each DB that has a pipeline_result.json in source, re-threshold and write.
    db_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    logger.info("Processing %d DBs from %s", len(db_dirs), src)

    summary = {
        "tau": args.tau,
        "source": str(src),
        "per_db": {},
        "totals": {"dbs_changed": 0, "cols_changed": 0, "cols_scored": 0,
                   "cols_blocked_by_threshold": 0},
    }

    for db_dir in db_dirs:
        db_id = db_dir.name
        pr_path = db_dir / "pipeline_result.json"
        if not pr_path.exists():
            continue
        with open(pr_path) as f:
            pr = json.load(f)

        if db_id not in schemas:
            logger.warning("Schema not loaded for %s, skipping", db_id)
            continue

        schema = schemas[db_id]
        results = _selection_results_from_pipeline_result(pr, schema, args.tau,
                                                          conservative=not args.no_conservative)

        # Conflict resolution
        resolved = resolve_conflicts(
            results, schema,
            max_rounds=args.conflict_rounds,
            min_delta=args.tau,
        )

        changed = [r for r in resolved if r.was_changed]
        blocked = sum(1 for r in resolved
                      if r.verification_method == "skipped_below_threshold")
        summary["totals"]["cols_scored"] += len(resolved)
        summary["totals"]["cols_changed"] += len(changed)
        summary["totals"]["cols_blocked_by_threshold"] += blocked
        if changed:
            summary["totals"]["dbs_changed"] += 1
        summary["per_db"][db_id] = {
            "cols_scored": len(resolved),
            "cols_changed": len(changed),
            "cols_blocked_by_threshold": blocked,
        }

        # Synthesize Phase 4 artifacts
        out_db_dir = dst / db_id
        out_db_dir.mkdir(parents=True, exist_ok=True)
        synth = synthesize(schema, resolved, output_dir=str(out_db_dir))

        # Also write a per-DB pipeline_result.json so downstream tools
        # that inspect it (e.g. audits) continue to work.
        pr_out = {
            "db_id": db_id,
            "refinements": [
                {
                    "table": r.column.table,
                    "column": r.column.name,
                    "original_name": r.column.name,
                    "refined_name": r.selected_name,
                    "delta": r.delta,
                    "was_changed": r.was_changed,
                    "verification_method": r.verification_method,
                    "all_scores": r.all_scores,
                }
                for r in resolved
            ],
            "view_definitions": synth["views"],
            "mapping": synth["mapping"],
            "reverse_mapping": synth["reverse_mapping"],
            "orig_table_map": synth["orig_table_map"],
            "statistics": {
                "columns_scored": len(resolved),
                "columns_refined": len(changed),
                "columns_kept_original": len(resolved) - len(changed),
                "columns_blocked_by_threshold": blocked,
                "min_delta": args.tau,
                "avg_delta": sum(r.delta for r in changed) / len(changed) if changed else 0.0,
            },
        }
        with open(out_db_dir / "pipeline_result.json", "w") as f:
            json.dump(pr_out, f, indent=2)

    with open(dst / "rethreshold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== τ=%.3f summary ===", args.tau)
    logger.info("  DBs with ≥1 change: %d", summary["totals"]["dbs_changed"])
    logger.info("  Total cols changed: %d", summary["totals"]["cols_changed"])
    logger.info("  Total cols blocked by threshold: %d",
                summary["totals"]["cols_blocked_by_threshold"])
    logger.info("  Output: %s", dst)


if __name__ == "__main__":
    main()
