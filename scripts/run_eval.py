#!/usr/bin/env python3
"""Stage 2 entry point: evaluate Text-to-SQL methods on original vs refined schemas.

Usage:
    # Both original and refined (default)
    PYTHONPATH=. python3 scripts/run_eval.py --refine-dir results/refine

    # Only original schema (no refine-dir needed)
    PYTHONPATH=. python3 scripts/run_eval.py --schema original --methods c3 dinsql macsql

    # Only refined schema
    PYTHONPATH=. python3 scripts/run_eval.py --refine-dir results/refine --schema refined

    # Refined eval with optimization: only re-eval queries that reference refined columns,
    # reuse original results for the rest
    PYTHONPATH=. python3 scripts/run_eval.py --refine-dir results/refine/bird --schema refined \\
        --reuse-original-dir results/eval/bird/main/per_db --methods c3 dinsql macsql

    # Specific databases and methods
    PYTHONPATH=. python3 scripts/run_eval.py --refine-dir results/refine --methods c3 --dbs financial

    # BEAVER benchmark
    PYTHONPATH=. python3 scripts/run_eval.py --benchmark beaver --schema original --methods c3
"""
import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.config import load_config
from egrefine.data.benchmark import BIRDLoader, DrSpiderLoader
from egrefine.eval.db_setup import copy_database, apply_views
from egrefine.eval.evaluator import (
    evaluate_method, evaluate_original_only, load_refined_schema,
    EvalMethodResult, QueryResult,
)
from egrefine.phase3.c3_runner import C3Text2SQL
from egrefine.phase3.dinsql_runner import DINSQLText2SQL
from egrefine.phase3.macsql_runner import MACSQLText2SQL
from egrefine.phase3.text2sql_runner import SimpleLLMText2SQL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY: Dict[str, type] = {
    "simple": SimpleLLMText2SQL,
    "c3": C3Text2SQL,
    "dinsql": DINSQLText2SQL,
    "macsql": MACSQLText2SQL,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_databases(refine_dir: str) -> List[str]:
    """Return db_ids that have Stage 1 artifacts in *refine_dir*."""
    refine_path = Path(refine_dir)
    if not refine_path.is_dir():
        return []
    db_ids = []
    for child in sorted(refine_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / "views.sql").exists() and (child / "refined_tables.json").exists():
            db_ids.append(child.name)
    return db_ids


def _load_benchmark(config: dict, benchmark: str):
    """Load benchmark data, return a loader object with .schemas, .get_pairs_for_db(), .get_db_path()."""
    if benchmark == "bird":
        return BIRDLoader(config["data"]["bird"]["path"])
    elif benchmark == "beaver":
        from egrefine.data.benchmark import BEAVERLoader
        beaver_cfg = config["data"]["beaver"]
        return BEAVERLoader(
            beaver_cfg["path"],
            split=beaver_cfg.get("split", "nw"),
            mysql_config=beaver_cfg.get("mysql"),
        )
    elif benchmark in ("drspider_abbr", "drspider_syn"):
        ds_cfg = config["data"][benchmark]
        return DrSpiderLoader(ds_cfg["path"])
    elif benchmark in ("drspider_abbr_pre", "drspider_syn_pre"):
        # Pre-perturbation: original Spider schema, same queries as post
        base_benchmark = benchmark.replace("_pre", "")  # drspider_abbr or drspider_syn
        ds_cfg = config["data"][base_benchmark]
        return DrSpiderLoader(ds_cfg["path"], pre=True)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def _get_refined_original_columns(refine_db_dir: Path) -> Set[str]:
    """Return the set of original column names that were refined (lowercased)."""
    refined_path = refine_db_dir / "refined_tables.json"
    if not refined_path.exists():
        return set()
    with open(refined_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cols = set()
    for t in data.get("tables", []):
        for c in t.get("columns", []):
            orig = c.get("original_name", "")
            if orig and c["name"] != orig:
                cols.add(orig.lower())
    return cols


def _query_references_columns(gold_sql: str, columns: Set[str]) -> bool:
    """Check if a gold SQL references any of the given column names."""
    sql_lower = gold_sql.lower()
    for col in columns:
        if re.search(r"\b" + re.escape(col) + r"\b", sql_lower):
            return True
    return False


def _load_original_results(original_dir: Path, db_id: str, method: str) -> Optional[dict]:
    """Load per-query original evaluation results from a T2 result file."""
    path = original_dir / db_id / f"{method}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_latex_table(all_results: List[EvalMethodResult]) -> str:
    """Generate a LaTeX comparison table from evaluation results."""
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Method & Database & ExAcc$_{\mathrm{orig}}$ & ExAcc$_{\mathrm{ref}}$ & $\Delta$ \\",
        r"\midrule",
    ]

    for r in all_results:
        delta_str = f"{r.delta:+.2f}" if r.delta != 0 else "0.00"
        lines.append(
            f"{r.method} & {r.db_id} & {r.exacc_original:.2f} & "
            f"{r.exacc_refined:.2f} & {delta_str} \\\\"
        )

    methods_seen = []
    for r in all_results:
        if r.method not in methods_seen:
            methods_seen.append(r.method)

    if len(all_results) > 1:
        lines.append(r"\midrule")
        for method in methods_seen:
            method_results = [r for r in all_results if r.method == method]
            if not method_results:
                continue
            total_q = sum(r.total_queries for r in method_results)
            if total_q == 0:
                continue
            avg_orig = sum(r.exacc_original * r.total_queries for r in method_results) / total_q
            avg_ref = sum(r.exacc_refined * r.total_queries for r in method_results) / total_q
            avg_delta = avg_ref - avg_orig
            delta_str = f"{avg_delta:+.2f}" if avg_delta != 0 else "0.00"
            lines.append(
                f"{method} & \\textit{{avg}} & {avg_orig:.2f} & "
                f"{avg_ref:.2f} & {delta_str} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EGRefine Stage 2: evaluate Text-to-SQL on original vs refined schemas",
    )
    parser.add_argument("--config", default="config/local.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--refine-dir", default=None,
                        help="Path to Stage 1 refine output directory (required for refined eval)")
    parser.add_argument("--schema", default="both", choices=["original", "refined", "both"],
                        help="Which schema to evaluate: original, refined, or both (default: both)")
    parser.add_argument("--benchmark", default="bird",
                        choices=["bird", "beaver", "drspider_abbr", "drspider_syn",
                                 "drspider_abbr_pre", "drspider_syn_pre"],
                        help="Benchmark dataset (default: bird)")
    parser.add_argument("--methods", nargs="*", default=["simple", "c3", "dinsql", "macsql"],
                        help="Text-to-SQL methods to evaluate")
    parser.add_argument("--dbs", nargs="*", default=None,
                        help="Specific databases to evaluate (default: all)")
    parser.add_argument("--reuse-original-dir", default=None,
                        help="Path to T2 per_db results to reuse for unaffected queries "
                             "(e.g. results/eval/bird/main/per_db)")
    parser.add_argument("--output", default="results/eval",
                        help="Output directory for results")
    parser.add_argument("--use-evidence", action="store_true", default=False,
                        help="Pass BIRD per-query evidence/hint to the Text-to-SQL "
                             "model. Default: off (enterprise-scenario framing). "
                             "Only meaningful for benchmarks that carry an evidence "
                             "field (BIRD).")
    parser.add_argument("--descriptions-json", default=None,
                        help="Path to column_descriptions.json produced by "
                             "scripts/desc_baseline/generate_descriptions.py. "
                             "When set, descriptions are injected inline as "
                             "SQL comments into the C3 schema serializers "
                             "(other methods ignore this flag). For refined "
                             "evaluation, description keys are automatically "
                             "remapped from original to refined column names "
                             "using the refine-dir's refined_tables.json.")
    args = parser.parse_args()

    # Validate: refined mode requires refine-dir
    eval_original = args.schema in ("original", "both")
    eval_refined = args.schema in ("refined", "both")

    if eval_refined and not args.refine_dir:
        logger.error("--refine-dir is required when --schema is 'refined' or 'both'")
        sys.exit(1)

    config = load_config(args.config)

    # ===== Validate methods =====
    for m in args.methods:
        if m not in METHOD_REGISTRY:
            logger.error("Unknown method %r. Available: %s", m, list(METHOD_REGISTRY.keys()))
            sys.exit(1)

    # ===== Load benchmark data =====
    loader = _load_benchmark(config, args.benchmark)

    # ===== Determine databases =====
    if eval_refined:
        available_dbs = discover_databases(args.refine_dir)
        if not available_dbs:
            logger.error("No Stage 1 artifacts found in %s", args.refine_dir)
            sys.exit(1)
    else:
        available_dbs = loader.db_ids

    if args.dbs:
        db_ids = [db for db in args.dbs if db in available_dbs]
        if not db_ids:
            logger.error("None of the requested databases are available")
            sys.exit(1)
    else:
        db_ids = available_dbs

    # ===== Instantiate models =====
    text2sql_config = config["models"]["text2sql"]
    if isinstance(text2sql_config, list):
        model_config = text2sql_config[0]
    else:
        model_config = text2sql_config

    # Optional: load column descriptions for the description-baseline experiment.
    # Keys are (db_id, table, column_name); for refined evaluation we
    # additionally insert (db_id, table, refined_name) entries pointing at the
    # same description so the C3 serializer can look up by whatever name appears
    # in the schema it is currently rendering.
    column_descriptions: Optional[Dict[tuple, str]] = None
    if args.descriptions_json:
        with open(args.descriptions_json, "r", encoding="utf-8") as f:
            desc_data = json.load(f)
        raw_descs = desc_data.get("descriptions", desc_data) if isinstance(desc_data, dict) else desc_data
        column_descriptions = {}
        for r in raw_descs:
            column_descriptions[(r["db_id"], r["table"], r["column"])] = r["description"]

        if eval_refined and args.refine_dir:
            refine_root = Path(args.refine_dir)
            remapped = 0
            for db_dir in refine_root.iterdir():
                if not db_dir.is_dir():
                    continue
                rt_path = db_dir / "refined_tables.json"
                if not rt_path.exists():
                    continue
                with open(rt_path, "r", encoding="utf-8") as f:
                    rt = json.load(f)
                rt_db_id = rt.get("db_id", db_dir.name)
                for t in rt.get("tables", []):
                    tname = t.get("name")
                    for c in t.get("columns", []):
                        orig = c.get("original_name", c["name"])
                        new = c["name"]
                        if new == orig:
                            continue
                        k_orig = (rt_db_id, tname, orig)
                        k_new = (rt_db_id, tname, new)
                        if k_orig in column_descriptions and k_new not in column_descriptions:
                            column_descriptions[k_new] = column_descriptions[k_orig]
                            remapped += 1
            logger.info("Loaded %d descriptions; added %d remapped (refined-name) entries",
                        len(raw_descs), remapped)
        else:
            logger.info("Loaded %d column descriptions for inline injection", len(raw_descs))

    models: Dict[str, Any] = {}
    for method_name in args.methods:
        cls = METHOD_REGISTRY[method_name]
        if method_name == "c3" and column_descriptions is not None:
            models[method_name] = cls(model_config, column_descriptions=column_descriptions)
        else:
            models[method_name] = cls(model_config)

    max_workers = config.get("concurrency", {}).get("max_workers", 1)

    # ===== Prepare output directory =====
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    reuse_dir = Path(args.reuse_original_dir) if args.reuse_original_dir else None

    print(f"\n{'='*70}")
    print(f"EGRefine Stage 2 — Evaluation")
    print(f"{'='*70}")
    print(f"Benchmark  : {args.benchmark}")
    print(f"Schema     : {args.schema}")
    print(f"Refine dir : {args.refine_dir or '(none)'}")
    print(f"Reuse dir  : {args.reuse_original_dir or '(none)'}")
    print(f"Databases  : {len(db_ids)} ({', '.join(db_ids)})")
    print(f"Methods    : {', '.join(args.methods)}")
    print(f"Concurrency: max_workers={max_workers}")
    print(f"Output     : {args.output}")
    print(f"{'='*70}\n")

    # ===== Run evaluation =====
    all_results: List[EvalMethodResult] = []

    with tempfile.TemporaryDirectory(prefix="egrefine_eval_") as tmpdir:
        for db_id in db_ids:
            pairs = loader.get_pairs_for_db(db_id)
            if not pairs:
                logger.warning("No NL-SQL pairs for %s, skipping", db_id)
                continue

            original_db_path = loader.get_db_path(db_id)
            # MySQL databases (BEAVER) use mysql:// URIs, not filesystem paths
            if not original_db_path.startswith("mysql://") and not Path(original_db_path).exists():
                logger.warning("Original database not found: %s, skipping", original_db_path)
                continue

            original_schema = loader.schemas.get(db_id)
            if original_schema is None:
                logger.warning("Schema not loaded for %s, skipping", db_id)
                continue

            # Load refined artifacts if needed
            refined_schema = None
            table_map: Dict[str, str] = {}
            tmp_db_path = None

            if eval_refined:
                refine_db_dir = Path(args.refine_dir) / db_id
                views_sql_path = refine_db_dir / "views.sql"
                refined_tables_path = refine_db_dir / "refined_tables.json"
                orig_table_map_path = refine_db_dir / "orig_table_map.json"

                # Compute refined columns once per DB (used for skip check + per-method reuse)
                refined_cols_for_db = _get_refined_original_columns(refine_db_dir)
                # Only skip DB setup in refined-only mode; --schema both still needs it
                skip_db_setup = not refined_cols_for_db and not eval_original

                if not skip_db_setup:
                    refined_schema = load_refined_schema(refined_tables_path)

                    if orig_table_map_path.exists():
                        with open(orig_table_map_path, "r", encoding="utf-8") as f:
                            table_map = json.load(f)

                    tmp_db_path = Path(tmpdir) / db_id / f"{db_id}.sqlite"
                    # For BEAVER (MySQL), copy_database exports MySQL → SQLite.
                    mysql_cfg = config.get("data", {}).get("beaver", {}).get("mysql")
                    copy_database(original_db_path, tmp_db_path, mysql_config=mysql_cfg)
                    apply_views(tmp_db_path, views_sql_path)

            print(f"--- {db_id} ({len(pairs)} pairs) ---")

            per_db_dir = output_dir / "per_db" / db_id
            per_db_dir.mkdir(parents=True, exist_ok=True)

            for method_name in args.methods:
                model = models[method_name]
                t0 = time.time()

                if eval_original and eval_refined:
                    # Both — use evaluate_method which runs two passes
                    result = evaluate_method(
                        model=model,
                        pairs=pairs,
                        original_schema=original_schema,
                        refined_schema=refined_schema,
                        original_db_path=original_db_path,
                        refined_db_path=str(tmp_db_path),
                        table_map=table_map,
                        method_name=method_name,
                        max_workers=max_workers,
                        use_evidence=args.use_evidence,
                    )
                elif eval_original:
                    # Original only
                    result = evaluate_original_only(
                        model=model,
                        pairs=pairs,
                        schema=original_schema,
                        db_path=original_db_path,
                        method_name=method_name,
                        max_workers=max_workers,
                        use_evidence=args.use_evidence,
                    )
                else:
                    # Refined only — with optional reuse optimization
                    refined_cols = refined_cols_for_db or set()
                    orig_data = _load_original_results(reuse_dir, db_id, method_name) if reuse_dir else None

                    if not refined_cols and orig_data:
                        # No columns refined → reuse NoRef results entirely
                        n_total = orig_data["total_queries"]
                        orig_exacc = orig_data["exacc_original"]
                        result = EvalMethodResult(
                            method=method_name, db_id=db_id,
                            exacc_original=0.0, exacc_refined=orig_exacc, delta=0.0,
                            total_queries=n_total,
                            refined_details=[
                                QueryResult(nl=d["nl"], gold_sql=d["gold_sql"],
                                            pred_sql=d["pred_sql"], match=d["match"])
                                for d in orig_data.get("original_details", [])
                            ],
                        )
                        logger.info("[%s] db=%s  no refined cols → reuse NoRef (exacc=%.3f, n=%d)",
                                    method_name, db_id, orig_exacc, n_total)
                    elif not refined_cols and not orig_data:
                        # No columns refined and no NoRef results to reuse → skip
                        logger.info("[%s] db=%s  no refined cols, skipping (use --reuse-original-dir to include)",
                                    method_name, db_id)
                        continue
                    elif orig_data and refined_cols:
                        # Partial reuse: only re-eval queries that reference refined columns
                        affected_pairs = []
                        affected_indices = set()
                        unaffected_matches = []

                        orig_details = orig_data.get("original_details", [])
                        for i, pair in enumerate(pairs):
                            if _query_references_columns(pair.gold_sql, refined_cols):
                                affected_pairs.append(pair)
                                affected_indices.add(i)
                            else:
                                if i < len(orig_details):
                                    unaffected_matches.append(orig_details[i]["match"])
                                else:
                                    unaffected_matches.append(False)

                        logger.info("[%s] db=%s  %d/%d queries affected by refined cols, re-evaluating",
                                    method_name, db_id, len(affected_pairs), len(pairs))

                        # Run eval only on affected queries
                        if affected_pairs:
                            partial_result = evaluate_original_only(
                                model=model, pairs=affected_pairs, schema=refined_schema,
                                db_path=str(tmp_db_path), method_name=method_name,
                                max_workers=max_workers, table_map=table_map, label="refined",
                                use_evidence=args.use_evidence,
                            )
                            affected_details = partial_result.refined_details
                        else:
                            affected_details = []

                        # Merge: rebuild full details list in original order
                        merged_details = []
                        affected_iter = iter(affected_details)
                        unaffected_idx = 0
                        for i, pair in enumerate(pairs):
                            if i in affected_indices:
                                merged_details.append(next(affected_iter))
                            else:
                                od = orig_details[i] if i < len(orig_details) else {}
                                merged_details.append(QueryResult(
                                    nl=pair.nl,
                                    gold_sql=od.get("gold_sql", pair.gold_sql),
                                    pred_sql=od.get("pred_sql", ""),
                                    match=unaffected_matches[unaffected_idx],
                                ))
                                unaffected_idx += 1

                        total_correct = sum(1 for d in merged_details if d.match)
                        exacc = total_correct / len(pairs) if pairs else 0.0

                        result = EvalMethodResult(
                            method=method_name, db_id=db_id,
                            exacc_original=0.0, exacc_refined=exacc, delta=0.0,
                            total_queries=len(pairs), refined_details=merged_details,
                        )
                    else:
                        # No reuse — full refined eval
                        result = evaluate_original_only(
                            model=model, pairs=pairs, schema=refined_schema,
                            db_path=str(tmp_db_path), method_name=method_name,
                            max_workers=max_workers, table_map=table_map, label="refined",
                            use_evidence=args.use_evidence,
                        )

                elapsed = time.time() - t0
                all_results.append(result)

                if eval_original and eval_refined:
                    print(
                        f"  {method_name:>10s}: "
                        f"orig={result.exacc_original:.3f} "
                        f"ref={result.exacc_refined:.3f} "
                        f"delta={result.delta:+.3f} "
                        f"({elapsed:.1f}s)"
                    )
                elif eval_original:
                    print(
                        f"  {method_name:>10s}: "
                        f"exacc={result.exacc_original:.3f} "
                        f"({elapsed:.1f}s)"
                    )
                else:
                    print(
                        f"  {method_name:>10s}: "
                        f"exacc={result.exacc_refined:.3f} "
                        f"({elapsed:.1f}s)"
                    )

                detail_path = per_db_dir / f"{method_name}.json"
                with open(detail_path, "w", encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

            print()

    # ===== Summary =====
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    if eval_original and eval_refined:
        print(f"{'Method':>10s}  {'Database':<20s}  {'Orig':>6s}  {'Ref':>6s}  {'Delta':>7s}")
        print("-" * 60)
        for r in all_results:
            print(
                f"{r.method:>10s}  {r.db_id:<20s}  "
                f"{r.exacc_original:>6.3f}  {r.exacc_refined:>6.3f}  {r.delta:>+7.3f}"
            )
    else:
        label = "ExAcc"
        print(f"{'Method':>10s}  {'Database':<20s}  {label:>8s}")
        print("-" * 45)
        for r in all_results:
            val = r.exacc_original if eval_original else r.exacc_refined
            print(f"{r.method:>10s}  {r.db_id:<20s}  {val:>8.3f}")

    # Per-method weighted averages
    methods_seen = []
    for r in all_results:
        if r.method not in methods_seen:
            methods_seen.append(r.method)

    if len(db_ids) > 1 and methods_seen:
        print("-" * 60)
        for method in methods_seen:
            method_results = [r for r in all_results if r.method == method]
            total_q = sum(r.total_queries for r in method_results)
            if total_q == 0:
                continue
            avg_orig = sum(r.exacc_original * r.total_queries for r in method_results) / total_q
            avg_ref = sum(r.exacc_refined * r.total_queries for r in method_results) / total_q
            avg_delta = avg_ref - avg_orig
            if eval_original and eval_refined:
                print(
                    f"{method:>10s}  {'(weighted avg)':20s}  "
                    f"{avg_orig:>6.3f}  {avg_ref:>6.3f}  {avg_delta:>+7.3f}"
                )
            elif eval_original:
                print(f"{method:>10s}  {'(weighted avg)':20s}  {avg_orig:>8.3f}")
            else:
                print(f"{method:>10s}  {'(weighted avg)':20s}  {avg_ref:>8.3f}")

    # ===== Save aggregate results =====
    aggregate = {
        "benchmark": args.benchmark,
        "schema_mode": args.schema,
        "databases": db_ids,
        "methods": args.methods,
        "results": [r.to_dict() for r in all_results],
        "summary": {},
    }
    for method in methods_seen:
        method_results = [r for r in all_results if r.method == method]
        total_q = sum(r.total_queries for r in method_results)
        if total_q == 0:
            aggregate["summary"][method] = {
                "avg_original": 0.0, "avg_refined": 0.0, "avg_delta": 0.0,
                "total_queries": 0,
            }
            continue
        avg_orig = sum(r.exacc_original * r.total_queries for r in method_results) / total_q
        avg_ref = sum(r.exacc_refined * r.total_queries for r in method_results) / total_q
        aggregate["summary"][method] = {
            "avg_original": round(avg_orig, 4),
            "avg_refined": round(avg_ref, 4),
            "avg_delta": round(avg_ref - avg_orig, 4),
            "total_queries": total_q,
        }

    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # ===== Save LaTeX table =====
    if eval_original and eval_refined:
        latex = generate_latex_table(all_results)
        tex_path = output_dir / "comparison.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"LaTeX table saved to {tex_path}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
