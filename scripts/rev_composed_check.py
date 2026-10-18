#!/usr/bin/env python3
"""E2 (Meta Item 1 / R2-W1): composed-schema re-verification.

For every committed rename c_i in a canonical refine artifact, score the pair
{original name, selected name} on Q(c_i) with M={C3, DIN-SQL} under two bases:

  ISO  — base = the original schema (replicates Phase 3's isolated setting,
         re-derived on THIS endpoint so the composed comparison is not
         confounded by the checkpoint change vs. the canonical run);
  COMP — base = the schema with ALL OTHER committed renames applied
         (the schema c_i actually lives in after Phase 4).

Reports per-column iso/comp deltas, direction agreement, and the flip list.

Fairness note: score_candidate() cannot be reused for COMP because it only
backmaps c_i's rename; under a composed base the predicted SQL references
other renamed columns too, so execution against the original SQLite requires
the FULL reverse mapping. This script maintains that mapping itself.

Resume: one JSON per column under <output>/cols/; existing files are skipped.

Usage:
  python scripts/rev_composed_check.py \
      --config config/revision_qwen35_27b.yaml --benchmark drspider_abbr \
      --refine-dir results/refine/drspider_abbr_multi \
      --output results/rev_analysis/composed_check_abbr
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml  # noqa: E402

from egrefine.data.benchmark import load_bird, load_drspider  # noqa: E402
from egrefine.data.query_index import build_query_index  # noqa: E402
from egrefine.phase3.c3_runner import C3Text2SQL  # noqa: E402
from egrefine.phase3.dinsql_runner import DINSQLText2SQL  # noqa: E402
from egrefine.phase3.executor import compare_results  # noqa: E402
from egrefine.phase4.backmapper import backmap  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rev_composed_check")
logger.setLevel(logging.INFO)


def load_committed(refine_dir: str) -> dict[str, dict[str, str]]:
    """{db_id: {"table.column": refined_name}} for was_changed columns."""
    out: dict[str, dict[str, str]] = {}
    for db_id in sorted(os.listdir(refine_dir)):
        pr = os.path.join(refine_dir, db_id, "pipeline_result.json")
        if not os.path.isfile(pr):
            continue
        with open(pr) as f:
            data = json.load(f)
        m = {
            f"{r['table']}.{r['original_name']}": r["refined_name"]
            for r in data.get("refinements", [])
            if r.get("was_changed")
        }
        if m:
            out[db_id] = m
    return out


def score_names(
    schema, base_mapping: dict[str, str], col, names: list[str],
    queries, models, db_path: str, query_workers: int = 24,
) -> dict[str, float]:
    """Score each name for `col` on top of base_mapping (other renames)."""
    scores: dict[str, float] = {}
    base_schema = schema.apply_refinement(base_mapping) if base_mapping else schema
    base_reverse = {v: k.split(".")[-1] for k, v in base_mapping.items()}

    for name in names:
        if name == col.name:
            mod_schema = base_schema
            reverse = dict(base_reverse)
        else:
            mod_schema = base_schema.apply_refinement({col.full_name: name})
            reverse = dict(base_reverse)
            reverse[name] = col.name

        def _eval(pair, _mod=mod_schema, _rev=reverse):
            pred = None
            try:
                pred = _model.generate(
                    pair.nl, _mod, db_path=db_path, column_mapping=_rev,
                    evidence="",
                )
                orig_sql = backmap(pred, _rev) if _rev else pred
                return 1 if compare_results(orig_sql, pair.gold_sql, db_path) else 0
            except Exception as e:  # noqa: BLE001
                logger.warning("query failed (%s): %s", type(e).__name__, e)
                return 0

        model_scores = []
        for _model in models:
            workers = min(query_workers, len(queries))
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    hits = list(pool.map(_eval, queries))
            else:
                hits = [_eval(q) for q in queries]
            model_scores.append(sum(hits) / len(queries))
        scores[name] = sum(model_scores) / len(model_scores)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--benchmark", choices=["drspider_abbr", "bird"], required=True)
    ap.add_argument("--refine-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--query-workers", type=int, default=24)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.benchmark == "drspider_abbr":
        path = config["data"]["drspider_abbr"]["path"]
        schemas, pairs = load_drspider(path)
        db_dir = os.path.join(path, "database_post_perturbation")
        get_db_path = lambda db: os.path.join(db_dir, db, f"{db}.sqlite")  # noqa: E731
        desc_dir = None
    else:
        path = config["data"]["bird"]["path"]
        schemas, pairs = load_bird(path)
        db_dir = os.path.join(path, "dev_databases")
        get_db_path = lambda db: os.path.join(db_dir, db, f"{db}.sqlite")  # noqa: E731
        desc_dir = db_dir

    tcs = config["models"]["text2sql"]
    if not isinstance(tcs, list):
        tcs = [tcs]
    models = []
    for tc in tcs:
        models.append(C3Text2SQL(tc, num_samples=3, sample_rows=3, desc_dir=desc_dir))
        models.append(DINSQLText2SQL(tc, self_correction=True))

    committed = load_committed(args.refine_dir)
    n_cols = sum(len(v) for v in committed.values())
    logger.info("Committed renames: %d across %d DBs", n_cols, len(committed))

    cols_dir = os.path.join(args.output, "cols")
    os.makedirs(cols_dir, exist_ok=True)

    by_db_pairs: dict[str, list] = {}
    for p in pairs:
        by_db_pairs.setdefault(p.db_id, []).append(p)

    done = 0
    t0 = time.time()
    for db_id, renames in committed.items():
        if db_id not in schemas:
            logger.warning("schema missing for %s", db_id)
            continue
        schema = schemas[db_id]
        db_path = get_db_path(db_id)
        qindex = build_query_index(by_db_pairs.get(db_id, []), schema)
        col_by_full = {c.full_name: c for c in schema.all_columns}

        for full_name, new_name in renames.items():
            out_path = os.path.join(
                cols_dir, f"{db_id}__{full_name.replace('.', '__')}.json")
            if os.path.exists(out_path):
                done += 1
                continue
            col = col_by_full.get(full_name)
            if col is None:
                logger.warning("column %s missing in %s", full_name, db_id)
                continue
            queries = qindex.get(full_name, [])
            if not queries:
                logger.warning("empty Q(c_i) for committed %s.%s", db_id, full_name)
                continue

            names = [col.name, new_name]
            iso = score_names(schema, {}, col, names, queries, models,
                              db_path, args.query_workers)
            other = {fn: nn for fn, nn in renames.items() if fn != full_name}
            if other:
                comp = score_names(schema, other, col, names, queries, models,
                                   db_path, args.query_workers)
                trivial = False
            else:
                comp = dict(iso)
                trivial = True   # single committed column: COMP == ISO by construction

            rec = {
                "db_id": db_id, "column": full_name, "selected": new_name,
                "n_queries": len(queries),
                "iso": iso, "comp": comp,
                "iso_delta": iso[new_name] - iso[col.name],
                "comp_delta": comp[new_name] - comp[col.name],
                "trivial_single_col": trivial,
            }
            with open(out_path, "w") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
            done += 1
            logger.info("[%d/%d] %s %s: iso Δ=%.3f comp Δ=%.3f (%.0fs elapsed)",
                        done, n_cols, db_id, full_name,
                        rec["iso_delta"], rec["comp_delta"], time.time() - t0)

    # aggregate
    recs = []
    for fn in sorted(os.listdir(cols_dir)):
        with open(os.path.join(cols_dir, fn)) as f:
            recs.append(json.load(f))
    nontrivial = [r for r in recs if not r["trivial_single_col"]]
    agree = [r for r in nontrivial
             if (r["comp_delta"] > 0) == (r["iso_delta"] > 0)]
    flips = [r for r in nontrivial if r["comp_delta"] <= 0 < r["iso_delta"]]
    summary = {
        "n_scored": len(recs),
        "n_nontrivial": len(nontrivial),
        "n_trivial_single_col": len(recs) - len(nontrivial),
        "direction_agreement_nontrivial": (
            len(agree) / len(nontrivial) if nontrivial else None),
        "n_flips_pos_to_nonpos": len(flips),
        "flips": [{"db_id": r["db_id"], "column": r["column"],
                   "selected": r["selected"], "iso_delta": r["iso_delta"],
                   "comp_delta": r["comp_delta"]} for r in flips],
        "mean_iso_delta": sum(r["iso_delta"] for r in recs) / len(recs) if recs else None,
        "mean_comp_delta": sum(r["comp_delta"] for r in recs) / len(recs) if recs else None,
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
