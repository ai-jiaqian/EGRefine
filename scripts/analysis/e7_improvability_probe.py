#!/usr/bin/env python3
"""E7: Phase-1 improvability probe — VLDB revision, Reviewer #4 D5.1.

Reviewer suggestion: Phase 1 should ask not only "might this column cause
confusion?" but also "is improvement possible?", pruning columns for which
no better name exists and saving Phase-2/3 cost.

This probe implements the reviewer's second question and runs it over every
column that passed Phase 1 in the k=3 rebase refine run on Dr.Spider-Abbr
(results/refine/rev_e3_k3_rebase_27b). Answers are cross-tabulated against
the Phase-3 execution ground truth (was_changed / delta, from each DB's
pipeline_result.json) to measure:

  - how many columns the second question would prune (Q2=NO);
  - how many of those actually committed a refinement (lost improvements),
    and what share of the total verified gain (sum delta x |Q(c_i)|) they carry;
  - how much Phase-2/Phase-3 cost the pruning would have saved.

Raw LLM responses are logged per column (run discipline: complete logs).
"""
import argparse
import itertools
import json
import logging
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from egrefine.data.benchmark import load_drspider
from egrefine.data.query_index import build_query_index
from egrefine.models.llm_client import LLMClient
from egrefine.phase1.llm_screener import _parse_yes_no
from egrefine.phase2.sampler import sample_column

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMPROVABILITY_PROMPT = """You are evaluating whether a database column's name can be improved for a Text-to-SQL model.

Database: {db_id}
Table: {table_name}
Column: {column_name} ({dtype})
Other columns in this table: {neighbor_names}
Sample values (5 rows): {sample_values}

This column was already flagged as potentially confusing for a Text-to-SQL model. Now judge whether its name can actually be improved: does a clearer, more standard alternative name exist that a Text-to-SQL model would more easily link natural-language questions to?

Keep in mind:
- Some names look confusing but have no clearly better alternative (e.g., domain-specific codes whose meaning is fixed by convention, or names that already match their values as well as any alternative).
- A better name must be one a user would naturally use when asking questions about this data.

Answer ONLY "YES" (a clearly better name exists) or "NO" (the current name is about as good as any alternative). Nothing else."""


def load_ground_truth(refine_dir: str):
    """Load per-column Phase-3 outcomes from a refinement directory.

    Returns:
        records: [{db_id, table, column, was_changed, delta,
                   verification_method, n_names}]
    """
    records = []
    for f in sorted(Path(refine_dir).glob("*/pipeline_result.json")):
        d = json.load(open(f))
        for r in d["refinements"]:
            records.append({
                "db_id": d["db_id"],
                "table": r["table"],
                "column": r["column"],
                "was_changed": r["was_changed"],
                "delta": r["delta"],
                "verification_method": r["verification_method"],
                "n_names": len(r.get("all_scores") or {}),
            })
    return records


def build_prompt(schema, table: str, column: str, db_path: str) -> str:
    table_obj = schema.get_table(table)
    col_obj = next((c for c in table_obj.columns if c.name == column), None)
    neighbor_names = [c.name for c in table_obj.columns if c.name != column]
    try:
        sample_values = sample_column(db_path, table, column, n=5)
    except Exception:
        sample_values = []
    return IMPROVABILITY_PROMPT.format(
        db_id=schema.db_id,
        table_name=table,
        column_name=column,
        dtype=col_obj.dtype if col_obj else "?",
        neighbor_names=", ".join(neighbor_names) if neighbor_names else "(none)",
        sample_values=", ".join(str(v) for v in sample_values[:5]) if sample_values else "(no data)",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine-dir", required=True)
    ap.add_argument("--drspider-path", required=True)
    ap.add_argument("--endpoints", required=True,
                    help="Comma-separated base URLs, used round-robin")
    ap.add_argument("--model", default="Qwen3.5-27B")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0,
                    help="Probe only the first N columns (0 means all)")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke mode: 5 committed columns and 10 seeded controls")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print prompt samples and counts without calling an LLM")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Phase-3 outcomes
    records = load_ground_truth(args.refine_dir)
    logger.info("ground truth: %d Phase-1-passed columns, %d commits",
                len(records), sum(1 for r in records if r["was_changed"]))

    # 2) Schema and query index for |Q(c_i)|
    schemas, pairs = load_drspider(args.drspider_path)
    db_dir = Path(args.drspider_path) / "database_post_perturbation"
    n_queries = {}
    for db_id in {r["db_id"] for r in records}:
        schema = schemas.get(db_id)
        if schema is None:
            logger.warning("schema missing for %s", db_id)
            continue
        db_pairs = [p for p in pairs if p.db_id == db_id]
        qidx = build_query_index(db_pairs, schema)
        for full_name, qs in qidx.items():
            n_queries[(db_id, full_name)] = len(qs)
    for r in records:
        r["n_queries"] = n_queries.get((r["db_id"], f"{r['table']}.{r['column']}"), 0)

    # 3) Select probe targets
    targets = list(records)
    if args.smoke:
        commits = [r for r in targets if r["was_changed"]][:5]
        others = [r for r in targets if not r["was_changed"]]
        random.Random(42).shuffle(others)
        targets = commits + others[:10]
    elif args.limit:
        targets = targets[: args.limit]
    logger.info("probing %d columns", len(targets))

    if args.dry_run:
        for r in targets[:3]:
            schema = schemas[r["db_id"]]
            db_path = str(db_dir / r["db_id"] / f"{r['db_id']}.sqlite")
            print("=" * 70)
            print(f"{r['db_id']}:{r['table']}.{r['column']}  "
                  f"changed={r['was_changed']} delta={r['delta']} |Q|={r['n_queries']}")
            print(build_prompt(schema, r["table"], r["column"], db_path))
        return

    # 4) Round-robin LLM clients
    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    clients = [
        LLMClient({
            "model_name": args.model,
            "api_key": "EMPTY",
            "base_url": url,
            "temperature": 0,
            "max_tokens": 8,
            "max_retries": 3,
            "timeout": 120,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        })
        for url in endpoints
    ]
    cycle = itertools.cycle(clients)
    cycle_lock = threading.Lock()

    def next_client():
        with cycle_lock:
            return next(cycle)

    def probe_one(rec):
        schema = schemas[rec["db_id"]]
        db_path = str(db_dir / rec["db_id"] / f"{rec['db_id']}.sqlite")
        prompt = build_prompt(schema, rec["table"], rec["column"], db_path)
        out = dict(rec)
        out["prompt"] = prompt
        try:
            resp = next_client().chat([{"role": "user", "content": prompt}])
            out["raw_response"] = resp
            out["q2"] = "YES" if _parse_yes_no(resp) else "NO"
            out["error"] = None
        except Exception as e:  # Conservatively keep the column on failure.
            out["raw_response"] = None
            out["q2"] = "YES"
            out["error"] = str(e)
        return out

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(probe_one, r) for r in targets]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 100 == 0:
                logger.info("progress: %d/%d", i, len(targets))

    # 5) Write per-column JSONL, including raw responses.
    jsonl_path = out_dir / "e7_probe_results.jsonl"
    with open(jsonl_path, "w") as f:
        for r in sorted(results, key=lambda x: (x["db_id"], x["table"], x["column"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 6) Aggregate the cross-tabulation.
    summary = summarize(results, args, endpoints)
    with open(out_dir / "e7_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def summarize(results, args, endpoints):
    n = len(results)
    n_yes = sum(1 for r in results if r["q2"] == "YES")
    n_no = n - n_yes
    n_err = sum(1 for r in results if r["error"])

    commits = [r for r in results if r["was_changed"]]
    pruned_commits = [r for r in commits if r["q2"] == "NO"]
    gain = lambda r: r["delta"] * r["n_queries"]  # Correct-query equivalents.
    total_gain = sum(gain(r) for r in commits)
    lost_gain = sum(gain(r) for r in pruned_commits)

    # Phase-3 cost unit = |Q| times the number of scored names. No-query
    # columns incur no Phase-3 execution cost. The denominator includes
    # skipped_below_threshold because those columns were fully scored before
    # tau_min rejected them.
    scored_cols = [r for r in results
                   if r["verification_method"] in ("execution", "skipped_below_threshold")]
    p3_total = sum(r["n_queries"] * r["n_names"] for r in scored_cols)
    p3_pruned = sum(r["n_queries"] * r["n_names"] for r in scored_cols if r["q2"] == "NO")

    yes_commit_rate = (sum(1 for r in commits if r["q2"] == "YES") / n_yes) if n_yes else 0
    no_commit_rate = (len(pruned_commits) / n_no) if n_no else 0

    return {
        "config": {
            "refine_dir": args.refine_dir,
            "endpoints": endpoints,
            "model": args.model,
            "smoke": args.smoke,
            "limit": args.limit,
            "prompt": IMPROVABILITY_PROMPT,
        },
        "n_passed": n,
        "n_errors": n_err,
        "q2_yes": n_yes,
        "q2_no": n_no,
        "q2_no_share": round(n_no / n, 4) if n else 0,
        "commits_total": len(commits),
        "commits_pruned_by_q2no": len(pruned_commits),
        "pruned_commit_list": [
            f"{r['db_id']}:{r['table']}.{r['column']} (delta={r['delta']}, |Q|={r['n_queries']})"
            for r in pruned_commits
        ],
        "total_verified_gain_delta_x_q": round(total_gain, 3),
        "lost_gain_delta_x_q": round(lost_gain, 3),
        "lost_gain_share": round(lost_gain / total_gain, 4) if total_gain else 0,
        "phase3_eval_units_total": p3_total,
        "phase3_eval_units_pruned": p3_pruned,
        "phase3_cost_saved_share": round(p3_pruned / p3_total, 4) if p3_total else 0,
        "phase2_calls_saved": n_no,
        "commit_rate_q2_yes": round(yes_commit_rate, 4),
        "commit_rate_q2_no": round(no_commit_rate, 4),
    }


if __name__ == "__main__":
    main()
