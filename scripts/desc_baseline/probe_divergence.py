#!/usr/bin/env python3
"""Compute endpoint divergence between two NoRef ABBR runs (per-DB).

Inputs: two `per_db/` directories each containing `{db_id}/c3.json` files
with per-query match labels.

Outputs:
- Aggregate weighted-avg ExAcc on the intersection of DBs
- Per-query flip count (canonical->probe)
- Pearson correlation between per-DB ExAcc values
- A short verdict: 'reuse', 'reuse-with-caveat', 'rerun-all'
"""
import argparse
import json
import sys
from pathlib import Path


def load_per_db(per_db_root: Path, method: str = "c3") -> dict:
    """Return {db_id: {'matches': [bool], 'nls': [str], 'exacc': float, 'n': int}}."""
    out = {}
    for db_dir in sorted(per_db_root.iterdir()):
        if not db_dir.is_dir():
            continue
        method_path = db_dir / f"{method}.json"
        if not method_path.exists():
            continue
        with open(method_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        details = data.get("original_details", [])
        matches = [bool(d["match"]) for d in details]
        nls = [d["nl"] for d in details]
        out[db_dir.name] = {
            "matches": matches,
            "nls": nls,
            "exacc": data.get("exacc_original", 0.0),
            "n": data.get("total_queries", len(matches)),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True,
                        help="Canonical NoRef per_db dir (e.g. results/eval/drspider_abbr/no_refinement/per_db)")
    parser.add_argument("--probe", required=True,
                        help="Probe NoRef per_db dir (e.g. results/eval/drspider_abbr/noref_full/per_db)")
    parser.add_argument("--method", default="c3")
    args = parser.parse_args()

    canonical = load_per_db(Path(args.canonical), args.method)
    probe = load_per_db(Path(args.probe), args.method)

    common_dbs = sorted(set(canonical) & set(probe))
    print(f"DBs canonical: {len(canonical)}, probe: {len(probe)}, common: {len(common_dbs)}")

    if not common_dbs:
        print("No common DBs — aborting.")
        sys.exit(1)

    # Aggregate weighted-avg over common DBs
    c_correct = c_total = 0
    p_correct = p_total = 0
    flips = 0
    cw_pc = cc_pw = 0  # canonical-wrong probe-correct, canonical-correct probe-wrong
    db_diffs = []
    n_align_fail = 0

    for db_id in common_dbs:
        c = canonical[db_id]
        p = probe[db_id]
        if len(c["matches"]) != len(p["matches"]):
            # length mismatch — align by NL when possible
            n_align_fail += 1
            n = min(len(c["matches"]), len(p["matches"]))
            cm = c["matches"][:n]
            pm = p["matches"][:n]
        else:
            cm = c["matches"]
            pm = p["matches"]
        c_correct += sum(cm)
        c_total += len(cm)
        p_correct += sum(pm)
        p_total += len(pm)
        for x, y in zip(cm, pm):
            if x != y:
                flips += 1
                if not x and y:
                    cw_pc += 1
                elif x and not y:
                    cc_pw += 1
        db_c = sum(cm) / len(cm) if cm else 0
        db_p = sum(pm) / len(pm) if pm else 0
        db_diffs.append((db_id, db_c, db_p, db_p - db_c, len(cm)))

    c_avg = 100.0 * c_correct / c_total if c_total else 0
    p_avg = 100.0 * p_correct / p_total if p_total else 0
    diff = p_avg - c_avg
    flip_rate = 100.0 * flips / c_total if c_total else 0

    print()
    print("=" * 70)
    print(f"Aggregate (n={c_total} queries on {len(common_dbs)} DBs)")
    print("=" * 70)
    print(f"  canonical ExAcc : {c_avg:.4f}%")
    print(f"  probe     ExAcc : {p_avg:.4f}%")
    print(f"  delta           : {diff:+.4f} pp")
    print(f"  per-query flips : {flips} ({flip_rate:.2f}% of all queries)")
    print(f"  canonical-wrong → probe-correct: {cw_pc}")
    print(f"  canonical-correct → probe-wrong: {cc_pw}")
    if n_align_fail:
        print(f"  ⚠️  {n_align_fail} DBs had length mismatch (truncated to common prefix)")

    # Per-DB diff sort
    print()
    print("Top 10 DBs by |Δ ExAcc|:")
    db_diffs.sort(key=lambda x: -abs(x[3]))
    print(f"  {'db_id':45} {'canonical':>10} {'probe':>10} {'Δ':>8}  n")
    for db_id, dc, dp, dd, n in db_diffs[:10]:
        print(f"  {db_id:45} {100*dc:>9.1f}% {100*dp:>9.1f}% {100*dd:>+7.1f}  {n}")

    # Verdict
    print()
    print("=" * 70)
    if abs(diff) < 0.3 and flip_rate < 5.0:
        print("VERDICT: reuse")
        print("  endpoint divergence is below noise floor; safely reuse "
              "canonical Cell 1/Cell 3 numbers and only run Cell 2 + Cell 4.")
    elif abs(diff) < 1.0:
        print("VERDICT: reuse-with-caveat")
        print(f"  endpoint divergence {diff:+.2f}pp is small but non-zero — "
              "reuse canonical baselines and add a footnote quantifying drift.")
    else:
        print("VERDICT: rerun-all")
        print(f"  endpoint divergence {diff:+.2f}pp is too large to mix; "
              "re-run Cell 1 + Cell 3 on the probe endpoint for "
              "internal consistency.")


if __name__ == "__main__":
    main()
