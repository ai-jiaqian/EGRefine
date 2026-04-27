#!/usr/bin/env python3
"""Aggregate cross-verifier ablation results.

Compares multiple refined-schema eval directories against a NoRef baseline and
a reference refinement, reporting Q-weighted ExAcc and refinement commit-set
Jaccard similarity.

Usage:
    python3 scripts/analysis/cross_verifier_aggregate.py \\
        --eval-root results/eval/drspider_abbr \\
        --new-refine results/refine/drspider_abbr_verifier_dinmac \\
        --old-refine results/refine/drspider_abbr_multi \\
        --rows "NoRef:noref_full,Ref M={C3,DIN}:egrefine_ref,M={DIN,MAC} C3:egrefine_dinmac_c3,M={DIN,MAC} DIN:egrefine_dinmac_din,M={DIN,MAC} MAC:egrefine_dinmac_mac"
"""
import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def q_weighted_exacc(per_db_dir: Path, method: str) -> tuple[float, int, int]:
    """Compute Q-weighted ExAcc from per_db/{db_id}/{method}.json files.

    Returns (avg_exacc * 100, n_correct, n_queries).
    """
    if not per_db_dir.is_dir():
        return float("nan"), 0, 0
    total_correct = 0
    total_q = 0
    for db_dir in sorted(per_db_dir.iterdir()):
        if not db_dir.is_dir():
            continue
        method_file = db_dir / f"{method}.json"
        if not method_file.exists():
            continue
        with open(method_file, "r") as f:
            data = json.load(f)
        # Accept both nested and flat structures
        details = data.get("refined_details") or data.get("details") or []
        if not details:
            # try original_details (only if eval was on original schema)
            details = data.get("original_details") or []
        n = len(details)
        c = sum(1 for d in details if d.get("match"))
        total_q += n
        total_correct += c
    avg = (total_correct / total_q * 100) if total_q else float("nan")
    return avg, total_correct, total_q


def load_commit_set(refine_dir: Path) -> set[tuple[str, str, str]]:
    """Read pipeline_result.json from each DB and collect committed (db_id, table, original_name)."""
    if not refine_dir.is_dir():
        return set()
    commits = set()
    for db_dir in sorted(refine_dir.iterdir()):
        if not db_dir.is_dir():
            continue
        pr_file = db_dir / "pipeline_result.json"
        if not pr_file.exists():
            continue
        with open(pr_file, "r") as f:
            data = json.load(f)
        db_id = data.get("db_id", db_dir.name)
        for r in data.get("refinements", []):
            if r.get("was_changed"):
                commits.add((db_id, r["table"], r["original_name"]))
    return commits


def parse_rows(rows_str: str) -> list[tuple[str, str]]:
    """Parse 'Label:dirname,Label2:dirname2,...' into list of (label, dirname)."""
    result = []
    for item in rows_str.split(","):
        item = item.strip()
        if ":" not in item:
            print(f"WARNING: skipping malformed row spec (missing ':'): {item!r}", file=sys.stderr)
            continue
        label, dirname = item.split(":", 1)
        result.append((label.strip(), dirname.strip()))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate cross-verifier ablation results.",
    )
    parser.add_argument("--eval-root", required=True,
                        help="Root eval directory (e.g. results/eval/drspider_abbr)")
    parser.add_argument("--new-refine", required=True,
                        help="New refinement dir for commit-set Jaccard (M={DIN,MAC})")
    parser.add_argument("--old-refine", required=True,
                        help="Reference refinement dir for commit-set Jaccard (M={C3,DIN})")
    parser.add_argument("--rows", required=True,
                        help="Comma-separated 'Label:dirname' pairs, e.g. 'NoRef:noref_full,EGRefine:egrefine_ref'")
    parser.add_argument("--title", default="Cross-Verifier Ablation Results",
                        help="Title printed in the report header")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    new_refine = Path(args.new_refine)
    old_refine = Path(args.old_refine)

    old_commits = load_commit_set(old_refine)
    new_commits = load_commit_set(new_refine)

    intersection = old_commits & new_commits
    union = old_commits | new_commits
    jaccard = len(intersection) / len(union) if union else float("nan")

    rows = parse_rows(args.rows)

    print("=" * 80)
    print(args.title)
    print("=" * 80)
    print(f"\nRefinement commit set sizes:")
    print(f"  Old refine ({old_refine.name}): {len(old_commits)} columns")
    print(f"  New refine ({new_refine.name}): {len(new_commits)} columns")
    print(f"  Intersection: {len(intersection)} | Union: {len(union)} | Jaccard: {jaccard:.3f}")

    print(f"\n{'Configuration':<40s}  {'C3':>10s}  {'DIN':>10s}  {'MAC':>10s}  {'n_q':>8s}")
    print("-" * 88)
    for label, dirname in rows:
        per_db = eval_root / dirname / "per_db"
        c3, _, n_c3 = q_weighted_exacc(per_db, "c3")
        din, _, n_din = q_weighted_exacc(per_db, "dinsql")
        mac, _, n_mac = q_weighted_exacc(per_db, "macsql")
        n_q = max(n_c3, n_din, n_mac)
        c3_s = f"{c3:.2f}" if c3 == c3 else "—"
        din_s = f"{din:.2f}" if din == din else "—"
        mac_s = f"{mac:.2f}" if mac == mac else "—"
        print(f"{label:<40s}  {c3_s:>10s}  {din_s:>10s}  {mac_s:>10s}  {n_q:>8d}")

    print()


if __name__ == "__main__":
    main()
