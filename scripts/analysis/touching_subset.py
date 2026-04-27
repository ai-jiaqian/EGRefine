"""Touching Subset ExAcc analysis.

For each (benchmark, backbone, algorithm, refinement_config) cell, partition
queries into:
  - touching: gold SQL references at least one column refined by EGRefine
  - non_touching: gold SQL references no refined columns

Compute ExAcc separately on each subset for NoRef and EGRefine, plus the
overall full-set numbers, so per-query effect can be decoupled from the
aggregate dilution caused by non-touching queries.

Reuses run_eval.py logic for column-level refined detection (refined_tables.json
where name != original_name) and word-boundary regex matching against gold SQL.

Output:
  - <label-dump>  : per-query labels + match flags (path supplied via --label-dump)
  - aggregate dict returned to caller / printed JSON when run as CLI
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def get_refined_original_columns(refine_db_dir: Path) -> Set[str]:
    """Lowercase set of original column names that were refined in this DB.

    Mirrors run_eval.py:_get_refined_original_columns. Reads refined_tables.json
    where every column has both `name` (refined or unchanged) and `original_name`.
    A column is "refined" iff name != original_name.
    """
    p = refine_db_dir / "refined_tables.json"
    if not p.exists():
        return set()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    cols: Set[str] = set()
    for t in data.get("tables", []):
        for c in t.get("columns", []):
            orig = c.get("original_name", "")
            if orig and c.get("name") != orig:
                cols.add(orig.lower())
    return cols


def query_references_columns(gold_sql: str, columns: Set[str]) -> Tuple[bool, List[str]]:
    """Word-boundary match. Returns (touched, list_of_matched_columns)."""
    if not columns:
        return False, []
    sql_lower = gold_sql.lower()
    matched: List[str] = []
    for col in columns:
        if re.search(r"\b" + re.escape(col) + r"\b", sql_lower):
            matched.append(col)
    return (len(matched) > 0), matched


def load_per_db_details(per_db_path: Path) -> List[dict]:
    """Read a per_db/<db>/<method>.json file and return the relevant details list.

    The eval pipeline writes:
      - NoRef runs:    original_details populated, refined_details = []
      - EGRefine runs: refined_details populated, original_details = [] (or
                       carried over from --reuse-original-dir)

    We unify by returning whichever list is non-empty (NoRef vs EGRefine).
    Each detail has keys: nl, gold_sql, pred_sql, match.
    """
    if not per_db_path.exists():
        return []
    with open(per_db_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    refined = d.get("refined_details") or []
    original = d.get("original_details") or []
    # Prefer refined when populated (this is an EGRefine eval file)
    return refined if refined else original


@dataclass
class CellSummary:
    method: str
    bucket: str  # full | touching | non_touching
    n: int
    exacc_noref: Optional[float]
    exacc_egrefine: Optional[float]
    delta: Optional[float]


def analyze_cell(
    noref_eval_dir: Path,
    egrefine_eval_dir: Path,
    refine_dir: Path,
    methods: Tuple[str, ...] = ("c3", "dinsql", "macsql"),
    label_dump_path: Optional[Path] = None,
    cell_name: str = "cell",
) -> Dict[str, Dict[str, dict]]:
    """Compute touching/non-touching ExAcc for one cell.

    Returns nested dict: {method: {bucket: {n, exacc_noref, exacc_egrefine, delta}}}
    """
    noref_per_db = noref_eval_dir / "per_db"
    egr_per_db = egrefine_eval_dir / "per_db"

    if not noref_per_db.exists():
        raise FileNotFoundError(f"missing per_db dir: {noref_per_db}")
    if not egr_per_db.exists():
        raise FileNotFoundError(f"missing per_db dir: {egr_per_db}")

    noref_dbs = {p.name for p in noref_per_db.iterdir() if p.is_dir()}
    egr_dbs = {p.name for p in egr_per_db.iterdir() if p.is_dir()}
    common_dbs = sorted(noref_dbs & egr_dbs)

    only_noref = noref_dbs - egr_dbs
    only_egr = egr_dbs - noref_dbs

    # Per-method buckets store list of (match_noref:bool, match_egrefine:bool)
    buckets: Dict[str, Dict[str, List[Tuple[bool, bool]]]] = {
        m: {"full": [], "touching": [], "non_touching": []} for m in methods
    }

    # Per-query labels for dump
    query_labels: List[dict] = []

    for db_id in common_dbs:
        refine_db_dir = refine_dir / db_id
        refined_cols = get_refined_original_columns(refine_db_dir) if refine_db_dir.exists() else set()

        for method in methods:
            noref_dets = load_per_db_details(noref_per_db / db_id / f"{method}.json")
            egr_dets = load_per_db_details(egr_per_db / db_id / f"{method}.json")

            n = min(len(noref_dets), len(egr_dets))
            if n == 0:
                continue
            if len(noref_dets) != len(egr_dets):
                # Different counts: pair only the prefix overlap. Log warning to stderr.
                print(
                    f"  warn: {db_id}/{method}: NoRef={len(noref_dets)} EGR={len(egr_dets)} -> using min={n}",
                    file=sys.stderr,
                )

            for i in range(n):
                gold_sql = noref_dets[i].get("gold_sql", "")
                touching, matched = query_references_columns(gold_sql, refined_cols)

                m_noref = bool(noref_dets[i].get("match"))
                m_egr = bool(egr_dets[i].get("match"))

                buckets[method]["full"].append((m_noref, m_egr))
                bucket_name = "touching" if touching else "non_touching"
                buckets[method][bucket_name].append((m_noref, m_egr))

                if label_dump_path is not None:
                    query_labels.append({
                        "cell": cell_name,
                        "db_id": db_id,
                        "method": method,
                        "query_idx": i,
                        "touching": touching,
                        "matched_refined_cols": matched,
                        "match_noref": m_noref,
                        "match_egrefine": m_egr,
                    })

    # Aggregate
    summary: Dict[str, Dict[str, dict]] = {}
    for method, bbs in buckets.items():
        summary[method] = {}
        for bucket, pairs in bbs.items():
            n = len(pairs)
            if n == 0:
                summary[method][bucket] = {"n": 0, "exacc_noref": None, "exacc_egrefine": None, "delta": None}
                continue
            ex_noref = 100.0 * sum(p[0] for p in pairs) / n
            ex_egr = 100.0 * sum(p[1] for p in pairs) / n
            summary[method][bucket] = {
                "n": n,
                "exacc_noref": round(ex_noref, 2),
                "exacc_egrefine": round(ex_egr, 2),
                "delta": round(ex_egr - ex_noref, 2),
            }

    if label_dump_path is not None:
        label_dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_dump_path, "w", encoding="utf-8") as f:
            json.dump({
                "cell": cell_name,
                "noref_eval_dir": str(noref_eval_dir),
                "egrefine_eval_dir": str(egrefine_eval_dir),
                "refine_dir": str(refine_dir),
                "common_dbs": common_dbs,
                "only_in_noref": sorted(only_noref),
                "only_in_egrefine": sorted(only_egr),
                "queries": query_labels,
            }, f, indent=2, ensure_ascii=False)

    return summary


def format_table(summary: Dict[str, Dict[str, dict]], cell_name: str) -> str:
    """Pretty markdown table for one cell."""
    lines = [
        f"### {cell_name}",
        "",
        "| Algorithm | Subset | n | ExAcc_NoRef | ExAcc_EGRefine | Δ |",
        "|-----------|--------|---|-------------|----------------|---|",
    ]
    method_label = {"c3": "C3", "dinsql": "DIN-SQL", "macsql": "MAC-SQL", "simple": "Simple"}
    bucket_label = {"full": "Full", "touching": "Touching", "non_touching": "Non-touching"}
    for method in ("c3", "dinsql", "macsql", "simple"):
        if method not in summary:
            continue
        for bucket in ("full", "touching", "non_touching"):
            row = summary[method][bucket]
            n = row["n"]
            if n == 0:
                lines.append(f"| {method_label[method]} | {bucket_label[bucket]} | 0 | — | — | — |")
            else:
                lines.append(
                    f"| {method_label[method]} | {bucket_label[bucket]} | {n} | "
                    f"{row['exacc_noref']:.2f} | {row['exacc_egrefine']:.2f} | "
                    f"{row['delta']:+.2f} |"
                )
    return "\n".join(lines)


def load_labels_from_file(labels_file: Path) -> Optional[List[dict]]:
    """Load pre-computed per-query labels from a JSON file.

    The file may be in one of two forms:
      1. Multi-cell: {"cell_name": {"queries": [...], ...}, ...}
      2. Single-cell: {"queries": [...], ...}

    Returns the list of query-label dicts, or None if the file cannot be read.
    """
    if not labels_file.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_file}. "
            "Supply your own labels in JSON format; see docs/architecture.md."
        )
    with open(labels_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Support both {cell_name: {...}} multi-cell form and single-cell form
    if isinstance(raw, dict):
        # Try single-cell form first (has "queries" key at top level)
        if "queries" in raw:
            return raw["queries"]
        # Otherwise assume first value is the cell data
        for v in raw.values():
            if isinstance(v, dict) and "queries" in v:
                return v["queries"]
    return None


def main():
    parser = argparse.ArgumentParser(description="Touching Subset ExAcc analysis")
    parser.add_argument("--noref-dir", required=True, help="Eval dir with NoRef baseline")
    parser.add_argument("--egrefine-dir", required=True, help="Eval dir with EGRefine results")
    parser.add_argument("--refine-dir", required=True, help="Refine output dir (for refined_tables.json)")
    parser.add_argument("--methods", nargs="*", default=["c3", "dinsql", "macsql"])
    parser.add_argument("--cell-name", default="cell", help="Name for this cell (used in output)")
    parser.add_argument("--label-dump", default=None, help="Optional path to dump per-query labels JSON")
    parser.add_argument("--labels-file", default=None,
                        help="Optional path to a pre-computed labels JSON file (skips recomputation). "
                             "Supports both single-cell {queries:[...]} and multi-cell {cell:{queries:[...]}} formats.")
    parser.add_argument("--print-table", action="store_true", help="Print markdown table")
    args = parser.parse_args()

    # If --labels-file is provided, load pre-computed labels and print summary
    if args.labels_file is not None:
        labels_path = Path(args.labels_file)
        queries = load_labels_from_file(labels_path)
        if queries is None:
            print(f"ERROR: could not parse labels from {labels_path}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps({"cell": args.cell_name, "labels_file": str(labels_path),
                          "n_queries": len(queries)}, indent=2))
        return

    summary = analyze_cell(
        noref_eval_dir=Path(args.noref_dir),
        egrefine_eval_dir=Path(args.egrefine_dir),
        refine_dir=Path(args.refine_dir),
        methods=tuple(args.methods),
        label_dump_path=Path(args.label_dump) if args.label_dump else None,
        cell_name=args.cell_name,
    )

    print(json.dumps({"cell": args.cell_name, "summary": summary}, indent=2))
    if args.print_table:
        print()
        print(format_table(summary, args.cell_name))


if __name__ == "__main__":
    main()
