#!/usr/bin/env python3
"""Analyze E6b shared-verification refinements on held-out queries."""
import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.run_refine import _load_forced_targets
from egrefine.config import load_config
from egrefine.data.benchmark import BIRDLoader, DrSpiderLoader
from egrefine.data.query_index import extract_columns_from_sql


def exact_mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Two-sided exact McNemar test for paired binary outcomes."""
    if len(a) != len(b):
        raise ValueError("Paired vectors have different lengths")
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum(y and not x for x, y in zip(a, b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, i) for i in range(min(a_only, b_only) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def _load_loader(config: dict, benchmark: str):
    if benchmark == "bird":
        return BIRDLoader(config["data"]["bird"]["path"])
    return DrSpiderLoader(config["data"][benchmark]["path"])


def _load_matches(eval_dir: str, db_id: str, method: str) -> List[bool]:
    path = os.path.join(eval_dir, "per_db", db_id, f"{method}.json")
    with open(path, "r") as f:
        payload = json.load(f)
    details = payload.get("original_details") or payload.get("refined_details")
    if details is None:
        raise ValueError(f"No per-query details in {path}")
    return [bool(row["match"]) for row in details]


def _metrics(matches: Sequence[bool]) -> dict:
    total = len(matches)
    correct = sum(matches)
    return {
        "correct": correct,
        "total": total,
        "exacc": correct / total if total else None,
    }


def _slice(values: Sequence[bool], indices: Sequence[int]) -> List[bool]:
    return [values[index] for index in indices]


def _comparison(
    original: Sequence[bool],
    standard: Sequence[bool],
    conditioned: Sequence[bool],
) -> dict:
    original_metrics = _metrics(original)
    standard_metrics = _metrics(standard)
    conditioned_metrics = _metrics(conditioned)
    return {
        "original": original_metrics,
        "standard": standard_metrics,
        "conditioned": conditioned_metrics,
        "standard_minus_original": (
            standard_metrics["exacc"] - original_metrics["exacc"]
            if original else None
        ),
        "conditioned_minus_original": (
            conditioned_metrics["exacc"] - original_metrics["exacc"]
            if original else None
        ),
        "conditioned_minus_standard": (
            conditioned_metrics["exacc"] - standard_metrics["exacc"]
            if original else None
        ),
        "mcnemar_standard_vs_original": exact_mcnemar(original, standard),
        "mcnemar_conditioned_vs_original": exact_mcnemar(original, conditioned),
        "mcnemar_conditioned_vs_standard": exact_mcnemar(standard, conditioned),
    }


def _percent(value):
    return "--" if value is None else f"{100 * value:.2f}"


def _write_markdown(path: str, result: dict) -> None:
    lines = [
        "# E6b held-out analysis",
        "",
        f"- Benchmark: `{result['benchmark']}`",
        f"- Frozen targets: {result['coverage']['target_columns']}",
        f"- Targets with held-out support: "
        f"{result['coverage']['targets_with_heldout_support']}",
        f"- Held-out target-touching queries: "
        f"{result['coverage']['target_touching_queries']}",
        "",
    ]
    for method, method_result in result["methods"].items():
        lines.extend([
            f"## {method}",
            "",
            "| Scope | NoRef | Standard | Query-conditioned | WC-Std | p (WC vs Std) |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for scope in ("all_target_db_queries", "target_touching_queries"):
            row = method_result[scope]
            lines.append(
                f"| {scope} (n={row['original']['total']}) "
                f"| {_percent(row['original']['exacc'])} "
                f"| {_percent(row['standard']['exacc'])} "
                f"| {_percent(row['conditioned']['exacc'])} "
                f"| {_percent(row['conditioned_minus_standard'])} "
                f"| {row['mcnemar_conditioned_vs_standard']['p_value']:.4g} |"
            )
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["bird", "drspider_abbr", "drspider_syn"],
    )
    parser.add_argument("--targets", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--original-eval", required=True)
    parser.add_argument("--standard-eval", required=True)
    parser.add_argument("--conditioned-eval", required=True)
    parser.add_argument("--methods", nargs="+", default=["c3", "dinsql"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    loader = _load_loader(config, args.benchmark)
    targets_by_db = _load_forced_targets(args.targets)
    with open(args.audit, "r") as f:
        audit = json.load(f)

    target_query_indices: Dict[str, List[int]] = {}
    target_support: Dict[str, int] = OrderedDict()
    all_query_count = 0
    all_target_query_count = 0
    for db_id, target_names in targets_by_db.items():
        schema = loader.schemas[db_id]
        target_set = set(target_names)
        indices = []
        support = {name: 0 for name in target_names}
        for index, pair in enumerate(loader.get_pairs_for_db(db_id)):
            referenced = {
                column.full_name
                for column in extract_columns_from_sql(pair.gold_sql, schema)
            }
            touching = referenced & target_set
            if touching:
                indices.append(index)
                for name in touching:
                    support[name] += 1
        target_query_indices[db_id] = indices
        target_support.update(
            (f"{db_id}:{name}", count) for name, count in support.items()
        )
        all_query_count += len(loader.get_pairs_for_db(db_id))
        all_target_query_count += len(indices)

    result = {
        "benchmark": args.benchmark,
        "coverage": {
            "target_columns": sum(len(names) for names in targets_by_db.values()),
            "targets_with_heldout_support": sum(
                count > 0 for count in target_support.values()
            ),
            "target_databases": len(targets_by_db),
            "all_queries_in_target_databases": all_query_count,
            "target_touching_queries": all_target_query_count,
            "per_target_query_count": target_support,
        },
        "candidate_generation": audit.get("rows", []),
        "methods": {},
    }

    for method in args.methods:
        original_all: List[bool] = []
        standard_all: List[bool] = []
        conditioned_all: List[bool] = []
        original_touching: List[bool] = []
        standard_touching: List[bool] = []
        conditioned_touching: List[bool] = []
        for db_id in targets_by_db:
            original = _load_matches(args.original_eval, db_id, method)
            standard = _load_matches(args.standard_eval, db_id, method)
            conditioned = _load_matches(args.conditioned_eval, db_id, method)
            expected = len(loader.get_pairs_for_db(db_id))
            if not (len(original) == len(standard) == len(conditioned) == expected):
                raise ValueError(f"Per-query length mismatch for {db_id}:{method}")
            indices = target_query_indices[db_id]
            original_all.extend(original)
            standard_all.extend(standard)
            conditioned_all.extend(conditioned)
            original_touching.extend(_slice(original, indices))
            standard_touching.extend(_slice(standard, indices))
            conditioned_touching.extend(_slice(conditioned, indices))

        result["methods"][method] = {
            "all_target_db_queries": _comparison(
                original_all, standard_all, conditioned_all
            ),
            "target_touching_queries": _comparison(
                original_touching, standard_touching, conditioned_touching
            ),
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    _write_markdown(os.path.splitext(args.output)[0] + ".md", result)
    print(json.dumps({"coverage": result["coverage"], "methods": result["methods"]}, indent=2))


if __name__ == "__main__":
    main()
