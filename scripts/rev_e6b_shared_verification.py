#!/usr/bin/env python3
"""Revision E6b: query-conditioned candidates with shared Phase-3 scoring.

This runner isolates the effect of user-side query signals in candidate
generation.  For every pre-registered target column it:

1. generates a standard (schema/data-only) candidate pool;
2. generates a workload-conditioned pool using only refinement-split queries;
3. scores the union of both pools exactly once in Phase 3; and
4. selects conservatively inside each arm using subsets of the shared scores.

The shared score dictionary removes independent-verification noise from the
comparison.  Artifacts for both arms are directly consumable by run_eval.py.
"""
import argparse
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_refine import (  # noqa: E402
    _load_forced_targets,
    _resolve_forced_columns,
    _save_artifacts,
)
from egrefine.config import load_config  # noqa: E402
from egrefine.data.benchmark import BIRDLoader, DrSpiderLoader  # noqa: E402
from egrefine.data.query_index import build_query_index  # noqa: E402
from egrefine.models.llm_client import LLMClient  # noqa: E402
from egrefine.phase2.generator import CandidateGenerator  # noqa: E402
from egrefine.phase2.prompts import CandidateName  # noqa: E402
from egrefine.phase3.c3_runner import C3Text2SQL  # noqa: E402
from egrefine.phase3.conflict import resolve_conflicts  # noqa: E402
from egrefine.phase3.dinsql_runner import DINSQLText2SQL  # noqa: E402
from egrefine.phase3.macsql_runner import MACSQLText2SQL  # noqa: E402
from egrefine.phase3.scorer import SelectionResult, score_all_candidates, select_best  # noqa: E402
from egrefine.phase4.view_synthesis import synthesize  # noqa: E402
from egrefine.pipeline import PipelineResult  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "e6b-shared-union-v1"


def deduplicate_candidates(
    pools: Iterable[Sequence[CandidateName]],
    original_name: str,
) -> List[CandidateName]:
    """Return first-occurrence, exact-name union, excluding the original."""
    result: List[CandidateName] = []
    seen = {original_name}
    for pool in pools:
        for candidate in pool:
            if candidate.name in seen:
                continue
            seen.add(candidate.name)
            result.append(candidate)
    return result


def restrict_scores(
    scores: Dict[str, float],
    original_name: str,
    candidates: Sequence[CandidateName],
) -> Dict[str, float]:
    """Restrict shared scores to original + one arm's candidate pool.

    ``select_best`` operates on every key in the supplied score dictionary, so
    explicit restriction here is essential to prevent candidates crossing arms.
    """
    allowed = {original_name, *(c.name for c in candidates)}
    return {name: score for name, score in scores.items() if name in allowed}


def pool_best_score(
    scores: Dict[str, float],
    original_name: str,
    candidates: Sequence[CandidateName],
) -> float:
    subset = restrict_scores(scores, original_name, candidates)
    return max(subset.values())


def compare_candidate_pools(
    original_name: str,
    standard: Sequence[CandidateName],
    conditioned: Sequence[CandidateName],
    shared_scores: Dict[str, float],
    min_delta: float,
) -> Dict[str, object]:
    """Compute proposal-level outcomes before conservative selection."""
    original_score = shared_scores[original_name]
    standard_best = pool_best_score(shared_scores, original_name, standard)
    conditioned_best = pool_best_score(shared_scores, original_name, conditioned)
    standard_pass = standard_best - original_score >= min_delta
    conditioned_pass = conditioned_best - original_score >= min_delta
    return {
        "original_score": original_score,
        "standard_best_score": standard_best,
        "conditioned_best_score": conditioned_best,
        "conditioned_minus_standard_best": conditioned_best - standard_best,
        "proposal_rescue": bool(conditioned_pass and not standard_pass),
        "proposal_reversal": bool(standard_pass and not conditioned_pass),
    }


class SharedScoreCache:
    """File-backed cache that reuses scores only for the exact candidate union."""

    def __init__(self, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, "shared_phase3_scores.json")
        if os.path.isfile(self.path):
            with open(self.path, "r") as f:
                payload = json.load(f)
            if payload.get("protocol") != PROTOCOL_VERSION:
                raise ValueError(
                    f"Incompatible shared-score cache protocol in {self.path}"
                )
            self.data = payload.get("entries", {})
        else:
            self.data = {}

    @staticmethod
    def _key(db_id: str, full_name: str) -> str:
        return f"{db_id}:{full_name}"

    def get(
        self,
        db_id: str,
        full_name: str,
        expected_names: Sequence[str],
    ) -> Optional[Dict[str, float]]:
        entry = self.data.get(self._key(db_id, full_name))
        if not entry or entry.get("candidate_names") != list(expected_names):
            return None
        scores = entry.get("scores", {})
        if set(scores) != set(expected_names):
            return None
        return {name: float(score) for name, score in scores.items()}

    def put(
        self,
        db_id: str,
        full_name: str,
        candidate_names: Sequence[str],
        scores: Dict[str, float],
    ) -> None:
        self.data[self._key(db_id, full_name)] = {
            "candidate_names": list(candidate_names),
            "scores": scores,
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(
                {"protocol": PROTOCOL_VERSION, "entries": self.data},
                f,
                indent=2,
            )
        os.replace(tmp_path, self.path)


def _build_loader(benchmark: str, config: dict):
    if benchmark == "bird":
        return BIRDLoader(config["data"]["bird"]["path"])
    return DrSpiderLoader(config["data"][benchmark]["path"])


def _build_models(
    config: dict,
    benchmark: str,
    verify_algorithms: str,
) -> List[object]:
    text2sql_configs = config["models"]["text2sql"]
    if not isinstance(text2sql_configs, list):
        text2sql_configs = [text2sql_configs]

    desc_dir = None
    if benchmark == "bird":
        desc_dir = os.path.join(
            config["data"]["bird"]["path"], "dev_databases"
        )

    models: List[object] = []
    for algorithm in verify_algorithms.split("+"):
        for model_config in text2sql_configs:
            if algorithm == "c3":
                models.append(
                    C3Text2SQL(
                        model_config,
                        num_samples=3,
                        sample_rows=3,
                        desc_dir=desc_dir,
                    )
                )
            elif algorithm == "dinsql":
                models.append(DINSQLText2SQL(model_config, self_correction=True))
            elif algorithm == "macsql":
                models.append(MACSQLText2SQL(model_config))
            else:
                raise ValueError(f"Unknown verification algorithm: {algorithm}")
    return models


def _candidate_dict(candidates: Sequence[CandidateName]) -> List[dict]:
    return [{"name": c.name, "reason": c.reason} for c in candidates]


def _selection_dict(result: SelectionResult) -> dict:
    selected_score = result.all_scores.get(result.selected_name)
    return {
        "selected_name": result.selected_name,
        "selected_score": selected_score,
        "delta": result.delta,
        "was_changed": result.was_changed,
        "verification_method": result.verification_method,
    }


def _make_pipeline_result(
    schema,
    results: List[SelectionResult],
    elapsed: float,
) -> PipelineResult:
    synthesis = synthesize(schema, results)
    changed = [result for result in results if result.was_changed]
    statistics = {
        "protocol": PROTOCOL_VERSION,
        "total_columns": len(schema.all_columns),
        "candidates_after_pruning": len(results),
        "columns_scored": len(results),
        "columns_refined": len(changed),
        "columns_kept_original": len(results) - len(changed),
        "fallback_count": 0,
        "skipped_below_threshold": sum(
            result.verification_method == "skipped_below_threshold"
            for result in results
        ),
        "avg_delta": (
            sum(result.delta for result in changed) / len(changed)
            if changed else 0.0
        ),
        "max_delta": max((result.delta for result in changed), default=0.0),
        "timings": {"shared_phase2_phase3": round(elapsed, 2)},
    }
    return PipelineResult(
        db_id=schema.db_id,
        refinements=results,
        views=synthesis["views"],
        mapping=synthesis["mapping"],
        reverse_mapping=synthesis["reverse_mapping"],
        orig_table_map=synthesis["orig_table_map"],
        statistics=statistics,
    )


def _write_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _aggregate(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"target_columns": 0}
    gains = [row["conditioned_minus_standard_selected"] for row in rows]
    return {
        "target_columns": len(rows),
        "proposal_rescues": sum(row["proposal_rescue"] for row in rows),
        "proposal_reversals": sum(row["proposal_reversal"] for row in rows),
        "commit_rescues": sum(row["commit_rescue"] for row in rows),
        "commit_reversals": sum(row["commit_reversal"] for row in rows),
        "conditioned_better": sum(gain > 0 for gain in gains),
        "standard_better": sum(gain < 0 for gain in gains),
        "selected_score_ties": sum(gain == 0 for gain in gains),
        "mean_conditioned_minus_standard_selected": sum(gains) / len(gains),
        "standard_columns_refined": sum(
            row["standard_final_selection"]["was_changed"] for row in rows
        ),
        "conditioned_columns_refined": sum(
            row["conditioned_final_selection"]["was_changed"] for row in rows
        ),
        "standard_unique_candidates": sum(
            len({c["name"] for c in row["standard_candidates"]}) for row in rows
        ),
        "conditioned_unique_candidates": sum(
            len({c["name"] for c in row["conditioned_candidates"]}) for row in rows
        ),
        "conditioned_novel_candidates": sum(
            len(row["conditioned_novel_names"]) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["bird", "drspider_abbr", "drspider_syn"],
    )
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--candidate-temperature",
        type=float,
        default=None,
        help="Override Phase-2 temperature for a deterministic sensitivity run",
    )
    parser.add_argument("--dbs", nargs="*", default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument(
        "--verify-algorithms",
        default="c3+dinsql",
        choices=[
            "c3", "dinsql", "macsql", "c3+dinsql", "c3+macsql",
            "dinsql+macsql",
        ],
    )
    parser.add_argument("--use-evidence", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    loader = _build_loader(args.benchmark, config)
    targets_by_db = _load_forced_targets(args.targets)
    if args.dbs:
        unknown = [db_id for db_id in args.dbs if db_id not in targets_by_db]
        if unknown:
            parser.error("Requested DBs without targets: " + ", ".join(unknown))
        targets_by_db = OrderedDict(
            (db_id, targets_by_db[db_id]) for db_id in args.dbs
        )

    if args.max_targets:
        remaining = args.max_targets
        limited = OrderedDict()
        for db_id, names in targets_by_db.items():
            if remaining <= 0:
                break
            limited[db_id] = names[:remaining]
            remaining -= len(limited[db_id])
        targets_by_db = limited

    missing_dbs = [db_id for db_id in targets_by_db if db_id not in loader.schemas]
    if missing_dbs:
        raise ValueError("Target DBs missing from benchmark: " + ", ".join(missing_dbs))

    target_tag = os.path.splitext(os.path.basename(args.targets))[0]
    config_tag = os.path.splitext(os.path.basename(args.config))[0]
    algorithm_tag = args.verify_algorithms.replace("+", "_")
    cache_dir = args.cache_dir or (
        config["output"].get("cache_dir", "./cache")
        + f"_e6b_shared_{config_tag}_{target_tag}_{algorithm_tag}"
        + ("_evidence" if args.use_evidence else "")
    )

    llm_config = dict(config["models"]["candidate_llm"])
    if args.candidate_temperature is not None:
        llm_config["temperature"] = args.candidate_temperature
    llm_client = LLMClient(llm_config)
    desc_base_path = (
        config["data"]["bird"]["path"] if args.benchmark == "bird" else None
    )
    standard_generator = CandidateGenerator(
        llm_client,
        cache_dir=os.path.join(cache_dir, "phase2_standard"),
        sample_rows=config["phase2"].get("sample_rows", 20),
        max_retries=llm_config.get("max_retries", 3),
        desc_base_path=desc_base_path,
    )
    conditioned_generator = CandidateGenerator(
        llm_client,
        cache_dir=os.path.join(cache_dir, "phase2_conditioned"),
        sample_rows=config["phase2"].get("sample_rows", 20),
        max_retries=llm_config.get("max_retries", 3),
        desc_base_path=desc_base_path,
    )
    score_cache = SharedScoreCache(cache_dir)
    models = _build_models(config, args.benchmark, args.verify_algorithms)

    max_workers = max(1, config.get("concurrency", {}).get("max_workers", 1))
    k = config["phase2"].get("k", 3)
    phase3_config = config["phase3"]
    conservative = phase3_config.get("conservative", True)
    min_delta = phase3_config.get("min_delta", 0.0)
    max_rounds = phase3_config.get("conflict_resolution_rounds", 2)

    os.makedirs(args.output, exist_ok=True)
    all_rows: List[dict] = []
    db_summaries: List[dict] = []
    total_start = time.time()

    logger.info(
        "Protocol=%s benchmark=%s targets=%d models=%s cache=%s",
        PROTOCOL_VERSION,
        args.benchmark,
        sum(len(names) for names in targets_by_db.values()),
        [type(model).__name__ for model in models],
        cache_dir,
    )

    for db_index, (db_id, target_names) in enumerate(targets_by_db.items(), 1):
        db_start = time.time()
        schema = loader.schemas[db_id]
        db_path = loader.get_db_path(db_id)
        columns = _resolve_forced_columns(schema, target_names)
        pairs = loader.get_pairs_for_db(db_id)
        query_index = build_query_index(pairs, schema)
        logger.info(
            "[%d/%d] %s: %d targets, %d refinement queries",
            db_index, len(targets_by_db), db_id, len(columns), len(pairs),
        )

        phase2: Dict[str, Tuple[List[CandidateName], List[CandidateName]]] = {}

        def _generate(column):
            questions = [
                pair.nl for pair in query_index.get(column.full_name, [])[:3]
            ]
            if not questions:
                raise ValueError(
                    f"Pre-registered target has no refinement queries: "
                    f"{db_id}:{column.full_name}"
                )
            standard = standard_generator.generate(
                column, schema, db_path, k=k, sample_questions=None
            )
            conditioned = conditioned_generator.generate(
                column, schema, db_path, k=k, sample_questions=questions
            )
            if not standard or not conditioned:
                raise RuntimeError(
                    f"Empty candidate pool for {db_id}:{column.full_name}"
                )
            return column.full_name, standard, conditioned

        generation_workers = min(max_workers, len(columns))
        if generation_workers > 1:
            with ThreadPoolExecutor(max_workers=generation_workers) as pool:
                futures = [pool.submit(_generate, column) for column in columns]
                for future in as_completed(futures):
                    full_name, standard, conditioned = future.result()
                    phase2[full_name] = (standard, conditioned)
        else:
            for column in columns:
                full_name, standard, conditioned = _generate(column)
                phase2[full_name] = (standard, conditioned)

        standard_results: List[SelectionResult] = []
        conditioned_results: List[SelectionResult] = []
        db_rows: List[dict] = []

        for column in columns:
            standard, conditioned = phase2[column.full_name]
            union = deduplicate_candidates(
                [standard, conditioned], original_name=column.name
            )
            expected_names = [column.name] + [c.name for c in union]
            q_ci = query_index[column.full_name]
            shared_scores = score_cache.get(
                db_id, column.full_name, expected_names
            )
            cache_hit = shared_scores is not None
            if shared_scores is None:
                shared_scores, _ = score_all_candidates(
                    column=column,
                    candidates=union,
                    queries=q_ci,
                    models=models,
                    schema=schema,
                    db_path=db_path,
                    collect_details=False,
                    max_workers=len(expected_names),
                    query_workers=max_workers,
                    use_evidence=args.use_evidence,
                )
                missing_names = [
                    name for name in expected_names if name not in shared_scores
                ]
                if missing_names:
                    raise RuntimeError(
                        f"Incomplete shared scores for {db_id}:{column.full_name}; "
                        f"missing {missing_names}"
                    )
                score_cache.put(
                    db_id, column.full_name, expected_names, shared_scores
                )

            standard_scores = restrict_scores(
                shared_scores, column.name, standard
            )
            conditioned_scores = restrict_scores(
                shared_scores, column.name, conditioned
            )
            standard_result = select_best(
                column,
                standard,
                standard_scores,
                conservative=conservative,
                min_delta=min_delta,
            )
            conditioned_result = select_best(
                column,
                conditioned,
                conditioned_scores,
                conservative=conservative,
                min_delta=min_delta,
            )
            standard_results.append(standard_result)
            conditioned_results.append(conditioned_result)

            comparison = compare_candidate_pools(
                column.name,
                standard,
                conditioned,
                shared_scores,
                min_delta,
            )
            standard_names = {candidate.name for candidate in standard}
            row = {
                "db_id": db_id,
                "table": column.table,
                "column": column.name,
                "full_name": column.full_name,
                "query_count": len(q_ci),
                "sample_questions": [pair.nl for pair in q_ci[:3]],
                "standard_candidates": _candidate_dict(standard),
                "conditioned_candidates": _candidate_dict(conditioned),
                "conditioned_novel_names": [
                    candidate.name
                    for candidate in conditioned
                    if candidate.name not in standard_names
                    and candidate.name != column.name
                ],
                "union_candidate_names": [candidate.name for candidate in union],
                "shared_scores": shared_scores,
                "shared_score_cache_hit": cache_hit,
                **comparison,
                "standard_pre_conflict_selection": _selection_dict(standard_result),
                "conditioned_pre_conflict_selection": _selection_dict(conditioned_result),
            }
            db_rows.append(row)

        standard_resolved = resolve_conflicts(
            standard_results, schema, max_rounds=max_rounds, min_delta=min_delta
        )
        conditioned_resolved = resolve_conflicts(
            conditioned_results, schema, max_rounds=max_rounds, min_delta=min_delta
        )

        for row, standard_result, conditioned_result in zip(
            db_rows, standard_resolved, conditioned_resolved
        ):
            standard_final = _selection_dict(standard_result)
            conditioned_final = _selection_dict(conditioned_result)
            standard_score = standard_result.all_scores[standard_result.selected_name]
            conditioned_score = conditioned_result.all_scores[
                conditioned_result.selected_name
            ]
            row.update({
                "standard_final_selection": standard_final,
                "conditioned_final_selection": conditioned_final,
                "conditioned_minus_standard_selected": (
                    conditioned_score - standard_score
                ),
                "commit_rescue": bool(
                    conditioned_result.was_changed and not standard_result.was_changed
                ),
                "commit_reversal": bool(
                    standard_result.was_changed and not conditioned_result.was_changed
                ),
            })

        elapsed = time.time() - db_start
        standard_pipeline = _make_pipeline_result(
            schema, standard_resolved, elapsed
        )
        conditioned_pipeline = _make_pipeline_result(
            schema, conditioned_resolved, elapsed
        )
        _save_artifacts(
            standard_pipeline,
            schema,
            os.path.join(args.output, "standard", db_id),
        )
        _save_artifacts(
            conditioned_pipeline,
            schema,
            os.path.join(args.output, "workload_conditioned", db_id),
        )

        all_rows.extend(db_rows)
        db_summary = {
            "db_id": db_id,
            "elapsed": round(elapsed, 2),
            **_aggregate(db_rows),
        }
        db_summaries.append(db_summary)
        _write_json(
            os.path.join(args.output, "audit.json"),
            {
                "protocol": PROTOCOL_VERSION,
                "benchmark": args.benchmark,
                "config": args.config,
                "targets": args.targets,
                "verify_algorithms": args.verify_algorithms,
                "candidate_temperature": llm_config.get("temperature", 0.0),
                "min_delta": min_delta,
                "rows": all_rows,
            },
        )
        logger.info("[%s] complete: %s", db_id, db_summary)

    total_elapsed = time.time() - total_start
    summary = {
        "protocol": PROTOCOL_VERSION,
        "benchmark": args.benchmark,
        "config": args.config,
        "targets": args.targets,
        "verify_algorithms": args.verify_algorithms,
        "candidate_temperature": llm_config.get("temperature", 0.0),
        "use_evidence": args.use_evidence,
        "total_elapsed": round(total_elapsed, 2),
        "aggregate": _aggregate(all_rows),
        "databases": db_summaries,
    }
    _write_json(os.path.join(args.output, "summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
