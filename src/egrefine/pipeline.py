"""EGRefine Pipeline: Phase 1 → 2 → 3 → 4 orchestration."""
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from egrefine.data.schema import Schema, NLSQLPair
from egrefine.data.query_index import build_query_index
from egrefine.phase1.pruner import prune, prune_llm
from egrefine.phase2.generator import CandidateGenerator
from egrefine.phase2.prompts import CandidateName
from egrefine.phase3.scorer import score_all_candidates, select_best, SelectionResult, ScoringDetail
from egrefine.phase3.conflict import resolve_conflicts
from egrefine.phase3.text2sql_runner import Text2SQLModel
from egrefine.phase4.view_synthesis import generate_views, generate_mapping, synthesize

logger = logging.getLogger(__name__)


class Phase3Cache:
    """Thread-safe file-backed cache for Phase 3 scoring results.

    Stores per-column scores so that interrupted runs can resume
    without re-running expensive LLM calls + SQL execution.
    """

    def __init__(self, cache_dir: str):
        self._path = os.path.join(cache_dir, "phase3_scores.json")
        self._lock = threading.Lock()
        if os.path.isfile(self._path):
            with open(self._path, "r") as f:
                self._data = json.load(f)
            logger.info("Loaded %d Phase 3 cached entries from %s", len(self._data), self._path)
        else:
            self._data = {}

    def _key(self, db_id: str, full_name: str) -> str:
        return f"{db_id}:{full_name}"

    def get(self, db_id: str, full_name: str) -> Optional[Dict[str, float]]:
        return self._data.get(self._key(db_id, full_name))

    def put(self, db_id: str, full_name: str, scores: Dict[str, float]):
        with self._lock:
            self._data[self._key(db_id, full_name)] = scores
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._data, f)


@dataclass
class PipelineResult:
    """Complete pipeline output for one database."""
    db_id: str
    refinements: List[SelectionResult]
    views: List[str]
    mapping: Dict[str, str]
    reverse_mapping: Dict[str, str]
    orig_table_map: Dict[str, str]
    statistics: Dict

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "db_id": self.db_id,
            "refinements": [
                {
                    "table": r.column.table,
                    "column": r.column.name,
                    "original_name": r.column.name,
                    "refined_name": r.selected_name,
                    "delta": round(r.delta, 4),
                    "was_changed": r.was_changed,
                    "verification_method": r.verification_method,
                    "all_scores": {k: round(v, 4) for k, v in r.all_scores.items()},
                }
                for r in self.refinements
            ],
            "view_definitions": self.views,
            "mapping": self.mapping,
            "reverse_mapping": self.reverse_mapping,
            "orig_table_map": self.orig_table_map,
            "statistics": self.statistics,
        }


def run_pipeline(
    schema: Schema,
    pairs: List[NLSQLPair],
    db_path: str,
    models: List[Text2SQLModel],
    phase1_config: dict,
    phase2_config: dict,
    phase3_config: dict,
    phase4_config: dict,
    candidate_generator: CandidateGenerator,
    embedding_client=None,
    detail_logger=None,
    override_candidates: Optional[List] = None,
    max_workers: int = 1,
    cache_dir: Optional[str] = None,
    phase1_llm_client=None,
    use_evidence: bool = False,
) -> PipelineResult:
    """Run the full EGRefine pipeline on one database.

    Args:
        schema: Database schema.
        pairs: NL-SQL pairs for this database.
        db_path: Path to SQLite database file.
        models: Text-to-SQL models for Phase 3 scoring.
        phase1_config: Phase 1 configuration.
        phase2_config: Phase 2 configuration (k, sample_rows).
        phase3_config: Phase 3 configuration (conservative, conflict_resolution_rounds).
        phase4_config: Phase 4 configuration.
        candidate_generator: Pre-configured CandidateGenerator instance.
        embedding_client: Optional embedding client for Phase 1 S2 signal.
        detail_logger: Optional DetailLogger for saving per-phase outputs.
        override_candidates: If provided, skip Phase 1 and use these columns
            as candidates directly (for ablation: w/o Phase 1).

    Returns:
        PipelineResult with all outputs and statistics.
    """
    collect_details = detail_logger is not None
    timings = {}

    if detail_logger:
        detail_logger.save_schema_info(schema)

    # ===== Phase 1: Pruning =====
    t0 = time.time()
    if override_candidates is not None:
        from egrefine.phase1.pruner import PruneResult
        prune_result = PruneResult(
            candidates=override_candidates,
            total_columns=len(schema.all_columns),
            signal_hits={},
            skipped_pks=[],
        )
        logger.info(
            "[%s] Phase 1 SKIPPED (override): %d/%d candidates",
            schema.db_id, prune_result.candidate_count, prune_result.total_columns,
        )
    elif phase1_config.get("method") == "llm" and phase1_llm_client is not None:
        p1_concurrency = phase1_config.get("concurrency", 64)
        prune_result = prune_llm(
            schema, phase1_config, phase1_llm_client, db_path,
            concurrency=p1_concurrency,
        )
        logger.info(
            "[%s] Phase 1 (LLM): %d/%d candidates (%.1f%%)",
            schema.db_id, prune_result.candidate_count, prune_result.total_columns,
            prune_result.compression_ratio * 100,
        )
    else:
        prune_result = prune(schema, phase1_config, embedding_client=embedding_client)
        logger.info(
            "[%s] Phase 1: %d/%d candidates (%.1f%%)",
            schema.db_id, prune_result.candidate_count, prune_result.total_columns,
            prune_result.compression_ratio * 100,
        )
    timings["phase1"] = time.time() - t0

    if detail_logger:
        detail_logger.save_phase1(schema.db_id, prune_result, schema)

    # ===== Build query index =====
    db_pairs = [p for p in pairs if p.db_id == schema.db_id]
    query_index = build_query_index(db_pairs, schema)

    # ===== Phase 2: Candidate Generation (parallel) =====
    t0 = time.time()
    k = phase2_config.get("k", 3)
    phase2_results: Dict[str, List[CandidateName]] = {}

    if max_workers > 1 and len(prune_result.candidates) > 1:
        def _gen(col):
            return col.full_name, candidate_generator.generate(col, schema, db_path, k=k)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_gen, col) for col in prune_result.candidates]
            for fut in as_completed(futures):
                full_name, candidates = fut.result()
                phase2_results[full_name] = candidates
    else:
        for col in prune_result.candidates:
            candidates = candidate_generator.generate(col, schema, db_path, k=k)
            phase2_results[col.full_name] = candidates

    timings["phase2"] = time.time() - t0
    total_candidates = sum(len(v) for v in phase2_results.values())
    logger.info(
        "[%s] Phase 2: %d candidates generated for %d columns",
        schema.db_id, total_candidates, len(phase2_results),
    )

    if detail_logger:
        detail_logger.save_phase2(schema.db_id, phase2_results)

    # ===== Phase 3: Scoring + Selection (parallel) =====
    t0 = time.time()
    selection_results: List[SelectionResult] = []
    all_scoring_details: Dict[str, Dict[str, ScoringDetail]] = {}
    conservative = phase3_config.get("conservative", True)
    min_delta = phase3_config.get("min_delta", 0.0)

    # Phase 3 cache: resume from interrupted runs
    p3_cache = Phase3Cache(cache_dir) if cache_dir else None

    # Build list of columns to score (those with candidates and queries)
    cols_to_score = []
    for col in prune_result.candidates:
        candidates = phase2_results.get(col.full_name, [])
        if candidates:
            cols_to_score.append(col)

    # Score columns serially; parallelism is at the query level inside
    # score_candidate (I/O-bound LLM calls).  query_workers controls
    # how many queries run concurrently per candidate.
    query_workers = max(max_workers, 1)

    for col in cols_to_score:
        candidates = phase2_results[col.full_name]
        q_ci = query_index.get(col.full_name, [])

        # Check cache first
        cached_scores = p3_cache.get(schema.db_id, col.full_name) if p3_cache else None
        if cached_scores is not None:
            logger.info("  [cached] %s: %d scores", col.full_name, len(cached_scores))
            result = select_best(col, candidates, cached_scores, conservative=conservative, min_delta=min_delta)
            selection_results.append(result)
            continue

        scores, scoring_details = score_all_candidates(
            column=col,
            candidates=candidates,
            queries=q_ci,
            models=models,
            schema=schema,
            db_path=db_path,
            collect_details=collect_details,
            max_workers=len(candidates) + 1,  # all candidates + original in parallel
            query_workers=query_workers,
            use_evidence=use_evidence,
        )

        # Save to cache immediately
        if p3_cache and scores:
            p3_cache.put(schema.db_id, col.full_name, scores)

        result = select_best(col, candidates, scores, conservative=conservative, min_delta=min_delta)
        selection_results.append(result)
        if scoring_details:
            all_scoring_details[col.full_name] = scoring_details

    # Conflict resolution
    max_rounds = phase3_config.get("conflict_resolution_rounds", 2)
    resolved = resolve_conflicts(selection_results, schema, max_rounds=max_rounds, min_delta=min_delta)
    timings["phase3"] = time.time() - t0

    changed = [r for r in resolved if r.was_changed]
    logger.info(
        "[%s] Phase 3: %d/%d columns refined",
        schema.db_id, len(changed), len(resolved),
    )

    if detail_logger:
        detail_logger.save_phase3(schema.db_id, resolved, all_scoring_details)
        detail_logger.save_phase3_conflicts(schema.db_id, {
            "conflicts_found": 0,  # TODO: capture from resolve_conflicts
            "resolutions": [],
        })

    # ===== Phase 4: VIEW Synthesis =====
    t0 = time.time()
    synthesis = synthesize(schema, resolved)
    timings["phase4"] = time.time() - t0

    if detail_logger:
        detail_logger.save_phase4(schema.db_id, synthesis)

    # ===== Statistics =====
    fallback_count = sum(1 for r in resolved if r.verification_method == "llm_fallback")
    skipped_threshold = sum(1 for r in resolved if r.verification_method == "skipped_below_threshold")
    deltas = [r.delta for r in changed]
    statistics = {
        "total_columns": prune_result.total_columns,
        "candidates_after_pruning": prune_result.candidate_count,
        "columns_scored": len(resolved),
        "columns_refined": len(changed),
        "columns_kept_original": len(resolved) - len(changed),
        "fallback_count": fallback_count,
        "skipped_below_threshold": skipped_threshold,
        "min_delta": min_delta,
        "avg_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "max_delta": max(deltas) if deltas else 0.0,
        "signal_hits": prune_result.signal_hits,
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }

    logger.info("[%s] Pipeline complete. Stats: %s", schema.db_id, statistics)

    return PipelineResult(
        db_id=schema.db_id,
        refinements=resolved,
        views=synthesis["views"],
        mapping=synthesis["mapping"],
        reverse_mapping=synthesis["reverse_mapping"],
        orig_table_map=synthesis["orig_table_map"],
        statistics=statistics,
    )
