"""Baseline: LLM Direct — take first LLM candidate without execution verification."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from egrefine.data.schema import Schema, NLSQLPair
from egrefine.phase1.pruner import prune, prune_llm
from egrefine.phase2.generator import CandidateGenerator
from egrefine.phase2.prompts import CandidateName
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase4.view_synthesis import synthesize
from egrefine.pipeline import PipelineResult

logger = logging.getLogger(__name__)


def run_llm_direct(
    schema: Schema,
    pairs: List[NLSQLPair],
    db_path: str,
    phase1_config: dict,
    phase2_config: dict,
    candidate_generator: CandidateGenerator,
    embedding_client=None,
    max_workers: int = 1,
    phase1_llm_client=None,
) -> PipelineResult:
    """Baseline: Phase 1 + Phase 2, then directly take first LLM candidate.

    Skips Phase 3 (execution verification). This tests whether LLM's
    implicit confidence ranking is sufficient without execution feedback.
    """
    # Phase 1: Pruning
    if phase1_config.get("method") == "llm" and phase1_llm_client is not None:
        p1_concurrency = phase1_config.get("concurrency", 64)
        prune_result = prune_llm(schema, phase1_config, phase1_llm_client, db_path, concurrency=p1_concurrency)
    else:
        prune_result = prune(schema, phase1_config, embedding_client=embedding_client)
    logger.info(
        "[%s] llm_direct Phase 1: %d candidates",
        schema.db_id, prune_result.candidate_count,
    )

    # Phase 2: Generate candidates
    k = phase2_config.get("k", 3)
    results: List[SelectionResult] = []

    def _process_col(col):
        candidates = candidate_generator.generate(col, schema, db_path, k=k)
        if not candidates:
            return None
        return SelectionResult(
            column=col,
            selected_name=candidates[0].name,
            delta=0.0,
            was_changed=True,
            all_scores={},
            verification_method="llm_direct",
        )

    if max_workers > 1 and len(prune_result.candidates) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_col, col): col for col in prune_result.candidates}
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    results.append(r)
    else:
        for col in prune_result.candidates:
            r = _process_col(col)
            if r is not None:
                results.append(r)

    # Phase 4: Synthesize
    synthesis = synthesize(schema, results)

    changed = [r for r in results if r.was_changed]
    statistics = {
        "method": "llm_direct",
        "total_columns": prune_result.total_columns,
        "candidates_after_pruning": prune_result.candidate_count,
        "columns_refined": len(changed),
        "columns_kept_original": prune_result.candidate_count - len(changed),
    }

    logger.info("[%s] llm_direct: %d columns refined", schema.db_id, len(changed))

    return PipelineResult(
        db_id=schema.db_id,
        refinements=results,
        views=synthesis["views"],
        mapping=synthesis["mapping"],
        reverse_mapping=synthesis["reverse_mapping"],
        orig_table_map=synthesis["orig_table_map"],
        statistics=statistics,
    )
