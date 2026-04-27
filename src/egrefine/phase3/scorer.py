"""Phase 3: Candidate Scorer — ExAcc-based scoring with conservative selection."""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from egrefine.data.schema import Column, NLSQLPair, Schema
from egrefine.phase2.prompts import CandidateName
from egrefine.phase3.executor import compare_results
from egrefine.phase3.text2sql_runner import Text2SQLModel
from egrefine.phase4.backmapper import backmap

logger = logging.getLogger(__name__)


@dataclass
class QueryDetail:
    """Per-query scoring detail for logging."""
    nl: str
    gold_sql: str
    pred_sql: str
    backmapped_sql: Optional[str]
    match: bool

    def to_dict(self) -> dict:
        d = {
            "nl": self.nl,
            "gold_sql": self.gold_sql,
            "pred_sql": self.pred_sql,
            "match": self.match,
        }
        if self.backmapped_sql is not None:
            d["backmapped_sql"] = self.backmapped_sql
        return d


@dataclass
class SelectionResult:
    """Result of conservative candidate selection for one column."""
    column: Column
    selected_name: str
    delta: float
    was_changed: bool
    all_scores: Dict[str, float]
    verification_method: str = "execution"  # "execution" or "llm_fallback"


@dataclass
class ScoringDetail:
    """Detailed scoring record for one candidate name."""
    candidate_name: str
    exacc: Optional[float]
    query_details: List[QueryDetail]

    def to_dict(self) -> dict:
        return {
            "exacc": self.exacc,
            "queries": [q.to_dict() for q in self.query_details],
        }


def score_candidate(
    column: Column,
    candidate_name: str,
    queries: List[NLSQLPair],
    model: Text2SQLModel,
    schema: Schema,
    db_path: str,
    collect_details: bool = False,
    query_workers: int = 32,
    use_evidence: bool = False,
) -> Tuple[Optional[float], List[QueryDetail]]:
    """Score a single candidate name for a column using one Text-to-SQL model.

    Builds a modified schema with the candidate name, runs Text-to-SQL on each
    query in Q(c_i), backmaps the predicted SQL, and computes ExAcc.

    Args:
        collect_details: If True, return per-query details.
        query_workers: Max threads for parallel query evaluation.
        use_evidence: If True, forward per-query BIRD evidence to the model.

    Returns:
        Tuple of (ExAcc score or None, list of QueryDetail).
        QueryDetail list is empty if collect_details is False.
    """
    if not queries:
        return None, []

    # Build modified schema: replace column name with candidate
    is_original = (candidate_name == column.name)
    if is_original:
        modified_schema = schema
        reverse_mapping = {}
    else:
        mapping = {column.full_name: candidate_name}
        modified_schema = schema.apply_refinement(mapping)
        reverse_mapping = {candidate_name: column.name}

    def _eval_query(pair: NLSQLPair):
        evidence = pair.evidence if use_evidence else ""
        pred_sql = model.generate(
            pair.nl, modified_schema,
            db_path=db_path, column_mapping=reverse_mapping,
            evidence=evidence,
        )
        if reverse_mapping:
            original_sql = backmap(pred_sql, reverse_mapping)
        else:
            original_sql = pred_sql
        match = compare_results(original_sql, pair.gold_sql, db_path)
        return pair, pred_sql, original_sql if reverse_mapping else None, match

    correct = 0
    details: List[QueryDetail] = []
    actual_workers = min(query_workers, len(queries))

    if actual_workers > 1:
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            for pair, pred_sql, backmapped_sql, match in pool.map(_eval_query, queries):
                if match:
                    correct += 1
                if collect_details:
                    details.append(QueryDetail(
                        nl=pair.nl, gold_sql=pair.gold_sql,
                        pred_sql=pred_sql, backmapped_sql=backmapped_sql,
                        match=match,
                    ))
    else:
        for pair in queries:
            pair, pred_sql, backmapped_sql, match = _eval_query(pair)
            if match:
                correct += 1
            if collect_details:
                details.append(QueryDetail(
                    nl=pair.nl, gold_sql=pair.gold_sql,
                    pred_sql=pred_sql, backmapped_sql=backmapped_sql,
                    match=match,
                ))

    return correct / len(queries), details


def _score_one_candidate(
    column: Column,
    name: str,
    queries: List[NLSQLPair],
    models: List[Text2SQLModel],
    schema: Schema,
    db_path: str,
    collect_details: bool = False,
    query_workers: int = 32,
    use_evidence: bool = False,
) -> Tuple[str, Optional[float], List[QueryDetail]]:
    """Score a single candidate across all models. Returns (name, avg_score, details)."""
    model_scores = []
    last_details: List[QueryDetail] = []

    if len(models) > 1:
        # Multi-model: run models in parallel to maximize LLM throughput
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run_model(model):
            return model, score_candidate(
                column, name, queries, model, schema, db_path,
                collect_details=collect_details,
                query_workers=query_workers,
                use_evidence=use_evidence,
            )

        with ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = [pool.submit(_run_model, m) for m in models]
            for fut in as_completed(futures):
                model, (s, details) = fut.result()
                if s is not None:
                    model_scores.append(s)
                    model_name = type(model).__name__
                    logger.debug(
                        "    %s.%s -> '%s': %s ExAcc=%.3f",
                        column.table, column.name, name, model_name, s,
                    )
                if details:
                    last_details = details
    else:
        # Single model: no threading overhead
        s, details = score_candidate(
            column, name, queries, models[0], schema, db_path,
            collect_details=collect_details,
            query_workers=query_workers,
            use_evidence=use_evidence,
        )
        if s is not None:
            model_scores.append(s)
        if details:
            last_details = details

    if model_scores:
        avg = sum(model_scores) / len(model_scores)
        return name, avg, last_details
    return name, None, last_details


def score_all_candidates(
    column: Column,
    candidates: List[CandidateName],
    queries: List[NLSQLPair],
    models: List[Text2SQLModel],
    schema: Schema,
    db_path: str,
    collect_details: bool = False,
    max_workers: int = 4,
    query_workers: int = 32,
    use_evidence: bool = False,
) -> Tuple[Dict[str, float], Dict[str, ScoringDetail]]:
    """Score original name + all candidates, averaging across models.

    Candidates are scored in parallel using threads (LLM calls are I/O-bound).

    Returns:
        Tuple of:
        - Dict mapping candidate name -> averaged ExAcc score (empty if no queries).
        - Dict mapping candidate name -> ScoringDetail (empty if collect_details is False).
    """
    if not queries:
        logger.warning("No queries for %s, cannot score", column.full_name)
        return {}, {}

    all_names = [column.name] + [c.name for c in candidates]
    scores: Dict[str, float] = {}
    all_details: Dict[str, ScoringDetail] = {}

    # Score candidates — parallel when max_workers > 1, serial otherwise
    actual_workers = min(max_workers, len(all_names))
    if actual_workers > 1:
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            futures = {
                pool.submit(
                    _score_one_candidate,
                    column, cand_name, queries, models, schema, db_path,
                    collect_details, query_workers, use_evidence,
                ): cand_name
                for cand_name in all_names
            }
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    failed = futures[future]
                    logger.error("Scoring failed for %s.%s -> '%s'",
                                 column.table, column.name, failed, exc_info=True)
    else:
        results = [
            _score_one_candidate(column, n, queries, models, schema, db_path,
                                 collect_details, query_workers,
                                 use_evidence=use_evidence)
            for n in all_names
        ]

    for cand_name, avg, last_details in results:
        if avg is not None:
            scores[cand_name] = avg
            logger.info(
                "  %s.%s -> '%s': ExAcc=%.3f (avg of %d models)",
                column.table, column.name, cand_name, avg, len(models),
            )
            if collect_details:
                all_details[cand_name] = ScoringDetail(
                    candidate_name=cand_name, exacc=avg, query_details=last_details,
                )

    return scores, all_details


def select_best(
    column: Column,
    candidates: List[CandidateName],
    scores: Dict[str, float],
    conservative: bool = True,
    min_delta: float = 0.0,
) -> SelectionResult:
    """Select the best candidate name for a column.

    Args:
        conservative: If True (default), only pick a candidate if it strictly
            beats the original name's score. If False, always pick the best
            candidate regardless.
        min_delta: Minimum ExAcc improvement required to accept a rename.
            Prevents marginal delta changes driven by LLM randomness.

    If scores is empty (Q(c_i) was empty), falls back to the first LLM candidate.
    """
    # Q(c_i) was empty: no execution verification possible, keep original name
    if not scores:
        logger.info(
            "No queries for %s: keeping original name (skipped_no_queries)",
            column.full_name,
        )
        return SelectionResult(
            column=column,
            selected_name=column.name,
            delta=0.0,
            was_changed=False,
            all_scores={},
            verification_method="skipped_no_queries",
        )

    original_score = scores.get(column.name, 0.0)
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]

    delta = best_score - original_score

    # Should we pick the best candidate?
    if conservative:
        # Conservative rule: strict improvement required + minimum delta
        should_change = (
            best_name != column.name
            and best_score > original_score
            and delta >= min_delta
        )
    else:
        # No conservative rule: always pick best (even if tied or worse)
        should_change = best_name != column.name

    if should_change:
        return SelectionResult(
            column=column,
            selected_name=best_name,
            delta=delta,
            was_changed=True,
            all_scores=scores,
        )

    # Detect when min_delta blocks an otherwise positive rename
    blocked_by_threshold = (
        best_name != column.name
        and best_score > original_score
        and delta < min_delta
    )
    if blocked_by_threshold:
        logger.info(
            "  %s: skipped rename '%s'→'%s' (delta=%.3f < min_delta=%.3f)",
            column.full_name, column.name, best_name, delta, min_delta,
        )

    return SelectionResult(
        column=column,
        selected_name=column.name,
        delta=0.0,
        was_changed=False,
        all_scores=scores,
        verification_method="skipped_below_threshold" if blocked_by_threshold else "execution",
    )
