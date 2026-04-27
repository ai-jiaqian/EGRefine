"""Phase 3: Global conflict resolution — detect and resolve naming collisions within scope."""
import logging
from typing import Dict, List, Set, Tuple

from egrefine.data.schema import Column, Schema
from egrefine.phase3.scorer import SelectionResult

logger = logging.getLogger(__name__)


def resolve_conflicts(
    results: List[SelectionResult],
    schema: Schema,
    max_rounds: int = 2,
    min_delta: float = 0.0,
) -> List[SelectionResult]:
    """Detect and resolve naming conflicts among refinement results.

    Two columns conflict if they are in the same scope (same table or FK-related)
    and both refined to the same name. The column with lower delta reverts to its
    next best candidate (or original name if no better candidate exists).

    Args:
        results: SelectionResult list from scorer.
        schema: Database schema (for scope computation).
        max_rounds: Maximum conflict resolution iterations.
        min_delta: Minimum delta threshold for accepting a rename.

    Returns:
        Updated list of SelectionResult with conflicts resolved.
    """
    if not results:
        return []

    # Work on mutable copies
    resolved = list(results)

    for round_num in range(max_rounds):
        conflicts = _find_conflicts(resolved, schema)
        if not conflicts:
            logger.info("Conflict resolution: no conflicts after round %d", round_num)
            break

        logger.info("Conflict resolution round %d: %d conflicts", round_num + 1, len(conflicts))
        for i, j in conflicts:
            _resolve_pair(resolved, i, j, min_delta=min_delta)

    # Final check
    remaining = _find_conflicts(resolved, schema)
    if remaining:
        logger.warning("Unresolved conflicts after %d rounds: %d", max_rounds, len(remaining))

    return resolved


def _find_conflicts(
    results: List[SelectionResult],
    schema: Schema,
) -> List[Tuple[int, int]]:
    """Find pairs of results that conflict (same scope, same selected name)."""
    conflicts = []
    for i in range(len(results)):
        if not results[i].was_changed:
            continue
        for j in range(i + 1, len(results)):
            if not results[j].was_changed:
                continue
            if results[i].selected_name != results[j].selected_name:
                continue
            if _in_same_scope(results[i].column, results[j].column, schema):
                conflicts.append((i, j))
    return conflicts


def _in_same_scope(col_a: Column, col_b: Column, schema: Schema) -> bool:
    """Check if two columns are in the same scope (same table or FK-related)."""
    if col_a.table == col_b.table:
        return True
    scope_a = {c.full_name for c in schema.scope(col_a)}
    return col_b.full_name in scope_a


def _resolve_pair(results: List[SelectionResult], i: int, j: int, min_delta: float = 0.0):
    """Resolve a conflict between results[i] and results[j].

    The one with lower delta reverts to its next best candidate.
    """
    if results[i].delta >= results[j].delta:
        winner_idx, loser_idx = i, j
    else:
        winner_idx, loser_idx = j, i

    loser = results[loser_idx]
    logger.info(
        "Conflict: '%s' claimed by %s (delta=%.3f) and %s (delta=%.3f). "
        "Reverting %s.",
        loser.selected_name,
        results[winner_idx].column.full_name, results[winner_idx].delta,
        loser.column.full_name, loser.delta,
        loser.column.full_name,
    )

    # Find next best candidate for loser
    next_name = _next_best(loser, exclude={loser.selected_name})
    original_name = loser.column.name
    original_score = loser.all_scores.get(original_name, 0.0)
    next_score = loser.all_scores.get(next_name, 0.0) if next_name else 0.0

    # Conservative: next best must still beat original by min_delta
    if next_name and next_score > original_score and (next_score - original_score) >= min_delta:
        results[loser_idx] = SelectionResult(
            column=loser.column,
            selected_name=next_name,
            delta=next_score - original_score,
            was_changed=True,
            all_scores=loser.all_scores,
            verification_method=loser.verification_method,
        )
    else:
        # Revert to original
        results[loser_idx] = SelectionResult(
            column=loser.column,
            selected_name=original_name,
            delta=0.0,
            was_changed=False,
            all_scores=loser.all_scores,
            verification_method=loser.verification_method,
        )


def _next_best(result: SelectionResult, exclude: Set[str]) -> str:
    """Find the next best candidate name from all_scores, excluding certain names."""
    original = result.column.name
    candidates = [
        (name, score) for name, score in result.all_scores.items()
        if name not in exclude and name != original
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]
