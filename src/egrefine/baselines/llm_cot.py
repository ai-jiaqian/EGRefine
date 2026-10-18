"""Baseline: LLM CoT — use Chain-of-Thought reasoning to select best candidate."""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from egrefine.data.schema import Column, Schema, NLSQLPair
from egrefine.models.llm_client import LLMClient
from egrefine.phase1.pruner import prune, prune_llm
from egrefine.phase2.generator import CandidateGenerator
from egrefine.phase2.prompts import CandidateName
from egrefine.phase3.scorer import SelectionResult
from egrefine.phase4.view_synthesis import synthesize
from egrefine.pipeline import PipelineResult

logger = logging.getLogger(__name__)


def _build_cot_prompt(column: Column, schema: Schema, candidates: List[CandidateName]) -> str:
    """Build a CoT prompt for LLM to select the best candidate name."""
    table = schema.get_table(column.table)
    neighbors = [c for c in table.columns if c.name != column.name] if table else []

    neighbor_str = ", ".join(c.name for c in neighbors) or "(none)"
    candidate_str = "\n".join(
        f"  {i+1}. `{c.name}` — {c.reason}" for i, c in enumerate(candidates)
    )

    return f"""You are a database schema expert. A column needs a better name.

## Current Column
- Name: `{column.name}`
- Table: `{column.table}`
- Type: {column.dtype}
- Neighboring columns: {neighbor_str}

## Candidates
{candidate_str}

## Task
Think step by step about which candidate name is the best choice for improving clarity and consistency. Consider:
1. How descriptive and unambiguous is each name?
2. How well does it fit with neighboring column names?
3. Would a Text-to-SQL system understand queries better with this name?

Respond in JSON format:
{{"selected": "chosen_name", "reasoning": "your step-by-step reasoning"}}"""


def _parse_cot_response(text: str, candidates: List[CandidateName]) -> Optional[str]:
    """Parse LLM CoT response, return selected candidate name or None."""
    # Strip thinking tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Try JSON parse
    try:
        # Try direct parse
        data = json.loads(text)
        selected = data.get("selected", "")
        valid_names = {c.name for c in candidates}
        if selected in valid_names:
            return selected
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1).strip())
            selected = data.get("selected", "")
            valid_names = {c.name for c in candidates}
            if selected in valid_names:
                return selected
        except (json.JSONDecodeError, AttributeError):
            pass

    # Try finding JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            selected = data.get("selected", "")
            valid_names = {c.name for c in candidates}
            if selected in valid_names:
                return selected
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse CoT response: %s", text[:200])
    return None


def run_llm_cot(
    schema: Schema,
    pairs: List[NLSQLPair],
    db_path: str,
    phase1_config: dict,
    phase2_config: dict,
    candidate_generator: CandidateGenerator,
    llm_client: LLMClient,
    embedding_client=None,
    max_workers: int = 1,
    phase1_llm_client=None,
) -> PipelineResult:
    """Baseline: Phase 1 + Phase 2 + LLM CoT selection (no execution verification).

    Uses Chain-of-Thought prompting to let the LLM reason about which
    candidate name is best. This is a stronger LLM-only baseline.
    """
    # Phase 1: Pruning
    if phase1_config.get("method") == "llm" and phase1_llm_client is not None:
        p1_concurrency = phase1_config.get("concurrency", 64)
        prune_result = prune_llm(schema, phase1_config, phase1_llm_client, db_path, concurrency=p1_concurrency)
    else:
        prune_result = prune(schema, phase1_config, embedding_client=embedding_client)
    logger.info(
        "[%s] llm_cot Phase 1: %d candidates",
        schema.db_id, prune_result.candidate_count,
    )

    # Phase 2: Generate candidates + CoT selection
    k = phase2_config.get("k", 3)
    results: List[SelectionResult] = []

    def _process_col(col):
        candidates = candidate_generator.generate(col, schema, db_path, k=k)
        if not candidates:
            return None

        prompt = _build_cot_prompt(col, schema, candidates)
        try:
            response = llm_client.chat([{"role": "user", "content": prompt}])
            selected = _parse_cot_response(response, candidates)
        except Exception as e:
            logger.warning("CoT call failed for %s: %s", col.full_name, e)
            selected = None

        if selected is None:
            selected = candidates[0].name
            logger.info("CoT fallback for %s: using first candidate '%s'", col.full_name, selected)

        return SelectionResult(
            column=col,
            selected_name=selected,
            delta=0.0,
            was_changed=True,
            all_scores={},
            verification_method="llm_cot",
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
        "method": "llm_cot",
        "total_columns": prune_result.total_columns,
        "candidates_after_pruning": prune_result.candidate_count,
        "columns_refined": len(changed),
        "columns_kept_original": prune_result.candidate_count - len(changed),
    }

    logger.info("[%s] llm_cot: %d columns refined", schema.db_id, len(changed))

    return PipelineResult(
        db_id=schema.db_id,
        refinements=results,
        views=synthesis["views"],
        mapping=synthesis["mapping"],
        reverse_mapping=synthesis["reverse_mapping"],
        orig_table_map=synthesis["orig_table_map"],
        statistics=statistics,
    )
