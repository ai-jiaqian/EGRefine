"""Baseline: No Refinement — use original schema as-is."""
import logging
from typing import List

from egrefine.data.schema import Schema, NLSQLPair
from egrefine.pipeline import PipelineResult

logger = logging.getLogger(__name__)


def run_no_refinement(
    schema: Schema,
    pairs: List[NLSQLPair],
    db_path: str,
) -> PipelineResult:
    """Baseline that does nothing: returns original schema unchanged.

    This provides the lower bound for comparison.
    """
    logger.info("[%s] no_refinement: returning original schema", schema.db_id)

    return PipelineResult(
        db_id=schema.db_id,
        refinements=[],
        views=[],
        mapping={},
        reverse_mapping={},
        orig_table_map={},
        statistics={
            "method": "no_refinement",
            "total_columns": len(schema.all_columns),
            "columns_refined": 0,
            "columns_kept_original": 0,
        },
    )
