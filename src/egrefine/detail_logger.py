"""Detailed result logger — saves per-phase outputs for data analysis."""
import json
import logging
import os
import shutil
import yaml
from typing import Dict, List, Optional

from egrefine.data.schema import Schema
from egrefine.phase1.pruner import PruneResult
from egrefine.phase2.prompts import CandidateName
from egrefine.phase3.scorer import SelectionResult, ScoringDetail
from egrefine.evaluate import EvalDetails

logger = logging.getLogger(__name__)


class DetailLogger:
    """Collects and saves detailed per-DB, per-phase outputs to disk."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _db_dir(self, db_id: str) -> str:
        d = os.path.join(self.output_dir, "per_db", db_id)
        os.makedirs(d, exist_ok=True)
        return d

    def save_config_snapshot(self, config: dict):
        """Save the config used for this experiment run."""
        path = os.path.join(self.output_dir, "config_snapshot.yaml")
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def setup_file_logging(self):
        """Add a file handler to root logger for full DEBUG output."""
        log_path = os.path.join(self.output_dir, "experiment.log")
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logging.getLogger().addHandler(handler)
        logger.info("File logging enabled: %s", log_path)
        return handler

    def save_schema_info(self, schema: Schema):
        """Save schema structure for reference."""
        d = self._db_dir(schema.db_id)
        info = {
            "db_id": schema.db_id,
            "tables": [
                {
                    "name": t.name,
                    "columns": [
                        {
                            "name": c.name,
                            "dtype": c.dtype,
                            "is_pk": c.is_pk,
                            "fk_target": c.fk_target,
                        }
                        for c in t.columns
                    ],
                }
                for t in schema.tables
            ],
            "foreign_keys": [list(fk) for fk in schema.foreign_keys],
            "total_columns": len(schema.all_columns),
        }
        _write_json(os.path.join(d, "schema_info.json"), info)

    def save_phase1(self, db_id: str, prune_result: PruneResult, schema: Schema):
        """Save Phase 1 pruning details."""
        d = self._db_dir(db_id)

        # Per-column signal breakdown
        all_columns_info = []
        for col in schema.all_columns:
            signals = []
            for sig_name, hits in prune_result.signal_hits.items():
                if col.full_name in hits:
                    signals.append(sig_name)
            all_columns_info.append({
                "table": col.table,
                "column": col.name,
                "dtype": col.dtype,
                "is_pk": col.is_pk,
                "signals": signals,
                "is_candidate": col.full_name in {c.full_name for c in prune_result.candidates},
            })

        info = {
            "total_columns": prune_result.total_columns,
            "candidate_count": prune_result.candidate_count,
            "compression_ratio": round(prune_result.compression_ratio, 4),
            "candidates": [c.full_name for c in prune_result.candidates],
            "signal_hits": {k: v for k, v in prune_result.signal_hits.items()},
            "skipped_pks": prune_result.skipped_pks,
            "all_columns": all_columns_info,
        }
        _write_json(os.path.join(d, "phase1_pruning.json"), info)

    def save_phase2(
        self,
        db_id: str,
        phase2_results: Dict[str, List[CandidateName]],
        sample_values: Optional[Dict[str, List[str]]] = None,
    ):
        """Save Phase 2 candidate generation details."""
        d = self._db_dir(db_id)

        entries = []
        for col_full_name, candidates in phase2_results.items():
            entry = {
                "column": col_full_name,
                "candidates": [
                    {"name": c.name, "reason": c.reason} for c in candidates
                ],
            }
            if sample_values and col_full_name in sample_values:
                entry["sample_values"] = sample_values[col_full_name]
            entries.append(entry)

        _write_json(os.path.join(d, "phase2_candidates.json"), entries)

    def save_phase3(
        self,
        db_id: str,
        selection_results: List[SelectionResult],
        scoring_details: Optional[Dict[str, Dict[str, ScoringDetail]]] = None,
    ):
        """Save Phase 3 scoring and selection details.

        Args:
            selection_results: Final selection results per column.
            scoring_details: {col_full_name: {candidate_name: ScoringDetail}}.
        """
        d = self._db_dir(db_id)

        entries = []
        for r in selection_results:
            entry = {
                "column": r.column.full_name,
                "original_name": r.column.name,
                "selected_name": r.selected_name,
                "delta": round(r.delta, 4),
                "was_changed": r.was_changed,
                "verification_method": r.verification_method,
                "all_scores": {k: round(v, 4) for k, v in r.all_scores.items()},
            }
            # Add per-query details if available
            if scoring_details and r.column.full_name in scoring_details:
                col_details = scoring_details[r.column.full_name]
                entry["scoring"] = {
                    name: detail.to_dict()
                    for name, detail in col_details.items()
                }
            entries.append(entry)

        _write_json(os.path.join(d, "phase3_scoring.json"), entries)

    def save_phase3_conflicts(self, db_id: str, conflicts_info: dict):
        """Save Phase 3 conflict resolution details."""
        d = self._db_dir(db_id)
        _write_json(os.path.join(d, "phase3_conflicts.json"), conflicts_info)

    def save_phase4(self, db_id: str, synthesis: dict):
        """Save Phase 4 VIEW definitions and mappings.

        Args:
            db_id: Database identifier.
            synthesis: Full synthesis dict returned by ``synthesize()``.
        """
        d = self._db_dir(db_id)

        views = synthesis.get("views", [])
        if views:
            with open(os.path.join(d, "phase4_views.sql"), "w") as f:
                f.write("\n\n".join(views))

        # Mapping
        _write_json(os.path.join(d, "phase4_mapping.json"), {
            "mapping": synthesis.get("mapping", {}),
            "reverse_mapping": synthesis.get("reverse_mapping", {}),
            "orig_table_map": synthesis.get("orig_table_map", {}),
        })

        # Refined tables JSON
        refined_tables = synthesis.get("refined_tables")
        if refined_tables:
            _write_json(os.path.join(d, "phase4_refined_tables.json"), refined_tables)

    def save_evaluation(self, db_id: str, method: str, eval_details: EvalDetails):
        """Save per-method evaluation details with per-query predictions."""
        d = self._db_dir(db_id)
        eval_dir = os.path.join(d, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        eval_details.method = method
        _write_json(os.path.join(eval_dir, f"{method}.json"), eval_details.to_dict())


def _write_json(path: str, data):
    """Write JSON with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug("Saved %s", path)
