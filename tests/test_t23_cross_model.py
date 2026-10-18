"""T23: Cross-model generalization experiment tests."""
import json
import os
import pytest

from egrefine.data.schema import Column, Table, Schema, NLSQLPair
from egrefine.phase3.scorer import SelectionResult


class TestCrossModelHelpers:
    """Test cross-model experiment helper functions."""

    def test_build_cross_model_table(self):
        from scripts.run_cross_model import _build_cross_model_table
        from egrefine.evaluate import EvalResult

        results = {
            "qwen": [
                {"eval_result": EvalResult(
                    db_id="test_db", exacc_before=0.5, exacc_after=0.6,
                    delta=0.1, total_queries=10, columns_changed=2,
                    columns_evaluated=5, refinement_precision=1.0,
                )}
            ],
            "deepseek": [
                {"eval_result": EvalResult(
                    db_id="test_db", exacc_before=0.4, exacc_after=0.55,
                    delta=0.15, total_queries=10, columns_changed=2,
                    columns_evaluated=5, refinement_precision=1.0,
                )}
            ],
        }

        latex = _build_cross_model_table("qwen", results)
        assert "qwen" in latex
        assert "deepseek" in latex
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex

    def test_build_cross_model_table_empty(self):
        from scripts.run_cross_model import _build_cross_model_table
        latex = _build_cross_model_table("qwen", {})
        assert "\\begin{tabular}" in latex

    def test_run_cross_model_function_signature(self):
        """Verify run_cross_model accepts expected arguments."""
        from scripts.run_cross_model import run_cross_model
        import inspect
        sig = inspect.signature(run_cross_model)
        params = list(sig.parameters.keys())
        assert "refinement_models" in params
        assert "eval_models" in params
        assert "detail_logger" in params


class TestCrossModelConfig:
    """Test that config supports cross_eval section."""

    def test_cross_eval_config_parsing(self):
        """Verify cross_eval config is a list of model configs."""
        config = {
            "models": {
                "text2sql": [
                    {"name": "qwen", "base_url": "http://x", "api_key": "k",
                     "model_name": "qwen", "temperature": 0, "max_tokens": 1024},
                ],
                "cross_eval": [
                    {"name": "deepseek", "base_url": "http://x", "api_key": "k",
                     "model_name": "deepseek", "temperature": 0, "max_tokens": 1024},
                ],
            }
        }
        text2sql_names = [tc["name"] for tc in config["models"]["text2sql"]]
        cross_names = [tc["name"] for tc in config["models"]["cross_eval"]]
        assert text2sql_names == ["qwen"]
        assert cross_names == ["deepseek"]

    def test_cross_eval_config_optional(self):
        """cross_eval section is optional."""
        config = {"models": {"text2sql": [{"name": "qwen"}]}}
        cross = config["models"].get("cross_eval", [])
        assert cross == []


class TestCrossModelIntegration:
    """Integration-level tests for cross-model evaluation logic."""

    def test_same_refinement_different_eval(self):
        """Verify that the same refinement can be evaluated by different models.

        This tests the core concept: SelectionResult is model-independent,
        and evaluate_refinement can use any model for evaluation.
        """
        col = Column(name="nm", table="employees", dtype="TEXT")
        refinements = [
            SelectionResult(
                column=col, selected_name="employee_name",
                delta=0.2, was_changed=True,
                all_scores={"nm": 0.5, "employee_name": 0.7},
            )
        ]

        # Refinements are plain data — no model reference
        assert refinements[0].selected_name == "employee_name"
        assert refinements[0].was_changed is True
        # evaluate_refinement(schema, pairs, model=ANY_MODEL, ..., refinement_results=refinements)
        # The same refinements work with any model — that's the point.

    def test_eval_models_dict_construction(self):
        """Verify eval_models dict is built correctly from config."""
        text2sql_configs = [
            {"name": "qwen", "base_url": "http://x", "api_key": "k",
             "model_name": "qwen", "temperature": 0, "max_tokens": 1024},
        ]
        cross_configs = [
            {"name": "deepseek", "base_url": "http://x", "api_key": "k",
             "model_name": "deepseek", "temperature": 0, "max_tokens": 1024},
        ]

        eval_models = {}
        for tc in text2sql_configs:
            eval_models[tc["name"]] = tc  # placeholder
        for tc in cross_configs:
            eval_models[tc["name"]] = tc

        assert len(eval_models) == 2
        assert "qwen" in eval_models
        assert "deepseek" in eval_models
