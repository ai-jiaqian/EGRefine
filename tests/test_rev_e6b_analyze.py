import pytest

from scripts.analysis.rev_e6b_analyze import exact_mcnemar


def test_exact_mcnemar_identical_vectors():
    result = exact_mcnemar([True, False], [True, False])
    assert result == {
        "a_only_correct": 0,
        "b_only_correct": 0,
        "discordant": 0,
        "p_value": 1.0,
    }


def test_exact_mcnemar_one_sided_improvements():
    result = exact_mcnemar([False] * 6, [True] * 6)
    assert result["a_only_correct"] == 0
    assert result["b_only_correct"] == 6
    assert result["p_value"] == pytest.approx(0.03125)
