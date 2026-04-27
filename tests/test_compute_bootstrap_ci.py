import json

import numpy as np
import pytest

from scripts import compute_bootstrap_ci as ci


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_paired_outcomes_aligns_by_question_text_and_counts_fallback(tmp_path):
    noref_dir = tmp_path / "noref"
    egr_dir = tmp_path / "egr"

    write_json(
        noref_dir / "db1" / "c3sql.json",
        {
            "original_details": [
                {"nl": "q1", "match": True},
                {"nl": "q2", "match": False},
                {"nl": "q3", "match": True},
            ],
        },
    )
    write_json(
        egr_dir / "db1" / "c3.json",
        {
            "refined_details": [
                {"nl": "q3", "match": False},
                {"nl": "q1", "match": True, "fallback_from_noref": True},
                {"nl": "q2", "match": True},
            ],
        },
    )

    noref, egrefine, fallback_count = ci.load_paired_outcomes(
        noref_dir,
        egr_dir,
        "c3",
    )

    assert noref.tolist() == [1, 0, 1]
    assert egrefine.tolist() == [1, 1, 0]
    assert fallback_count == 1


def test_load_paired_outcomes_falls_back_to_noref_when_egrefine_file_missing(tmp_path):
    noref_dir = tmp_path / "noref"
    egr_dir = tmp_path / "egr"

    write_json(
        noref_dir / "db1" / "c3.json",
        {
            "original_details": [
                {"nl": "q1", "match": True},
                {"nl": "q2", "match": False},
            ],
        },
    )
    (egr_dir / "db1").mkdir(parents=True)

    noref, egrefine, fallback_count = ci.load_paired_outcomes(
        noref_dir,
        egr_dir,
        "c3",
    )

    assert noref.tolist() == [1, 0]
    assert egrefine.tolist() == [1, 0]
    assert fallback_count == 2


def test_mcnemar_exact_counts_regressions_and_improvements():
    noref = np.array([1, 1, 0, 0, 1])
    egrefine = np.array([1, 0, 1, 0, 0])

    b, c, p_value = ci.mcnemar_exact(noref, egrefine)

    assert b == 2
    assert c == 1
    assert 0.0 <= p_value <= 1.0


def test_validate_expected_exacc_reports_diagnostics_on_mismatch(capsys):
    with pytest.raises(ValueError, match="ExAcc mismatch"):
        ci.validate_expected_exacc(
            cell_name="cell",
            noref_path="noref/path",
            egrefine_path="egr/path",
            noref=np.array([1, 0]),
            egrefine=np.array([1, 1]),
            expected_n=2,
            expected_noref=40.0,
            expected_egrefine=100.0,
        )

    captured = capsys.readouterr()
    assert "computed NoRef ExAcc" in captured.out
    assert "registry NoRef ExAcc" in captured.out
