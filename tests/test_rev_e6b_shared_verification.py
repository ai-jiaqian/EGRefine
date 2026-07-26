import pytest

from egrefine.phase2.prompts import CandidateName

from scripts.rev_e6b_shared_verification import (
    SharedScoreCache,
    compare_candidate_pools,
    deduplicate_candidates,
    restrict_scores,
)


def _candidate(name):
    return CandidateName(name=name, reason="test")


def test_union_deduplicates_and_excludes_original():
    standard = [_candidate("clear_name"), _candidate("x")]
    conditioned = [_candidate("clear_name"), _candidate("user_term"), _candidate("orig")]
    union = deduplicate_candidates([standard, conditioned], "orig")
    assert [candidate.name for candidate in union] == [
        "clear_name", "x", "user_term",
    ]


def test_restrict_scores_prevents_candidate_crossing_arms():
    shared = {"orig": 0.2, "standard_name": 0.4, "user_term": 0.9}
    standard = [_candidate("standard_name")]
    assert restrict_scores(shared, "orig", standard) == {
        "orig": 0.2,
        "standard_name": 0.4,
    }


def test_compare_candidate_pools_detects_query_signal_rescue():
    comparison = compare_candidate_pools(
        "orig",
        [_candidate("standard_name")],
        [_candidate("user_term")],
        {"orig": 0.4, "standard_name": 0.4, "user_term": 0.7},
        min_delta=0.05,
    )
    assert comparison["proposal_rescue"] is True
    assert comparison["proposal_reversal"] is False
    assert comparison["conditioned_minus_standard_best"] == pytest.approx(0.3)


def test_shared_score_cache_requires_exact_union(tmp_path):
    cache = SharedScoreCache(str(tmp_path))
    cache.put("db", "t.c", ["c", "a", "b"], {"c": 0.1, "a": 0.2, "b": 0.3})
    assert cache.get("db", "t.c", ["c", "a", "b"]) == {
        "c": 0.1, "a": 0.2, "b": 0.3,
    }
    assert cache.get("db", "t.c", ["c", "b", "a"]) is None
    assert cache.get("db", "t.c", ["c", "a"]) is None
