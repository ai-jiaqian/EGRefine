"""T12: Phase 2 Prompt template tests."""
import json
import pytest

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase2.prompts import build_candidate_prompt, parse_candidates, CandidateName


# ========== Fixtures ==========

@pytest.fixture
def financial_schema():
    return Schema(
        db_id="financial",
        tables=[
            Table(name="district", columns=[
                Column(name="district_id", table="district", dtype="INTEGER", is_pk=True),
                Column(name="A2", table="district", dtype="TEXT"),
                Column(name="A3", table="district", dtype="TEXT"),
                Column(name="A11", table="district", dtype="REAL"),
            ]),
            Table(name="account", columns=[
                Column(name="account_id", table="account", dtype="INTEGER", is_pk=True),
                Column(name="district_id", table="account", dtype="INTEGER",
                       fk_target="district.district_id"),
                Column(name="frequency", table="account", dtype="TEXT"),
                Column(name="date", table="account", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[
            ("account.district_id", "district.district_id"),
        ],
    )


@pytest.fixture
def target_column():
    return Column(name="A2", table="district", dtype="TEXT")


@pytest.fixture
def sample_values():
    return ["Prague", "Benesov", "Beroun", "Kladno", "Kolin"]


# ========== build_candidate_prompt ==========

class TestBuildCandidatePrompt:
    def test_contains_column_info(self, financial_schema, target_column, sample_values):
        prompt = build_candidate_prompt(target_column, financial_schema, sample_values, k=3)
        assert "`A2`" in prompt
        assert "district" in prompt
        assert "TEXT" in prompt

    def test_contains_neighbors(self, financial_schema, target_column, sample_values):
        prompt = build_candidate_prompt(target_column, financial_schema, sample_values, k=3)
        # Neighboring columns (same table, excluding A2 itself)
        assert "A3" in prompt
        assert "A11" in prompt
        assert "district_id" in prompt

    def test_excludes_self_from_neighbors(self, financial_schema, target_column, sample_values):
        prompt = build_candidate_prompt(target_column, financial_schema, sample_values, k=3)
        # A2 should appear in Column Information but not in Neighbors
        lines = prompt.split("\n")
        neighbor_section = False
        for line in lines:
            if "Neighboring Columns" in line:
                neighbor_section = True
                continue
            if neighbor_section and line.startswith("##"):
                break
            if neighbor_section and line.startswith("- "):
                assert "A2 (" not in line

    def test_contains_sample_values(self, financial_schema, target_column, sample_values):
        prompt = build_candidate_prompt(target_column, financial_schema, sample_values, k=3)
        assert "Prague" in prompt
        assert "Benesov" in prompt

    def test_contains_fk_info(self, financial_schema, sample_values):
        fk_col = Column(name="district_id", table="account", dtype="INTEGER",
                        fk_target="district.district_id")
        prompt = build_candidate_prompt(fk_col, financial_schema, [], k=3)
        assert "district.district_id" in prompt

    def test_k_value_in_prompt(self, financial_schema, target_column, sample_values):
        prompt = build_candidate_prompt(target_column, financial_schema, sample_values, k=5)
        assert "exactly 5" in prompt.lower()

    def test_json_format_instruction(self, financial_schema, target_column, sample_values):
        prompt = build_candidate_prompt(target_column, financial_schema, sample_values, k=3)
        assert "JSON format only" in prompt
        assert '"name"' in prompt
        assert '"reason"' in prompt

    def test_snake_case_instruction(self, financial_schema, target_column, sample_values):
        prompt = build_candidate_prompt(target_column, financial_schema, sample_values, k=3)
        assert "snake_case" in prompt

    def test_empty_sample_values(self, financial_schema, target_column):
        prompt = build_candidate_prompt(target_column, financial_schema, [], k=3)
        assert "no data available" in prompt

    def test_no_fk_columns(self, financial_schema, target_column, sample_values):
        """Column with no FK references should show None."""
        prompt = build_candidate_prompt(target_column, financial_schema, sample_values, k=3)
        # A2 has no FK, so FK section should show None
        assert "Foreign Key Related Columns" in prompt

    def test_prompt_is_all_english(self, financial_schema, target_column, sample_values):
        """Prompt should be entirely in English."""
        prompt = build_candidate_prompt(target_column, financial_schema, sample_values, k=3)
        # Check no Chinese characters
        for ch in prompt:
            assert ord(ch) < 0x4E00 or ord(ch) > 0x9FFF, f"Found Chinese char: {ch}"

    def test_pk_flag_shown(self, financial_schema, sample_values):
        pk_col = Column(name="district_id", table="district", dtype="INTEGER", is_pk=True)
        prompt = build_candidate_prompt(pk_col, financial_schema, [], k=3)
        assert "True" in prompt


# ========== parse_candidates ==========

class TestParseCandidates:
    def test_valid_json(self):
        raw = [
            {"name": "district_name", "reason": "More descriptive"},
            {"name": "city_name", "reason": "Represents city"},
        ]
        result = parse_candidates(raw)
        assert len(result) == 2
        assert result[0].name == "district_name"
        assert result[0].reason == "More descriptive"

    def test_strips_whitespace(self):
        raw = [{"name": "  foo_bar  ", "reason": "  reason  "}]
        result = parse_candidates(raw)
        assert result[0].name == "foo_bar"
        assert result[0].reason == "reason"

    def test_missing_reason_defaults_empty(self):
        raw = [{"name": "some_col"}]
        result = parse_candidates(raw)
        assert result[0].reason == ""

    def test_invalid_not_list(self):
        with pytest.raises(ValueError, match="JSON list"):
            parse_candidates({"name": "foo"})

    def test_invalid_item_not_dict(self):
        with pytest.raises(ValueError, match="dict"):
            parse_candidates(["not_a_dict"])

    def test_missing_name_field(self):
        with pytest.raises(ValueError, match="name"):
            parse_candidates([{"reason": "no name"}])

    def test_empty_name_field(self):
        with pytest.raises(ValueError, match="name"):
            parse_candidates([{"name": "", "reason": "empty"}])

    def test_empty_list(self):
        result = parse_candidates([])
        assert result == []

    def test_returns_candidate_name_type(self):
        raw = [{"name": "col_a", "reason": "r"}]
        result = parse_candidates(raw)
        assert isinstance(result[0], CandidateName)
