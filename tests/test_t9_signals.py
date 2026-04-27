"""T9: Phase 1 Heuristic Signals 测试"""
import pytest

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase1.signals import (
    s1_short_name,
    s2_high_similarity,
    s3_naming_inconsistency,
    s4_generic_vocabulary,
    _detect_style,
    GENERIC_VOCABULARY,
)


# ========== Fixtures ==========

@pytest.fixture
def mixed_style_schema():
    """混合命名风格的 schema: snake_case 主流 + camelCase 少数派。"""
    return Schema(
        db_id="mixed",
        tables=[
            Table(name="users", columns=[
                Column(name="user_id", table="users", dtype="INTEGER", is_pk=True),
                Column(name="first_name", table="users", dtype="TEXT"),
                Column(name="last_name", table="users", dtype="TEXT"),
                Column(name="emailAddress", table="users", dtype="TEXT"),  # camelCase
                Column(name="phoneNumber", table="users", dtype="TEXT"),   # camelCase
            ]),
        ],
        foreign_keys=[],
    )


@pytest.fixture
def uniform_style_schema():
    """统一命名风格的 schema。"""
    return Schema(
        db_id="uniform",
        tables=[
            Table(name="products", columns=[
                Column(name="product_id", table="products", dtype="INTEGER", is_pk=True),
                Column(name="product_name", table="products", dtype="TEXT"),
                Column(name="unit_price", table="products", dtype="REAL"),
                Column(name="stock_count", table="products", dtype="INTEGER"),
            ]),
        ],
        foreign_keys=[],
    )


@pytest.fixture
def allcaps_mixed_schema():
    """ALLCAPS + snake_case 混合 schema。"""
    return Schema(
        db_id="caps_mixed",
        tables=[
            Table(name="lab", columns=[
                Column(name="ID", table="lab", dtype="INTEGER", is_pk=True),
                Column(name="GOT", table="lab", dtype="REAL"),
                Column(name="GPT", table="lab", dtype="REAL"),
                Column(name="SEX", table="lab", dtype="TEXT"),
                Column(name="patient_name", table="lab", dtype="TEXT"),
                Column(name="admission_date", table="lab", dtype="TEXT"),
                Column(name="blood_type", table="lab", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[],
    )


# ========== _detect_style ==========

class TestDetectStyle:
    def test_snake_case(self):
        assert _detect_style("first_name") == "snake_case"
        assert _detect_style("user_id") == "snake_case"
        assert _detect_style("email_address_backup") == "snake_case"

    def test_camel_case(self):
        assert _detect_style("firstName") == "camelCase"
        assert _detect_style("emailAddress") == "camelCase"
        assert _detect_style("userId") == "camelCase"

    def test_allcaps(self):
        assert _detect_style("GOT") == "ALLCAPS"
        assert _detect_style("SEX") == "ALLCAPS"
        assert _detect_style("FIRST_NAME") == "ALLCAPS"
        assert _detect_style("A11") == "ALLCAPS"

    def test_alllower(self):
        assert _detect_style("name") == "alllower"
        assert _detect_style("status") == "alllower"
        assert _detect_style("id") == "alllower"
        assert _detect_style("lat") == "alllower"

    def test_single_char(self):
        # 单字符大写不算 ALLCAPS (需要 len > 1)
        result = _detect_style("A")
        assert result == "other"

    def test_other(self):
        # PascalCase 等不常见模式
        assert _detect_style("FirstName") == "other"


# ========== S1: short_name ==========

class TestS1ShortName:
    def test_short_name_2_chars(self):
        col = Column(name="nm", table="t", dtype="TEXT")
        assert s1_short_name(col) is True

    def test_short_name_3_chars(self):
        col = Column(name="amt", table="t", dtype="REAL")
        assert s1_short_name(col) is True

    def test_short_name_exact_boundary(self):
        col = Column(name="abc", table="t", dtype="TEXT")
        assert s1_short_name(col, max_length=3) is True

    def test_not_short_4_chars(self):
        col = Column(name="name", table="t", dtype="TEXT")
        assert s1_short_name(col) is False

    def test_long_name(self):
        col = Column(name="email_address", table="t", dtype="TEXT")
        assert s1_short_name(col) is False

    def test_custom_max_length(self):
        col = Column(name="name", table="t", dtype="TEXT")
        assert s1_short_name(col, max_length=4) is True
        assert s1_short_name(col, max_length=3) is False

    def test_single_char(self):
        col = Column(name="x", table="t", dtype="TEXT")
        assert s1_short_name(col) is True


# ========== S2: high_similarity ==========

class TestS2HighSimilarity:
    def test_column_has_high_similarity(self):
        matrix = {
            ("t.a", "t.b"): 0.90,
            ("t.a", "t.c"): 0.50,
        }
        col_a = Column(name="a", table="t", dtype="TEXT")
        assert s2_high_similarity(col_a, matrix, threshold=0.85) is True

    def test_column_no_high_similarity(self):
        matrix = {
            ("t.a", "t.b"): 0.60,
            ("t.a", "t.c"): 0.50,
        }
        col_a = Column(name="a", table="t", dtype="TEXT")
        assert s2_high_similarity(col_a, matrix, threshold=0.85) is False

    def test_column_as_second_in_pair(self):
        """列在 pair 的第二个位置也应被检测。"""
        matrix = {
            ("t.a", "t.b"): 0.92,
        }
        col_b = Column(name="b", table="t", dtype="TEXT")
        assert s2_high_similarity(col_b, matrix, threshold=0.85) is True

    def test_exact_threshold(self):
        matrix = {("t.a", "t.b"): 0.85}
        col_a = Column(name="a", table="t", dtype="TEXT")
        assert s2_high_similarity(col_a, matrix, threshold=0.85) is True

    def test_below_threshold(self):
        matrix = {("t.a", "t.b"): 0.849}
        col_a = Column(name="a", table="t", dtype="TEXT")
        assert s2_high_similarity(col_a, matrix, threshold=0.85) is False

    def test_empty_matrix(self):
        col = Column(name="a", table="t", dtype="TEXT")
        assert s2_high_similarity(col, {}, threshold=0.85) is False

    def test_column_not_in_matrix(self):
        matrix = {("t.x", "t.y"): 0.95}
        col = Column(name="a", table="t", dtype="TEXT")
        assert s2_high_similarity(col, matrix, threshold=0.85) is False


# ========== S3: naming_inconsistency ==========

class TestS3NamingInconsistency:
    def test_mixed_styles_returns_minority(self, mixed_style_schema):
        result = s3_naming_inconsistency(mixed_style_schema)
        names = {c.name for c in result}
        # camelCase 是少数派
        assert "emailAddress" in names
        assert "phoneNumber" in names
        # snake_case 主流不应出现
        assert "first_name" not in names
        assert "last_name" not in names

    def test_uniform_style_returns_empty(self, uniform_style_schema):
        result = s3_naming_inconsistency(uniform_style_schema)
        assert result == []

    def test_allcaps_mixed(self, allcaps_mixed_schema):
        result = s3_naming_inconsistency(allcaps_mixed_schema)
        names = {c.name for c in result}
        # snake_case 有 3 个，ALLCAPS 有 4 个 (ID, GOT, GPT, SEX)
        # 少数派是 snake_case
        assert "patient_name" in names or "GOT" in names

    def test_empty_schema(self):
        schema = Schema(db_id="empty", tables=[], foreign_keys=[])
        assert s3_naming_inconsistency(schema) == []

    def test_single_column(self):
        schema = Schema(
            db_id="single",
            tables=[Table(name="t", columns=[
                Column(name="id", table="t", dtype="INTEGER"),
            ])],
            foreign_keys=[],
        )
        assert s3_naming_inconsistency(schema) == []

    def test_two_styles_both_significant(self):
        """两种风格各占一半，都算少数派？不——列数多的是主流。"""
        schema = Schema(
            db_id="half",
            tables=[Table(name="t", columns=[
                Column(name="user_name", table="t", dtype="TEXT"),
                Column(name="user_email", table="t", dtype="TEXT"),
                Column(name="user_phone", table="t", dtype="TEXT"),
                Column(name="userName2", table="t", dtype="TEXT"),
                Column(name="userEmail2", table="t", dtype="TEXT"),
            ])],
            foreign_keys=[],
        )
        result = s3_naming_inconsistency(schema)
        names = {c.name for c in result}
        # snake_case: 3, camelCase: 2 -> camelCase 是少数
        assert "userName2" in names
        assert "userEmail2" in names


# ========== S4: generic_vocabulary ==========

class TestS4GenericVocabulary:
    def test_status_is_generic(self):
        col = Column(name="status", table="t", dtype="TEXT")
        assert s4_generic_vocabulary(col) is True

    def test_type_is_generic(self):
        col = Column(name="type", table="t", dtype="TEXT")
        assert s4_generic_vocabulary(col) is True

    def test_case_insensitive(self):
        col = Column(name="Status", table="t", dtype="TEXT")
        assert s4_generic_vocabulary(col) is True
        col2 = Column(name="TYPE", table="t", dtype="TEXT")
        assert s4_generic_vocabulary(col2) is True

    def test_non_generic(self):
        col = Column(name="email_address", table="t", dtype="TEXT")
        assert s4_generic_vocabulary(col) is False

    def test_partial_match_not_generic(self):
        """'status_code' 不在列表中，只匹配完整词。"""
        col = Column(name="status_code", table="t", dtype="TEXT")
        assert s4_generic_vocabulary(col) is False

    def test_all_generic_words(self):
        """验证所有泛化词都能被识别。"""
        for word in GENERIC_VOCABULARY:
            col = Column(name=word, table="t", dtype="TEXT")
            assert s4_generic_vocabulary(col) is True, f"{word} should be generic"

    def test_id_is_generic(self):
        col = Column(name="id", table="t", dtype="INTEGER")
        assert s4_generic_vocabulary(col) is True

    def test_descriptive_name_not_generic(self):
        col = Column(name="transaction_frequency", table="t", dtype="TEXT")
        assert s4_generic_vocabulary(col) is False
