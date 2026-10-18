"""T5 测试: Text-to-SQL 系统封装"""
import os
import sqlite3
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.data.schema import Column, Table, Schema
from egrefine.phase3.text2sql_runner import Text2SQLModel, SimpleLLMText2SQL


def _make_schema():
    return Schema(
        db_id="test",
        tables=[
            Table(name="employees", columns=[
                Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
                Column(name="name", table="employees", dtype="TEXT"),
                Column(name="salary", table="employees", dtype="INTEGER"),
                Column(name="dept_id", table="employees", dtype="INTEGER",
                       fk_target="departments.id"),
            ]),
            Table(name="departments", columns=[
                Column(name="id", table="departments", dtype="INTEGER", is_pk=True),
                Column(name="dept_name", table="departments", dtype="TEXT"),
            ]),
        ],
        foreign_keys=[("employees.dept_id", "departments.id")],
    )


# ====== Text2SQLModel 接口测试 ======

class TestText2SQLModelInterface:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            Text2SQLModel()

    def test_subclass_must_implement_generate(self):
        class Broken(Text2SQLModel):
            pass
        with pytest.raises(TypeError):
            Broken()


# ====== SimpleLLMText2SQL 测试 ======

class TestSimpleLLMText2SQL:
    def _make_config(self):
        return {
            "base_url": "http://localhost:8000/v1",
            "api_key": "test-key",
            "model_name": "test-model",
            "temperature": 0,
            "max_tokens": 1024,
        }

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_generate_returns_sql(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="SELECT name FROM employees WHERE salary > 50000"
            ))]
        )
        model = SimpleLLMText2SQL(self._make_config())
        schema = _make_schema()
        result = model.generate("Who earns more than 50k?", schema)
        assert "SELECT" in result
        assert "employees" in result

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_generate_strips_markdown(self, mock_openai_cls):
        """LLM 返回 markdown 代码块时应正确提取 SQL"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="```sql\nSELECT name FROM employees\n```"
            ))]
        )
        model = SimpleLLMText2SQL(self._make_config())
        result = model.generate("List names", _make_schema())
        assert result.strip() == "SELECT name FROM employees"
        assert "```" not in result

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_generate_strips_plain_codeblock(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="```\nSELECT 1\n```"
            ))]
        )
        model = SimpleLLMText2SQL(self._make_config())
        result = model.generate("test", _make_schema())
        assert result.strip() == "SELECT 1"

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_prompt_contains_schema(self, mock_openai_cls):
        """prompt 应包含 CREATE TABLE 格式的 schema"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SELECT 1"))]
        )
        model = SimpleLLMText2SQL(self._make_config())
        model.generate("test", _make_schema())

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        prompt_text = messages[0]["content"]
        assert "CREATE TABLE" in prompt_text
        assert "employees" in prompt_text
        assert "departments" in prompt_text

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_prompt_contains_fk(self, mock_openai_cls):
        """prompt 应包含外键信息"""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SELECT 1"))]
        )
        model = SimpleLLMText2SQL(self._make_config())
        model.generate("test", _make_schema())

        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "FOREIGN KEY" in prompt_text or "REFERENCES" in prompt_text

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_prompt_contains_question(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SELECT 1"))]
        )
        model = SimpleLLMText2SQL(self._make_config())
        model.generate("How many departments exist?", _make_schema())

        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "How many departments exist?" in prompt_text


# ====== format_schema 测试 ======

class TestFormatSchema:
    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_format_creates_valid_ddl(self, mock_openai_cls):
        model = SimpleLLMText2SQL({
            "base_url": "http://x/v1", "api_key": "k", "model_name": "m",
        })
        schema = _make_schema()
        ddl = model._format_schema(schema)
        assert "CREATE TABLE employees" in ddl
        assert "CREATE TABLE departments" in ddl
        assert "id INTEGER PRIMARY KEY" in ddl
        assert "name TEXT" in ddl

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_format_includes_fk(self, mock_openai_cls):
        model = SimpleLLMText2SQL({
            "base_url": "http://x/v1", "api_key": "k", "model_name": "m",
        })
        schema = _make_schema()
        ddl = model._format_schema(schema)
        assert "REFERENCES" in ddl


# ====== Sample Data 测试 ======

class TestSampleDataInPrompt:
    @pytest.fixture
    def sample_db(self):
        fd, path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary INTEGER)"
        )
        conn.executemany(
            "INSERT INTO employees VALUES (?, ?, ?)",
            [(1, "Alice", 50000), (2, "Bob", 60000), (3, "Carol", 55000)],
        )
        conn.commit()
        conn.close()
        yield path
        os.unlink(path)

    @pytest.fixture
    def simple_schema(self):
        return Schema(
            db_id="test",
            tables=[
                Table(name="employees", columns=[
                    Column(name="id", table="employees", dtype="INTEGER", is_pk=True),
                    Column(name="name", table="employees", dtype="TEXT"),
                    Column(name="salary", table="employees", dtype="INTEGER"),
                ]),
            ],
            foreign_keys=[],
        )

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_prompt_includes_sample_data(self, mock_openai_cls, simple_schema, sample_db):
        """When db_path is provided, prompt should contain sample data."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SELECT 1"))]
        )
        model = SimpleLLMText2SQL(
            {"base_url": "http://x/v1", "api_key": "k", "model_name": "m"},
            sample_rows=3,
        )
        model.generate("test", simple_schema, db_path=sample_db)

        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "Sample data" in prompt_text
        assert "Alice" in prompt_text
        assert "employees" in prompt_text

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_no_sample_without_db_path(self, mock_openai_cls, simple_schema):
        """Without db_path, no sample data in prompt."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SELECT 1"))]
        )
        model = SimpleLLMText2SQL(
            {"base_url": "http://x/v1", "api_key": "k", "model_name": "m"},
        )
        model.generate("test", simple_schema)

        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "Sample data" not in prompt_text

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_sample_rows_configurable(self, mock_openai_cls, simple_schema, sample_db):
        """sample_rows=0 disables sample data."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SELECT 1"))]
        )
        model = SimpleLLMText2SQL(
            {"base_url": "http://x/v1", "api_key": "k", "model_name": "m"},
            sample_rows=0,
        )
        model.generate("test", simple_schema, db_path=sample_db)

        call_args = mock_client.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        assert "Sample data" not in prompt_text
