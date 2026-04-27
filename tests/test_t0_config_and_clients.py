"""T0 测试: 配置系统 + 模型客户端"""
import os
import sys
import pytest
import yaml
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.models.llm_client import LLMClient
from egrefine.models.embedding_client import EmbeddingClient
from egrefine.config import load_config


# ====== 配置加载测试 ======

class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({"models": {"candidate_llm": {"base_url": "http://localhost:8000/v1"}}}))
        cfg = load_config(str(cfg_file))
        assert cfg["models"]["candidate_llm"]["base_url"] == "http://localhost:8000/v1"

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_load_default_config_exists(self):
        """default.yaml 必须存在且可解析"""
        cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
        assert "models" in cfg
        assert "phase1" in cfg


# ====== LLMClient 测试 ======

class TestLLMClient:
    def _make_config(self, **overrides):
        base = {
            "base_url": "http://localhost:8000/v1",
            "api_key": "test-key",
            "model_name": "test-model",
            "temperature": 0.7,
            "max_tokens": 512,
            "max_retries": 3,
        }
        base.update(overrides)
        return base

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_init_stores_config(self, mock_openai_cls):
        cfg = self._make_config()
        client = LLMClient(cfg)
        assert client.model_name == "test-model"
        assert client.temperature == 0.7
        assert client.max_tokens == 512
        assert client.max_retries == 3

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_init_defaults(self, mock_openai_cls):
        """缺少 optional 字段时使用默认值"""
        cfg = {"base_url": "http://x/v1", "api_key": "k", "model_name": "m"}
        client = LLMClient(cfg)
        assert client.temperature == 0.0
        assert client.max_tokens == 1024
        assert client.max_retries == 3

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_chat_returns_string(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="SELECT 1"))]
        )
        client = LLMClient(self._make_config())
        result = client.chat([{"role": "user", "content": "hello"}])
        assert result == "SELECT 1"
        mock_client.chat.completions.create.assert_called_once()

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_chat_passes_correct_params(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        client = LLMClient(self._make_config(temperature=0.3, max_tokens=256))
        client.chat([{"role": "user", "content": "hi"}])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 256

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_chat_retries_on_failure(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        # 前两次抛异常，第三次成功
        mock_client.chat.completions.create.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))]),
        ]
        client = LLMClient(self._make_config(max_retries=3))
        result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "ok"
        assert mock_client.chat.completions.create.call_count == 3

    @patch("egrefine.models.llm_client.openai.OpenAI")
    def test_chat_raises_after_max_retries(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("timeout")
        client = LLMClient(self._make_config(max_retries=2))
        with pytest.raises(Exception, match="timeout"):
            client.chat([{"role": "user", "content": "hi"}])
        assert mock_client.chat.completions.create.call_count == 2


# ====== EmbeddingClient 测试 ======

class TestEmbeddingClient:
    def _api_config(self):
        return {
            "base_url": "http://localhost:8000/v1",
            "api_key": "test-key",
            "model_name": "text-embedding-3-large",
        }

    def _local_config(self):
        return {
            "type": "local",
            "local_model": "all-MiniLM-L6-v2",
        }

    @patch("egrefine.models.embedding_client.openai.OpenAI")
    def test_init_api_mode(self, mock_openai_cls):
        client = EmbeddingClient(self._api_config())
        assert client.mode == "api"

    def test_init_local_mode(self):
        client = EmbeddingClient(self._local_config())
        assert client.mode == "local"

    @patch("egrefine.models.embedding_client.openai.OpenAI")
    def test_api_embed_returns_list_of_vectors(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[0.1, 0.2, 0.3]),
                MagicMock(embedding=[0.4, 0.5, 0.6]),
            ]
        )
        client = EmbeddingClient(self._api_config())
        result = client.embed(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @patch("egrefine.models.embedding_client.openai.OpenAI")
    def test_api_embed_passes_model_name(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1])]
        )
        client = EmbeddingClient(self._api_config())
        client.embed(["test"])
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["model"] == "text-embedding-3-large"

    def test_cosine_similarity(self):
        """测试 cosine similarity 辅助函数"""
        sim = EmbeddingClient.cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 1e-6
        sim = EmbeddingClient.cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(sim - 0.0) < 1e-6
        sim = EmbeddingClient.cosine_similarity([1, 0], [-1, 0])
        assert abs(sim - (-1.0)) < 1e-6
