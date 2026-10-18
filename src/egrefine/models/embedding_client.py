"""统一 Embedding 调用客户端 (API / local 两种模式)"""
import math
import logging
from typing import List

import openai

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Embedding 客户端，支持 API 模式和 local 模式。"""

    def __init__(self, config: dict):
        self.mode = config.get("type", "api")
        if self.mode == "api":
            self.model_name = config["model_name"]
            self._client = openai.OpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"],
            )
            self._local_model = None
        else:
            # local 模式: 延迟加载 sentence-transformers
            self._client = None
            self.model_name = config.get("local_model", "all-MiniLM-L6-v2")
            self._local_model = None  # 延迟加载

    def _load_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model: %s", self.model_name)
            self._local_model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """计算文本列表的 embedding 向量。"""
        if self.mode == "api":
            response = self._client.embeddings.create(
                input=texts,
                model=self.model_name,
            )
            return [item.embedding for item in response.data]
        else:
            self._load_local_model()
            embeddings = self._local_model.encode(texts, convert_to_numpy=True)
            return [vec.tolist() for vec in embeddings]

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """计算两个向量的 cosine similarity。"""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
