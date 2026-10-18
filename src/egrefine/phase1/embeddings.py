"""Phase 1: Embedding 计算与 pairwise similarity matrix 构建。"""
import logging
from typing import Dict, List, Tuple

from egrefine.data.schema import Column, Schema
from egrefine.models.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


def compute_column_embeddings(
    schema: Schema,
    embedding_client: EmbeddingClient,
) -> Dict[str, List[float]]:
    """对 schema 中所有列计算 embedding。

    列名太短时拼接表名以提供更多语义信息:
        "table_name.column_name" 格式。

    返回:
        {column.full_name: embedding_vector}
    """
    columns = schema.all_columns
    if not columns:
        return {}

    # 构造 embedding 输入文本: 短名字拼接表名
    texts = []
    keys = []
    for col in columns:
        if len(col.name) <= 5:
            text = f"{col.table}.{col.name}"
        else:
            text = col.name
        texts.append(text)
        keys.append(col.full_name)

    logger.info(
        "Computing embeddings for %d columns in schema '%s'",
        len(texts), schema.db_id,
    )

    vectors = embedding_client.embed(texts)

    return dict(zip(keys, vectors))


def build_similarity_matrix(
    embeddings: Dict[str, List[float]],
) -> Dict[Tuple[str, str], float]:
    """构建同 schema 内所有列对的 cosine similarity matrix。

    返回:
        {(col_a_full_name, col_b_full_name): similarity}
        只存 a < b 的对（避免重复）。
    """
    keys = sorted(embeddings.keys())
    matrix = {}

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sim = EmbeddingClient.cosine_similarity(
                embeddings[keys[i]], embeddings[keys[j]]
            )
            matrix[(keys[i], keys[j])] = sim

    return matrix


def find_high_similarity_pairs(
    similarity_matrix: Dict[Tuple[str, str], float],
    threshold: float = 0.85,
) -> List[Tuple[str, str, float]]:
    """找出 similarity 超过阈值的列对。

    返回:
        [(col_a, col_b, similarity), ...] 按 similarity 降序排列
    """
    pairs = [
        (a, b, sim)
        for (a, b), sim in similarity_matrix.items()
        if sim >= threshold
    ]
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs
