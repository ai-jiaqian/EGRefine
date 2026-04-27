"""Phase 1: Pruner — 整合四个 heuristic signals，筛选候选列。

支持两种模式：
- method="rule": 经典 S1-S4 规则筛选（默认）
- method="llm": Structural Exclusion + LLM YES/NO 筛选
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from egrefine.data.schema import Column, Schema
from egrefine.models.embedding_client import EmbeddingClient
from egrefine.models.llm_client import LLMClient
from egrefine.phase1.embeddings import (
    build_similarity_matrix,
    compute_column_embeddings,
)
from egrefine.phase1.signals import (
    s1_short_name,
    s2_high_similarity,
    s3_naming_inconsistency,
    s4_generic_vocabulary,
)

logger = logging.getLogger(__name__)


@dataclass
class PruneResult:
    """Phase 1 剪枝结果。"""
    candidates: List[Column]              # 候选列（union of all signals）
    total_columns: int                    # schema 总列数
    signal_hits: Dict[str, List[str]]     # signal_name -> [col.full_name, ...]
    skipped_pks: List[str]                # 被跳过的主键列
    skipped_fks: List[str] = field(default_factory=list)   # 被跳过的外键列
    skipped_shared: List[str] = field(default_factory=list) # 被跳过的跨表同名列

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def compression_ratio(self) -> float:
        if self.total_columns == 0:
            return 0.0
        return self.candidate_count / self.total_columns

    def summary(self) -> str:
        lines = [
            f"Phase 1 Pruning: {self.candidate_count}/{self.total_columns} "
            f"columns selected ({self.compression_ratio:.0%})",
        ]
        for sig, cols in self.signal_hits.items():
            lines.append(f"  {sig}: {len(cols)} columns")
        if self.skipped_pks:
            lines.append(f"  Skipped PKs: {len(self.skipped_pks)}")
        if self.skipped_fks:
            lines.append(f"  Skipped FKs: {len(self.skipped_fks)}")
        if self.skipped_shared:
            lines.append(f"  Skipped shared-name: {len(self.skipped_shared)}")
        return "\n".join(lines)


def prune(
    schema: Schema,
    config: dict,
    embedding_client: Optional[EmbeddingClient] = None,
) -> PruneResult:
    """对 schema 执行 Phase 1 剪枝，返回候选列集合。

    Args:
        schema: 数据库 schema
        config: phase1 配置 (config["phase1"])
        embedding_client: Embedding 客户端（S2 需要，若 S2 disabled 可为 None）
    """
    signals_cfg = config.get("signals", {})
    skip_pks = config.get("skip_primary_keys", True)

    all_columns = schema.all_columns
    total = len(all_columns)

    # 收集各 signal 的命中
    signal_hits: Dict[str, List[str]] = {
        "S1_short_name": [],
        "S2_high_similarity": [],
        "S3_naming_inconsistency": [],
        "S4_generic_vocabulary": [],
    }

    # 跳过的 PK
    skipped_pks: List[str] = []
    if skip_pks:
        skipped_pks = [c.full_name for c in all_columns if c.is_pk]

    # 跳过的 FK 列（JOIN key 不应被 rename）
    skipped_fks: List[str] = []
    fk_full_names: Set[str] = set()
    for src, tgt in schema.foreign_keys:
        fk_full_names.add(src)
        fk_full_names.add(tgt)
    for col in all_columns:
        if col.fk_target:
            fk_full_names.add(col.full_name)
    skipped_fks = [fn for fn in fk_full_names if fn not in set(skipped_pks)]

    # 跳过跨表同名列（隐式 JOIN key）
    skipped_shared: List[str] = []
    name_to_tables: Dict[str, List[str]] = {}
    for col in all_columns:
        name_to_tables.setdefault(col.name, []).append(col.table)
    shared_names = {name for name, tbls in name_to_tables.items() if len(tbls) > 1}
    for col in all_columns:
        if col.name in shared_names and col.full_name not in set(skipped_pks) | fk_full_names:
            skipped_shared.append(col.full_name)

    # 可跳过的列集合
    skip_set: Set[str] = set(skipped_pks) | set(skipped_fks) | set(skipped_shared)

    # --- S1: Short name ---
    s1_cfg = signals_cfg.get("short_name", {})
    if s1_cfg.get("enabled", True):
        max_len = s1_cfg.get("max_length", 3)
        for col in all_columns:
            if col.full_name in skip_set:
                continue
            if s1_short_name(col, max_length=max_len):
                signal_hits["S1_short_name"].append(col.full_name)

    # --- S2: High similarity ---
    s2_cfg = signals_cfg.get("high_similarity", {})
    similarity_matrix: Dict[Tuple[str, str], float] = {}
    if s2_cfg.get("enabled", True):
        if embedding_client is None:
            logger.warning("S2 enabled but no embedding_client provided, skipping S2")
        else:
            threshold = s2_cfg.get("threshold", 0.85)
            embeddings = compute_column_embeddings(schema, embedding_client)
            similarity_matrix = build_similarity_matrix(embeddings)
            for col in all_columns:
                if col.full_name in skip_set:
                    continue
                if s2_high_similarity(col, similarity_matrix, threshold=threshold):
                    signal_hits["S2_high_similarity"].append(col.full_name)

    # --- S3: Naming inconsistency ---
    s3_cfg = signals_cfg.get("naming_inconsistency", {})
    if s3_cfg.get("enabled", True):
        minority_cols = s3_naming_inconsistency(schema)
        for col in minority_cols:
            if col.full_name in skip_set:
                continue
            signal_hits["S3_naming_inconsistency"].append(col.full_name)

    # --- S4: Generic vocabulary ---
    s4_cfg = signals_cfg.get("generic_vocabulary", {})
    if s4_cfg.get("enabled", True):
        for col in all_columns:
            if col.full_name in skip_set:
                continue
            if s4_generic_vocabulary(col):
                signal_hits["S4_generic_vocabulary"].append(col.full_name)

    # Union of all signals
    candidate_names: Set[str] = set()
    for hits in signal_hits.values():
        candidate_names.update(hits)

    # 保持原始列顺序
    candidates = [c for c in all_columns if c.full_name in candidate_names]

    result = PruneResult(
        candidates=candidates,
        total_columns=total,
        signal_hits=signal_hits,
        skipped_pks=skipped_pks,
        skipped_fks=skipped_fks,
        skipped_shared=skipped_shared,
    )

    logger.info(result.summary())
    return result


def prune_llm(
    schema: Schema,
    config: dict,
    llm_client: LLMClient,
    db_path: str,
    concurrency: int = 64,
) -> PruneResult:
    """Phase 1 LLM 模式：Structural Exclusion + LLM YES/NO 筛选。

    Args:
        schema: 数据库 schema
        config: phase1 配置
        llm_client: Phase 1 专用 LLM 客户端（4B 模型）
        db_path: 数据库文件路径
        concurrency: LLM 并发调用数
    """
    from egrefine.phase1.llm_screener import structural_exclusion, screen_columns_llm

    skip_pks = config.get("skip_primary_keys", True)

    all_columns = schema.all_columns
    total = len(all_columns)

    # Step 1: Structural Exclusion
    skip_set, skipped_pks, skipped_fks, skipped_shared = structural_exclusion(
        schema, skip_pks=skip_pks,
    )
    columns_to_screen = [c for c in all_columns if c.full_name not in skip_set]

    logger.info(
        "[%s] Phase 1 LLM: %d columns after structural exclusion (%d PKs, %d FKs, %d shared skipped)",
        schema.db_id, len(columns_to_screen), len(skipped_pks), len(skipped_fks), len(skipped_shared),
    )

    # Step 2: LLM Screening
    candidates = screen_columns_llm(
        schema=schema,
        db_path=db_path,
        columns=columns_to_screen,
        llm_client=llm_client,
        concurrency=concurrency,
    )

    result = PruneResult(
        candidates=candidates,
        total_columns=total,
        signal_hits={"LLM_screening": [c.full_name for c in candidates]},
        skipped_pks=skipped_pks,
        skipped_fks=skipped_fks,
        skipped_shared=skipped_shared,
    )

    logger.info(result.summary())
    return result
