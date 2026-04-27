"""Phase 1 LLM Screener — 用轻量 LLM 替代规则筛选候选列。

Structural Exclusion（PK/FK/跨表同名列）仍用规则，
剩余列用 4B 模型做 YES/NO 判断。
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from egrefine.data.schema import Column, Schema
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.sampler import sample_column

logger = logging.getLogger(__name__)

SCREENING_PROMPT = """You are evaluating whether a database column name might confuse a Text-to-SQL model.

Database: {db_id}
Table: {table_name}
Column: {column_name} ({dtype})
Other columns in this table: {neighbor_names}
Sample values (5 rows): {sample_values}

Could this column name cause confusion for a Text-to-SQL model? Consider:
- Is it an unclear abbreviation? (e.g., cntry, stdm, dept, acct)
- Is it a single letter or code? (e.g., A2, X, N)
- Is it too generic without context? (e.g., status, type, value, code)
- Does it use non-standard or unusual wording that a user would not naturally use in a question? (e.g., "heaviness" instead of "weight", "spot" instead of "location", "charge_of_curative" instead of "treatment_cost")
- Does the name NOT match what the sample values suggest? (e.g., column named "content" but values are country names)
- Is it an uncommon synonym that differs from the standard domain term? (e.g., "vessel" for ship_type, "craft_classification" for ship_type, "bulgarian_head" for commander)

Answer YES if a clearer, more standard name exists. Answer ONLY "YES" or "NO". Nothing else."""


def structural_exclusion(
    schema: Schema,
    **_kwargs,
) -> Tuple[Set[str], List[str], List[str], List[str]]:
    """Structural Exclusion: 排除 FK 和跨表同名同类型列。

    变更（2026-04-13）:
    - PK 列不再排除，允许进入 Phase 1-3 正常流程
    - FK 列仍然排除（Phase 4 PK→FK 传播驱动）
    - 跨表同名列改为"同名同类型"才排除（大概率是隐式 JOIN key）
      同名不同类型则放开

    Returns:
        skip_set: 应跳过的列 full_name 集合
        skipped_pks: 被跳过的 PK 列（始终为空）
        skipped_fks: 被跳过的 FK 列
        skipped_shared: 被跳过的跨表同名同类型列
    """
    all_columns = schema.all_columns

    # PK: 不再排除
    skipped_pks: List[str] = []

    # FK: 仍然排除（但 PK 即使是 FK 目标也不排除）
    pk_full_names = {c.full_name for c in all_columns if c.is_pk}
    fk_full_names: Set[str] = set()
    for src, tgt in schema.foreign_keys:
        fk_full_names.add(src)
        fk_full_names.add(tgt)
    for col in all_columns:
        if col.fk_target:
            fk_full_names.add(col.full_name)
    # PK 列即使是 FK 目标也应进入检测
    fk_full_names -= pk_full_names
    skipped_fks = sorted(fk_full_names)

    # 跨表同名列: 只排除同名且同类型的（大概率是隐式 JOIN key）
    name_to_table_types: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for col in all_columns:
        name_to_table_types[col.name].append((col.table, col.dtype.upper()))

    shared_skip: Set[str] = set()
    for name, table_types in name_to_table_types.items():
        if len(table_types) <= 1:
            continue
        types = {dt for _, dt in table_types}
        if len(types) == 1:
            # 同名同类型 → 跳过
            for tbl, _ in table_types:
                fn = f"{tbl}.{name}"
                if fn not in fk_full_names and fn not in pk_full_names:
                    shared_skip.add(fn)

    skipped_shared = sorted(shared_skip)

    skip_set = fk_full_names | shared_skip
    return skip_set, skipped_pks, skipped_fks, skipped_shared


def _build_prompt(
    col: Column,
    schema: Schema,
    db_path: str,
) -> str:
    """为单个列构造 screening prompt。"""
    # 获取同表邻居列名
    table_obj = schema.get_table(col.table)
    neighbor_names = []
    if table_obj:
        neighbor_names = [c.name for c in table_obj.columns if c.name != col.name]

    # 采样 5 行数据
    try:
        sample_values = sample_column(db_path, col.table, col.name, n=5)
    except Exception:
        sample_values = []

    return SCREENING_PROMPT.format(
        db_id=schema.db_id,
        table_name=col.table,
        column_name=col.name,
        dtype=col.dtype,
        neighbor_names=", ".join(neighbor_names) if neighbor_names else "(none)",
        sample_values=", ".join(str(v) for v in sample_values[:5]) if sample_values else "(no data)",
    )


def _parse_yes_no(response: str) -> bool:
    """解析 LLM 返回的 YES/NO，兼容 reasoning 型模型。

    Reasoning 模型（GLM-5.1 / MiniMax M2.7 等）会先输出一大段 CoT，
    再给出最终答案，通常形如 `…</think>YES` 或 `…Final Decision: YES`。
    策略：
      1. 如含 `</think>`，只看它之后的内容（最终回答区）
      2. 查找最后一次出现的 YES / NO 词 token
      3. 若完全未出现，默认 YES（保守，不漏掉可能有问题的列）
    """
    import re as _re
    if not response:
        return True
    text = response
    # 1) 若有思考标签，只保留其后
    tag_pos = text.rfind("</think>")
    if tag_pos >= 0:
        text = text[tag_pos + len("</think>"):]
    text_upper = text.upper()
    # 2) 查找词边界的 YES / NO（取最后一次，因为最终结论通常在末尾）
    matches = list(_re.finditer(r"\b(YES|NO)\b", text_upper))
    if matches:
        return matches[-1].group(1) == "YES"
    # 3) 找不到明确答案，fallback 到旧行为：看第一个词
    first_word = text_upper.strip().split()[0] if text_upper.strip() else ""
    first_word = first_word.strip(".,!?;:")
    if first_word == "YES":
        return True
    if first_word == "NO":
        return False
    # 完全无法判定，保守返回 YES（加入候选，由 Phase 3 验证筛掉）
    return True


def screen_columns_llm(
    schema: Schema,
    db_path: str,
    columns: List[Column],
    llm_client: LLMClient,
    concurrency: int = 64,
) -> List[Column]:
    """用 LLM 对列做 YES/NO 筛选。

    Args:
        schema: 数据库 schema
        db_path: 数据库文件路径
        columns: 待筛选的列（已排除 PK/FK/shared）
        llm_client: Phase 1 专用 LLM 客户端（4B 模型）
        concurrency: 并发数

    Returns:
        被 LLM 判定为 YES（可能有歧义）的列
    """
    if not columns:
        return []

    results: Dict[str, bool] = {}
    lock = threading.Lock()

    def _screen_one(col: Column) -> Tuple[str, bool]:
        prompt = _build_prompt(col, schema, db_path)
        try:
            response = llm_client.chat([{"role": "user", "content": prompt}])
            is_candidate = _parse_yes_no(response)
        except Exception as e:
            logger.warning("LLM screening failed for %s: %s, defaulting to YES", col.full_name, e)
            is_candidate = True  # 失败时保守处理，加入候选
        return col.full_name, is_candidate

    if concurrency > 1 and len(columns) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_screen_one, col) for col in columns]
            for fut in as_completed(futures):
                full_name, is_candidate = fut.result()
                with lock:
                    results[full_name] = is_candidate
    else:
        for col in columns:
            full_name, is_candidate = _screen_one(col)
            results[full_name] = is_candidate

    # 保持原始列顺序
    candidates = [col for col in columns if results.get(col.full_name, True)]

    yes_count = sum(1 for v in results.values() if v)
    logger.info(
        "LLM screening: %d/%d columns flagged as candidates",
        yes_count, len(columns),
    )

    return candidates
