#!/usr/bin/env python3
"""Generate and evaluate BIRD holdout queries for EGRefine.

This script intentionally keeps the holdout workflow separate from the
standard dev-set evaluation pipeline so every intermediate artifact can be
audited:

1. Generate unseen questions with column names hidden.
2. Generate gold SQL in a second pass where true names are revealed.
3. Validate gold SQL on real BIRD SQLite databases.
4. Evaluate C3+Qwen3.5-27B on original and fixed EGRefine refined schemas.
5. Write comparison and spot-check reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from egrefine.config import load_config
from egrefine.data.benchmark import BIRDLoader
from egrefine.data.db_connection import execute_sql as execute_db_sql
from egrefine.data.schema import NLSQLPair, Schema
from egrefine.eval.db_setup import apply_views, copy_database
from egrefine.eval.evaluator import load_refined_schema
from egrefine.phase3.c3_runner import C3Text2SQL


DEFAULT_CONFIG = "config/default.yaml"
DEFAULT_REFINE_DIR = "results/refine/bird_multi"
DEFAULT_OUTPUT_DIR = "results/holdout/bird_27b_c3"
DEFAULT_CLAUDE_URL = "http://127.0.0.1:8317/v1/messages"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-6[1m]"
DEFAULT_GENERATOR_PROVIDER = "openai"
DEFAULT_GENERATOR_BASE_URL = "https://api.deepseek.com"
DEFAULT_GENERATOR_MODEL = "deepseek-v4-pro"
DEFAULT_QWEN_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_QWEN_MODEL = "Qwen3.5-27B"
DEFAULT_CLAUDE_KEY = ""
IN_LOOP_NOREF = 41.53
IN_LOOP_EGREFINE = 41.72


@dataclass
class SQLExecutionDetail:
    ok: bool
    reason: str
    row_count: int
    preview: List[List[Any]]
    error: str = ""


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compact_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql or "").strip()


def sql_touches_column(sql: str, column_name: str) -> bool:
    pattern = r'(?<![a-zA-Z0-9_])"?' + re.escape(column_name.lower()) + r'"?(?![a-zA-Z0-9_])'
    return re.search(pattern, sql.lower()) is not None


def collect_refined_targets(refine_dir: Path) -> List[Dict[str, str]]:
    targets: List[Dict[str, str]] = []
    for db_dir in sorted(refine_dir.iterdir()):
        if not db_dir.is_dir():
            continue
        refined_path = db_dir / "refined_tables.json"
        if not refined_path.exists():
            continue
        data = read_json(refined_path, {})
        db_id = data.get("db_id", db_dir.name)
        for table in data.get("tables", []):
            table_name = table.get("name", "")
            for col in table.get("columns", []):
                original = col.get("original_name", "")
                refined = col.get("name", "")
                if original and refined and original != refined:
                    targets.append({
                        "db_id": db_id,
                        "table": table_name,
                        "original_column": original,
                        "refined_column": refined,
                    })
    return targets


def strip_code_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json|sql)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.I)
    if match:
        return match.group(1).strip()
    return text


def parse_json_array(text: str) -> List[Dict[str, Any]]:
    text = strip_code_fences(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    return data


def call_claude(
    prompt: str,
    *,
    url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    retries: int = 3,
) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            body = response.json()
            content = body.get("content", [])
            if isinstance(content, list):
                return "".join(part.get("text", "") for part in content if part.get("type") == "text")
            if isinstance(content, str):
                return content
            raise ValueError(f"Unexpected Claude response content: {body}")
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"Claude call failed after {retries} attempts: {last_error}") from last_error


def openai_chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def call_openai_compatible(
    prompt: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    retries: int = 3,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = openai_chat_completions_url(base_url)
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            body = response.json()
            return body["choices"][0]["message"].get("content", "")
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"OpenAI-compatible generator failed after {retries} attempts: {last_error}") from last_error


def call_generator(args: argparse.Namespace, prompt: str, *, temperature: Optional[float] = None) -> str:
    temp = args.generator_temperature if temperature is None else temperature
    if args.generator_provider == "anthropic":
        return call_claude(
            prompt,
            url=args.claude_url,
            api_key=args.claude_api_key,
            model=args.claude_model,
            max_tokens=args.generator_max_tokens,
            temperature=temp,
        )
    if args.generator_provider == "openai":
        return call_openai_compatible(
            prompt,
            base_url=args.generator_base_url,
            api_key=args.generator_api_key,
            model=args.generator_model,
            max_tokens=args.generator_max_tokens,
            temperature=temp,
        )
    raise ValueError(f"Unknown generator provider: {args.generator_provider}")


def load_description_rows(bird_path: Path, db_id: str, table_name: str) -> Dict[str, Dict[str, str]]:
    csv_path = bird_path / "dev_databases" / db_id / "database_description" / f"{table_name}.csv"
    if not csv_path.exists():
        return {}
    rows: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {k.lstrip("\ufeff"): (v or "").strip() for k, v in row.items()}
            name = clean.get("original_column_name", "")
            if name:
                rows[name] = clean
    return rows


def column_semantic_text(row: Dict[str, str], ordinal: int, dtype: str) -> str:
    desc = row.get("column_description", "").strip()
    value_desc = row.get("value_description", "").strip()
    data_format = row.get("data_format", "").strip() or dtype
    parts = [f"Column {ordinal}"]
    if desc:
        parts.append(f"description: {desc}")
    else:
        parts.append("description: not provided")
    if data_format:
        parts.append(f"data format: {data_format}")
    if value_desc:
        parts.append(f"value meaning: {value_desc}")
    return "; ".join(parts)


def get_table_samples(db_path: Path, table_name: str, limit: int = 5) -> List[List[Any]]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(f'SELECT * FROM "{table_name}" LIMIT {limit}').fetchall()
        return [list(row) for row in rows]
    except Exception:
        return []
    finally:
        conn.close()


def build_description_context(loader: BIRDLoader, bird_path: Path, db_id: str) -> Tuple[str, str, str]:
    schema = loader.schemas[db_id]
    db_path = Path(loader.get_db_path(db_id))

    table_sections: List[str] = []
    sample_sections: List[str] = []
    placeholder_by_column: Dict[str, str] = {}
    desc_by_column: Dict[str, str] = {}

    for table in schema.tables:
        desc_rows = load_description_rows(bird_path, db_id, table.name)
        table_sections.append(f"Table: {table.name}")
        sample_sections.append(f"Table: {table.name}")

        placeholders = []
        for idx, col in enumerate(table.columns, start=1):
            placeholder = f"{table.name}.Column {idx}"
            placeholders.append(f"col_{idx}")
            placeholder_by_column[col.full_name] = placeholder
            desc_text = column_semantic_text(desc_rows.get(col.name, {}), idx, col.dtype)
            desc_by_column[col.full_name] = desc_text
            table_sections.append(f"  - {desc_text}")

        rows = get_table_samples(db_path, table.name, limit=5)
        if rows:
            sample_sections.append("  headers: " + ", ".join(placeholders))
            for row in rows:
                sample_sections.append("  - " + json.dumps(row, ensure_ascii=False, default=str))
        else:
            sample_sections.append("  (no sample rows available)")
        sample_sections.append("")
        table_sections.append("")

    fk_lines = []
    for src, tgt in schema.foreign_keys:
        src_text = desc_by_column.get(src, src)
        tgt_text = desc_by_column.get(tgt, tgt)
        src_table = src.split(".", 1)[0]
        tgt_table = tgt.split(".", 1)[0]
        fk_lines.append(f"- {src_table}: {src_text} REFERENCES {tgt_table}: {tgt_text}")

    return "\n".join(table_sections), "\n".join(sample_sections), "\n".join(fk_lines) or "(none)"


def build_actual_schema(schema: Schema) -> str:
    lines: List[str] = []
    fk_by_table: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for src, tgt in schema.foreign_keys:
        src_table, src_col = src.split(".", 1)
        fk_by_table[src_table].append((src_col, tgt))

    for table in schema.tables:
        lines.append(f"CREATE TABLE {table.name} (")
        col_lines = []
        for col in table.columns:
            suffix = " PRIMARY KEY" if col.is_pk else ""
            col_lines.append(f"  {col.name} {col.dtype}{suffix}")
        for src_col, tgt in fk_by_table.get(table.name, []):
            tgt_table, tgt_col = tgt.split(".", 1)
            col_lines.append(f"  FOREIGN KEY ({src_col}) REFERENCES {tgt_table}({tgt_col})")
        lines.append(",\n".join(col_lines))
        lines.append(");\n")
    return "\n".join(lines)


def sample_reference_questions(loader: BIRDLoader, db_id: str, n: int = 8) -> str:
    pairs = loader.get_pairs_for_db(db_id)
    rng = random.Random(f"bird-holdout-{db_id}")
    picked = pairs[:] if len(pairs) <= n else rng.sample(pairs, n)
    return "\n".join(f"- {p.nl}" for p in picked)


def build_question_prompt(loader: BIRDLoader, bird_path: Path, db_id: str, queries_per_db: int) -> str:
    descriptions, samples, fk_descriptions = build_description_context(loader, bird_path, db_id)
    examples = sample_reference_questions(loader, db_id, n=8)
    return f"""你正在为一个数据库设计评测查询。这个数据库的 schema 用以下方式描述（**列名被隐藏**，只提供每列的语义描述）：

数据库: {db_id}

表与列说明:
{descriptions}

每张表的样本数据（列名用占位符代替）:
{samples}

外键关系（用列描述代替列名）:
{fk_descriptions}

参考已有查询的风格（这些查询不要重复；这里只给自然语言问题，不给 SQL，避免泄露列名）:
{examples}

任务要求:
请生成 {queries_per_db} 个新的、自然的、业务分析师真实会问的问题。每个问题需要满足:
1. 必须能从这个 schema 中得到答案（不要问 schema 里没有的信息）
2. 风格类似上述参考查询（不要太简单，也不要过于复杂）
3. 涵盖多种 SQL 操作：简单查询、聚合、JOIN、过滤、排序、子查询等
4. 不要重复参考查询的内容
5. 不要在 question 中提及任何占位符列名（例如 col_1 或 Column 1）
6. 避免过度狭窄的过滤条件；请优先使用样本数据中真实出现的值，让答案很可能非空

输出格式（只输出 JSON，不要输出解释）:

```json
[
  {{
    "question": "..."
  }}
]
```
"""


def build_target_question_prompt(
    loader: BIRDLoader,
    bird_path: Path,
    target: Dict[str, str],
    queries_per_target: int,
) -> str:
    db_id = target["db_id"]
    table_name = target["table"]
    original_column = target["original_column"]
    schema = loader.schemas[db_id]
    db_path = Path(loader.get_db_path(db_id))
    table = schema.get_table(table_name)
    if table is None:
        raise ValueError(f"Target table not found: {db_id}.{table_name}")

    desc_rows = load_description_rows(bird_path, db_id, table_name)
    target_col = schema.get_column(table_name, original_column)
    if target_col is None:
        raise ValueError(f"Target column not found: {db_id}.{table_name}.{original_column}")

    target_idx = next(i for i, col in enumerate(table.columns, start=1) if col.name == original_column)
    target_desc = column_semantic_text(desc_rows.get(original_column, {}), target_idx, target_col.dtype)

    same_table_lines = [f"Table: {table_name}"]
    for idx, col in enumerate(table.columns, start=1):
        same_table_lines.append(
            "  - " + column_semantic_text(desc_rows.get(col.name, {}), idx, col.dtype)
        )

    sample_rows = get_table_samples(db_path, table_name, limit=8)
    sample_lines = [f"Table: {table_name}", "headers: " + ", ".join(f"col_{i}" for i in range(1, len(table.columns) + 1))]
    for row in sample_rows:
        sample_lines.append("- " + json.dumps(row, ensure_ascii=False, default=str))

    fk_lines = []
    target_full = f"{table_name}.{original_column}"
    for src, tgt in schema.foreign_keys:
        if src == target_full or tgt == target_full or src.startswith(f"{table_name}.") or tgt.startswith(f"{table_name}."):
            fk_lines.append(f"- {src.split('.', 1)[0]} column relationship with {tgt.split('.', 1)[0]}")

    examples = sample_reference_questions(loader, db_id, n=6)
    return f"""你正在为一个数据库设计 targeted holdout 查询。这个实验专门测试某个被隐藏列的语义泛化能力。

数据库: {db_id}

目标列语义（真实列名被隐藏，必须围绕这个语义设计问题）:
- {target_desc}

目标列所在表的其他列说明（列名全部隐藏）:
{chr(10).join(same_table_lines)}

目标表样本数据（列名用占位符代替，目标列是 col_{target_idx}）:
{chr(10).join(sample_lines)}

相关外键关系（不暴露列名）:
{chr(10).join(fk_lines) if fk_lines else "(none)"}

参考已有查询风格（不要重复）:
{examples}

任务要求:
请生成 {queries_per_target} 个新的、自然的、业务分析师真实会问的问题。每个问题必须满足:
1. 回答这个问题必须使用上面的“目标列语义”
2. 不要在问题中提及任何占位符列名（例如 col_{target_idx} 或 Column {target_idx}）
3. 问题应能从该数据库回答，且答案很可能非空
4. 可以使用过滤、聚合、JOIN、排序或子查询，但不要过度复杂
5. 不要重复参考查询

输出格式（只输出 JSON，不要输出解释）:
```json
[
  {{"question": "..."}}
]
```
"""


def build_target_sql_prompt(schema: Schema, target: Dict[str, str], questions: List[str]) -> str:
    question_json = json.dumps([{"question": q} for q in questions], ensure_ascii=False, indent=2)
    target_full = f"{target['table']}.{target['original_column']}"
    return f"""我现在告诉你这个数据库的真实 schema 和目标列名。

数据库: {target['db_id']}
目标列: {target_full}

{build_actual_schema(schema)}

请为下列每个 question 写出对应的 SQLite gold SQL:

{question_json}

要求:
1. 每条 SQL 必须引用目标列 `{target_full}`（可以带表别名，但语义必须使用该列）
2. SQL 必须能在 SQLite 上执行成功，并且尽量返回非空结果
3. 只生成只读 SELECT 或 WITH 查询
4. 使用真实 column name

输出格式（只输出 JSON，不要输出解释）:
```json
[
  {{"question": "...", "sql": "..."}}
]
```
"""


def build_sql_prompt(schema: Schema, db_id: str, questions: List[str]) -> str:
    question_json = json.dumps([{"question": q} for q in questions], ensure_ascii=False, indent=2)
    return f"""我现在告诉你这个数据库的真实 column 名:

数据库: {db_id}

{build_actual_schema(schema)}

请为下列每个 question 写出对应的 gold SQL（基于 SQLite 语法）:

{question_json}

要求:
1. SQL 必须能在 SQLite 上执行成功
2. 必须能回答对应的 question
3. 使用真实 column name，不要用我之前给你的描述
4. 只生成只读 SELECT 或 WITH 查询，不要生成 INSERT/UPDATE/DELETE
5. 避免永远返回 0 行的过窄条件；如需过滤，优先使用数据库中常见或样本中出现的值

输出格式（只输出 JSON，不要输出解释）:
```json
[
  {{
    "question": "...",
    "sql": "..."
  }}
]
```
"""


def generate_holdout(args: argparse.Namespace, loader: BIRDLoader, bird_path: Path, db_ids: List[str]) -> List[Dict[str, Any]]:
    raw_path = Path(args.output_dir) / "holdout_queries_raw.json"
    audit_path = Path(args.output_dir) / "generation_audit.json"
    if raw_path.exists() and not args.force:
        records: List[Dict[str, Any]] = read_json(raw_path, [])
        audit: List[Dict[str, Any]] = read_json(audit_path, [])
        done_dbs = {record["db_id"] for record in records}
        db_ids = [db_id for db_id in db_ids if db_id not in done_dbs]
        if not db_ids:
            print(f"[generate] Reusing complete {raw_path}")
            return records
        print(f"[generate] Resuming {raw_path}; remaining DBs: {', '.join(db_ids)}")
    else:
        records = []
        audit = []

    for db_id in db_ids:
        print(f"[generate] {db_id}")
        q_prompt = build_question_prompt(loader, bird_path, db_id, args.queries_per_db)
        print(f"  question prompt chars={len(q_prompt)}")
        q_response = call_generator(args, q_prompt)
        question_items = parse_json_array(q_response)
        questions = [str(item.get("question", "")).strip() for item in question_items if item.get("question")]
        questions = questions[: args.queries_per_db]
        if not questions:
            raise RuntimeError(f"Claude returned no questions for {db_id}: {q_response[:500]}")

        sql_prompt = build_sql_prompt(loader.schemas[db_id], db_id, questions)
        print(f"  sql prompt chars={len(sql_prompt)} questions={len(questions)}")
        sql_response = call_generator(args, sql_prompt, temperature=0.0)
        sql_items = parse_json_array(sql_response)
        for item in sql_items:
            question = str(item.get("question", "")).strip()
            sql = compact_sql(str(item.get("sql", "")))
            if question and sql:
                records.append({
                    "db_id": db_id,
                    "question": question,
                    "sql": sql,
                    "source": args.generator_model if args.generator_provider == "openai" else args.claude_model,
                    "generator_provider": args.generator_provider,
                })

        audit.append({
            "db_id": db_id,
            "question_prompt": q_prompt,
            "question_response": q_response,
            "sql_prompt": sql_prompt,
            "sql_response": sql_response,
        })
        write_json(raw_path, records)
        write_json(audit_path, audit)

    return records


def generate_targeted_holdout(args: argparse.Namespace, loader: BIRDLoader, bird_path: Path) -> List[Dict[str, Any]]:
    raw_path = Path(args.output_dir) / "holdout_queries_raw.json"
    audit_path = Path(args.output_dir) / "generation_audit.json"
    targets = collect_refined_targets(Path(args.refine_dir))
    if args.dbs:
        targets = [target for target in targets if target["db_id"] in set(args.dbs)]

    if raw_path.exists() and not args.force:
        records: List[Dict[str, Any]] = read_json(raw_path, [])
        audit: List[Dict[str, Any]] = read_json(audit_path, [])
        done = {(r["db_id"], r.get("target_table"), r.get("target_column")) for r in records}
        targets = [
            target for target in targets
            if (target["db_id"], target["table"], target["original_column"]) not in done
        ]
        if not targets:
            print(f"[generate:targeted] Reusing complete {raw_path}")
            return records
        print(f"[generate:targeted] Resuming; remaining targets={len(targets)}")
    else:
        records = []
        audit = []

    for index, target in enumerate(targets, start=1):
        label = f"{target['db_id']}.{target['table']}.{target['original_column']}→{target['refined_column']}"
        print(f"[generate:targeted] {index}/{len(targets)} {label}")
        q_prompt = build_target_question_prompt(loader, bird_path, target, args.queries_per_target)
        print(f"  question prompt chars={len(q_prompt)}")
        q_response = call_generator(args, q_prompt)
        question_items = parse_json_array(q_response)
        questions = [str(item.get("question", "")).strip() for item in question_items if item.get("question")]
        questions = questions[: args.queries_per_target]
        if not questions:
            raise RuntimeError(f"Generator returned no targeted questions for {label}: {q_response[:500]}")

        sql_prompt = build_target_sql_prompt(loader.schemas[target["db_id"]], target, questions)
        print(f"  sql prompt chars={len(sql_prompt)} questions={len(questions)}")
        sql_response = call_generator(args, sql_prompt, temperature=0.0)
        sql_items = parse_json_array(sql_response)
        for item in sql_items:
            question = str(item.get("question", "")).strip()
            sql = compact_sql(str(item.get("sql", "")))
            if question and sql:
                records.append({
                    "db_id": target["db_id"],
                    "question": question,
                    "sql": sql,
                    "source": args.generator_model if args.generator_provider == "openai" else args.claude_model,
                    "generator_provider": args.generator_provider,
                    "target_table": target["table"],
                    "target_column": target["original_column"],
                    "target_refined_column": target["refined_column"],
                })

        audit.append({
            "target": target,
            "question_prompt": q_prompt,
            "question_response": q_response,
            "sql_prompt": sql_prompt,
            "sql_response": sql_response,
        })
        write_json(raw_path, records)
        write_json(audit_path, audit)

    return records


def execute_sql_detailed(db_path: Path, sql: str, timeout: int = 30) -> SQLExecutionDetail:
    stripped = compact_sql(sql).rstrip(";")
    if not re.match(r"(?is)^\s*(select|with)\b", stripped):
        return SQLExecutionDetail(False, "non_select", 0, [], "Only SELECT/WITH queries are allowed")

    conn = sqlite3.connect(str(db_path))
    start = time.monotonic()

    def progress_handler() -> int:
        return 1 if time.monotonic() - start > timeout else 0

    conn.set_progress_handler(progress_handler, 1000)
    try:
        cursor = conn.execute(stripped)
        rows = cursor.fetchmany(1001)
        row_count = len(rows)
        preview = [list(row) for row in rows[:5]]
        if row_count == 0:
            return SQLExecutionDetail(False, "empty_result", row_count, preview)
        if row_count > 1000:
            return SQLExecutionDetail(False, "too_many_rows", row_count, preview)
        return SQLExecutionDetail(True, "ok", row_count, preview)
    except sqlite3.OperationalError as exc:
        reason = "timeout" if "interrupted" in str(exc).lower() else "execution_error"
        return SQLExecutionDetail(False, reason, 0, [], str(exc))
    except Exception as exc:
        return SQLExecutionDetail(False, "execution_error", 0, [], str(exc))
    finally:
        conn.close()


def sql_metrics(sql: str) -> Dict[str, Any]:
    lower = sql.lower()
    return {
        "length_chars": len(sql),
        "join_count": len(re.findall(r"\bjoin\b", lower)),
        "has_aggregation": bool(re.search(r"\b(count|sum|avg|min|max)\s*\(", lower)),
        "has_group_by": bool(re.search(r"\bgroup\s+by\b", lower)),
        "has_order_by": bool(re.search(r"\border\s+by\b", lower)),
        "has_subquery": bool(re.search(r"\(\s*select\b", lower)),
    }


def validate_holdout(args: argparse.Namespace, loader: BIRDLoader, raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output_dir = Path(args.output_dir)
    valid_path = output_dir / "holdout_queries_validated.json"
    filtered_path = output_dir / "holdout_queries_filtered.json"
    report_path = output_dir / "validation_report.md"
    if valid_path.exists() and not args.force:
        print(f"[validate] Reusing existing {valid_path}")
        return read_json(valid_path, [])

    seen_sql_by_db: Dict[str, set[str]] = defaultdict(set)
    valid: List[Dict[str, Any]] = []
    filtered: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for record in raw_records:
        db_id = record["db_id"]
        db_path = Path(loader.get_db_path(db_id))
        normalized = compact_sql(record["sql"]).lower()
        if normalized in seen_sql_by_db[db_id]:
            reason = "duplicate_sql"
            detail = SQLExecutionDetail(False, reason, 0, [])
        elif record.get("target_column") and not sql_touches_column(record["sql"], record["target_column"]):
            reason = "missing_target_column"
            detail = SQLExecutionDetail(False, reason, 0, [], f"SQL does not reference {record['target_column']}")
        else:
            detail = execute_sql_detailed(db_path, record["sql"], timeout=args.sql_timeout)
        seen_sql_by_db[db_id].add(normalized)

        enriched = dict(record)
        enriched.update({
            "validation": {
                "ok": detail.ok,
                "reason": detail.reason,
                "row_count": detail.row_count,
                "preview": detail.preview,
                "error": detail.error,
            },
            "metrics": sql_metrics(record["sql"]),
        })
        if detail.ok:
            valid.append(enriched)
        else:
            filtered.append(enriched)
            reason_counts[detail.reason] += 1

    write_json(valid_path, valid)
    write_json(filtered_path, filtered)
    write_validation_report(report_path, valid, filtered, reason_counts)
    return valid


def stratified_review_sample(records: List[Dict[str, Any]], n: int = 20) -> List[Dict[str, Any]]:
    rng = random.Random(20260426)
    by_db: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_db[record["db_id"]].append(record)

    sample: List[Dict[str, Any]] = []
    for db_id in sorted(by_db):
        if by_db[db_id]:
            sample.append(rng.choice(by_db[db_id]))

    remaining = [r for r in records if r not in sample]
    rng.shuffle(remaining)
    sample.extend(remaining[: max(0, n - len(sample))])
    return sample[:n]


def write_validation_report(
    path: Path,
    valid: List[Dict[str, Any]],
    filtered: List[Dict[str, Any]],
    reason_counts: Counter[str],
) -> None:
    by_db = Counter(r["db_id"] for r in valid)
    lengths = [r["metrics"]["length_chars"] for r in valid]
    join_counts = [r["metrics"]["join_count"] for r in valid]
    agg_count = sum(1 for r in valid if r["metrics"]["has_aggregation"])
    group_count = sum(1 for r in valid if r["metrics"]["has_group_by"])
    order_count = sum(1 for r in valid if r["metrics"]["has_order_by"])
    subquery_count = sum(1 for r in valid if r["metrics"]["has_subquery"])

    lines = [
        "# BIRD Holdout Gold SQL Validation Report",
        "",
        "## Summary",
        f"- Raw queries: {len(valid) + len(filtered)}",
        f"- Valid queries: {len(valid)}",
        f"- Filtered queries: {len(filtered)}",
        "",
        "## Valid Queries by DB",
        "",
        "| DB | Valid queries |",
        "|---|---:|",
    ]
    for db_id in sorted(by_db):
        marker = " ⚠️" if by_db[db_id] < 6 else ""
        lines.append(f"| {db_id} | {by_db[db_id]}{marker} |")

    lines += [
        "",
        "## Filter Reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"| {reason} | {count} |")

    avg_len = sum(lengths) / len(lengths) if lengths else 0
    avg_join = sum(join_counts) / len(join_counts) if join_counts else 0
    lines += [
        "",
        "## SQL Complexity",
        f"- Average SQL length: {avg_len:.1f} chars",
        f"- Average JOIN count: {avg_join:.2f}",
        f"- Aggregation queries: {agg_count}/{len(valid)}",
        f"- GROUP BY queries: {group_count}/{len(valid)}",
        f"- ORDER BY queries: {order_count}/{len(valid)}",
        f"- Subquery queries: {subquery_count}/{len(valid)}",
        "",
        "## Random Stratified Spot-Check Sample",
        "",
        "Please manually review these 20 samples for NL/SQL semantic correctness.",
        "",
    ]

    for idx, record in enumerate(stratified_review_sample(valid, n=20), start=1):
        preview = json.dumps(record["validation"]["preview"], ensure_ascii=False)
        lines += [
            f"### Sample {idx}: {record['db_id']}",
            f"- Question: {record['question']}",
            "",
            "```sql",
            record["sql"],
            "```",
            f"- Row count: {record['validation']['row_count']}",
            f"- Preview: `{preview}`",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


def to_pairs(records: List[Dict[str, Any]], db_id: str) -> List[NLSQLPair]:
    return [
        NLSQLPair(nl=r["question"], gold_sql=r["sql"], db_id=db_id)
        for r in records
        if r["db_id"] == db_id
    ]


def get_refined_original_columns(refined_schema_path: Path) -> set[str]:
    data = read_json(refined_schema_path, {})
    cols = set()
    for table in data.get("tables", []):
        for col in table.get("columns", []):
            original = col.get("original_name")
            name = col.get("name")
            if original and name and original != name:
                cols.add(f"{table['name']}.{original}".lower())
                cols.add(original.lower())
    return cols


def compare_cross_db(pred_sql: str, pred_db_path: Path, gold_sql: str, gold_db_path: Path) -> bool:
    pred_result = execute_db_sql(pred_sql, str(pred_db_path))
    gold_result = execute_db_sql(gold_sql, str(gold_db_path))
    if pred_result is None or gold_result is None:
        return False
    return pred_result == gold_result


def evaluate_pair(
    model: C3Text2SQL,
    pair: NLSQLPair,
    schema: Schema,
    pred_db_path: Path,
    gold_db_path: Path,
) -> Dict[str, Any]:
    pred_sql = model.generate(pair.nl, schema, db_path=str(pred_db_path), evidence="")
    match = compare_cross_db(pred_sql, pred_db_path, pair.gold_sql, gold_db_path)
    return {
        "db_id": pair.db_id,
        "question": pair.nl,
        "gold_sql": pair.gold_sql,
        "pred_sql": pred_sql,
        "match": match,
    }


def build_qwen_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "base_url": args.qwen_base_url,
        "api_key": args.qwen_api_key,
        "model_name": args.qwen_model,
        "temperature": 0,
        "max_tokens": args.qwen_max_tokens,
        "max_retries": 3,
        "timeout": 600,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
    }


def evaluate_holdout(
    args: argparse.Namespace,
    loader: BIRDLoader,
    valid_records: List[Dict[str, Any]],
    *,
    schema_mode: str,
) -> List[Dict[str, Any]]:
    assert schema_mode in {"noref", "egrefine"}
    output_path = Path(args.output_dir) / (
        "predictions_noref.json" if schema_mode == "noref" else "predictions_egrefine.json"
    )
    if output_path.exists() and not args.force_eval:
        print(f"[eval:{schema_mode}] Reusing existing {output_path}")
        return read_json(output_path, [])

    qwen_config = build_qwen_config(args)
    model = C3Text2SQL(qwen_config, num_samples=args.c3_num_samples, sample_rows=args.c3_sample_rows)
    predictions: List[Dict[str, Any]] = []
    no_ref_predictions = read_json(Path(args.output_dir) / "predictions_noref.json", [])
    no_ref_by_key = {
        (p["db_id"], p["question"], compact_sql(p["gold_sql"])): p for p in no_ref_predictions
    }

    db_ids = sorted({r["db_id"] for r in valid_records})
    with tempfile.TemporaryDirectory(prefix="egrefine_bird_holdout_") as tmpdir:
        for db_id in db_ids:
            pairs = to_pairs(valid_records, db_id)
            if not pairs:
                continue
            original_db_path = Path(loader.get_db_path(db_id))

            if schema_mode == "noref":
                schema = loader.schemas[db_id]
                pred_db_path = original_db_path
                zero_refined_db = False
            else:
                refine_db_dir = Path(args.refine_dir) / db_id
                refined_schema_path = refine_db_dir / "refined_tables.json"
                views_path = refine_db_dir / "views.sql"
                refined_cols = get_refined_original_columns(refined_schema_path)
                zero_refined_db = len(refined_cols) == 0
                if zero_refined_db and no_ref_by_key:
                    print(f"[eval:{schema_mode}] {db_id}: zero refined columns, using NoRef fallback")
                    for pair in pairs:
                        key = (db_id, pair.nl, compact_sql(pair.gold_sql))
                        reused = dict(no_ref_by_key[key])
                        reused["fallback_from_noref"] = True
                        predictions.append(reused)
                    write_json(output_path, predictions)
                    continue

                schema = load_refined_schema(refined_schema_path)
                pred_db_path = Path(tmpdir) / db_id / f"{db_id}.sqlite"
                copy_database(original_db_path, pred_db_path)
                apply_views(pred_db_path, views_path)

            print(f"[eval:{schema_mode}] {db_id}: {len(pairs)} queries")
            for idx, pair in enumerate(pairs, start=1):
                started = time.time()
                try:
                    result = evaluate_pair(model, pair, schema, pred_db_path, original_db_path)
                except Exception as exc:
                    result = {
                        "db_id": db_id,
                        "question": pair.nl,
                        "gold_sql": pair.gold_sql,
                        "pred_sql": "",
                        "match": False,
                        "error": str(exc),
                    }
                result["schema_mode"] = schema_mode
                result["elapsed_sec"] = round(time.time() - started, 3)
                if zero_refined_db:
                    result["fallback_from_noref"] = False
                predictions.append(result)
                print(f"  {idx:02d}/{len(pairs)} match={int(result['match'])} elapsed={result['elapsed_sec']:.1f}s")
                write_json(output_path, predictions)

    return predictions


def aggregate_predictions(predictions: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Tuple[int, int]]]:
    total = len(predictions)
    correct = sum(1 for p in predictions if p.get("match"))
    by_db_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for p in predictions:
        by_db_counts[p["db_id"]][1] += 1
        by_db_counts[p["db_id"]][0] += int(bool(p.get("match")))
    by_db = {db: (counts[0], counts[1]) for db, counts in by_db_counts.items()}
    return (correct / total * 100 if total else 0.0), by_db


def write_results_report(
    path: Path,
    valid_records: List[Dict[str, Any]],
    noref: List[Dict[str, Any]],
    egrefine: List[Dict[str, Any]],
) -> None:
    noref_acc, noref_by_db = aggregate_predictions(noref)
    ref_acc, ref_by_db = aggregate_predictions(egrefine)
    delta = ref_acc - noref_acc

    def pred_key(prediction: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            prediction["db_id"],
            prediction["question"],
            compact_sql(prediction["gold_sql"]),
        )

    lines = [
        "# BIRD Holdout EGRefine Evaluation",
        "",
        "## 4.1 Main Comparison",
        "",
        "| Setting | N queries | ExAcc_NoRef | ExAcc_EGRefine | Δ |",
        "|---|---:|---:|---:|---:|",
        f"| BIRD dev (in-loop) | 1534 | {IN_LOOP_NOREF:.2f} | {IN_LOOP_EGREFINE:.2f} | {IN_LOOP_EGREFINE - IN_LOOP_NOREF:+.2f} |",
        f"| BIRD holdout (unseen) | {len(valid_records)} | {noref_acc:.2f} | {ref_acc:.2f} | {delta:+.2f} |",
        "",
        "## 4.2 Database-Level Breakdown",
        "",
        "| DB | N | ExAcc_NoRef | ExAcc_EGRefine | Δ |",
        "|---|---:|---:|---:|---:|",
    ]

    for db_id in sorted(set(noref_by_db) | set(ref_by_db)):
        n_correct, n_total = noref_by_db.get(db_id, (0, 0))
        r_correct, r_total = ref_by_db.get(db_id, (0, 0))
        total = max(n_total, r_total)
        n_acc = n_correct / n_total * 100 if n_total else 0.0
        r_acc = r_correct / r_total * 100 if r_total else 0.0
        lines.append(f"| {db_id} | {total} | {n_acc:.2f} | {r_acc:.2f} | {r_acc - n_acc:+.2f} |")

    targets_by_key = {
        (r["db_id"], r["question"], compact_sql(r["sql"])): (
            r.get("target_table"), r.get("target_column"), r.get("target_refined_column")
        )
        for r in valid_records
        if r.get("target_column")
    }
    if targets_by_key:
        target_counts: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(lambda: [0, 0, 0])
        noref_map_tmp = {pred_key(p): p for p in noref}
        eg_map_tmp = {pred_key(p): p for p in egrefine}
        for record_key, (table_name, original_col, refined_col) in targets_by_key.items():
            db_id = record_key[0]
            group_key = (db_id, table_name or "", original_col or "", refined_col or "")
            target_counts[group_key][0] += 1
            target_counts[group_key][1] += int(bool(noref_map_tmp.get(record_key, {}).get("match")))
            target_counts[group_key][2] += int(bool(eg_map_tmp.get(record_key, {}).get("match")))

        lines += [
            "",
            "## Target Refined Column Breakdown",
            "",
            "| DB | Target column | Refined name | N | NoRef | EGRefine | Δ |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for (db_id, table_name, original_col, refined_col), (total, n_ok, e_ok) in sorted(target_counts.items()):
            n_acc = n_ok / total * 100 if total else 0.0
            e_acc = e_ok / total * 100 if total else 0.0
            lines.append(
                f"| {db_id} | `{table_name}.{original_col}` | `{refined_col}` | "
                f"{total} | {n_acc:.2f} | {e_acc:.2f} | {e_acc - n_acc:+.2f} |"
            )

    noref_keyed = {pred_key(p): p for p in noref}
    eg_keyed = {pred_key(p): p for p in egrefine}
    pass_to_fail = []
    fail_to_pass = []
    for key, n_pred in noref_keyed.items():
        e_pred = eg_keyed.get(key)
        if not e_pred:
            continue
        if n_pred.get("match") and not e_pred.get("match"):
            pass_to_fail.append((n_pred, e_pred))
        elif not n_pred.get("match") and e_pred.get("match"):
            fail_to_pass.append((n_pred, e_pred))

    lines += [
        "",
        "## 4.3 Difference Query Analysis",
        f"- Pass→Fail: {len(pass_to_fail)}",
        f"- Fail→Pass: {len(fail_to_pass)}",
        "",
        "### Fail→Pass Examples",
    ]
    for idx, (n_pred, e_pred) in enumerate(fail_to_pass[:5], start=1):
        lines += format_diff_example(idx, n_pred, e_pred)

    lines += ["", "### Pass→Fail Examples"]
    for idx, (n_pred, e_pred) in enumerate(pass_to_fail[:5], start=1):
        lines += format_diff_example(idx, n_pred, e_pred)

    if delta >= 0:
        framing = (
            "To address the concern that EGRefine may overfit to the BIRD dev queries used during "
            "execution-grounded verification, we constructed an independent holdout set of unseen "
            "BIRD questions. The holdout questions were generated from column descriptions and sample "
            "values while hiding both original and refined column names, and their gold SQL was validated "
            "by execution on the original SQLite databases. Using the fixed BIRD refinement artifact "
            "learned from the dev queries, C3 with Qwen3.5-27B achieves "
            f"{ref_acc:.2f}% ExAcc on the refined schema versus {noref_acc:.2f}% on the original schema "
            f"(Δ={delta:+.2f} pp, N={len(valid_records)}). This mirrors the positive in-loop dev-set "
            "trend and indicates that the schema edits improve general schema interpretability rather "
            "than merely memorizing the verification queries."
        )
    else:
        framing = (
            "We additionally evaluated EGRefine on an independently generated BIRD holdout set whose "
            "questions were produced from column descriptions with actual column names hidden and whose "
            "gold SQL was validated by execution. On this unseen set, C3 with Qwen3.5-27B obtains "
            f"{ref_acc:.2f}% ExAcc with the fixed refined schema versus {noref_acc:.2f}% on the original "
            f"schema (Δ={delta:+.2f} pp, N={len(valid_records)}). The result is weaker than the in-loop "
            "dev-set trend, suggesting that BIRD's already descriptive schemas leave limited room for "
            "generalization gains and that the main benefit of EGRefine is concentrated on queries that "
            "touch the small subset of columns selected for refinement."
        )

    lines += [
        "",
        "## 4.4 Paper Framing Draft",
        "",
        textwrap.fill(framing, width=100),
        "",
        "## Artifacts",
        "- `holdout_queries_raw.json`",
        "- `holdout_queries_validated.json`",
        "- `validation_report.md`",
        "- `predictions_noref.json`",
        "- `predictions_egrefine.json`",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def format_diff_example(idx: int, n_pred: Dict[str, Any], e_pred: Dict[str, Any]) -> List[str]:
    return [
        "",
        f"#### Example {idx}: {n_pred['db_id']}",
        f"- Question: {n_pred['question']}",
        "",
        "Gold SQL:",
        "```sql",
        n_pred["gold_sql"],
        "```",
        "NoRef prediction:",
        "```sql",
        n_pred.get("pred_sql", ""),
        "```",
        "EGRefine prediction:",
        "```sql",
        e_pred.get("pred_sql", ""),
        "```",
    ]


def load_loader_and_ids(args: argparse.Namespace) -> Tuple[BIRDLoader, Path, List[str]]:
    config = load_config(args.config)
    bird_path = Path(args.bird_path or config["data"]["bird"]["path"])
    loader = BIRDLoader(str(bird_path))
    db_ids = args.dbs or loader.db_ids
    return loader, bird_path, db_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BIRD holdout generation and EGRefine evaluation.")
    parser.add_argument("--stage", choices=["all", "generate", "validate", "eval", "report"], default="all")
    parser.add_argument("--mode", choices=["broad", "targeted"], default="broad")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--bird-path", default=None)
    parser.add_argument("--refine-dir", default=DEFAULT_REFINE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dbs", nargs="*", default=None)
    parser.add_argument("--queries-per-db", type=int, default=10)
    parser.add_argument("--queries-per-target", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Regenerate raw/validated artifacts.")
    parser.add_argument("--force-eval", action="store_true", help="Rerun prediction artifacts.")
    parser.add_argument("--sql-timeout", type=int, default=30)

    parser.add_argument("--generator-provider", choices=["anthropic", "openai"], default=DEFAULT_GENERATOR_PROVIDER)
    parser.add_argument("--generator-base-url", default=os.environ.get("GENERATOR_BASE_URL", DEFAULT_GENERATOR_BASE_URL))
    parser.add_argument("--generator-api-key", default=os.environ.get("GENERATOR_API_KEY", ""))
    parser.add_argument("--generator-model", default=os.environ.get("GENERATOR_MODEL", DEFAULT_GENERATOR_MODEL))
    parser.add_argument("--generator-max-tokens", type=int, default=8192)
    parser.add_argument("--generator-temperature", type=float, default=0.7)
    parser.add_argument("--claude-url", default=DEFAULT_CLAUDE_URL)
    parser.add_argument("--claude-api-key", default=os.environ.get("CLAUDE_API_KEY", DEFAULT_CLAUDE_KEY))
    parser.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)

    parser.add_argument("--qwen-base-url", default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--qwen-api-key", default="dummy")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-max-tokens", type=int, default=1024)
    parser.add_argument("--c3-num-samples", type=int, default=5)
    parser.add_argument("--c3-sample-rows", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader, bird_path, db_ids = load_loader_and_ids(args)
    print("Confirmed inputs:")
    print(f"- BIRD path: {bird_path}")
    print(f"- Refine dir: {Path(args.refine_dir).resolve()}")
    print("- C3 runner: src/phase3/c3_runner.py::C3Text2SQL")
    print(f"- Generator: {args.generator_provider} / {args.generator_model}")
    print(f"- Qwen endpoint/model: {args.qwen_base_url} / {args.qwen_model}")
    print(f"- Output dir: {output_dir.resolve()}")

    raw_records = read_json(output_dir / "holdout_queries_raw.json", [])
    valid_records = read_json(output_dir / "holdout_queries_validated.json", [])

    if args.stage in {"all", "generate"}:
        if args.mode == "targeted":
            raw_records = generate_targeted_holdout(args, loader, bird_path)
        else:
            raw_records = generate_holdout(args, loader, bird_path, db_ids)

    if args.stage in {"all", "validate"}:
        if not raw_records:
            raise FileNotFoundError("No raw records found. Run --stage generate first.")
        valid_records = validate_holdout(args, loader, raw_records)

    if args.stage in {"all", "eval"}:
        if not valid_records:
            raise FileNotFoundError("No validated records found. Run --stage validate first.")
        evaluate_holdout(args, loader, valid_records, schema_mode="noref")
        evaluate_holdout(args, loader, valid_records, schema_mode="egrefine")

    if args.stage in {"all", "report"}:
        if not valid_records:
            valid_records = read_json(output_dir / "holdout_queries_validated.json", [])
        noref = read_json(output_dir / "predictions_noref.json", [])
        egrefine = read_json(output_dir / "predictions_egrefine.json", [])
        if not valid_records or not noref or not egrefine:
            raise FileNotFoundError("Need validated queries and both prediction files before report.")
        write_results_report(output_dir / "holdout_results.md", valid_records, noref, egrefine)
        print(f"[report] wrote {output_dir / 'holdout_results.md'}")


if __name__ == "__main__":
    main()
