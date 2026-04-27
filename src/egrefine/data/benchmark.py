"""Benchmark 数据加载器 (BIRD / BEAVER / Dr.Spider)"""
import json
import os
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple

from egrefine.data.schema import Column, Table, Schema, NLSQLPair

logger = logging.getLogger(__name__)


def load_schema_from_sqlite(db_id: str, db_path: str) -> Schema:
    """从 SQLite 文件加载 Schema，包括表、列、主键、外键信息。"""
    conn = sqlite3.connect(db_path)

    # 获取所有用户表（排除 sqlite 内部表）
    table_names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]

    tables = []
    all_fks: List[Tuple[str, str]] = []

    for tname in table_names:
        # 列信息: (cid, name, type, notnull, default_value, pk)
        col_rows = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()

        # 外键信息: (id, seq, table, from, to, ...)
        fk_rows = conn.execute(f'PRAGMA foreign_key_list("{tname}")').fetchall()
        fk_map = {}  # from_col -> "ref_table.ref_col"
        for fk in fk_rows:
            ref_table = fk[2]
            from_col = fk[3]
            to_col = fk[4]
            fk_map[from_col] = f"{ref_table}.{to_col}"
            all_fks.append((f"{tname}.{from_col}", f"{ref_table}.{to_col}"))

        columns = []
        for row in col_rows:
            cname = row[1]
            dtype = row[2] if row[2] else "TEXT"
            is_pk = row[5] > 0
            fk_target = fk_map.get(cname)
            columns.append(Column(
                name=cname,
                table=tname,
                dtype=dtype,
                is_pk=is_pk,
                fk_target=fk_target,
            ))

        tables.append(Table(name=tname, columns=columns))

    conn.close()
    return Schema(db_id=db_id, tables=tables, foreign_keys=all_fks)


def load_bird(bird_path: str) -> Tuple[Dict[str, Schema], List[NLSQLPair]]:
    """加载 BIRD 数据集（支持 minidev 和标准 dev）。

    返回:
        schemas: {db_id: Schema}
        pairs: List[NLSQLPair]
    """
    # 自动检测 JSON 文件名
    json_path = _find_json(bird_path)
    db_dir = os.path.join(bird_path, "dev_databases")

    with open(json_path, "r") as f:
        data = json.load(f)

    pairs = [
        NLSQLPair(
            nl=d["question"], gold_sql=d["SQL"], db_id=d["db_id"],
            evidence=d.get("evidence", ""),
            difficulty=d.get("difficulty", ""),
        )
        for d in data
    ]

    # 加载每个数据库的 schema
    db_ids = sorted(set(p.db_id for p in pairs))
    schemas = {}
    for db_id in db_ids:
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            logger.warning("Database file not found: %s", db_path)
            continue
        schemas[db_id] = load_schema_from_sqlite(db_id, db_path)

    logger.info("Loaded %d databases, %d NL-SQL pairs", len(schemas), len(pairs))
    return schemas, pairs


def _find_json(bird_path: str) -> str:
    """自动检测 BIRD JSON 文件（兼容 minidev 和标准 dev）。"""
    candidates = [
        "mini_dev_sqlite.json",  # minidev
        "dev.json",              # 标准 dev
    ]
    for name in candidates:
        path = os.path.join(bird_path, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No BIRD JSON file found in {bird_path}. "
        f"Expected one of: {candidates}"
    )


class _BaseBenchmarkLoader:
    """所有 benchmark loader 的公共基类。

    子类只需在 __init__ 中设置 self.schemas / self.pairs，
    并实现 get_db_path()。
    """

    def _build_pairs_index(self) -> None:
        self._pairs_by_db: Dict[str, List[NLSQLPair]] = {}
        for p in self.pairs:
            self._pairs_by_db.setdefault(p.db_id, []).append(p)

    @property
    def db_ids(self) -> List[str]:
        return sorted(self.schemas.keys())

    def get_pairs_for_db(self, db_id: str) -> List[NLSQLPair]:
        return self._pairs_by_db.get(db_id, [])

    def get_db_path(self, db_id: str) -> str:
        raise NotImplementedError


class BIRDLoader(_BaseBenchmarkLoader):
    """BIRD benchmark loader."""

    def __init__(self, bird_path: str):
        self.bird_path = bird_path
        self.schemas, self.pairs = load_bird(bird_path)
        self._build_pairs_index()

    def get_db_path(self, db_id: str) -> str:
        return os.path.join(
            self.bird_path, "dev_databases", db_id, f"{db_id}.sqlite"
        )


def load_beaver(
    beaver_path: str,
    split: str = "nw",
    mysql_config: Optional[Dict] = None,
) -> Tuple[Dict[str, Schema], List[NLSQLPair]]:
    """Load BEAVER benchmark dataset.

    BEAVER uses MySQL databases. Schema is loaded from INFORMATION_SCHEMA,
    pairs from dev_nw.json or dev_dw.json.

    Args:
        beaver_path: Path to BEAVER dataset root.
        split: "nw" (5 DBs, 88 pairs) or "dw" (1 DB, 121 pairs) or "all".
        mysql_config: MySQL connection params {host, user, password, port}.

    Returns:
        schemas: {db_id: Schema}
        pairs: List[NLSQLPair]
    """
    from egrefine.data.db_connection import load_schema_from_mysql

    pairs = []

    if split in ("nw", "all"):
        nw_path = os.path.join(beaver_path, "dev_nw.json")
        if os.path.exists(nw_path):
            with open(nw_path, "r") as f:
                data = json.load(f)
            pairs.extend(
                NLSQLPair(nl=d["question"], gold_sql=d["sql"], db_id=d["db_id"])
                for d in data
            )

    if split in ("dw", "all"):
        dw_path = os.path.join(beaver_path, "dev_dw.json")
        if os.path.exists(dw_path):
            with open(dw_path, "r") as f:
                data = json.load(f)
            pairs.extend(
                NLSQLPair(nl=d["question"], gold_sql=d["sql"], db_id=d["db_id"])
                for d in data
            )

    # Load schemas from MySQL
    db_ids = sorted(set(p.db_id for p in pairs))
    schemas = {}
    for db_id in db_ids:
        try:
            schemas[db_id] = load_schema_from_mysql(db_id, db_id, mysql_config)
        except Exception as e:
            logger.warning("Failed to load MySQL schema for %s: %s", db_id, e)

    logger.info(
        "Loaded BEAVER (%s): %d databases, %d NL-SQL pairs",
        split, len(schemas), len(pairs),
    )
    return schemas, pairs


class BEAVERLoader(_BaseBenchmarkLoader):
    """BEAVER benchmark loader — MySQL-backed enterprise databases."""

    def __init__(
        self,
        beaver_path: str,
        split: str = "nw",
        mysql_config: Optional[Dict] = None,
    ):
        self.beaver_path = beaver_path
        self.mysql_config = mysql_config or {}
        self.split = split
        self.schemas, self.pairs = load_beaver(beaver_path, split, mysql_config)
        self._build_pairs_index()

    def get_db_path(self, db_id: str) -> str:
        return f"mysql://{db_id}"


# =====================================================================
# Dr.Spider
# =====================================================================

def load_drspider(
    drspider_path: str,
) -> Tuple[Dict[str, Schema], List[NLSQLPair]]:
    """加载 Dr.Spider DB 扰动测试集 (schema_abbreviation 或 schema_synonym).

    Dr.Spider 的 DB 扰动测试集结构:
      - questions_post_perturbation.json — 扰动后的 NL-SQL pairs
      - database_post_perturbation/{db_id}/{db_id}.sqlite — 扰动后的 SQLite 数据库
      - gold_post_perturbation.sql — 扰动后的 gold SQL (TAB 分隔: sql\\tdb_id)

    每个 db_id 形如 "concert_singer_0"，是基础 DB 的一个扰动变体。

    Args:
        drspider_path: Dr.Spider 某个测试集的根目录
            (如 .../dr_spider/DB_schema_abbreviation)

    Returns:
        schemas: {db_id: Schema}
        pairs: List[NLSQLPair]
    """
    questions_path = os.path.join(drspider_path, "questions_post_perturbation.json")
    db_dir = os.path.join(drspider_path, "database_post_perturbation")

    if not os.path.exists(questions_path):
        raise FileNotFoundError(
            f"Dr.Spider questions file not found: {questions_path}"
        )

    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = [
        NLSQLPair(
            nl=d["question"],
            gold_sql=d["query"],
            db_id=d["db_id"],
        )
        for d in data
    ]

    # 加载每个扰动数据库的 schema
    db_ids = sorted(set(p.db_id for p in pairs))
    schemas = {}
    for db_id in db_ids:
        db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            logger.warning("Dr.Spider database not found: %s", db_path)
            continue
        schemas[db_id] = load_schema_from_sqlite(db_id, db_path)

    logger.info(
        "Loaded Dr.Spider: %d databases, %d NL-SQL pairs from %s",
        len(schemas), len(pairs), drspider_path,
    )
    return schemas, pairs


def load_drspider_pre(
    drspider_path: str,
    spider_db_dir: str,
) -> Tuple[Dict[str, Schema], List[NLSQLPair]]:
    """加载 Dr.Spider pre-perturbation 数据（原始 Spider schema + 对应 queries）。

    Pre-perturbation 使用原始 Spider 的 DB 和列名，作为性能上限参照。
    queries 数量与 post-perturbation 相同（同一组 NL 问题）。

    Args:
        drspider_path: Dr.Spider 测试集根目录
            (如 .../dr_spider/DB_schema_abbreviation)
        spider_db_dir: 原始 Spider 数据库目录
            (如 .../dr_spider/Spider-dev/databases)

    Returns:
        schemas: {db_id: Schema}
        pairs: List[NLSQLPair]
    """
    questions_path = os.path.join(drspider_path, "questions_pre_perturbation.json")

    if not os.path.exists(questions_path):
        raise FileNotFoundError(
            f"Dr.Spider pre-perturbation questions not found: {questions_path}"
        )

    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = [
        NLSQLPair(
            nl=d["question"],
            gold_sql=d["query"],
            db_id=d["db_id"],
        )
        for d in data
    ]

    db_ids = sorted(set(p.db_id for p in pairs))
    schemas = {}
    for db_id in db_ids:
        db_path = os.path.join(spider_db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            logger.warning("Spider database not found: %s", db_path)
            continue
        schemas[db_id] = load_schema_from_sqlite(db_id, db_path)

    logger.info(
        "Loaded Dr.Spider pre-perturbation: %d databases, %d NL-SQL pairs",
        len(schemas), len(pairs),
    )
    return schemas, pairs


class DrSpiderLoader(_BaseBenchmarkLoader):
    """Dr.Spider DB 扰动测试集加载器.

    支持 DB_schema_abbreviation 和 DB_schema_synonym 两个测试集。
    pre=True 时加载 pre-perturbation（原始 Spider schema），用于上限参照。
    """

    def __init__(self, drspider_path: str, pre: bool = False,
                 spider_db_dir: str | None = None):
        self.drspider_path = drspider_path
        self.pre = pre
        if pre:
            if spider_db_dir is None:
                # Default: dr_spider/Spider-dev/databases/
                spider_db_dir = os.path.join(
                    os.path.dirname(drspider_path), "Spider-dev", "databases",
                )
            self.spider_db_dir = spider_db_dir
            self.schemas, self.pairs = load_drspider_pre(drspider_path, spider_db_dir)
        else:
            self.schemas, self.pairs = load_drspider(drspider_path)
        self._build_pairs_index()

    def get_db_path(self, db_id: str) -> str:
        if self.pre:
            return os.path.join(self.spider_db_dir, db_id, f"{db_id}.sqlite")
        return os.path.join(
            self.drspider_path, "database_post_perturbation",
            db_id, f"{db_id}.sqlite",
        )
