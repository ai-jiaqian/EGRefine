"""Schema 数据结构定义"""
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


@dataclass
class Column:
    name: str               # 列的 surface name (如 "nm", "sal")
    table: str              # 所属表名
    dtype: str              # 数据类型 (如 "INTEGER", "VARCHAR", "TEXT")
    is_pk: bool = False     # 是否主键
    fk_target: Optional[str] = None  # 外键指向 (如 "other_table.column")

    @property
    def full_name(self) -> str:
        return f"{self.table}.{self.name}"


@dataclass
class Table:
    name: str
    columns: List[Column]

    @property
    def primary_keys(self) -> List[Column]:
        return [c for c in self.columns if c.is_pk]


@dataclass
class Schema:
    db_id: str
    tables: List[Table]
    foreign_keys: List[Tuple[str, str]]  # [(col1_full_name, col2_full_name)]

    @property
    def all_columns(self) -> List[Column]:
        return [c for t in self.tables for c in t.columns]

    def get_table(self, table_name: str) -> Optional[Table]:
        for t in self.tables:
            if t.name == table_name:
                return t
        return None

    def get_column(self, table_name: str, column_name: str) -> Optional[Column]:
        t = self.get_table(table_name)
        if t is None:
            return None
        for c in t.columns:
            if c.name == column_name:
                return c
        return None

    def scope(self, column: Column) -> List[Column]:
        """论文 Eq.1: 同表列 + FK 关联列"""
        same_table = [c for c in self.all_columns if c.table == column.table]
        fk_related = []
        col_full = column.full_name
        for c1, c2 in self.foreign_keys:
            if c1 == col_full:
                fk_related.extend(
                    c for c in self.all_columns if c.full_name == c2
                )
            if c2 == col_full:
                fk_related.extend(
                    c for c in self.all_columns if c.full_name == c1
                )
        return list({id(c): c for c in same_table + fk_related}.values())

    def apply_refinement(self, mapping: Dict[str, str]) -> 'Schema':
        """返回应用 refinement 后的新 Schema（不修改原 Schema）。

        mapping: {"table.old_col": "new_col", ...}
        """
        new_tables = []
        for table in self.tables:
            new_cols = []
            for col in table.columns:
                key = col.full_name
                if key in mapping:
                    new_cols.append(Column(
                        name=mapping[key],
                        table=col.table,
                        dtype=col.dtype,
                        is_pk=col.is_pk,
                        fk_target=col.fk_target,
                    ))
                else:
                    new_cols.append(Column(
                        name=col.name,
                        table=col.table,
                        dtype=col.dtype,
                        is_pk=col.is_pk,
                        fk_target=col.fk_target,
                    ))
            new_tables.append(Table(name=table.name, columns=new_cols))
        return Schema(
            db_id=self.db_id,
            tables=new_tables,
            foreign_keys=list(self.foreign_keys),
        )


@dataclass
class NLSQLPair:
    nl: str         # 自然语言问题
    gold_sql: str   # gold SQL 查询
    db_id: str      # 对应的数据库 ID
    evidence: str = ""       # BIRD: hint/evidence text
    difficulty: str = ""     # BIRD: simple/moderate/challenging
