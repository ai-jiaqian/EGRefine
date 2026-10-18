"""临时 VIEW SQL 构造"""
from egrefine.data.schema import Schema, Column


def build_view(
    schema: Schema,
    column: Column,
    candidate_name: str,
    view_prefix: str = "refined_",
) -> str:
    """构造 CREATE VIEW SQL，将指定列重命名为 candidate_name。

    其他列保持原名不变。
    """
    table = schema.get_table(column.table)
    select_parts = []
    for col in table.columns:
        if col.name == column.name and candidate_name != column.name:
            select_parts.append(f"  {col.name} AS {candidate_name}")
        else:
            select_parts.append(f"  {col.name}")

    view_name = f"{view_prefix}{table.name}"
    return (
        f"CREATE VIEW {view_name} AS\n"
        f"SELECT\n"
        f"{','.join(chr(10) + p if i > 0 else p for i, p in enumerate(select_parts))}\n"
        f"FROM {table.name};"
    )
