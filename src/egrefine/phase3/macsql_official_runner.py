"""Faithful deterministic MAC-SQL port, upstream 31a9df5e0d520be4769be57a4b9022e5e34a14f4.

Templates, parsing, value filtering and schema pruning retain upstream semantics.
Only the data adapter and OpenAI-compatible transport belong to EGRefine.
"""
import csv
import json
import logging
import sqlite3
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from egrefine.data.schema import Schema, Table, Column
from egrefine.models.llm_client import LLMClient
from egrefine.phase3.text2sql_runner import Text2SQLModel
from egrefine.phase3.macsql_official_utils import (
    parse_json, parse_sql_from_string, add_prefix, is_email, is_valid_date_column,
)
from egrefine.phase3 import macsql_official_prompts as prompts
from egrefine.phase4.backmapper import backmap

OFFICIAL_COMMIT = "31a9df5e0d520be4769be57a4b9022e5e34a14f4"
MAX_ROUND = 3


class MACSQLOfficialText2SQL(Text2SQLModel):
    def __init__(self, config: dict, dataset_name: str = "spider",
                 desc_dir: Optional[str] = None, detail_logger=None, tables_json: Optional[str] = None,
                 column_descriptions: Optional[dict] = None):
        config = deepcopy(config)
        config["temperature"] = 0
        config.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        self.llm = LLMClient(config)
        self.dataset_name = dataset_name
        self.desc_dir = desc_dir
        self.column_descriptions = column_descriptions or {}
        self.metadata = {}
        if tables_json:
            self.metadata = {d["db_id"]: d for d in json.loads(Path(tables_json).read_text())}
        self.trace_dir = Path(detail_logger.output_dir if detail_logger else
                              config.get("trace_dir", "logs/macsql_official"))

    def generate(
        self,
        nl: str,
        schema: Schema,
        db_path: Optional[str] = None,
        column_mapping: Optional[dict] = None,
        evidence: str = "",
    ) -> str:
        started = time.monotonic()
        record = {"official_commit": OFFICIAL_COMMIT, "db_id": schema.db_id,
                  "nl": nl, "evidence": evidence, "column_mapping": column_mapping or {},
                  "dataset_name": self.dataset_name, "calls": [], "executions": []}
        try:
            info = self._schema_info(schema, db_path, column_mapping or {})
            desc, fk, chosen = self._get_db_desc_str(schema.db_id, info, {})
            record["full_desc_str"], record["fk_str"] = desc, fk
            record["pruned"] = self._is_need_prune(schema, info)
            if record["pruned"]:
                prompt = prompts.selector_template.format(db_id=schema.db_id, query=nl,
                            evidence=evidence, desc_str=desc, fk_str=fk)
                try:
                    extracted = parse_json(self._call("Selector", prompt, record))
                except Exception as exc:
                    record["selector_error"] = str(exc)
                    extracted = {}
                desc, fk, chosen = self._get_db_desc_str(schema.db_id, info, extracted)
                record["extracted_schema"] = extracted
            record["desc_str"], record["chosen_db_schem_dict"] = desc, chosen
            template = (prompts.decompose_template_bird if self.dataset_name == "bird"
                        else prompts.decompose_template_spider)
            prompt = template.format(query=nl, evidence=evidence, desc_str=desc, fk_str=fk)
            reply = self._call("Decomposer", prompt, record).strip()
            sql = parse_sql_from_string(reply)
            record["qa_pairs"] = reply
            for _ in range(MAX_ROUND):
                if "error" in sql or not db_path:
                    break
                result = self._execute_sql(sql, db_path, column_mapping or {})
                record["executions"].append(result.copy())
                if result.get("timeout") or not self._is_need_refine(result):
                    break
                prompt = prompts.refiner_template.format(query=nl, evidence=evidence,
                            desc_str=desc, fk_str=fk, sql=add_prefix(result.get("sql")),
                            sqlite_error=result.get("sqlite_error"),
                            exception_class=result.get("exception_class"))
                sql = parse_sql_from_string(self._call("Refiner", prompt, record))
            record["pred_sql"] = sql
            return sql
        except Exception as exc:
            record["error"] = repr(exc)
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            directory = self.trace_dir / "per_db" / schema.db_id / "macsql_calls"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{uuid.uuid4().hex}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def _call(self, agent, prompt, record):
        call = {"agent": agent, "api_calls": []}
        record["calls"].append(call)
        started = time.monotonic()
        messages = [{"role": "user", "content": prompt}]
        call["messages"] = messages
        raw = self.llm.chat(messages, trace=call["api_calls"])
        call["raw_response"] = raw
        call["seconds"] = time.monotonic() - started
        return raw

    @staticmethod
    def _is_need_prune(schema, info=None):
        counts = [len(cols) for cols in info["desc_dict"].values()] if info else [len(t.columns) for t in schema.tables]
        total = sum(counts)
        return not (total // len(counts) <= 6 and total <= 30)

    def _is_need_refine(self, exec_result):
        if self.dataset_name == "spider":
            return "data" not in exec_result
        data = exec_result.get("data", None)
        if data is not None:
            if len(data) == 0:
                exec_result["sqlite_error"] = "no data selected"
                return True
            for t in data:
                for n in t:
                    if n is None:
                        exec_result["sqlite_error"] = "exist None value, you can add `NOT NULL` in SQL"
                        return True
            return False
        return True

    @staticmethod
    def _execute_sql(sql, db_path, column_mapping):
        # Phase 3 uses original storage; Stage 2 already supplies a VIEW database.
        original_sql = backmap(sql, column_mapping) if column_mapping else sql
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        started = time.monotonic()
        conn.set_progress_handler(lambda: int(time.monotonic() - started > 120), 1000)
        try:
            result = conn.execute(original_sql).fetchall()
            return {"sql": str(sql), "data": result[:5], "sqlite_error": "", "exception_class": ""}
        except sqlite3.Error as er:
            return {"sql": str(sql), "sqlite_error": str(' '.join(er.args)),
                    "exception_class": str(er.__class__), "timeout": time.monotonic() - started > 120}
        finally:
            conn.close()

    def _schema_info(self, schema, db_path, reverse):
        info = {"desc_dict": {}, "value_dict": {}, "pk_dict": {}, "fk_dict": {}}
        # Use only current names: no BIRD original-name lookup on renamed columns,
        # matching C3's description filter. DIN has no BIRD CSV descriptions.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) if db_path else None
        if conn:
            conn.text_factory = lambda b: b.decode(errors="ignore")
        metadata = self.metadata.get(schema.db_id)
        if not metadata and db_path and self.dataset_name == "spider":
            metadata_path = Path(db_path).parent.parent.parent / "tables.json"
            if metadata_path.is_file():
                metadata = next((d for d in json.loads(metadata_path.read_text()) if d["db_id"] == schema.db_id), None)
        full_names = {}
        if metadata and self.dataset_name == "spider":
            for (ti, name), (_, full) in zip(metadata["column_names_original"], metadata["column_names"]):
                if ti >= 0:
                    full_names[(metadata["table_names_original"][ti], name)] = full.replace('_', ' ')
        tables = list(schema.tables)
        # Official world_1 metadata exposes sqlite_sequence. The generic project
        # loader omits it; restore this observed metadata entry for upstream parity.
        if metadata and conn and "sqlite_sequence" in metadata["table_names_original"] and not schema.get_table("sqlite_sequence"):
            columns = conn.execute("PRAGMA table_info(`sqlite_sequence`)").fetchall()
            tables.append(Table("sqlite_sequence", [Column(c[1], "sqlite_sequence", c[2], bool(c[5])) for c in columns]))
        fks = schema.foreign_keys
        if metadata:
            order = {name: i for i, name in enumerate(metadata["table_names_original"])}
            tables = sorted((t for t in tables if t.name in order), key=lambda t: order[t.name])
            fk_order = []
            for si, ti in metadata["foreign_keys"]:
                st, sc = metadata["column_names_original"][si]
                tt, tc = metadata["column_names_original"][ti]
                fk_order.append((f"{metadata['table_names_original'][st]}.{sc}", f"{metadata['table_names_original'][tt]}.{tc}"))
            fks = sorted(fks, key=lambda fk: fk_order.index(fk) if fk in fk_order else len(fk_order))
        key_columns = {c.full_name for c in schema.all_columns if c.is_pk}
        key_columns.update(c for pair in schema.foreign_keys for c in pair)
        primary_columns = {c.full_name for c in schema.all_columns if c.is_pk}
        if metadata:
            primary_ids = [i for pk in metadata["primary_keys"] for i in (pk if isinstance(pk, list) else [pk])]
            key_ids = set(primary_ids + [i for pair in metadata["foreign_keys"] for i in pair])
            original_keys = {(metadata["table_names_original"][metadata["column_names_original"][i][0]], metadata["column_names_original"][i][1]) for i in key_ids}
            original_pks = {(metadata["table_names_original"][metadata["column_names_original"][i][0]], metadata["column_names_original"][i][1]) for i in primary_ids}
            key_columns = {c.full_name for t in tables for c in t.columns if (c.table, reverse.get(c.name,c.name)) in original_keys}
            primary_columns = {c.full_name for t in tables for c in t.columns if (c.table, reverse.get(c.name,c.name)) in original_pks}
            declared_columns = set(full_names)
            # Stage 2 supplies a VIEW and no backmapping: its renamed PK/FK
            # identities are already in Schema, while metadata has old names.
            primary_columns.update(c.full_name for c in schema.all_columns if c.is_pk and (c.table,reverse.get(c.name,c.name)) not in declared_columns)
            key_columns.update(primary_columns)
            key_columns.update(name for pair in schema.foreign_keys for name in pair if tuple(name.split('.',1)) not in declared_columns)
        if reverse:
            current_names = {(c.table, reverse.get(c.name,c.name)): c.name for c in schema.all_columns}
            for name in list(key_columns):
                table_name, column_name = name.split('.',1)
                key_columns.add(f"{table_name}.{current_names.get((table_name,column_name),column_name)}")
            visible_fks = []
            for source, target in fks:
                st, sc = source.split('.',1)
                tt, tc = target.split('.',1)
                visible_fks.append((f"{st}.{current_names.get((st,sc),sc)}", f"{tt}.{current_names.get((tt,tc),tc)}"))
            fks = visible_fks
        try:
            visible_tables = {t.name for t in tables}
            fks = [fk for fk in fks if fk[0].split(".", 1)[0] in visible_tables and fk[1].split(".", 1)[0] in visible_tables]
            for table in tables:
                descriptions = {}
                if self.desc_dir:
                    path = Path(self.desc_dir) / schema.db_id / "database_description" / f"{table.name}.csv"
                    if path.is_file():
                        with path.open(encoding="utf-8", errors="replace") as f:
                            for row in csv.DictReader(f):
                                name = row.get("original_column_name", "").strip()
                                desc = row.get("column_description", "").strip()
                                value_desc = row.get("value_description", "").strip()
                                if desc:
                                    descriptions[name] = desc + (f" ({value_desc})" if value_desc else "")
                info["desc_dict"][table.name] = []
                info["value_dict"][table.name] = []
                info["pk_dict"][table.name] = [c.name for c in table.columns if c.full_name in primary_columns]
                info["fk_dict"][table.name] = []
                for col in table.columns:
                    full_name = full_names.get((table.name, col.name), col.name.lower().replace('_', ' '))
                    description = self.column_descriptions.get((schema.db_id, table.name, col.name))
                    if description:
                        # Same per-column text and SQL comment sanitization as DIN.
                        full_name += " /* " + description.replace("*/", " ").replace("\n", " ") + " */"
                    info["desc_dict"][table.name].append([col.name, full_name,
                                                         descriptions.get(col.name, '')])
                    values_str = ''
                    if conn and col.full_name not in key_columns and not col.name.lower().endswith(('id', 'email', 'url')):
                        db_col = reverse.get(col.name, col.name)
                        quoted_col = '`' + db_col.replace('`', '``') + '`'
                        quoted_table = '`' + table.name.replace('`', '``') + '`'
                        values = [r[0] for r in conn.execute(f"SELECT {quoted_col} FROM {quoted_table} GROUP BY {quoted_col} ORDER BY COUNT(*) DESC")]
                        values_str = self._get_value_examples_str(values, col.dtype)
                    info["value_dict"][table.name].append([col.name, values_str])
            for source, target in fks:
                st, sc = source.split('.', 1)
                tt, tc = target.split('.', 1)
                info["fk_dict"][st].append((sc, tt, tc))
        finally:
            if conn:
                conn.close()
        return info

    def _get_value_examples_str(self, values: List[object], col_type: str):
            if not values:
                return ''
            if len(values) > 10 and col_type in ['INTEGER', 'REAL', 'NUMERIC', 'FLOAT', 'INT']:
                return ''

            vals = []
            has_null = False
            for v in values:
                if v is None:
                    has_null = True
                else:
                    tmp_v = str(v).strip()
                    if tmp_v == '':
                        continue
                    else:
                        vals.append(v)
            if not vals:
                return ''

            # drop meaningless values
            if col_type in ['TEXT', 'VARCHAR']:
                new_values = []

                for v in vals:
                    if not isinstance(v, str):

                        new_values.append(v)
                    else:
                        if self.dataset_name == 'spider':
                            v = v.strip()
                        if v == '': # exclude empty string
                            continue
                        elif ('https://' in v) or ('http://' in v): # exclude url
                            return ''
                        elif is_email(v): # exclude email
                            return ''
                        else:
                            new_values.append(v)
                vals = new_values
                tmp_vals = [len(str(a)) for a in vals]
                if not tmp_vals:
                    return ''
                max_len = max(tmp_vals)
                if max_len > 50:
                    return ''

            if not vals:
                return ''

            vals = vals[:6]

            is_date_column = is_valid_date_column(vals)
            if is_date_column:
                vals = vals[:1]

            if has_null:
                vals.insert(0, None)

            val_str = str(vals)
            return val_str

    def _build_bird_table_schema_list_str(self, table_name, new_columns_desc, new_columns_val):
            schema_desc_str = ''
            schema_desc_str += f"# Table: {table_name}\n"
            extracted_column_infos = []
            for (col_name, full_col_name, col_extra_desc), (_, col_values_str) in zip(new_columns_desc, new_columns_val):
                col_extra_desc = 'And ' + str(col_extra_desc) if col_extra_desc != '' and str(col_extra_desc) != 'nan' else ''
                col_extra_desc = col_extra_desc[:100]

                col_line_text = ''
                col_line_text += f'  ('
                col_line_text += f"{col_name},"

                if full_col_name != '':
                    full_col_name = full_col_name.strip()
                    col_line_text += f" {full_col_name}."
                if col_values_str != '':
                    col_line_text += f" Value examples: {col_values_str}."
                if col_extra_desc != '':
                    col_line_text += f" {col_extra_desc}"
                col_line_text += '),'
                extracted_column_infos.append(col_line_text)
            schema_desc_str += '[\n' + '\n'.join(extracted_column_infos).strip(',') + '\n]' + '\n'
            return schema_desc_str

    def _get_db_desc_str(self,
                             db_id: str,
                             db_info: dict,
                             extracted_schema: dict,
                             use_gold_schema: bool = False) -> List[str]:
            """
            Add foreign keys, and value descriptions of focused columns.
            :param db_id: name of sqlite database
            :param extracted_schema: {table_name: "keep_all" or "drop_all" or ['col_a', 'col_b']}
            :return: Detailed columns info of db; foreign keys info of db
            """
            desc_info = db_info['desc_dict']  # table:str -> columns[(column_name, full_column_name, extra_column_desc): str]
            value_info = db_info['value_dict']  # table:str -> columns[(column_name, value_examples_str): str]
            pk_info = db_info['pk_dict']  # table:str -> primary keys[column_name: str]
            fk_info = db_info['fk_dict']  # table:str -> foreign keys[(column_name, to_table, to_column): str]
            tables_1, tables_2, tables_3 = desc_info.keys(), value_info.keys(), fk_info.keys()
            assert set(tables_1) == set(tables_2)
            assert set(tables_2) == set(tables_3)

            # print(f"desc_info: {desc_info}\n\n")

            # schema_desc_str = f"[db_id]: {db_id}\n"
            schema_desc_str = ''  # for concat
            db_fk_infos = []  # use list type for unique check in db

            # print(f"extracted_schema:\n")
            # pprint(extracted_schema)
            # print()

            print(f"db_id: {db_id}")
            # For selector recall and compression rate calculation
            chosen_db_schem_dict = {} # {table_name: ['col_a', 'col_b'], ..}
            for (table_name, columns_desc), (_, columns_val), (_, fk_info), (_, pk_info) in \
                    zip(desc_info.items(), value_info.items(), fk_info.items(), pk_info.items()):

                table_decision = extracted_schema.get(table_name, '')
                if table_decision == '' and use_gold_schema:
                    continue

                # columns_desc = [(column_name, full_column_name, extra_column_desc): str]
                # columns_val = [(column_name, value_examples_str): str]
                # fk_info = [(column_name, to_table, to_column): str]
                # pk_info = [column_name: str]

                all_columns = [name for name, _, _ in columns_desc]
                primary_key_columns = [name for name in pk_info]
                foreign_key_columns = [name for name, _, _ in fk_info]

                important_keys = primary_key_columns + foreign_key_columns

                new_columns_desc = []
                new_columns_val = []

                print(f"table_name: {table_name}")
                if table_decision == "drop_all":
                    new_columns_desc = deepcopy(columns_desc[:6])
                    new_columns_val = deepcopy(columns_val[:6])
                elif table_decision == "keep_all" or table_decision == '':
                    new_columns_desc = deepcopy(columns_desc)
                    new_columns_val = deepcopy(columns_val)
                else:
                    llm_chosen_columns = table_decision
                    print(f"llm_chosen_columns: {llm_chosen_columns}")
                    append_col_names = []
                    for idx, col in enumerate(all_columns):
                        if col in important_keys:
                            new_columns_desc.append(columns_desc[idx])
                            new_columns_val.append(columns_val[idx])
                            append_col_names.append(col)
                        elif col in llm_chosen_columns:
                            new_columns_desc.append(columns_desc[idx])
                            new_columns_val.append(columns_val[idx])
                            append_col_names.append(col)
                        else:
                            pass

                    # todo: check if len(new_columns_val) ≈ 6
                    if len(all_columns) > 6 and len(new_columns_val) < 6:
                        for idx, col in enumerate(all_columns):
                            if len(append_col_names) >= 6:
                                break
                            if col not in append_col_names:
                                new_columns_desc.append(columns_desc[idx])
                                new_columns_val.append(columns_val[idx])
                                append_col_names.append(col)

                # 统计经过 Selector 筛选后的表格信息
                chosen_db_schem_dict[table_name] = [col_name for col_name, _, _ in new_columns_desc]

                # 1. Build schema part of prompt
                # schema_desc_str += self._build_bird_table_schema_sqlite_str(table_name, new_columns_desc, new_columns_val)
                schema_desc_str += self._build_bird_table_schema_list_str(table_name, new_columns_desc, new_columns_val)

                # 2. Build foreign key part of prompt
                for col_name, to_table, to_col in fk_info:
                    from_table = table_name
                    if '`' not in str(col_name):
                        col_name = f"`{col_name}`"
                    if '`' not in str(to_col):
                        to_col = f"`{to_col}`"
                    fk_link_str = f"{from_table}.{col_name} = {to_table}.{to_col}"
                    if fk_link_str not in db_fk_infos:
                        db_fk_infos.append(fk_link_str)
            fk_desc_str = '\n'.join(db_fk_infos)
            schema_desc_str = schema_desc_str.strip()
            fk_desc_str = fk_desc_str.strip()

            return schema_desc_str, fk_desc_str, chosen_db_schem_dict
