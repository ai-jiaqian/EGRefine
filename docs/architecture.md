# EGRefine Architecture

## Overview

EGRefine (Execution-Grounded Refinement) is a plug-and-play preprocessing layer
that improves Text-to-SQL accuracy by automatically renaming ambiguous database
columns before inference. Real-world schemas frequently use abbreviations (`nm`,
`sal`, `dept`), generic tokens (`status`, `type`, `code`), or mixed naming
conventions — all of which confuse schema-linking in Text-to-SQL systems.
EGRefine identifies these columns, proposes clearer names via an LLM, and
selects the best name using downstream SQL *execution results* as the signal.
No manual annotation is required.

<!-- TODO: pipeline figure (paper Figure 2) -->

**High-level data flow:**
```
Schema (SQLite) ──► Phase 1: Prune ──► Phase 2: Generate candidates
                                            │
                                            ▼
                              Phase 3: Execution-grounded verify
                                            │
                                            ▼
                              Phase 4: Emit VIEW + back-mapper
                                            │
                                            ▼
                         Refined schema (VIEW layer, zero DB mutation)
                                            │
                                            ▼
                               Text-to-SQL ──► SQL ──► ExAcc
```

---

## Theoretical Framework

### Optimization Problem

EGRefine formalises schema refinement as a constrained optimisation over a
column-name mapping:

```
r* = argmax_{r ∈ R(S)}  Quality(r(S), M, Q)

where:
  S       = original database schema (set of tables and columns)
  r       = refinement mapping  (old column name → new column name)
  R(S)    = all valid refinement mappings (uniqueness, SQL identifier constraints)
  M       = {M_1, ..., M_l}   a set of Text-to-SQL models
  Q       = {(nl_j, sql*_j)}  a benchmark of NL-SQL pairs
  Quality(S, M, Q) = (1/|M|) × Σ_j ExAcc(M_j, S, Q)
  ExAcc   = execution accuracy (predicted SQL result set == gold result set)
```

### Why Column-wise Decomposition

The joint search space is O(k^m) — k candidates × m columns — and exact
optimisation is NP-hard. EGRefine decomposes the problem column by column:

```
For each column c_i:
    c_i* = argmax_{c' ∈ Candidates(c_i) ∪ {original}}
                    Quality(S[c_i → c'], M, Q(c_i))

where Q(c_i) = the subset of Q whose gold SQL references column c_i
```

This is exact when column refinements are independent, and near-optimal in
practice because schema-linking errors are typically column-local.

### Conservative Rule

**If the best candidate's ExAcc does not strictly exceed the original name's
ExAcc, keep the original name.** Formally:

```
if max_score > original_score:
    select best candidate, delta = max_score − original_score
else:
    revert to original name, delta = 0
```

This guarantees monotone non-regression: EGRefine never makes things worse on
the columns it evaluates. Combined with a minimum delta threshold `min_delta`
(default 0.05), it also filters out noisy marginal improvements.

### VIEW Equivalence (Theorem 1)

Refinement is implemented via SQL VIEWs — the original database is never
mutated:

```sql
-- Original table: employees(nm, sal, dept, dt)
-- After refinement, original table is renamed to _orig_employees

CREATE VIEW employees AS
SELECT
  nm   AS employee_name,
  sal  AS annual_salary,
  dept AS department_name,
  dt   AS hire_date
FROM _orig_employees;
```

The downstream model sees the VIEW's column names and generates SQL using
the refined names. A back-mapper translates predicted SQL back to original
column names before execution. Theorem 1 proves that for any SQL query on
the VIEW, the back-mapped query on the base table returns an identical result
set.

---

## Phase 1: Pruning

**Goal:** Filter the schema's m columns down to n candidate columns
(n ≪ m) that are likely ambiguous, reducing Phase 2 / Phase 3 cost.

**Input:** `Schema` object  
**Output:** `candidate_columns: List[Column]`

Phase 1 operates in two modes (set via `phase1.method` in config):

### Rule mode (S1–S4 heuristics)

Four signals whose union forms the candidate set. False positives are
automatically filtered by Phase 3 (conservative rule), so recall is
prioritised over precision.

| Signal | Name | Trigger condition |
|--------|------|-------------------|
| S1 | Short name | `len(column.name) <= 3` — single-character or abbreviation tokens |
| S2 | High similarity | Another column in the same schema has embedding cosine similarity > 0.85 — indicates potential confusion between semantically close names. Short names are prefixed with table name before embedding: `{table}.{col}` |
| S3 | Naming inconsistency | The schema mixes ≥2 naming conventions (camelCase / snake_case / ALLCAPS / alllower) — minority-style columns are flagged |
| S4 | Generic vocabulary | Column name is in the generic-token set: `{status, type, code, value, flag, name, num, desc, id, date, text, info, data, level, state, category, group, class, kind, mode, label, title, result, count, amount, total, number, index, key, note, comment, remark}` |

Design notes:
- Primary key columns (`is_pk=True`) are skipped by default
  (`phase1.skip_primary_keys: true`).
- S2 requires an embedding endpoint; if not configured, it is automatically
  disabled. S1 + S3 + S4 alone cover most schema-quality problems.

### LLM mode

Two-step structural exclusion + YES/NO prompting:

1. **Structural exclusion**: automatically skip columns that are PKs, FKs,
   or share an identical name with a column in another table (cross-table
   shared names are unlikely to benefit from renaming).
2. **LLM YES/NO**: for each remaining column send a single-turn prompt to the
   same model used in Phases 2–3, asking whether the column name is potentially
   ambiguous. Response is parsed from the first token (`yes` / `no`).

LLM mode configuration (under `phase1.model`) uses the same
OpenAI-compatible endpoint as Phase 2–3, so no second model deployment is
needed. Concurrency is controlled by `phase1.concurrency` (recommended: 48).

---

## Phase 2: Candidate Generation

**Goal:** For each candidate column, produce k alternative names using an LLM.

**Input:** `candidate_column: Column`, `schema: Schema`, `k: int` (default 3)  
**Output:** `List[CandidateName]` where each entry is `{name: str, reason: str}`

### Prompt template

```
You are a database schema expert. Given a column in a database,
propose {k} clearer, more descriptive alternative names.

## Column Information
- Current name: `{column.name}`
- Table: `{column.table}`
- Data type: `{column.dtype}`
- Is primary key: {column.is_pk}
- Foreign key target: {column.fk_target or "None"}

## Neighboring Columns (same table)
- {col.name} ({col.dtype})
  ...

## Foreign Key Related Columns
- {fk_col.table}.{fk_col.name}
  ...

## Sample Data Values (20 rows)
{sample_values}

## Task
Propose exactly {k} alternative column names that are:
1. More descriptive and unambiguous
2. Following snake_case convention
3. Consistent with neighboring column naming patterns

Respond in JSON format only, no other text:
[{"name": "proposed_name", "reason": "brief justification"}, ...]
```

### Implementation notes

- **Sample data**: 20 rows of actual column values are included in the prompt.
  Phase 3 verification also requires sample data in the schema context —
  omitting them significantly degrades SQL generation quality.
- **Retry**: JSON parse failures are retried up to `max_retries` times
  (default 3). Candidate names that are not valid SQL identifiers (contain
  spaces, start with digits, match SQL reserved words) are filtered out.
- **Caching**: results are keyed by `(db_id, table_name, column_name)` and
  persisted to `{cache_dir}/phase2/`. Re-runs skip columns already in cache.
- **Temperature**: 0.7 (to promote diverse candidates). Text-to-SQL generation
  in Phase 3 uses temperature 0.

---

## Phase 3: Execution-Grounded Verification

**Goal:** Score each candidate name by running downstream Text-to-SQL on the
queries that reference the column, then apply the conservative selection rule.

**Input:**
- `column: Column`
- `candidates: List[CandidateName]` — from Phase 2
- `Q_ci: List[NLSQLPair]` — benchmark queries whose gold SQL references `column`
- `models: List[Text2SQLModel]`
- `db_path: str`

**Output:**
- `selected_name: str`
- `delta: float` — ExAcc gain vs original name
- `was_changed: bool`
- `all_scores: Dict[str, float]`

### Verification loop (pseudocode)

```python
def verify_column(column, candidates, Q_ci, models, db_path, schema):
    all_candidates = [column.name] + [c.name for c in candidates]
    scores = {}

    for cand in all_candidates:
        exacc_per_model = []
        for model in models:
            correct = 0
            for (nl, gold_sql) in Q_ci:
                modified_schema = schema.apply_refinement({column.name: cand})
                pred_sql = model.generate(nl, modified_schema)
                original_sql = backmap(pred_sql, {cand: column.name})
                if execute_and_compare(original_sql, gold_sql, db_path):
                    correct += 1
            exacc_per_model.append(correct / len(Q_ci))
        scores[cand] = sum(exacc_per_model) / len(exacc_per_model)

    original_score = scores[column.name]
    best_cand = max(scores, key=scores.get)
    best_score = scores[best_cand]

    # Conservative rule with min_delta threshold
    if best_score - original_score > min_delta:
        return best_cand, best_score - original_score, True, scores
    else:
        return column.name, 0.0, False, scores
```

### Conservative selection and min_delta

The default `min_delta` threshold is **0.05** (5 percentage points). Empirical
sensitivity analysis on Dr.Spider-Abbr shows:
- `min_delta ∈ [0.03, 0.10]` → downstream ExAcc varies < 1.8 pp across all
  three Text-to-SQL algorithms (within the LLM noise floor).
- `min_delta = 0.01` → clearly too aggressive: DIN-SQL ExAcc regresses on all
  tested settings.
- `min_delta = 0.05` is the shared optimal point for C3, DIN-SQL, and MAC-SQL.

### Q(c_i)-empty fallback

If a column is never referenced by any gold SQL query (`Q_ci = []`),
execution verification is impossible. In this case EGRefine falls back to the
first candidate returned by Phase 2 (the LLM's implicit top-ranked choice) and
logs `verification_method = "llm_fallback"`. The fallback fraction is reported
in `statistics.json`.

### Conflict resolution (Phase 3d)

After all columns are scored independently, a conflict-resolution pass checks
for name collisions within the same scope (same table, or FK-linked tables).
If two columns are assigned the same refined name:
- The column with the higher `delta` retains the name.
- The other column reverts to the next-best candidate from `all_scores`, or to
  the original name if no alternative exists.

This pass runs `phase3.conflict_resolution_rounds` times (default 2).

---

## Phase 4: VIEW Synthesis + Back-Mapping

### VIEW generation

For each table where at least one column was refined, Phase 4 emits a CREATE
VIEW statement. The original table is renamed to `_orig_{table_name}` so the
VIEW can use the original table name — downstream models see an unchanged table
namespace.

```python
def generate_views(schema, refinements):
    views = []
    for table in schema.tables:
        changed_cols = [c for c in table.columns if c in refinements]
        if not changed_cols:
            continue   # table unchanged, no VIEW needed

        select_parts = []
        for col in table.columns:
            if col in refinements:
                select_parts.append(f"  {col.name} AS {refinements[col]}")
            else:
                select_parts.append(f"  {col.name}")

        view_sql = (
            f"CREATE VIEW {table.name} AS\n"
            f"SELECT\n"
            f"{',\n'.join(select_parts)}\n"
            f"FROM _orig_{table.name};"
        )
        views.append(view_sql)
    return views
```

### Back-mapping

Back-mapping translates predicted SQL (which uses refined column names) back
to original column names before execution against the original database.

Two modes (set via `phase4.backmapper`):

**`simple`** — naive string replace:
```python
sorted_mapping = sorted(reverse_mapping.items(),
                        key=lambda x: len(x[0]), reverse=True)
for refined_name, original_name in sorted_mapping:
    sql = sql.replace(refined_name, original_name)
```

**`regex`** (default, recommended) — word-boundary replacement to avoid
substring collisions:
```python
import re
for refined_name, original_name in sorted_mapping:
    sql = re.sub(r'\b' + re.escape(refined_name) + r'\b', original_name, sql)
```

**Critical ordering**: replacements must be applied in descending order of
refined name length. Otherwise a shorter name (e.g. `name`) can corrupt a
longer name (e.g. `employee_name`) before the longer name is processed.

### Theorem 1: Query equivalence

For any SQL query Q_refined that references the refined VIEW, the back-mapped
query Q_orig on the original (renamed) base table returns an identical result
set. This holds because:
1. The VIEW selects all columns of the original table with only alias
   substitutions — no filtering, aggregation, or join is introduced.
2. Back-mapping is a syntactic renaming of identifiers.
3. SQLite resolves VIEW queries by substituting the VIEW definition at
   query parse time.

The test `tests/test_view_equivalence.py` programmatically validates this
property on the BIRD formula_1 database.

---

## Two-Stage Execution Model

EGRefine separates *refinement* from *evaluation* into two fully decoupled
stages. The same refined schema can be evaluated against many Text-to-SQL
systems and backbone LLMs without re-running the (expensive) Phase 1–3.

### Stage 1 — `egrefine-refine`

Runs Phases 1–4. Writes four files per database:

| File | Description |
|------|-------------|
| `views.sql` | `ALTER TABLE ... RENAME TO _orig_...` + `CREATE VIEW ...` |
| `refined_tables.json` | Schema description with original table names and refined column names |
| `orig_table_map.json` | Table rename map `{"district": "_orig_district", ...}` |
| `statistics.json` | Per-phase statistics (columns pruned, candidates generated, delta distribution, fallback count) |

### Stage 2 — `egrefine-eval`

1. Copies the SQLite database to a temp directory.
2. Executes `views.sql` on the copy (renames original tables, creates VIEWs).
3. Runs the Text-to-SQL model using `refined_tables.json` as the schema
   description.
4. For gold SQL evaluation: replaces table names in gold SQL using
   `orig_table_map.json` (`district` → `_orig_district`) so gold SQL
   executes on the original base tables — which still hold the data.
5. Compares predicted and gold result sets to compute ExAcc.

**Why this design?** Refinement (Phase 3) is by far the most expensive step —
it runs the Text-to-SQL model O(k × |Q(c_i)|) times per candidate column.
Evaluation is comparatively cheap. Decoupling lets researchers ablate different
evaluation models (C3 / DIN-SQL / MAC-SQL) or backbone sizes (9B / 27B) against
the same refined schema without repeating Phase 3.

---

## Data Structures

### Schema, Table, Column

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

@dataclass
class Column:
    name: str               # surface name, e.g. "nm", "sal"
    table: str              # parent table name
    dtype: str              # "INTEGER", "VARCHAR", "TEXT", ...
    is_pk: bool = False
    fk_target: Optional[str] = None   # "other_table.column" or None

@dataclass
class Table:
    name: str
    columns: List[Column]

@dataclass
class Schema:
    db_id: str
    tables: List[Table]
    foreign_keys: List[Tuple[str, str]]  # [(col1_full, col2_full)]

    @property
    def all_columns(self) -> List[Column]: ...

    def scope(self, column: Column) -> List[Column]:
        """Paper Eq.1: same-table columns + FK-linked columns."""
        ...

    def apply_refinement(self, mapping: Dict[str, str]) -> 'Schema':
        """Returns a new Schema with column names substituted (non-mutating)."""
        ...
```

### NLSQLPair

```python
@dataclass
class NLSQLPair:
    nl: str           # natural language question
    gold_sql: str     # gold SQL query
    db_id: str
```

### Refinement output JSON

```json
{
  "db_id": "financial",
  "refinements": [
    {
      "table": "account",
      "column": "freq",
      "original_name": "freq",
      "refined_name": "transaction_frequency",
      "delta": 0.08,
      "verification_method": "execution",
      "all_scores": {
        "freq": 0.62,
        "transaction_frequency": 0.70,
        "frequency": 0.66,
        "payment_frequency": 0.64
      }
    }
  ],
  "view_definitions": ["CREATE VIEW refined_account AS ..."],
  "statistics": {
    "total_columns": 55,
    "candidates_after_pruning": 12,
    "columns_refined": 7,
    "columns_kept_original": 5,
    "fallback_count": 2,
    "avg_delta": 0.053
  }
}
```

---

## Configuration Reference

All LLM and embedding models use the same three-parameter OpenAI-compatible
interface: `base_url`, `api_key`, `model_name`. The tool is provider-agnostic.

| Config key | Type | Description | Example |
|---|---|---|---|
| `models.candidate_llm.base_url` | str | OpenAI-compatible endpoint for Phase 2 candidate generation | `"http://localhost:8000/v1"` |
| `models.candidate_llm.api_key` | str | API key (any value for local deployments) | `"token-abc"` |
| `models.candidate_llm.model_name` | str | Model name as registered on the endpoint | `"Qwen2.5-Coder-32B"` |
| `models.candidate_llm.temperature` | float | Sampling temperature for candidate diversity | `0.7` |
| `models.candidate_llm.max_tokens` | int | Max output tokens for candidate JSON | `512` |
| `models.candidate_llm.max_retries` | int | JSON parse retry limit | `3` |
| `models.text2sql` | list | List of Text-to-SQL model configs for Phase 3 verification | see below |
| `models.text2sql[*].name` | str | Identifier used in logs and result files | `"model_A"` |
| `models.text2sql[*].base_url` | str | Endpoint for this verification model | `"http://localhost:8000/v1"` |
| `models.text2sql[*].temperature` | float | Should be 0 for deterministic verification | `0` |
| `models.text2sql[*].max_tokens` | int | Max SQL output length | `1024` |
| `models.embedding.base_url` | str | Endpoint supporting `/v1/embeddings` (Phase 1 S2) | `"http://localhost:8000/v1"` |
| `models.embedding.model_name` | str | Embedding model name | `"bge-m3"` |
| `models.embedding.type` | str | `"api"` (default) or `"local"` (sentence-transformers) | `"api"` |
| `models.embedding.local_model` | str | HuggingFace model name (only if type=local) | `"all-MiniLM-L6-v2"` |
| `phase1.method` | str | `"rule"` (S1–S4) or `"llm"` | `"llm"` |
| `phase1.model` | dict | LLM config for Phase 1 screening (method=llm only); same fields as candidate_llm | — |
| `phase1.concurrency` | int | Concurrent LLM calls in LLM mode | `48` |
| `phase1.signals.short_name.enabled` | bool | Enable S1 | `true` |
| `phase1.signals.short_name.max_length` | int | Name length threshold for S1 | `3` |
| `phase1.signals.high_similarity.enabled` | bool | Enable S2 | `true` |
| `phase1.signals.high_similarity.threshold` | float | Cosine similarity threshold | `0.85` |
| `phase1.signals.naming_inconsistency.enabled` | bool | Enable S3 | `true` |
| `phase1.signals.generic_vocabulary.enabled` | bool | Enable S4 | `true` |
| `phase1.skip_primary_keys` | bool | Skip PK columns | `true` |
| `phase2.k` | int | Candidates to generate per column | `3` |
| `phase2.sample_rows` | int | Data rows to include in Phase 2 prompt | `20` |
| `phase3.conservative` | bool | Enable conservative rule | `true` |
| `phase3.min_delta` | float | Minimum ExAcc improvement to accept a rename | `0.05` |
| `phase3.conflict_resolution_rounds` | int | Rounds of name-collision resolution | `2` |
| `phase4.view_prefix` | str | Prefix for VIEW names (legacy; current implementation uses original table name) | `"refined_"` |
| `phase4.backmapper` | str | `"simple"` or `"regex"` | `"regex"` |
| `data.bird.path` | str | Path to BIRD dataset root | `"/path/to/BIRD"` |
| `data.bird.split` | str | `"dev"` or `"train"` | `"dev"` |
| `data.drspider_abbr.path` | str | Path to Dr.Spider-Abbr dataset root | `"/path/to/drspider"` |
| `data.drspider_syn.path` | str | Path to Dr.Spider-Syn dataset root | `"/path/to/drspider"` |
| `data.beaver.path` | str | Path to BEAVER dataset root | `"/path/to/beaver"` |
| `concurrency.max_workers` | int | Thread pool size for parallel DB evaluation | `8` |
| `output.dir` | str | Root output directory | `"./results"` |
| `output.cache_dir` | str | Cache for Phase 2/3 intermediate results | `"./cache"` |
| `output.save_intermediate` | bool | Persist per-phase outputs to disk | `true` |

See `config/example_local_vllm.yaml` for a complete annotated example.

---

## Touching-Labels File Format

The *touching-subset* analysis partitions benchmark queries into those whose
gold SQL references at least one refined column ("touching") and those that
do not ("non-touching"). The delta on touching queries is typically 4–10×
larger than on the full set, providing strong per-column evidence of the
refinement's effect.

To reproduce this analysis, provide a labels file in the following JSON format:

```json
{
  "financial": {
    "touching": [
      "What is the average balance of accounts with monthly frequency?",
      "List all accounts ordered by transaction frequency"
    ],
    "non_touching": [
      "How many accounts are there in total?",
      "What is the maximum account ID?"
    ]
  },
  "formula_1": {
    "touching": ["..."],
    "non_touching": ["..."]
  }
}
```

The keys are `db_id` values matching the benchmark. Each entry lists the
natural-language question strings that do or do not reference a refined column.
(These labels must be generated or annotated externally; see
`scripts/analysis/touching_subset.py --help` for the expected format.)

Usage:

```bash
python scripts/analysis/touching_subset.py \
    --eval-dir results/eval/bird/egrefine/ \
    --noref-dir results/eval/bird/no_refinement/ \
    --labels-file /path/to/touching_labels.json \
    --output results/analyses/touching_subset.md
```

The script emits a markdown report with per-DB and aggregate ExAcc broken down
by touching / non-touching subsets, plus the ratio Δ_touching / Δ_full.
