"""Phase 2: Prompt templates for candidate column name generation."""
import csv
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from egrefine.data.schema import Column, Schema

logger = logging.getLogger(__name__)


@dataclass
class CandidateName:
    """A proposed alternative column name with justification."""
    name: str
    reason: str


def load_column_descriptions(
    db_id: str,
    bird_path: str,
) -> Dict[str, Dict[str, str]]:
    """Load column descriptions from BIRD database_description/ CSVs.

    Returns:
        Dict mapping "table.column" -> {"description": str, "value_description": str}
    """
    db_desc_dir = os.path.join(
        bird_path, "dev_databases", db_id, "database_description"
    )
    if not os.path.isdir(db_desc_dir):
        return {}

    result: Dict[str, Dict[str, str]] = {}
    try:
        for fname in os.listdir(db_desc_dir):
            if not fname.endswith(".csv"):
                continue
            table_name = fname[:-4]
            csv_path = os.path.join(db_desc_dir, fname)
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    col_name = row.get("original_column_name", "").strip()
                    desc = row.get("column_description", "").strip()
                    val_desc = row.get("value_description", "").strip()
                    if col_name:
                        key = f"{table_name}.{col_name}"
                        result[key] = {
                            "description": desc,
                            "value_description": val_desc,
                        }
    except Exception as e:
        logger.warning("Failed to load database descriptions for %s: %s", db_id, e)

    return result


def build_candidate_prompt(
    column: Column,
    schema: Schema,
    sample_values: List[str],
    k: int = 3,
    column_descriptions: Optional[Dict[str, Dict[str, str]]] = None,
    neighbor_samples: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Build the LLM prompt for generating candidate column names.

    Args:
        column: The target column to rename.
        schema: The full database schema (for context).
        sample_values: Sample data values from this column.
        k: Number of candidates to generate.
        column_descriptions: Dict from load_column_descriptions().
        neighbor_samples: Dict mapping neighbor col name -> sample values.

    Returns:
        The formatted prompt string.
    """
    descs = column_descriptions or {}

    # Same-table columns (excluding the target column itself)
    table = schema.get_table(column.table)
    if table:
        neighbors = [c for c in table.columns if c.name != column.name]
    else:
        neighbors = []

    # FK-related columns
    fk_related = []
    col_full = column.full_name
    for c1, c2 in schema.foreign_keys:
        if c1 == col_full:
            fk_related.append(c2)
        elif c2 == col_full:
            fk_related.append(c1)

    # Format neighbor lines with optional descriptions and sample values
    neighbor_lines_parts = []
    for c in neighbors[:10]:  # cap at 10 neighbors to control token usage
        line = f"- {c.name} ({c.dtype})"
        desc_info = descs.get(f"{c.table}.{c.name}", {})
        if desc_info.get("description"):
            desc_text = desc_info['description'][:80]
            line += f": {desc_text}"
        if neighbor_samples and c.name in neighbor_samples:
            vals = neighbor_samples[c.name][:5]
            if vals:
                line += f" [samples: {', '.join(str(v) for v in vals)}]"
        neighbor_lines_parts.append(line)
    neighbor_lines = "\n".join(neighbor_lines_parts) or "- (none)"

    fk_lines = "\n".join(
        f"- {ref}" for ref in fk_related
    ) or "- None"

    sample_lines = "\n".join(
        f"- {v}" for v in sample_values
    ) or "- (no data available)"

    # Target column description
    target_desc = descs.get(f"{column.table}.{column.name}", {})
    target_desc_line = ""
    if target_desc.get("description"):
        target_desc_line = f"\n- Column description: {target_desc['description'][:120]}"
    if target_desc.get("value_description"):
        target_desc_line += f"\n- Value semantics: {target_desc['value_description'][:120]}"

    prompt = f"""You are a database schema expert helping improve column names for better Text-to-SQL performance.

## Database Context
- Database: `{schema.db_id}`

## Target Column
- Current name: `{column.table}`.`{column.name}`
- Data type: {column.dtype}
- Is primary key: {column.is_pk}
- Foreign key target: {column.fk_target or "None"}{target_desc_line}

## Neighboring Columns (same table)
{neighbor_lines}

## Foreign Key Related Columns
{fk_lines}

## Sample Data Values of Target Column
{sample_lines}

## Task
Propose exactly {k} alternative column names. Follow these rules:

1. DOMAIN AWARENESS: Interpret data values in light of the database domain. Use column descriptions and value semantics above if available.

2. CONSERVATIVE RENAMING: If the current name is already clear and unambiguous in context, include the original name as one of your {k} candidates. Not every column needs renaming.

3. NO OVER-SPECIFICATION: Do NOT add unnecessary prefixes or suffixes to already-descriptive names. Bad: "element" → "element_symbol". Good: "A2" → "city_name".

4. SNAKE_CASE: Use snake_case convention, consistent with neighboring columns.

5. TEXT-TO-SQL FRIENDLY: The name should help a language model correctly understand what this column stores when generating SQL queries.

Respond in JSON format only, no other text:
[{{"name": "proposed_name", "reason": "brief justification"}}, ...]"""

    return prompt


# Valid SQL identifier: letters, digits, underscores only (no dots, backticks, spaces, etc.)
_VALID_COLUMN_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _is_valid_column_name(name: str) -> bool:
    """Check if a candidate name is a valid SQL column identifier."""
    return bool(_VALID_COLUMN_RE.match(name))


def parse_candidates(raw_json: list) -> List[CandidateName]:
    """Parse the LLM JSON response into CandidateName objects.

    Filters out candidates with invalid column names (containing dots,
    backticks, spaces, or other special characters).

    Args:
        raw_json: Parsed JSON list from LLM response.

    Returns:
        List of CandidateName objects.

    Raises:
        ValueError: If the JSON structure is invalid.
    """
    if not isinstance(raw_json, list):
        raise ValueError(f"Expected a JSON list, got {type(raw_json).__name__}")

    candidates = []
    for item in raw_json:
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict in list, got {type(item).__name__}")
        name = item.get("name")
        reason = item.get("reason", "")
        if not name or not isinstance(name, str):
            raise ValueError(f"Missing or invalid 'name' field: {item}")
        name = name.strip()
        if not _is_valid_column_name(name):
            logger.warning("Filtered invalid candidate name: %r", name)
            continue
        candidates.append(CandidateName(name=name, reason=str(reason).strip()))

    return candidates
