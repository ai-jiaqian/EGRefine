"""Phase 2: Candidate Generator — LLM-based column name candidate generation with caching."""
import json
import logging
import os
import re
import threading
from typing import Dict, List, Optional

from egrefine.data.schema import Column, Schema
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.prompts import CandidateName, build_candidate_prompt, load_column_descriptions, parse_candidates
from egrefine.phase2.sampler import sample_column, sample_table_columns

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """Generate candidate column names using an LLM.

    Integrates prompt building, data sampling, LLM calling, JSON parsing,
    and file-based caching. Thread-safe for concurrent generation.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        cache_dir: Optional[str] = None,
        sample_rows: int = 20,
        max_retries: int = 3,
        desc_base_path: Optional[str] = None,
    ):
        self.llm = llm_client
        self.sample_rows = sample_rows
        self.max_retries = max_retries
        self.desc_base_path = desc_base_path
        self._lock = threading.Lock()

        # Per-DB description cache (loaded lazily)
        self._desc_cache: Dict[str, Dict[str, Dict[str, str]]] = {}

        # File-based cache
        self._cache: Dict[str, List[dict]] = {}
        self._cache_path: Optional[str] = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._cache_path = os.path.join(cache_dir, "phase2_candidates.json")
            if os.path.exists(self._cache_path):
                with open(self._cache_path) as f:
                    self._cache = json.load(f)
                logger.info("Loaded %d cached entries from %s", len(self._cache), self._cache_path)

    def _get_descriptions(self, db_id: str) -> Dict[str, Dict[str, str]]:
        """Get column descriptions for a database (cached, thread-safe)."""
        with self._lock:
            if db_id not in self._desc_cache:
                if self.desc_base_path:
                    self._desc_cache[db_id] = load_column_descriptions(db_id, self.desc_base_path)
                else:
                    self._desc_cache[db_id] = {}
            return self._desc_cache[db_id]

    def _cache_key(self, db_id: str, table: str, column: str) -> str:
        return f"{db_id}:{table}:{column}"

    def _save_cache(self):
        if self._cache_path:
            with self._lock:
                with open(self._cache_path, "w") as f:
                    json.dump(self._cache, f, indent=2)

    def generate(
        self,
        column: Column,
        schema: Schema,
        db_path: str,
        k: int = 3,
    ) -> List[CandidateName]:
        """Generate k candidate names for a column.

        Args:
            column: Target column to rename.
            schema: Full database schema for context.
            db_path: Path to SQLite database for sampling.
            k: Number of candidates to generate.

        Returns:
            List of CandidateName objects.
        """
        # Check cache (thread-safe)
        key = self._cache_key(schema.db_id, column.table, column.name)
        with self._lock:
            if key in self._cache:
                logger.info("[cached] %s -> %d candidates", key, len(self._cache[key]))
                return parse_candidates(self._cache[key])

        # Sample data for target column
        sample_values = sample_column(db_path, column.table, column.name, n=self.sample_rows)

        # Sample neighbor columns (up to 5 values each, for context)
        table_obj = schema.get_table(column.table)
        neighbor_samples: Dict[str, List[str]] = {}
        if table_obj:
            neighbor_names = [c.name for c in table_obj.columns if c.name != column.name][:10]
            if neighbor_names:
                neighbor_samples = sample_table_columns(
                    db_path, column.table, neighbor_names, n=5,
                )

        # Load column descriptions
        col_descs = self._get_descriptions(schema.db_id)

        # Build prompt
        prompt = build_candidate_prompt(
            column, schema, sample_values, k=k,
            column_descriptions=col_descs,
            neighbor_samples=neighbor_samples,
        )

        # Call LLM with retry on JSON parse failure
        candidates = None
        last_error = None
        for attempt in range(self.max_retries):
            try:
                raw_text = self.llm.chat([{"role": "user", "content": prompt}])
                raw_json = _extract_json(raw_text)
                candidates = parse_candidates(raw_json)
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "Candidate generation failed for %s (attempt %d/%d): %s",
                    key, attempt + 1, self.max_retries, e,
                )

        if candidates is None:
            logger.error("All retries failed for %s: %s", key, last_error)
            return []

        # Save to cache (thread-safe)
        with self._lock:
            self._cache[key] = [{"name": c.name, "reason": c.reason} for c in candidates]
        self._save_cache()

        logger.info("Generated %d candidates for %s", len(candidates), key)
        return candidates


def _extract_json(text: str) -> list:
    """Extract a JSON array from LLM response text.

    Handles common cases:
    - Pure JSON response
    - JSON inside markdown code blocks
    - JSON with thinking/reasoning prefix
    """
    # Strip thinking tags (some models wrap in <think>...</think>)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try finding first [ ... ] block
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON array from LLM response: {text[:200]}")
