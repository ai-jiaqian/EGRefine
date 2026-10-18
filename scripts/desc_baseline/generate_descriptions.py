#!/usr/bin/env python3
"""Generate one-sentence column descriptions for the description-baseline experiment.

For every Phase-1-screened column in the canonical EGRefine refinement output
(by default `results/refine/drspider_abbr_multi/`), call Qwen3.5-27B once
to produce a SQL-comment-style English description.

The output is later injected as inline `-- comment` annotations into the
schema string passed to the C3 algorithm (see scripts/desc_baseline/inject.py
and the patched src/phase3/c3_runner.py).

Usage:
    PYTHONPATH=. python3 scripts/desc_baseline/generate_descriptions.py \
        --refine-dir results/refine/drspider_abbr_multi \
        --benchmark drspider_abbr \
        --config config/default.yaml \
        --output results/desc_baseline/column_descriptions.json \
        --sample-size 30
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from egrefine.config import load_config
from egrefine.data.benchmark import BIRDLoader, DrSpiderLoader
from egrefine.models.llm_client import LLMClient
from egrefine.phase2.sampler import sample_column

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """\
You are a database expert writing a SQL comment for a column in a database table.

Database: {db_id}
Table: {table_name}
All columns in this table: {table_columns_list}

The column you need to describe: {target_column_name} (type: {column_type})
Sample values from this column (up to 20): {sample_values}

Task: Write a single concise but complete English SQL comment describing what this column stores. The comment will be added to the schema documentation to help downstream consumers (including LLMs) understand the column's semantics.

Requirements:
1. The comment must be a complete, natural English clause.
2. Length must be between 8 and 25 words.
3. Where helpful, include example values drawn from the sample data.
4. Do not repeat the column name itself — assume the reader cannot see the name and only your description should make the column understandable.
5. Output the comment directly. No surrounding quotes, no "Description:" prefix, no markdown.

Output (one line of comment):"""


def _load_loader(config: dict, benchmark: str):
    if benchmark in ("drspider_abbr", "drspider_syn"):
        return DrSpiderLoader(config["data"][benchmark]["path"])
    if benchmark == "bird":
        return BIRDLoader(config["data"]["bird"]["path"])
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _collect_candidates(refine_dir: Path) -> List[Dict]:
    """Read every per-DB pipeline_result.json under *refine_dir* and return
    the Phase-1 candidate column list.

    Each candidate is the (db_id, table, original_column_name) tuple, plus
    enough schema info to build the description prompt.
    """
    out = []
    for db_dir in sorted(refine_dir.iterdir()):
        if not db_dir.is_dir():
            continue
        pipeline_path = db_dir / "pipeline_result.json"
        refined_tables_path = db_dir / "refined_tables.json"
        if not pipeline_path.exists() or not refined_tables_path.exists():
            continue

        with open(pipeline_path, "r", encoding="utf-8") as f:
            pipeline = json.load(f)
        with open(refined_tables_path, "r", encoding="utf-8") as f:
            refined = json.load(f)

        # Build dtype/neighbors lookup keyed by original column name.
        by_table: Dict[str, Dict[str, str]] = {}
        neighbors_by_table: Dict[str, List[str]] = {}
        for t in refined.get("tables", []):
            tname = t["name"]
            cols = t.get("columns", [])
            by_table[tname] = {
                c.get("original_name", c["name"]): c.get("dtype", "")
                for c in cols
            }
            # Neighbors = all original column names in the same table.
            neighbors_by_table[tname] = [
                c.get("original_name", c["name"]) for c in cols
            ]

        for ref in pipeline.get("refinements", []):
            tname = ref["table"]
            orig = ref.get("original_name") or ref["column"]
            dtype = by_table.get(tname, {}).get(orig, "")
            neighbors = [
                n for n in neighbors_by_table.get(tname, []) if n != orig
            ]
            out.append({
                "db_id": pipeline["db_id"],
                "table": tname,
                "column": orig,
                "dtype": dtype,
                "neighbors": neighbors,
            })

    return out


def _fetch_sample_values(
    candidate: Dict, db_path: str, n: int = 20,
) -> List[str]:
    """Pull up to *n* non-NULL sample values for the column from SQLite."""
    try:
        return sample_column(db_path, candidate["table"], candidate["column"], n=n)
    except Exception as e:
        logger.warning(
            "sample_column failed for %s.%s.%s: %s",
            candidate["db_id"], candidate["table"], candidate["column"], e,
        )
        return []


def _build_prompt(candidate: Dict, samples: List[str]) -> str:
    cols_str = ", ".join(candidate["neighbors"] + [candidate["column"]])
    sample_str = (
        ", ".join(repr(v) for v in samples[:20]) if samples else "(no sample values available)"
    )
    return PROMPT_TEMPLATE.format(
        db_id=candidate["db_id"],
        table_name=candidate["table"],
        table_columns_list=cols_str,
        target_column_name=candidate["column"],
        column_type=candidate["dtype"] or "UNKNOWN",
        sample_values=sample_str,
    )


def _generate_one(client: LLMClient, candidate: Dict, db_path: str) -> Dict:
    samples = _fetch_sample_values(candidate, db_path)
    prompt = _build_prompt(candidate, samples)
    description = client.chat([{"role": "user", "content": prompt}]).strip()
    # Tighten: strip surrounding quotes / common prefixes the model may slip in
    if description.startswith(("'", '"', '`')) and description.endswith(("'", '"', '`')):
        description = description[1:-1].strip()
    for prefix in ("Description:", "Comment:", "Note:"):
        if description.lower().startswith(prefix.lower()):
            description = description[len(prefix):].strip()
    # Collapse to a single line
    description = " ".join(description.split())
    return {
        "db_id": candidate["db_id"],
        "table": candidate["table"],
        "column": candidate["column"],
        "dtype": candidate["dtype"],
        "description": description,
        "sample_values_used": samples[:5],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate column descriptions for the description baseline.",
    )
    parser.add_argument("--refine-dir", required=True,
                        help="Phase 1 refinement output directory (per-DB pipeline_result.json)")
    parser.add_argument("--benchmark", default="drspider_abbr",
                        help="Benchmark name (drives loader for db_path lookup)")
    parser.add_argument("--config", default="config/default.yaml",
                        help="LLM config (candidate LLM endpoint)")
    parser.add_argument("--output", required=True,
                        help="Output JSON path for column_descriptions.json")
    parser.add_argument("--sample-md", default=None,
                        help="If set, write a markdown file with N random descriptions for human review")
    parser.add_argument("--sample-size", type=int, default=30,
                        help="Number of samples in the human-review markdown (default 30)")
    parser.add_argument("--concurrency", type=int, default=48,
                        help="Parallel LLM calls (default 48)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="LLM temperature for description gen (default 0.3 per spec)")
    parser.add_argument("--max-tokens", type=int, default=80,
                        help="LLM max_tokens for description gen (default 80 per spec)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for sampling the markdown subset")
    args = parser.parse_args()

    config = load_config(args.config)

    # Override LLM config to enforce description-gen sampling params.
    candidate_cfg = dict(config["models"]["candidate_llm"])
    candidate_cfg["temperature"] = args.temperature
    candidate_cfg["max_tokens"] = args.max_tokens
    client = LLMClient(candidate_cfg)

    loader = _load_loader(config, args.benchmark)

    refine_dir = Path(args.refine_dir)
    candidates = _collect_candidates(refine_dir)
    logger.info("Collected %d Phase-1 candidate columns from %s",
                len(candidates), refine_dir)

    if not candidates:
        logger.error("No candidates found — aborting.")
        sys.exit(1)

    # Resolve DB paths up front; skip candidates whose DB cannot be located.
    db_path_cache: Dict[str, str] = {}
    valid_candidates = []
    for cand in candidates:
        db_id = cand["db_id"]
        if db_id not in db_path_cache:
            try:
                db_path_cache[db_id] = loader.get_db_path(db_id)
            except Exception as e:
                logger.warning("Cannot locate DB for %s: %s — skipping its columns", db_id, e)
                db_path_cache[db_id] = ""
        if db_path_cache[db_id]:
            valid_candidates.append(cand)

    logger.info("Generating descriptions for %d columns (concurrency=%d, temp=%.1f)",
                len(valid_candidates), args.concurrency, args.temperature)

    t_start = time.time()
    results: List[Dict] = []
    failures: List[Dict] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_generate_one, client, cand, db_path_cache[cand["db_id"]]): cand
            for cand in valid_candidates
        }
        done = 0
        for fut in as_completed(futures):
            cand = futures[fut]
            try:
                rec = fut.result()
                if rec["description"]:
                    results.append(rec)
                else:
                    failures.append({**cand, "error": "empty_response"})
            except Exception as e:
                logger.warning("Description gen failed for %s.%s.%s: %s",
                               cand["db_id"], cand["table"], cand["column"], e)
                failures.append({**cand, "error": str(e)})
            done += 1
            if done % 100 == 0 or done == len(futures):
                logger.info("  progress: %d/%d (%.1f%%)",
                            done, len(futures), 100.0 * done / len(futures))

    t_elapsed = time.time() - t_start
    logger.info("Generation done in %.1fs (%d ok, %d failed)",
                t_elapsed, len(results), len(failures))

    # Write output JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "refine_dir": str(refine_dir),
                "benchmark": args.benchmark,
                "config": args.config,
                "model": candidate_cfg["model_name"],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "n_candidates": len(valid_candidates),
                "n_descriptions": len(results),
                "n_failures": len(failures),
                "wall_clock_seconds": round(t_elapsed, 1),
            },
            "descriptions": results,
            "failures": failures,
        }, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", out_path)

    # Optional human-review markdown
    if args.sample_md:
        import random
        rng = random.Random(args.seed)
        sample_n = min(args.sample_size, len(results))
        sample = rng.sample(results, sample_n)
        md_path = Path(args.sample_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Description Quality Audit — random sample of {sample_n}\n\n")
            f.write(f"Source: `{out_path}`\n\n")
            f.write(f"Model: `{candidate_cfg['model_name']}` "
                    f"@ `{candidate_cfg['base_url']}`, "
                    f"T={args.temperature}, max_tokens={args.max_tokens}\n\n")
            f.write("---\n\n")
            for i, rec in enumerate(sample, 1):
                f.write(f"## {i}. `{rec['db_id']}.{rec['table']}.{rec['column']}` "
                        f"({rec['dtype']})\n\n")
                if rec["sample_values_used"]:
                    sv = ", ".join(repr(v) for v in rec["sample_values_used"])
                    f.write(f"**Sample values:** {sv}\n\n")
                f.write(f"**Description:** {rec['description']}\n\n")
                f.write("---\n\n")
        logger.info("Wrote audit markdown to %s", md_path)


if __name__ == "__main__":
    main()
