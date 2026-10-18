#!/usr/bin/env python3
"""Compute paired bootstrap CIs and McNemar tests for core BIRD cells.

This script only reads existing per-query evaluation JSON files. It does not
rerun any Text-to-SQL model or SQL execution.
"""

from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs" / "bootstrap_ci_results.md"
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42
EXACC_TOLERANCE_PP = 0.05

METHOD_ALIASES = {
    "c3": ("c3", "c3sql"),
    "din": ("din", "dinsql"),
    "dinsql": ("dinsql", "din"),
    "mac": ("mac", "macsql"),
    "macsql": ("macsql", "mac"),
}

FALLBACK_KEYS = (
    "fallback_from_noref",
    "fallback",
    "reused_from_original",
    "reuse_original",
    "copied_from_noref",
)


@dataclass(frozen=True)
class CellSpec:
    name: str
    noref_path: Path
    egrefine_path: Path
    method: str
    expected_n: int
    expected_noref: float
    expected_egrefine: float


@dataclass(frozen=True)
class QueryRecord:
    key: str
    exec_match: int
    fallback_from_noref: bool


@dataclass(frozen=True)
class CellResult:
    spec: CellSpec
    n: int
    noref_pass: int
    egrefine_pass: int
    noref_exacc: float
    egrefine_exacc: float
    delta: float
    ci_low: float
    ci_high: float
    b: int
    c: int
    p_value: float
    fallback_count: int


def paired_bootstrap_ci(
    noref: np.ndarray,
    egrefine: np.ndarray,
    B: int = BOOTSTRAP_ITERATIONS,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return observed delta and percentile bootstrap CI in percentage points."""
    n = len(noref)
    delta_obs = (egrefine.mean() - noref.mean()) * 100
    deltas = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        d = (egrefine[idx].mean() - noref[idx].mean()) * 100
        deltas.append(d)
    deltas = np.array(deltas)
    ci_low = np.percentile(deltas, 100 * alpha / 2)
    ci_high = np.percentile(deltas, 100 * (1 - alpha / 2))
    return float(delta_obs), float(ci_low), float(ci_high)


def mcnemar_exact(noref: np.ndarray, egrefine: np.ndarray) -> Tuple[int, int, float]:
    """Return regressions, improvements, and two-sided exact McNemar p-value."""
    b = int(((noref == 1) & (egrefine == 0)).sum())
    c = int(((noref == 0) & (egrefine == 1)).sum())
    if b + c == 0:
        return b, c, 1.0
    p_value = binomtest(b, n=b + c, p=0.5, alternative="two-sided").pvalue
    return b, c, float(p_value)


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_exec_match(row: Mapping[str, Any]) -> int:
    for key in ("exec_match", "match", "execution_match", "correct"):
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value != 0)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y"}:
                return 1
            if normalized in {"0", "false", "no", "n"}:
                return 0
        raise ValueError(f"Cannot normalize {key}={value!r}")
    raise KeyError(f"No exec_match/match field found in row keys: {sorted(row.keys())}")


def is_fallback(row: Mapping[str, Any]) -> bool:
    return any(bool(row.get(key)) for key in FALLBACK_KEYS)


def select_detail_rows(payload: Any, prefer_refined: bool) -> List[Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON list or object, got {type(payload).__name__}")

    preferred_keys = (
        ("refined_details", "details", "queries", "results", "predictions")
        if prefer_refined
        else ("original_details", "details", "queries", "results", "predictions")
    )
    fallback_keys = (
        ("original_details", "refined_details")
        if prefer_refined
        else ("refined_details", "original_details")
    )

    for key in preferred_keys + fallback_keys:
        value = payload.get(key)
        if isinstance(value, list) and value:
            return value
    for key in preferred_keys + fallback_keys:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    raise KeyError(f"No query-detail list found in payload keys: {sorted(payload.keys())}")


def raw_query_key(row: Mapping[str, Any], db_id: str, idx: int) -> str:
    explicit_id = row.get("question_id", row.get("query_id", row.get("id")))
    if explicit_id is not None:
        return f"{db_id}::id::{explicit_id}"

    question = row.get("nl", row.get("question", row.get("utterance")))
    if question:
        return f"{db_id}::question::{compact_text(question)}"

    gold_sql = row.get("gold_sql", row.get("SQL"))
    if gold_sql:
        return f"{db_id}::gold::{compact_text(gold_sql)}"

    return f"{db_id}::idx::{idx}"


def uniquify_records(rows: Iterable[Mapping[str, Any]], db_id: str) -> "OrderedDict[str, QueryRecord]":
    records: "OrderedDict[str, QueryRecord]" = OrderedDict()
    seen: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        key = raw_query_key(row, db_id, idx)
        seen[key] += 1
        if seen[key] > 1:
            key = f"{key}::occ::{seen[key]}"
        records[key] = QueryRecord(
            key=key,
            exec_match=normalize_exec_match(row),
            fallback_from_noref=is_fallback(row),
        )
    return records


def find_method_file(db_dir: Path, method: str) -> Path:
    aliases = METHOD_ALIASES.get(method, (method,))
    for alias in aliases:
        path = db_dir / f"{alias}.json"
        if path.exists():
            return path
    expected = ", ".join(f"{alias}.json" for alias in aliases)
    raise FileNotFoundError(f"No method JSON in {db_dir}; expected one of: {expected}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_per_db_records(per_db_dir: Path, method: str, prefer_refined: bool) -> "OrderedDict[str, QueryRecord]":
    if not per_db_dir.is_dir():
        raise NotADirectoryError(f"per_db path does not exist or is not a directory: {per_db_dir}")

    all_records: "OrderedDict[str, QueryRecord]" = OrderedDict()
    db_dirs = sorted(child for child in per_db_dir.iterdir() if child.is_dir())
    if not db_dirs:
        raise ValueError(f"No database subdirectories found under {per_db_dir}")

    for db_dir in db_dirs:
        path = find_method_file(db_dir, method)
        payload = load_json(path)
        db_id = payload.get("db_id", db_dir.name) if isinstance(payload, dict) else db_dir.name
        rows = select_detail_rows(payload, prefer_refined=prefer_refined)
        records = uniquify_records(rows, str(db_id))
        for key, record in records.items():
            if key in all_records:
                raise ValueError(f"Duplicate query key across DBs: {key}")
            all_records[key] = record
    return all_records


def load_per_db_paired_records(
    noref_dir: Path,
    egrefine_dir: Path,
    method: str,
) -> Tuple["OrderedDict[str, QueryRecord]", "OrderedDict[str, QueryRecord]"]:
    """Load paired per-db records, using NoRef as EGRefine fallback for missing files."""
    if not noref_dir.is_dir():
        raise NotADirectoryError(f"NoRef per_db path is not a directory: {noref_dir}")
    if not egrefine_dir.is_dir():
        raise NotADirectoryError(f"EGRefine per_db path is not a directory: {egrefine_dir}")

    noref_all: "OrderedDict[str, QueryRecord]" = OrderedDict()
    egrefine_all: "OrderedDict[str, QueryRecord]" = OrderedDict()
    db_dirs = sorted(child for child in noref_dir.iterdir() if child.is_dir())
    if not db_dirs:
        raise ValueError(f"No database subdirectories found under {noref_dir}")

    for noref_db_dir in db_dirs:
        db_name = noref_db_dir.name
        noref_file = find_method_file(noref_db_dir, method)
        noref_payload = load_json(noref_file)
        db_id = noref_payload.get("db_id", db_name) if isinstance(noref_payload, dict) else db_name
        noref_rows = select_detail_rows(noref_payload, prefer_refined=False)
        noref_records = uniquify_records(noref_rows, str(db_id))

        egrefine_db_dir = egrefine_dir / db_name
        try:
            egrefine_file = find_method_file(egrefine_db_dir, method)
        except FileNotFoundError:
            egrefine_records = OrderedDict(
                (
                    key,
                    QueryRecord(
                        key=key,
                        exec_match=record.exec_match,
                        fallback_from_noref=True,
                    ),
                )
                for key, record in noref_records.items()
            )
        else:
            egrefine_payload = load_json(egrefine_file)
            egrefine_db_id = (
                egrefine_payload.get("db_id", db_name)
                if isinstance(egrefine_payload, dict)
                else db_name
            )
            egrefine_rows = select_detail_rows(egrefine_payload, prefer_refined=True)
            egrefine_records = uniquify_records(egrefine_rows, str(egrefine_db_id))

        for key, record in noref_records.items():
            if key in noref_all:
                raise ValueError(f"Duplicate query key across NoRef DBs: {key}")
            noref_all[key] = record
        for key, record in egrefine_records.items():
            if key in egrefine_all:
                raise ValueError(f"Duplicate query key across EGRefine DBs: {key}")
            egrefine_all[key] = record

    return noref_all, egrefine_all


def load_flat_records(path: Path) -> "OrderedDict[str, QueryRecord]":
    payload = load_json(path)
    rows = select_detail_rows(payload, prefer_refined=False)
    grouped: Dict[str, List[Mapping[str, Any]]] = OrderedDict()
    for row in rows:
        db_id = str(row.get("db_id", "holdout"))
        grouped.setdefault(db_id, []).append(row)

    all_records: "OrderedDict[str, QueryRecord]" = OrderedDict()
    for db_id, db_rows in grouped.items():
        records = uniquify_records(db_rows, db_id)
        for key, record in records.items():
            if key in all_records:
                raise ValueError(f"Duplicate query key in flat file {path}: {key}")
            all_records[key] = record
    return all_records


def load_records(path: Path, method: str, prefer_refined: bool) -> "OrderedDict[str, QueryRecord]":
    if path.is_file():
        return load_flat_records(path)
    return load_per_db_records(path, method, prefer_refined=prefer_refined)


def diagnose_alignment(
    noref_path: Path,
    egrefine_path: Path,
    noref_records: Mapping[str, QueryRecord],
    egrefine_records: Mapping[str, QueryRecord],
) -> None:
    noref_keys = set(noref_records)
    egrefine_keys = set(egrefine_records)
    missing_in_egrefine = sorted(noref_keys - egrefine_keys)
    missing_in_noref = sorted(egrefine_keys - noref_keys)
    print("Question alignment failed.")
    print(f"  NoRef path:    {noref_path}")
    print(f"  EGRefine path: {egrefine_path}")
    print(f"  NoRef queries:    {len(noref_records)}")
    print(f"  EGRefine queries: {len(egrefine_records)}")
    print(f"  Missing in EGRefine ({len(missing_in_egrefine)}):")
    for key in missing_in_egrefine[:20]:
        print(f"    - {key}")
    print(f"  Missing in NoRef ({len(missing_in_noref)}):")
    for key in missing_in_noref[:20]:
        print(f"    - {key}")


def load_paired_outcomes(
    noref_path: Path | str,
    egrefine_path: Path | str,
    method: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    noref_path = Path(noref_path)
    egrefine_path = Path(egrefine_path)
    if noref_path.is_dir() and egrefine_path.is_dir():
        noref_records, egrefine_records = load_per_db_paired_records(
            noref_path,
            egrefine_path,
            method,
        )
    else:
        noref_records = load_records(noref_path, method, prefer_refined=False)
        egrefine_records = load_records(egrefine_path, method, prefer_refined=True)

    if set(noref_records) != set(egrefine_records):
        diagnose_alignment(noref_path, egrefine_path, noref_records, egrefine_records)
        raise ValueError("question_id alignment mismatch")

    noref_values = []
    egrefine_values = []
    fallback_count = 0
    fallback_mismatches = []
    for key in noref_records:
        noref_record = noref_records[key]
        egrefine_record = egrefine_records[key]
        noref_values.append(noref_record.exec_match)
        egrefine_values.append(egrefine_record.exec_match)
        if egrefine_record.fallback_from_noref:
            fallback_count += 1
            if noref_record.exec_match != egrefine_record.exec_match:
                fallback_mismatches.append(key)

    if fallback_mismatches:
        print("fallback_from_noref consistency failed.")
        print(f"  NoRef path:    {noref_path}")
        print(f"  EGRefine path: {egrefine_path}")
        print(f"  Mismatched fallback queries ({len(fallback_mismatches)}):")
        for key in fallback_mismatches[:20]:
            print(f"    - {key}")
        raise ValueError("fallback_from_noref rows differ from NoRef rows")

    return (
        np.array(noref_values, dtype=int),
        np.array(egrefine_values, dtype=int),
        fallback_count,
    )


def validate_expected_exacc(
    cell_name: str,
    noref_path: Path | str,
    egrefine_path: Path | str,
    noref: np.ndarray,
    egrefine: np.ndarray,
    expected_n: int,
    expected_noref: float,
    expected_egrefine: float,
) -> None:
    actual_n = len(noref)
    if actual_n != len(egrefine):
        print(f"n mismatch for {cell_name}:")
        print(f"  NoRef path:    {noref_path}")
        print(f"  EGRefine path: {egrefine_path}")
        print(f"  NoRef n:    {len(noref)}")
        print(f"  EGRefine n: {len(egrefine)}")
        raise ValueError("paired n mismatch")

    if actual_n != expected_n:
        print(f"unexpected n for {cell_name}:")
        print(f"  NoRef path:    {noref_path}")
        print(f"  EGRefine path: {egrefine_path}")
        print(f"  computed n: {actual_n}")
        print(f"  expected n: {expected_n}")
        raise ValueError("expected n mismatch")

    actual_noref = noref.mean() * 100
    actual_egrefine = egrefine.mean() * 100
    diff_noref = abs(actual_noref - expected_noref)
    diff_egrefine = abs(actual_egrefine - expected_egrefine)
    if diff_noref >= EXACC_TOLERANCE_PP or diff_egrefine >= EXACC_TOLERANCE_PP:
        print(f"ExAcc validation failed for {cell_name}:")
        print(f"  NoRef path:    {noref_path}")
        print(f"  EGRefine path: {egrefine_path}")
        print(f"  query count: {actual_n}")
        print(f"  computed NoRef ExAcc:    {actual_noref:.4f}")
        print(f"  registry NoRef ExAcc:    {expected_noref:.4f}")
        print(f"  NoRef difference:        {diff_noref:.4f} pp")
        print(f"  computed EGRefine ExAcc: {actual_egrefine:.4f}")
        print(f"  registry EGRefine ExAcc: {expected_egrefine:.4f}")
        print(f"  EGRefine difference:     {diff_egrefine:.4f} pp")
        raise ValueError("ExAcc mismatch")


def resolve_holdout_paths() -> Tuple[Path, Path]:
    candidates = [
        (
            ROOT / "results/eval/bird_holdout/no_refinement/per_db",
            ROOT / "results/eval/bird_holdout/egrefine_multi/per_db",
        ),
        (
            ROOT / "results/holdout/bird_27b_c3/predictions_noref.json",
            ROOT / "results/holdout/bird_27b_c3/predictions_egrefine.json",
        ),
    ]
    for noref, egrefine in candidates:
        if noref.exists() and egrefine.exists():
            return noref, egrefine
    print("Could not locate holdout paths. Searched:")
    for noref, egrefine in candidates:
        print(f"  NoRef:    {noref}")
        print(f"  EGRefine: {egrefine}")
    raise FileNotFoundError("No holdout result paths found")


def build_cell_specs() -> List[CellSpec]:
    holdout_noref, holdout_egrefine = resolve_holdout_paths()
    return [
        CellSpec(
            name="BIRD 27B x C3",
            noref_path=ROOT / "results/eval/bird/no_refinement/per_db",
            egrefine_path=ROOT / "results/eval/bird/egrefine_multi/per_db",
            method="c3",
            expected_n=1534,
            expected_noref=41.53,
            expected_egrefine=41.72,
        ),
        CellSpec(
            name="BIRD 27B x DIN",
            noref_path=ROOT / "results/eval/bird/no_refinement/per_db",
            egrefine_path=ROOT / "results/eval/bird/egrefine_multi/per_db",
            method="din",
            expected_n=1534,
            expected_noref=38.59,
            expected_egrefine=39.24,
        ),
        CellSpec(
            name="BIRD 27B x MAC",
            noref_path=ROOT / "results/eval/bird/no_refinement/per_db",
            egrefine_path=ROOT / "results/eval/bird/egrefine_multi/per_db",
            method="mac",
            expected_n=1534,
            expected_noref=29.99,
            expected_egrefine=31.23,
        ),
        CellSpec(
            name="BIRD 9B x C3",
            noref_path=ROOT / "results/eval_9b/bird/no_refinement/per_db",
            egrefine_path=ROOT / "results/eval_9b/bird/egrefine/per_db",
            method="c3",
            expected_n=1534,
            expected_noref=28.75,
            expected_egrefine=30.25,
        ),
        CellSpec(
            name="BIRD 9B x DIN",
            noref_path=ROOT / "results/eval_9b/bird/no_refinement/per_db",
            egrefine_path=ROOT / "results/eval_9b/bird/egrefine/per_db",
            method="din",
            expected_n=1534,
            expected_noref=31.75,
            expected_egrefine=33.25,
        ),
        CellSpec(
            name="BIRD 9B x MAC",
            noref_path=ROOT / "results/eval_9b/bird/no_refinement/per_db",
            egrefine_path=ROOT / "results/eval_9b/bird/egrefine/per_db",
            method="mac",
            expected_n=1534,
            expected_noref=20.60,
            expected_egrefine=20.14,
        ),
        CellSpec(
            name="Holdout x C3",
            noref_path=holdout_noref,
            egrefine_path=holdout_egrefine,
            method="c3",
            expected_n=96,
            expected_noref=47.92,
            expected_egrefine=48.96,
        ),
    ]


def compute_cell(spec: CellSpec) -> CellResult:
    noref, egrefine, fallback_count = load_paired_outcomes(
        spec.noref_path,
        spec.egrefine_path,
        spec.method,
    )
    validate_expected_exacc(
        cell_name=spec.name,
        noref_path=spec.noref_path,
        egrefine_path=spec.egrefine_path,
        noref=noref,
        egrefine=egrefine,
        expected_n=spec.expected_n,
        expected_noref=spec.expected_noref,
        expected_egrefine=spec.expected_egrefine,
    )

    delta, ci_low, ci_high = paired_bootstrap_ci(noref, egrefine)
    b, c, p_value = mcnemar_exact(noref, egrefine)
    return CellResult(
        spec=spec,
        n=len(noref),
        noref_pass=int(noref.sum()),
        egrefine_pass=int(egrefine.sum()),
        noref_exacc=float(noref.mean() * 100),
        egrefine_exacc=float(egrefine.mean() * 100),
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        b=b,
        c=c,
        p_value=p_value,
        fallback_count=fallback_count,
    )


def fmt_signed(value: float) -> str:
    return f"{value:+.2f}"


def fmt_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.4f}"


def verdict(result: CellResult) -> str:
    ci_significant = result.ci_low > 0 or result.ci_high < 0
    mcnemar_significant = result.p_value < 0.05
    if ci_significant and mcnemar_significant:
        return "significant at alpha=0.05"
    return "not significant at alpha=0.05"


def render_markdown(results: Sequence[CellResult]) -> str:
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Bootstrap CI Results for EGRefine BIRD Cells",
        "",
        f"Computed: {timestamp}",
        f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}, seed={BOOTSTRAP_SEED}",
        "",
        "## Summary Table",
        "",
        "| Cell | n | NoRef ExAcc | EGRefine ExAcc | Delta (pp) | 95% CI | McNemar p | b (regressions) | c (improvements) |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]

    for r in results:
        lines.append(
            "| "
            f"{r.spec.name} | {r.n} | {r.noref_exacc:.2f} | {r.egrefine_exacc:.2f} | "
            f"{fmt_signed(r.delta)} | [{r.ci_low:.2f}, {r.ci_high:.2f}] | "
            f"{fmt_p(r.p_value)} | {r.b} | {r.c} |"
        )

    lines.extend(["", "## Per-Cell Details", ""])
    for r in results:
        lines.extend(
            [
                f"### {r.spec.name}",
                f"- n = {r.n}",
                f"- NoRef: {r.noref_pass} pass / {r.n} = {r.noref_exacc:.2f}%",
                f"- EGRefine: {r.egrefine_pass} pass / {r.n} = {r.egrefine_exacc:.2f}%",
                f"- Observed Delta: {fmt_signed(r.delta)} pp",
                f"- 95% CI: [{r.ci_low:.2f}, {r.ci_high:.2f}] pp",
                f"- McNemar: b={r.b}, c={r.c}, p={fmt_p(r.p_value)}",
                f"- fallback_from_noref count: {r.fallback_count}",
                f"- **Verdict**: {verdict(r)}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- Cells with CI excluding 0 are statistically significant at alpha=0.05.",
            "- McNemar p<0.05 confirms a paired binary outcome difference.",
            "- These CIs are within-cell sample variance; the multi-source evidence (holdout, touching, 9B reproduction, coverage-linearity) addresses the broader concern of effect existence.",
            "- Rows marked fallback_from_noref are copied from the NoRef result, so they reduce the discordant-pair counts b and c without changing paired bootstrap validity.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = []
    for spec in build_cell_specs():
        print(f"Computing {spec.name}")
        print(f"  NoRef:    {spec.noref_path}")
        print(f"  EGRefine: {spec.egrefine_path}")
        result = compute_cell(spec)
        print(
            f"  n={result.n}, NoRef={result.noref_exacc:.2f}, "
            f"EGRefine={result.egrefine_exacc:.2f}, "
            f"Delta={fmt_signed(result.delta)} pp, "
            f"fallback_from_noref={result.fallback_count}"
        )
        results.append(result)

    markdown = render_markdown(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
