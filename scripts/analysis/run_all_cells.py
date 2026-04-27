"""Run touching subset analysis on all configured cells.

Outputs (written to --label-dir, default: <repo>/results/touching_labels):
  - <label_dir>/<cell_name>.json          (per-query labels per cell)
  - <label_dir>/_summary_all_cells.json  (aggregate summary, all cells)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from touching_subset import analyze_cell, format_table

ROOT = Path(__file__).resolve().parents[2]

# Each cell: (cell_name, noref_eval, egrefine_eval, refine_dir, methods)
# Cell names follow registry conventions: <benchmark>_<backbone>_<variant>
CELLS = [
    # ---- Main matrix: Dr.Spider-Abbr ----
    (
        "drspider_abbr_27b_egrefine",
        "results/eval/drspider_abbr/no_refinement",
        "results/eval/drspider_abbr/egrefine_multi",
        "results/refine/drspider_abbr_multi",
        ("c3", "dinsql", "macsql"),
    ),
    (
        "drspider_abbr_9b_egrefine",
        "results/eval_9b/drspider_abbr/no_refinement",
        "results/eval_9b/drspider_abbr/egrefine_multi",
        "results/refine/drspider_abbr_multi_9b",
        ("c3", "dinsql", "macsql"),
    ),
    # ---- Main matrix: BIRD (no evidence) ----
    (
        "bird_27b_egrefine",
        "results/eval/bird/no_refinement",
        "results/eval/bird/egrefine_multi",
        "results/refine/bird_multi",
        ("c3", "dinsql", "macsql"),
    ),
    (
        "bird_9b_egrefine",
        "results/eval_9b/bird/no_refinement",
        "results/eval_9b/bird/egrefine",
        "results/refine/bird_multi_9b",
        ("c3", "dinsql", "macsql"),
    ),
    # ---- BIRD + Evidence variants ----
    (
        "bird_27b_egrefine_evidence",
        "results/eval/bird/no_refinement_evidence",
        "results/eval/bird/egrefine_evidence",
        "results/refine/bird_multi",
        ("c3", "dinsql", "macsql"),
    ),
    (
        "bird_9b_egrefine_evidence",
        "results/eval_9b/bird/no_refinement_evidence",
        "results/eval_9b/bird/egrefine_evidence",
        "results/refine/bird_multi_9b",
        ("c3", "dinsql", "macsql"),
    ),
    # ---- BEAVER MiniMax ----
    (
        "beaver_minimax_egrefine",
        "results/eval/beaver/no_refinement_minimax",
        "results/eval/beaver/egrefine_minimax",
        "results/refine/beaver_minimax_m27",
        ("c3", "dinsql", "macsql"),
    ),
    # ---- Cross-model on Dr.Spider-Abbr ----
    # 9B refine source, 27B eval (refinement transferable to stronger model)
    (
        "drspider_abbr_27b_egrefine_9brefine",
        "results/eval/drspider_abbr/no_refinement",
        "results/eval/drspider_abbr/egrefine_9b_refine",
        "results/refine/drspider_abbr_multi_9b",
        ("c3", "dinsql", "macsql"),
    ),
    # 27B refine source, 9B eval (refinement transferable to weaker model)
    (
        "drspider_abbr_9b_egrefine_27brefine",
        "results/eval_9b/drspider_abbr/no_refinement",
        "results/eval_9b/drspider_abbr/egrefine_27b_refine",
        "results/refine/drspider_abbr_multi",
        ("c3", "dinsql", "macsql"),
    ),
    # ---- LLM-Direct (supplementary). Reference = EGRefine refined cols (apple-to-apples). ----
    (
        "drspider_abbr_27b_llmdirect",
        "results/eval/drspider_abbr/no_refinement",
        "results/eval/drspider_abbr/llm_direct",
        "results/refine/drspider_abbr_multi",
        ("c3", "dinsql", "macsql"),
    ),
    (
        "drspider_abbr_9b_llmdirect",
        "results/eval_9b/drspider_abbr/no_refinement",
        "results/eval_9b/drspider_abbr/llm_direct",
        "results/refine/drspider_abbr_multi_9b",
        ("c3", "dinsql", "macsql"),
    ),
    (
        "bird_27b_llmdirect",
        "results/eval/bird/no_refinement",
        "results/eval/bird/llm_direct_multi",
        "results/refine/bird_multi",
        ("c3", "dinsql", "macsql"),
    ),
    (
        "bird_9b_llmdirect",
        "results/eval_9b/bird/no_refinement",
        "results/eval_9b/bird/llm_direct",
        "results/refine/bird_multi_9b",
        ("c3", "dinsql", "macsql"),
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Run touching subset analysis on all configured cells.")
    parser.add_argument("--label-dir", default=None,
                        help="Directory to write per-query label JSON files "
                             "(default: <repo>/results/touching_labels)")
    args = parser.parse_args()

    label_dir = Path(args.label_dir) if args.label_dir else ROOT / "results" / "touching_labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for cell_name, noref_rel, egr_rel, refine_rel, methods in CELLS:
        noref = ROOT / noref_rel
        egr = ROOT / egr_rel
        refine = ROOT / refine_rel

        if not noref.exists() or not egr.exists() or not refine.exists():
            print(f"SKIP {cell_name}: missing path")
            print(f"  noref:  {'OK' if noref.exists() else 'MISS'} {noref}")
            print(f"  egr:    {'OK' if egr.exists() else 'MISS'} {egr}")
            print(f"  refine: {'OK' if refine.exists() else 'MISS'} {refine}")
            continue

        print(f"\n=== {cell_name} ===")
        try:
            summary = analyze_cell(
                noref_eval_dir=noref,
                egrefine_eval_dir=egr,
                refine_dir=refine,
                methods=methods,
                label_dump_path=label_dir / f"{cell_name}.json",
                cell_name=cell_name,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        all_summaries[cell_name] = {
            "noref_eval_dir": str(noref.relative_to(ROOT)),
            "egrefine_eval_dir": str(egr.relative_to(ROOT)),
            "refine_dir": str(refine.relative_to(ROOT)),
            "summary": summary,
        }
        print(format_table(summary, cell_name))

    out_path = label_dir / "_summary_all_cells.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nWrote summary: {out_path}")


if __name__ == "__main__":
    main()
