#!/usr/bin/env python3
"""Regenerate Figure "Cross-model results on Dr.Spider Schema-Abbreviation".

Revision fix for reviewer R4-D2 (VLDB 2026-07): the originally submitted figure
plotted C3 data while the caption and Section 5.4.1 text quote DIN-SQL numbers.
This script regenerates the figure from canonical registry numbers, defaulting
to DIN-SQL so the figure matches the caption/text.

Data provenance (docs/experiment_data_registry.md, all starred canonical):
  - 27B  self-refine : registry §1.1  (pre / no_refinement / egrefine_multi)
  - 9B   self-refine : registry §1.3
  - 9B + 27B-refined : registry §1.6.1 (egrefine_27b_refine + NoRef fallback)
  - Gemma3-27B       : registry §1.5  (eval_gemma3_27b, 27B-refined schema)
  - Phi-4-14B        : registry §1.4  (eval_phi4, 27B-refined schema)

Usage:
  .venv/bin/python scripts/plot_fig_model_strength.py [--algo DIN-SQL|C3|MAC-SQL] [--out PATH]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (clean, degraded, egrefine) per group, per algorithm — registry canonical.
DATA = {
    "C3": {
        "Qwen-9B\n(self)":     (64.70, 59.38, 63.34),
        "Qwen-9B\n(27B-ref)":  (64.70, 59.38, 62.36),
        "Qwen-27B":            (72.77, 70.87, 73.43),
        "Gemma3\n-27B":        (71.64, 69.33, 71.15),
        "Phi-4\n-14B":         (59.34, 56.85, 57.59),
    },
    "DIN-SQL": {
        "Qwen-9B\n(self)":     (78.23, 70.63, 75.81),
        "Qwen-9B\n(27B-ref)":  (78.23, 70.63, 75.60),
        "Qwen-27B":            (80.41, 76.73, 79.21),
        "Gemma3\n-27B":        (76.62, 75.67, 76.94),
        "Phi-4\n-14B":         (71.05, 69.54, 72.38),
    },
    "MAC-SQL": {
        "Qwen-9B\n(self)":     (55.84, 51.52, 53.63),
        "Qwen-9B\n(27B-ref)":  (55.84, 51.52, 53.49),
        "Qwen-27B":            (63.79, 59.59, 60.22),
        "Gemma3\n-27B":        (35.65, 34.77, 35.02),
        "Phi-4\n-14B":         (34.60, 34.00, 34.38),
    },
}

C_CLEAN, C_DEGRADED, C_EGREFINE = "#cdd7e1", "#f0c4c4", "#3d5a73"
C_GREEN, C_ORANGE = "#2e8b57", "#e07b39"


def recovery(clean: float, degraded: float, egrefine: float) -> float:
    return (egrefine - degraded) / (clean - degraded) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="DIN-SQL", choices=sorted(DATA))
    ap.add_argument("--out", default="artifacts/pvldb_revision/fig_model_strength.pdf")
    args = ap.parse_args()
    groups = DATA[args.algo]

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    width = 0.27
    for i, (label, (clean, degraded, egr)) in enumerate(groups.items()):
        ax.bar(i - width, clean, width, color=C_CLEAN, edgecolor="#444", linewidth=0.6,
               label="Clean" if i == 0 else None)
        ax.bar(i, degraded, width, color=C_DEGRADED, edgecolor="#444", linewidth=0.6,
               label="Degraded" if i == 0 else None)
        ax.bar(i + width, egr, width, color=C_EGREFINE, edgecolor="#444", linewidth=0.6,
               label="EGRefine" if i == 0 else None)
        rec = recovery(clean, degraded, egr)
        over100 = rec >= 100
        ax.annotate(f"{rec:.1f}%", xy=(i + width, egr), xytext=(0, 4),
                    textcoords="offset points", ha="center",
                    color=C_GREEN if rec >= 50 else C_ORANGE,
                    fontsize=11 if over100 else 10,
                    fontweight="bold" if over100 else "normal")

    lo = min(v for triple in groups.values() for v in triple)
    hi = max(v for triple in groups.values() for v in triple)
    ax.set_ylim(max(0, (lo // 5) * 5 - 5), (hi // 5) * 5 + 6.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups.keys())
    ax.set_ylabel("ExAcc (%) on Dr.Spider-Abbr")
    ax.set_title("Recovery Rate (%) shown above each EGRefine bar",
                 fontsize=11, pad=26)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.115),
              framealpha=0.95, fontsize=10, borderpad=0.35, columnspacing=1.4)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} [{args.algo}]")
    for label, (clean, degraded, egr) in groups.items():
        print(f"  {label.replace(chr(10), ' '):22s} clean={clean:5.2f} deg={degraded:5.2f} "
              f"egr={egr:5.2f} recovery={recovery(clean, degraded, egr):6.1f}%")


if __name__ == "__main__":
    main()
