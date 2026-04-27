# Reproducing Paper Results

This document gives concrete commands to reproduce every table and figure in
the EGRefine paper. Each section is self-contained: read the section for the
table you want, then run only those commands.

> **Note on expected numbers**: absolute ExAcc values marked `TBD` will be
> updated upon paper acceptance. Relative deltas (Δ vs NoRef) and ordinal
> claims (which method ranks best) are stable.

## Prerequisites

Install EGRefine and verify the CLI:

```bash
pip install -e .
egrefine-refine --help
egrefine-eval --help
```

Download benchmarks following the instructions in
[`docs/benchmarks.md`](benchmarks.md) and update config paths accordingly.
All examples below use `config/example_local_vllm.yaml` — substitute
`config/example_openai.yaml` if using the OpenAI API instead.

**Runtime estimates** assume an A800-class GPU (80 GB VRAM) running a 27B
FP8 model via vLLM. OpenAI gpt-4o-mini will be faster per-call but has
API rate limits; scale concurrency down to `concurrency.max_workers: 4` in
that case.

---

## Table II — Cross-algorithm main results (Dr.Spider-Abbr)

**What it shows:** ExAcc of C3 / DIN-SQL / MAC-SQL across five refinement
strategies (No Refinement, LLM-Direct, LLM-CoT, Column-Description, EGRefine)
on Dr.Spider-Abbr (2853 queries, 90 databases, 27B backbone).

**Estimated runtime:** ~8 hours total for all refinement + eval runs on a 27B
model (Phase 3 dominates). Eval-only reruns take ~1–2 h per configuration.

### Step 1: Run EGRefine refinement (once, shared by all eval configs)

```bash
egrefine-refine \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --output results/refine/drspider_abbr_multi \
    --method egrefine \
    --verify-algorithms c3+dinsql
```

### Step 2: Run No-Refinement baseline

```bash
egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --schema original \
    --methods c3 dinsql macsql \
    --output results/eval/drspider_abbr/no_refinement
```

### Step 3: Run LLM-Direct baseline

```bash
egrefine-refine \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --output results/refine/drspider_abbr_llm_direct \
    --method llm-direct

egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --schema refined \
    --refine-dir results/refine/drspider_abbr_llm_direct \
    --methods c3 dinsql macsql \
    --output results/eval/drspider_abbr/llm_direct
```

### Step 4: Run LLM-CoT baseline

```bash
egrefine-refine \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --output results/refine/drspider_abbr_llm_cot \
    --method llm-cot

egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --schema refined \
    --refine-dir results/refine/drspider_abbr_llm_cot \
    --methods c3 dinsql macsql \
    --output results/eval/drspider_abbr/llm_cot
```

### Step 5: Evaluate EGRefine

```bash
egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --schema refined \
    --refine-dir results/refine/drspider_abbr_multi \
    --methods c3 dinsql macsql \
    --output results/eval/drspider_abbr/egrefine_multi \
    --reuse-original-dir results/eval/drspider_abbr/no_refinement
```

### Expected numbers (27B backbone)

| Method | C3 | DIN-SQL | MAC-SQL |
|---|---|---|---|
| No Refinement | TBD | TBD | TBD |
| LLM-Direct | TBD | TBD | TBD |
| LLM-CoT | TBD | TBD | TBD |
| **EGRefine** | TBD | TBD | TBD |

EGRefine consistently ranks first across all three algorithms. The canonical
finding: EGRefine's C3 recovery rate (relative to the pre-perturbation upper
bound) exceeds 100%, meaning refined column names are clearer than even the
original Spider names for zero-shot C3.

---

## Table III — Cross-backbone results (Dr.Spider-Abbr)

**What it shows:** ExAcc of C3 across three LLM backbone sizes (9B / 27B /
Gemma3-27B) under No Refinement and EGRefine. Demonstrates that refinement
benefits transfer across model families.

**Estimated runtime:** ~3 h per backbone for refine + eval (9B is faster).

### 9B backbone

```bash
# Edit config to point to the 9B model endpoint, then:
egrefine-refine \
    --config config/example_local_vllm.yaml \   # set model_name to your 9B checkpoint
    --benchmark drspider_abbr \
    --output results/refine/drspider_abbr_multi_9b \
    --method egrefine \
    --verify-algorithms c3+dinsql

egrefine-eval \
    --config config/example_local_vllm.yaml \   # set model_name to your 9B checkpoint
    --benchmark drspider_abbr \
    --schema original \
    --methods c3 dinsql macsql \
    --output results/eval_9b/drspider_abbr/no_refinement

egrefine-eval \
    --config config/example_local_vllm.yaml \   # set model_name to your 9B checkpoint
    --benchmark drspider_abbr \
    --schema refined \
    --refine-dir results/refine/drspider_abbr_multi_9b \
    --methods c3 dinsql macsql \
    --output results/eval_9b/drspider_abbr/egrefine_multi \
    --reuse-original-dir results/eval_9b/drspider_abbr/no_refinement
```

### 27B backbone

Already produced in Table II. Reuse `results/eval/drspider_abbr/`.

### Gemma3-27B backbone (cross-family transfer)

```bash
# The 27B-refined schema is reused — no re-refinement needed
egrefine-eval \
    --config config/example_local_vllm.yaml \   # set model_name to gemma-3-27b-it
    --benchmark drspider_abbr \
    --schema original \
    --methods c3 dinsql macsql \
    --output results/eval_gemma3/drspider_abbr/no_refinement

egrefine-eval \
    --config config/example_local_vllm.yaml \   # set model_name to gemma-3-27b-it
    --benchmark drspider_abbr \
    --schema refined \
    --refine-dir results/refine/drspider_abbr_multi \
    --methods c3 dinsql macsql \
    --output results/eval_gemma3/drspider_abbr/egrefine_multi \
    --reuse-original-dir results/eval_gemma3/drspider_abbr/no_refinement
```

### Expected numbers

| Backbone | NoRef C3 | EGRefine C3 | Δ |
|---|---|---|---|
| 9B | TBD | TBD | TBD |
| 27B | TBD | TBD | TBD |
| Gemma3-27B | TBD | TBD | TBD |

All three backbones show positive Δ, confirming cross-family transferability.
Weaker backbones (9B) benefit at least as much as stronger ones.

---

## Table IV — Cross-benchmark results

**What it shows:** ExAcc on BIRD (1534 queries, 11 DBs) and BEAVER (88 queries,
5 DBs) under No Refinement, LLM-Direct, and EGRefine. Tests generalisation
beyond the Dr.Spider-Abbr domain.

**Estimated runtime:** ~2 h for BIRD (11 DBs), ~30 min for BEAVER (5 DBs).

### BIRD

```bash
egrefine-refine \
    --config config/example_local_vllm.yaml \
    --benchmark bird \
    --output results/refine/bird_multi \
    --method egrefine \
    --verify-algorithms c3+dinsql

egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark bird \
    --schema original \
    --methods c3 dinsql macsql \
    --output results/eval/bird/no_refinement

egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark bird \
    --schema refined \
    --refine-dir results/refine/bird_multi \
    --methods c3 dinsql macsql \
    --output results/eval/bird/egrefine_multi \
    --reuse-original-dir results/eval/bird/no_refinement
```

### BEAVER

```bash
egrefine-refine \
    --config config/example_local_vllm.yaml \
    --benchmark beaver \
    --output results/refine/beaver_multi \
    --method egrefine \
    --verify-algorithms c3+dinsql

egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark beaver \
    --schema original \
    --methods c3 dinsql macsql \
    --output results/eval/beaver/no_refinement

egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark beaver \
    --schema refined \
    --refine-dir results/refine/beaver_multi \
    --methods c3 dinsql macsql \
    --output results/eval/beaver/egrefine_multi \
    --reuse-original-dir results/eval/beaver/no_refinement
```

### Expected numbers

| Benchmark | Method | C3 | DIN-SQL | MAC-SQL |
|---|---|---|---|---|
| BIRD | NoRef | TBD | TBD | TBD |
| BIRD | EGRefine | TBD | TBD | TBD |
| BEAVER | NoRef | TBD | TBD | TBD |
| BEAVER | EGRefine | TBD | TBD | TBD |

---

## Table V — Ablation study

**What it shows:** Contribution of each EGRefine component on Dr.Spider-Abbr
(27B backbone, C3 + DIN-SQL). Four conditions:
- Full EGRefine
- w/o Execution verification (= LLM-Direct, already computed in Table II)
- w/o Conservative rule
- w/o Phase 1 Pruning

**Important**: ablation comparisons must not use `--reuse-original-dir` because
changing which columns are refined changes which queries are "affected". Each
ablation condition must be evaluated independently (no partial reuse).

```bash
# w/o Conservative rule
egrefine-refine \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --output results/refine/drspider_abbr_no_conservative \
    --method egrefine \
    --verify-algorithms c3+dinsql \
    --no-conservative

egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --schema refined \
    --refine-dir results/refine/drspider_abbr_no_conservative \
    --methods c3 dinsql macsql \
    --output results/eval/drspider_abbr/ablation_no_conservative

# w/o Phase 1 Pruning (evaluate all columns, no candidate filtering)
egrefine-refine \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --output results/refine/drspider_abbr_no_pruning \
    --method egrefine \
    --verify-algorithms c3+dinsql \
    --no-pruning

egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --schema refined \
    --refine-dir results/refine/drspider_abbr_no_pruning \
    --methods c3 dinsql macsql \
    --output results/eval/drspider_abbr/ablation_no_pruning
```

### Expected numbers (C3 ExAcc, 27B backbone, Dr.Spider-Abbr)

| Condition | C3 | DIN-SQL | MAC-SQL |
|---|---|---|---|
| Full EGRefine | TBD | TBD | TBD |
| w/o Conservative rule | TBD (↓) | TBD (↓) | TBD (↓) |
| w/o Phase 1 Pruning | TBD (↓) | TBD (↓) | TBD (↓) |
| w/o Execution (LLM-Direct) | TBD (↓) | TBD (↓) | TBD (↓) |

Each ablation degrades performance relative to full EGRefine. The largest
single-component contribution is execution-grounded verification (Phase 3).

---

## Figure 1 — Conservative-rule sensitivity (τ ablation)

**What it shows:** ExAcc as a function of min_delta threshold τ ∈ {0.01, 0.03,
0.05, 0.10} for C3, DIN-SQL, and MAC-SQL on Dr.Spider-Abbr. All three
algorithms peak at τ = 0.05; τ = 0.01 is clearly too aggressive (DIN-SQL
and MAC-SQL both regress).

The τ ablation reuses the Phase 3 scores already cached from the full EGRefine
run — no new LLM calls are needed. To re-threshold at a different τ, copy the
example config, change `phase3.min_delta` to the desired value, and re-run
`egrefine-refine` against the same output directory; the Phase 1/2/3 caches
will be reused (only Phase 4 VIEW-synthesis re-runs).

```bash
# Re-threshold by re-running with a different min_delta. Phase 1-3 caches are
# reused, so each τ value adds only ~1 minute (Phase 4 only).
for TAU in 0.01 0.03 0.10; do
    # Copy and patch the config in place (or maintain one config per τ)
    sed "s/min_delta:.*/min_delta: ${TAU}/" config/example_local_vllm.yaml \
        > tmp/config_tau${TAU//./}.yaml

    egrefine-refine \
        --config tmp/config_tau${TAU//./}.yaml \
        --benchmark drspider_abbr \
        --output results/refine/drspider_abbr_tau${TAU//./} \
        --method egrefine \
        --verify-algorithms c3+dinsql

    egrefine-eval \
        --config config/example_local_vllm.yaml \
        --benchmark drspider_abbr \
        --schema refined \
        --refine-dir results/refine/drspider_abbr_tau${TAU//./} \
        --methods c3 dinsql macsql \
        --output results/eval/drspider_abbr/egrefine_tau${TAU//./}
done

# Generate Figure 1
python scripts/compute_bootstrap_ci.py \
    --eval-dirs \
        results/eval/drspider_abbr/no_refinement \
        results/eval/drspider_abbr/egrefine_tau001 \
        results/eval/drspider_abbr/egrefine_tau003 \
        results/eval/drspider_abbr/egrefine_multi \
        results/eval/drspider_abbr/egrefine_tau010 \
    --labels "NoRef,τ=0.01,τ=0.03,τ=0.05,τ=0.10" \
    --output results/analyses/tau_sensitivity.png
```

**Key finding**: τ ∈ [0.03, 0.10] varies < 1.8 pp (within the LLM noise
floor). Only τ = 0.01 shows a clear regression on DIN-SQL and MAC-SQL.

---

## Figure 2 — Cross-verifier results

**What it shows:** Held-out transfer ExAcc when the Phase 3 verifier set M
is changed from M = {C3, DIN-SQL} (default) to the cross-paradigm
M = {DIN-SQL, MAC-SQL}. Demonstrates that EGRefine's gains are
verifier-agnostic in direction.

```bash
# Run refinement with cross-paradigm verifier set M = {DIN-SQL, MAC-SQL}
egrefine-refine \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --output results/refine/drspider_abbr_dinmac_verifier \
    --method egrefine \
    --verify-algorithms dinsql+macsql

# Evaluate held-out C3 (not in the verifier set) on the new refinement
egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --schema refined \
    --refine-dir results/refine/drspider_abbr_dinmac_verifier \
    --methods c3 \
    --output results/eval/drspider_abbr/egrefine_dinmac_c3 \
    --reuse-original-dir results/eval/drspider_abbr/no_refinement

# Generate Figure 2 / cross-verifier summary table
python scripts/analysis/cross_verifier_aggregate.py \
    --noref-dir results/eval/drspider_abbr/no_refinement \
    --canonical-dir results/eval/drspider_abbr/egrefine_multi \
    --crossverifier-dir results/eval/drspider_abbr/egrefine_dinmac_c3 \
    --output results/analyses/cross_verifier_results.md
```

**Expected finding**: both verifier sets produce positive held-out C3
transfer. M = {C3, DIN-SQL} produces larger magnitude (~3×) because C3 is
in the verifier set. See `docs/analyses/cross_verifier_results.md` for the
full analysis.

---

## Touching-subset analysis

**What it shows:** ExAcc split into queries whose gold SQL references at
least one refined column ("touching") vs. all others ("non-touching").
Delta on touching queries is 4–10× larger than on the full benchmark,
providing strong per-column evidence.

**This analysis requires a user-supplied labels file.** The labels file
maps each database to lists of touching / non-touching question strings.
See `docs/architecture.md` (Touching-Labels File Format) for the JSON
schema.

```bash
python scripts/analysis/touching_subset.py \
    --eval-dir results/eval/drspider_abbr/egrefine_multi \
    --noref-dir results/eval/drspider_abbr/no_refinement \
    --refine-dir results/refine/drspider_abbr_multi \
    --labels-file /path/to/touching_labels.json \
    --output results/analyses/touching_subset.md
```

The script emits a markdown table with per-DB and aggregate ExAcc for
touching / non-touching subsets, plus Δ_touching / Δ_full ratio.

---

## Description-baseline (4-cell)

**What it shows:** Isolates the contribution of *identifier rewriting*
(EGRefine) from *description annotation* (adding natural-language column
comments to the prompt). Responds to the question "could you get the same
gain just by adding column descriptions?".

The 4-cell design:
- Cell 1: No Refinement (original schema, no descriptions)
- Cell 2: Descriptions only (original schema, Phase-1-screened columns
  annotated with LLM-generated SQL comments)
- Cell 3: EGRefine only (refined VIEW, no descriptions)
- Cell 4: EGRefine + Descriptions

```bash
# Generate column descriptions for Phase-1-screened columns
python scripts/desc_baseline/generate_descriptions.py \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --refine-dir results/refine/drspider_abbr_multi \
    --output results/desc_baseline/column_descriptions.json

# Cell 1: already in results/eval/drspider_abbr/no_refinement/

# Cell 2: NoRef + Descriptions
egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --schema original \
    --methods c3 \
    --descriptions-json results/desc_baseline/column_descriptions.json \
    --output results/eval/drspider_abbr/noref_with_desc

# Cell 3: EGRefine only
# already in results/eval/drspider_abbr/egrefine_multi/

# Cell 4: EGRefine + Descriptions
egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark drspider_abbr \
    --schema refined \
    --refine-dir results/refine/drspider_abbr_multi \
    --methods c3 \
    --descriptions-json results/desc_baseline/column_descriptions.json \
    --output results/eval/drspider_abbr/egrefine_with_desc

# Generate analysis report
python scripts/desc_baseline/analyze_4cell.py \
    --cell1 results/eval/drspider_abbr/no_refinement \
    --cell2 results/eval/drspider_abbr/noref_with_desc \
    --cell3 results/eval/drspider_abbr/egrefine_multi \
    --cell4 results/eval/drspider_abbr/egrefine_with_desc \
    --output results/desc_baseline/description_baseline_results.md
```

### Expected numbers (C3, Dr.Spider-Abbr)

| Cell | Schema | Descriptions | C3 ExAcc | Δ vs Cell 1 |
|---|---|---|---|---|
| 1. NoRef | original | no | TBD | — |
| 2. NoRef + Desc | original | yes | TBD | TBD (ΔDesc) |
| 3. EGRefine | refined VIEW | no | TBD | TBD (ΔRename) |
| 4. EGRefine + Desc | refined VIEW | yes | TBD | TBD (ΔBoth) |

**Key findings**:
- ΔRename > ΔDesc on the canonical endpoint — pure identifier rewriting
  outperforms description annotation alone.
- ΔDesc ≈ LLM-Direct — broad LLM annotation without execution verification
  reaches a similar performance floor.
- ΔBoth > max(ΔDesc, ΔRename) — the two mechanisms are complementary and
  near-additive.
- W:C ratio (rescues per regression): EGRefine ~5.0 vs Description ~2.7 —
  description annotations introduce regressions at roughly 2× the rate of
  execution-verified renamings.

---

## Aggregating results

All `egrefine-eval` runs write `results.json` (global summary) and
`per_db/*.json` (per-database breakdown). The authoritative aggregate is
always the Q-weighted mean from `per_db/`:

```python
import json, pathlib

def weighted_avg(eval_dir, method="c3"):
    total_correct, total_queries = 0, 0
    for p in pathlib.Path(eval_dir, "per_db").glob(f"*/{method}.json"):
        d = json.loads(p.read_text())
        total_correct += d["n_correct"]
        total_queries += d["n_queries"]
    return total_correct / total_queries if total_queries else 0.0
```

`results.json` can be stale if a run was interrupted mid-way. Always
re-aggregate from `per_db/` when in doubt.
