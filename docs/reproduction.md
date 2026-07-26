# Reproducing the PVLDB Revision

This guide corresponds to the 12-page PVLDB revision in
[`EGRefine-PVLDB.pdf`](../EGRefine-PVLDB.pdf). Exact reported values for all
21 tables are frozen in
[`artifacts/pvldb_revision/confirmed_results.md`](../artifacts/pvldb_revision/confirmed_results.md).

The repository does not redistribute third-party benchmarks, model weights,
or SNAILS classifier outputs. Follow [`benchmarks.md`](benchmarks.md) and each
project's license before running the commands below.

## 1. Environment and configuration

```bash
git clone https://github.com/ai-jiaqian/EGRefine
cd EGRefine
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q
```

Start an OpenAI-compatible endpoint for the desired checkpoint, copy
[`config/revision_qwen35_27b.yaml`](../config/revision_qwen35_27b.yaml), and
replace only its endpoint/model identifiers and dataset paths. The paper's
default refinement configuration is:

- Qwen3.5-27B-FP8;
- Phase-1 LLM screening;
- `k=3` candidates and `N_s=20` sampled values;
- verifier set `C3+DIN-SQL`;
- `tau_min=0.05`;
- no BIRD evidence unless the experiment explicitly enables it.

The public config disables Qwen thinking through `extra_body`. If a serving
stack uses another mechanism, preserve the same decoding behavior.

## 2. Output contract and aggregation

Refinement writes one directory per database containing:

- `pipeline_result.json`;
- `views.sql`;
- `refined_tables.json`;
- `orig_table_map.json`;
- `statistics.json`.

Evaluation writes `per_db/<db_id>/<method>.json`. Query-weighted aggregation
must be derived from those files rather than trusting a possibly stale global
summary:

```python
import json
from pathlib import Path

def query_weighted_exacc(eval_dir, method):
    correct = total = 0
    for path in Path(eval_dir, "per_db").glob(f"*/{method}.json"):
        payload = json.loads(path.read_text())
        correct += payload["n_correct"]
        total += payload["n_queries"]
    return correct / total
```

The paper reports exact paired McNemar tests when per-query outcomes are
available and paired bootstrap intervals for BIRD aggregate sensitivity.

## 3. Canonical refine/evaluate workflow

The following produces the default Dr.Spider-Abbr schema used by Tables 1--5,
7--8, and downstream analyses:

```bash
python scripts/run_refine.py \
  --config config/revision_qwen35_27b.yaml \
  --benchmark drspider_abbr \
  --verify-algorithms c3+dinsql \
  --output results/refine/abbr_27b_k3

python scripts/run_eval.py \
  --config config/revision_qwen35_27b.yaml \
  --benchmark drspider_abbr \
  --schema original \
  --methods c3 dinsql macsql \
  --output results/eval/abbr_27b_noref

python scripts/run_eval.py \
  --config config/revision_qwen35_27b.yaml \
  --benchmark drspider_abbr \
  --schema refined \
  --refine-dir results/refine/abbr_27b_k3 \
  --methods c3 dinsql macsql \
  --output results/eval/abbr_27b_egrefine \
  --reuse-original-dir results/eval/abbr_27b_noref
```

Table 1 additionally evaluates the official pre-perturbation Dr.Spider split,
LLM-Direct, and SNAILS-style schemas. For the latter, first run the released
SNAILS naturalness classifier over every column, export its labels as JSON,
then run:

```bash
python scripts/run_snails_style.py \
  --config config/revision_qwen35_27b.yaml \
  --screened /path/to/snails_screened.json \
  --output results/refine/abbr_snails \
  --cache-dir cache_snails
```

Table 2 uses the phase counters and elapsed times recorded by the canonical
`pipeline_result.json` files. Counts are logical requests/comparisons;
transport retries and token usage were not logged.

## 4. Ablations and verifier robustness (Tables 3--6)

### Component ablations

```bash
# Without conservative selection
python scripts/run_refine.py --config config/revision_qwen35_27b.yaml \
  --benchmark drspider_abbr --verify-algorithms c3+dinsql \
  --no-conservative --output results/refine/abbr_no_conservative

# Without Phase-1 screening; structural exclusions remain enforced
python scripts/run_refine.py --config config/revision_qwen35_27b.yaml \
  --benchmark drspider_abbr --verify-algorithms c3+dinsql \
  --no-pruning --output results/refine/abbr_no_screening

# Single-algorithm verifier
python scripts/run_refine.py --config config/revision_qwen35_27b.yaml \
  --benchmark drspider_abbr --verify-algorithms c3 \
  --output results/refine/abbr_c3_verifier
```

LLM-Direct is produced with `--method llm-direct`. Evaluate every artifact
with the same three `run_eval.py` methods shown in Section 3.

### Threshold sensitivity

Table 4 reuses the canonical `all_scores`; no new Phase-3 calls are needed:

```bash
for tau in 0.01 0.03 0.10; do
  python scripts/rethreshold_refine.py \
    --source-refine results/refine/abbr_27b_k3 \
    --benchmark drspider_abbr \
    --config config/revision_qwen35_27b.yaml \
    --tau "$tau" \
    --output "results/refine/abbr_tau_${tau}"
done
```

### Alternative verifier set

Table 5 changes only the verifier set:

```bash
python scripts/run_refine.py --config config/revision_qwen35_27b.yaml \
  --benchmark drspider_abbr --verify-algorithms dinsql+macsql \
  --output results/refine/abbr_din_mac_verifier
```

### Candidate-count and sample-row ablation

For Table 6, make one sanitized config per arm with `phase2.k` in
`{1,2,3,5}` or `phase2.sample_rows=5`, use a distinct cache/output directory,
and repeat the canonical workflow. The `k=2` arm uses 25.1% fewer logical LLM
calls than `k=3`; exact values are in the confirmed result snapshot.

## 5. Transfer experiments (Tables 7--8)

Repeat the canonical workflow with Qwen3.5-9B-FP8 to obtain self-refined 9B
results. Then evaluate both the 9B- and 27B-refined schemas with the 9B and 27B
serving checkpoints for Table 7.

For Table 8, keep `--refine-dir results/refine/abbr_27b_k3` fixed and change
only the serving endpoint/model identifier to Qwen3.5-9B, Phi-4-14B, or
Gemma3-27B. The transferred schema is not re-verified.

`scripts/plot_fig_model_strength.py` contains the frozen cross-backbone values
and can regenerate the corresponding diagnostic plot.

## 6. BIRD and BEAVER (Tables 9--11)

Replace `--benchmark drspider_abbr` in the canonical workflow with `bird` or
`beaver`. For BIRD evidence cells, add `--use-evidence` to refinement and the
matching evidence option to evaluation; evidence and non-evidence runs must
use distinct caches. Run BIRD with both 9B and 27B checkpoints.

The reported BEAVER comparison uses the same four databases in both arms
(45 queries); the 1,269-column database exceeds the declared refinement budget
and is excluded from both arms. Qwen3.5-27B produces zero commits. The
MiniMax-M2.7 row changes only the refinement backbone; C3, DIN-SQL, and MAC-SQL
remain the serving systems.

## 7. Unseen queries and workload size (Tables 12--13)

Create split files before refinement so held-out queries are never visible to
Phases 1--3. Dr.Spider-Abbr splits by base-question pool with seed 42; BIRD
uses a stratified 50/50 split. The test sizes are 851, 1,430, and 770 for ABBR
70/30, ABBR 50/50, and BIRD 50/50.

Use [`scripts/run_bird_holdout.py`](../scripts/run_bird_holdout.py) for the
independent-authoring BIRD probe. It hides real identifiers while questions
are authored and reveals them only after each question is fixed for gold-SQL
construction.

For Table 13, use nested Dr.Spider-Abbr refinement subsets of 25%, 50%, 75%,
and 100% with seed 42 and shared Phase-2 candidates. Evaluate every resulting
schema on all 2,853 queries. Commit sets need not be nested because each arm
re-estimates the per-column delta on its own workload.

## 8. Description baseline (Table 14)

```bash
python scripts/desc_baseline/generate_descriptions.py \
  --config config/revision_qwen35_27b.yaml \
  --benchmark drspider_abbr \
  --refine-dir results/refine/abbr_27b_k3 \
  --output results/description/column_descriptions.json
```

Evaluate the four cells NoRef, NoRef+descriptions, EGRefine, and
EGRefine+descriptions with `scripts/desc_baseline/run_all_cells.sh`, then use
`scripts/desc_baseline/analyze_4cell.py` for the aggregate.

## 9. Revision audits (Tables 15--21)

### Query flips and touching subsets

[`scripts/analysis/run_all_cells.py`](../scripts/analysis/run_all_cells.py)
aggregates paired query outcomes for Table 15.
[`scripts/analysis/touching_subset.py`](../scripts/analysis/touching_subset.py)
uses gold-SQL column references and accepted rename mappings for Table 16.

### Candidate outcomes and user-side query signals

Table 17 is derived directly from per-column `all_scores` in the canonical
pipeline artifacts. Table 18 pre-registers the lexical-proxy targets, forces
them past screening, proposes standard and workload-conditioned candidate
pools, and scores their union once:

```bash
python scripts/rev_e6b_shared_verification.py \
  --config config/revision_qwen35_27b.yaml \
  --benchmark drspider_abbr \
  --targets /path/to/frozen_targets.json \
  --output results/refine/query_conditioning_shared

python scripts/analysis/rev_e6b_analyze.py --help
```

The target JSON format is:

```json
{
  "targets": [
    {"db_id": "database_id", "full_name": "table.column"}
  ]
}
```

### Composed-schema audit

```bash
python scripts/rev_composed_check.py \
  --config config/revision_qwen35_27b.yaml \
  --benchmark drspider_abbr \
  --refine-dir results/refine/abbr_27b_k3 \
  --output results/analysis/composition_abbr
```

The script scores each selected/original pair both in isolation and with all
other commits active, using one audit endpoint so absolute scores are not
mixed across checkpoints.

### Screening funnel and refiner strength

Table 20 aggregates Phase-1 counts and accepted refinements from
`pipeline_result.json`. For Table 21, repeat candidate generation once with
each named refiner, evaluate LLM-Direct and execution-verified selection from
the same candidate pool, and compare both with the same-endpoint NoRef.

## 10. Method invariants added by the revision

The default Phase-4 behavior propagates a PK rename only to an FK whose surface
name equals the original PK name. Role-named FKs retain their local semantics.
After propagation, Phase 4 enforces within-table name uniqueness as a hard
post-condition. SQLite evaluation uses read-only connections. The regression
tests for these invariants are in
[`tests/test_revision_invariants.py`](../tests/test_revision_invariants.py).

## 11. Result verification

After reproducing a cell:

1. aggregate from `per_db` with query weighting;
2. verify query counts and database coverage against the relevant table;
3. compare with the exact snapshot in
   [`confirmed_results.md`](../artifacts/pvldb_revision/confirmed_results.md);
4. preserve the config, git commit, model identifier, per-database JSON, and
   paired outcome vectors used for statistical testing.

Small endpoint-level variations should not be silently substituted into the
paper's canonical table. Report a reproduction separately unless it uses the
same checkpoint, prompt behavior, candidate cache, dataset split, and query
aggregation protocol.
