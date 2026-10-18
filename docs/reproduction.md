# Running experiments

This guide describes the supplied runtime and command-line entry points. It
does not redistribute benchmark data or promise exact agreement across model
serving stacks. Configure paths and endpoints as described in README (`../README.md`).

## Refine and evaluate

```bash
python scripts/run_refine.py --config config/local.yaml \
  --benchmark drspider_abbr --verify-algorithms c3+dinsql \
  --output results/refine/abbr

python scripts/run_eval.py --config config/local.yaml \
  --benchmark drspider_abbr --schema both --refine-dir results/refine/abbr \
  --methods simple c3 dinsql macsql --full-eval --output results/eval/abbr
```

The example config uses LLM screening, three candidate names, 20 sampled
values and a minimum execution-accuracy improvement of 0.05. Use one model
endpoint/configuration across the comparison arms. Held-out questions must
not enter screening, candidate generation or verification.

## Ablations

Run each arm into its own output/cache directory:

- `--method llm-direct`: direct candidate selection without execution verification.
- `--no-conservative`: disable conservative candidate selection.
- `--no-pruning`: bypass screening while preserving structural exclusions.
- `--verify-algorithms c3`: single-verifier refinement.
- `--workload-conditioned`: expose refinement-workload questions to candidate generation.

Change `phase2.k` or `phase2.sample_rows` in a copied config for candidate-count
or value-sampling ablations. `scripts/rethreshold_refine.py --help` describes
reselecting saved candidate scores at a different threshold without new model
calls. Evaluate each published schema with `--full-eval`; verify that all arms
cover the same databases and query identities.

## Records and statistics

Refinement writes `pipeline_result.json`, `views.sql`, `refined_tables.json`,
`orig_table_map.json`, and `statistics.json` per database. Evaluation stores
per-database predictions and correctness in `per_db/`. Use query-weighted
aggregation, rather than averaging database accuracies equally. Statistical
helpers are in `scripts/compute_bootstrap_ci.py` and
`scripts/analysis/rev_s_stats.py`.

To capture complete model responses and individual HTTP attempts, create the
log directory first and set:

```bash
mkdir -p logs
export EGREFINE_RAW_LOG=logs/model-calls.jsonl
export EGREFINE_HTTP_RAW_LOG=logs/http-attempts.jsonl
```

These logs can contain dataset contents and should remain private. Preserve
configuration, model/checkpoint identity, complete predictions and raw traces
for any experiment you conduct.
