# EGRefine

**Execution-Grounded Schema Refinement for Text-to-SQL**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/ai-jiaqian/EGRefine/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-jiaqian/EGRefine/actions)

EGRefine is a plug-and-play preprocessor that improves the column/table naming
quality of database schemas to lift downstream Text-to-SQL accuracy. Unlike
prior approaches that rely on LLM "semantic understanding" to guess better
names, EGRefine **verifies** every candidate by running downstream Text-to-SQL
on a benchmark and only accepts a rename when the execution accuracy actually
improves.

## How it works

EGRefine is a four-phase pipeline:

1. **Phase 1 — Pruning.** Identify columns whose names are likely problematic
   (short names, generic vocabulary, naming inconsistency, high similarity).
2. **Phase 2 — Candidate Generation.** Ask an LLM for `k` alternative names
   per column, given the surrounding schema context and sampled data values.
3. **Phase 3 — Execution-Grounded Verification.** For each candidate name,
   construct a temporary VIEW exposing the renamed schema, run a downstream
   Text-to-SQL model on benchmark queries that touch the column, back-map the
   generated SQL to the original schema, and execute. The candidate that
   maximizes execution accuracy wins — but only if it strictly beats the
   original name (the "conservative rule").
4. **Phase 4 — VIEW Synthesis + Back-Mapping.** Materialize the accepted
   refinements as SQL VIEWs (the original tables are never modified), and
   provide a back-mapper for translating downstream-generated SQL back to
   the original schema for execution.

## Why it works

Two ideas distinguish EGRefine from prior schema-refinement work:

- **Execution feedback** replaces "the LLM thinks this name is better" with
  "the downstream system actually performs better with this name on real
  queries". The downstream model becomes the arbiter.
- **The conservative rule** guarantees monotonic non-degradation: if no
  candidate beats the original, the original is kept. EGRefine never makes
  a schema worse.

## Installation

```bash
git clone https://github.com/ai-jiaqian/EGRefine
cd EGRefine
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

(EGRefine v0.1.0 supports editable installs from a git checkout. PyPI
publication is planned post-paper-acceptance.)

## Quickstart

1. **Configure an LLM endpoint.** Pick one of:
   - Local vLLM: edit `config/example_local_vllm.yaml`
   - OpenAI: edit `config/example_openai.yaml` and `export OPENAI_API_KEY=...`

2. **Download a benchmark.** See [`docs/benchmarks.md`](docs/benchmarks.md)
   for BIRD, Dr.Spider, and BEAVER setup. Update the `data:` paths in your
   config.

3. **Refine a database:**

   ```bash
   egrefine-refine \
     --config config/example_local_vllm.yaml \
     --benchmark bird \
     --dbs formula_1 \
     --output ./results/refine_demo
   ```

   This runs Phases 1–4 and emits `results/refine_demo/formula_1/`:
   `views.sql`, `refined_tables.json`, `orig_table_map.json`, `statistics.json`.

4. **Evaluate (NoRef baseline):**

   ```bash
   egrefine-eval \
     --config config/example_local_vllm.yaml \
     --benchmark bird \
     --dbs formula_1 \
     --schema original \
     --methods c3 \
     --output ./results/eval_noref
   ```

5. **Evaluate (refined schema):**

   ```bash
   egrefine-eval \
     --config config/example_local_vllm.yaml \
     --benchmark bird \
     --dbs formula_1 \
     --schema refined \
     --refine-dir ./results/refine_demo \
     --methods c3 \
     --output ./results/eval_refined
   ```

For a step-by-step walkthrough see [`docs/quickstart.md`](docs/quickstart.md).

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline
specification, data structures, and configuration reference.

## Reproducing the paper

See [`docs/reproduction.md`](docs/reproduction.md) for the table-by-table
reproduction guide (which config + which command produces which paper Table
or Figure, with estimated runtime and expected numbers).

## Benchmarks supported

- **BIRD** (1534 dev queries, 11 DBs)
- **Dr.Spider-Abbr** (2853 queries, 90 DBs) — main benchmark for schema
  ambiguity
- **Dr.Spider-Syn** (2619 queries, 92 DBs) — supplementary
- **BEAVER (NW)** (88 queries, 5 DBs) — enterprise-grade

Setup details in [`docs/benchmarks.md`](docs/benchmarks.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

## Contact

Wang Jiaqian — `24151111272@stu.xidian.edu.cn` — Xidian University

Issues and pull requests welcome at
[github.com/ai-jiaqian/EGRefine/issues](https://github.com/ai-jiaqian/EGRefine/issues).
