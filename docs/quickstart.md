# Quickstart: refine your first database in 10 minutes

This walkthrough takes a fresh checkout to a measurable EGRefine run on the
`formula_1` database from BIRD-dev. Total wall-clock time is ~10 minutes
(most spent waiting on the LLM).

## Prerequisites

- Python 3.10 or later
- ~10 GB free disk for BIRD-dev
- An OpenAI-compatible LLM endpoint:
  - **Option A (recommended):** local vLLM serving a coder model
  - **Option B:** OpenAI API with `OPENAI_API_KEY` set

## Step 1: Install

```bash
git clone https://github.com/ai-jiaqian/EGRefine
cd EGRefine
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify:

```bash
egrefine-refine --help
egrefine-eval --help
python -c "import egrefine; print(egrefine.__version__)"
```

## Step 2: Set up the LLM endpoint

### Option A — Local vLLM

In a separate terminal:

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.92
```

Wait until you see `INFO: Uvicorn running on http://0.0.0.0:8000`.

### Option B — OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

Use `config/example_openai.yaml` instead of `config/example_local_vllm.yaml`
in all commands below.

## Step 3: Download BIRD-dev

```bash
# Download from the official BIRD release page:
#   https://bird-bench.github.io/
# Place the unzipped contents at /path/to/BIRD/ such that:
#   /path/to/BIRD/dev_20240627/dev.json
#   /path/to/BIRD/dev_20240627/dev_databases/formula_1/formula_1.sqlite
```

Edit `config/example_local_vllm.yaml`:

```yaml
data:
  bird:
    path: "/path/to/BIRD/dev_20240627"   # <-- update this
```

## Step 4: Run refinement

```bash
egrefine-refine \
    --config config/example_local_vllm.yaml \
    --benchmark bird \
    --dbs formula_1 \
    --output ./results/refine_demo
```

What happens (~5 minutes):
- Phase 1 picks ~10-15 candidate columns from `formula_1`'s schema
- Phase 2 calls the LLM 3 times per candidate column (k=3 candidate names)
- Phase 3 verifies each candidate by running the downstream Text-to-SQL on
  the queries that reference each column
- Phase 4 emits `views.sql`

Check the output:

```bash
ls results/refine_demo/formula_1/
# views.sql  refined_tables.json  orig_table_map.json  statistics.json
cat results/refine_demo/formula_1/views.sql
```

## Step 5: Evaluate the no-refinement baseline

```bash
egrefine-eval \
    --config config/example_local_vllm.yaml \
    --benchmark bird \
    --dbs formula_1 \
    --schema original \
    --methods c3 \
    --output ./results/eval_noref
```

## Step 6: Evaluate the refined schema

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

## Step 7: Compare ExAcc

```bash
python -c "
import json
n = json.load(open('results/eval_noref/results.json'))
r = json.load(open('results/eval_refined/results.json'))
print(f'NoRef ExAcc:   {n[\"exacc\"]:.3f}')
print(f'Refined ExAcc: {r[\"exacc\"]:.3f}')
print(f'Delta:         {r[\"exacc\"] - n[\"exacc\"]:+.3f}')
"
```

If `Refined ExAcc > NoRef ExAcc`, EGRefine improved this database.
For `formula_1` with Qwen2.5-Coder-32B as backbone we typically see
+2 to +5 percentage points.

## Next steps

- Run on more BIRD databases (drop the `--dbs formula_1` to run on all 11)
- Try a different downstream model: `--methods dinsql` or `--methods macsql`
- Reproduce paper Tables: see [`reproduction.md`](reproduction.md)
- Understand what each phase does: see [`architecture.md`](architecture.md)
