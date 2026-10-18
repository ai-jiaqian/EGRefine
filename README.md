# EGRefine

Execution-Grounded Schema Refinement for Text-to-SQL.

This anonymous review artifact contains the implementation, regression tests,
example configurations, and refinement/evaluation scripts. It is a fixed
snapshot. Benchmarks, model weights, credentials, and private run logs are not
bundled.

## Install

Download and extract the ZIP from this anonymous repository, then enter its
root directory (the directory containing `pyproject.toml`). Python 3.10+ is
required; this snapshot is checked with Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest tests -q
egrefine-refine --help
egrefine-eval --help
```

Editable installation from the extracted source directory is supported. No
Git history or third-party repository checkout is needed. Tests for external
BIRD/BEAVER data are skipped unless the corresponding data is configured;
see benchmark setup (`docs/benchmarks.md`). Unit and pipeline tests use mocks
and temporary SQLite databases, without issuing real model requests.

## Run

Copy `config/revision_qwen35_27b.yaml` to `config/local.yaml`, then supply your
OpenAI-compatible model endpoint, model name, API key and dataset paths. Keep
`enable_thinking: false` for Qwen3.5 in each model configuration. Configuration
values are literal YAML; environment-variable placeholders are not expanded.

```bash
egrefine-refine --config config/local.yaml --benchmark bird \
  --dbs formula_1 --verify-algorithms c3+dinsql --output results/refine_demo

egrefine-eval --config config/local.yaml --benchmark bird \
  --dbs formula_1 --schema both --refine-dir results/refine_demo \
  --methods c3 dinsql macsql --full-eval --output results/eval_demo
```

The four phases screen columns, propose alternative names, verify candidates
by Text-to-SQL execution, and publish SQL VIEWs. Original database tables are
not renamed in place: evaluation constructs isolated database copies.

`macsql` uses the pinned upstream MAC-SQL flow and templates; `macsql_legacy`
retains the earlier approximation. `--full-eval` evaluates every query,
including databases with no accepted changes, and cannot be combined with
`--reuse-original-dir`.

See running experiments (`docs/reproduction.md`), quickstart (`docs/quickstart.md`),
and architecture (`docs/architecture.md`). Full experiments require your own
model endpoint and datasets; unit-test success does not establish reproduction
of every numerical result in the paper.

## License

Apache License 2.0; see LICENSE (`LICENSE`). Third-party dataset and model
licenses remain applicable to those separately obtained resources.
