# Quickstart

1. Download this anonymous artifact and follow the installation commands in
   [README](../README.md).
2. Obtain BIRD-dev from the official source described in
   [benchmark setup](benchmarks.md).
3. Copy `config/revision_qwen35_27b.yaml` to `config/local.yaml`. Set the
   endpoint, API key, exact model identifier and BIRD dataset path. The dataset
   directory should contain `dev.json` and `dev_databases/`.
4. Run the single-database refinement and complete evaluation commands in
   [README](../README.md).
5. Inspect `results/refine_demo/formula_1/` for `views.sql`,
   `refined_tables.json`, `orig_table_map.json`, `statistics.json` and
   `pipeline_result.json`. Inspect evaluation `per_db/` records for each method.

Runtime and accuracy depend on the serving model, endpoint load and dataset.
Measure a small run before committing to a full experiment. Keep all arms on
the same model and evaluation configuration.
