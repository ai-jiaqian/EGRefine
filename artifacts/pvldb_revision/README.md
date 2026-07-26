# PVLDB Revision Artifact

This directory is the public result snapshot for the PVLDB revision of
**EGRefine: An Execution-Grounded Optimization Framework for Text-to-SQL
Schema Refinement**.

## Contents

- [`../../EGRefine-PVLDB.pdf`](../../EGRefine-PVLDB.pdf): submitted revision
  PDF. SHA-256:
  `f507afc66dc638190b7ab3626acbf227ba9d90d1c280755e62af4a7c42d1332d`.
- [`confirmed_results.md`](confirmed_results.md): exact values reported in
  Tables 1--21 of the revision.
- [`../../docs/reproduction.md`](../../docs/reproduction.md): commands and
  table-to-script mapping.
- [`../../config/revision_qwen35_27b.yaml`](../../config/revision_qwen35_27b.yaml):
  sanitized default configuration. Dataset paths and the OpenAI-compatible
  endpoint are placeholders.

## Reproducibility boundary

The repository distributes source code, sanitized configurations, analysis
scripts, and confirmed aggregate results. BIRD, Dr.Spider, BEAVER, SNAILS,
and model weights are third-party artifacts and are not redistributed; their
official download locations and expected layouts are documented in
[`../../docs/benchmarks.md`](../../docs/benchmarks.md).

Results were produced with the model checkpoints named in the paper. Local
OpenAI-compatible endpoints may expose different model identifiers, so
reproduction requires mapping the public checkpoint name to the local server's
identifier without changing decoding parameters.

## Revision-specific implementations

The public code includes the method and audit changes used by the revision:

- role-preserving PK-to-FK propagation and a Phase-4 duplicate-name
  post-condition;
- read-only SQLite verification;
- forced target sets and optional workload-conditioned Phase-2 generation;
- shared-score query-conditioning analysis;
- composed-schema re-verification;
- threshold re-selection without repeating Phase 3;
- the SNAILS-style baseline and refiner-strength plotting utilities.

The blue text in the revision PDF identifies changes relative to the original
submission. The response letter is not part of the public artifact.
