# 2026-10-18 runtime synchronization

The research runtime and public package share the same implementation. The
research package prefix is `src`; the public prefix is `egrefine`. The public
package keeps its existing editable-install CLI and portable example configs.

Changes:

- `macsql` uses the MAC-SQL flow and templates from upstream commit
  `31a9df5e0d520be4769be57a4b9022e5e34a14f4`; the previous approximation remains
  available as `macsql_legacy`.
- C3 retains foreign-key columns during schema filtering, passes sampling
  temperature per call, and backmaps candidate SQL before Phase 3 execution voting.
- Renamed schemas update both ends of foreign-key metadata; DIN-SQL extracts
  multiline SQL; the simple baseline displays actual SQLite rows.
- Optional descriptions reach C3, DIN-SQL, simple and MAC-SQL prompts.
- `--full-eval` includes databases with no renamed columns and rejects partial
  reuse. Forced target replay and no-pruning ablations retain structural exclusions.
- `EGREFINE_RAW_LOG` records raw model responses; `EGREFINE_HTTP_RAW_LOG` records
  individual HTTP attempts, including retries. Set either to a writable file path
  with an existing parent directory. These logs may contain input data and stay local.
- Existing public Phase 4 collision handling is retained and was also copied back
  to the research runtime.

## Installation and checks

Use Python 3.10+; synchronization was validated on Python 3.12.

```bash
pip install -e '.[dev]'
python -m pytest tests -q
egrefine-refine --help
egrefine-eval --help
```

Tests use local fixtures and mocks by default. Real BIRD and BEAVER loader tests
require `EGREFINE_TEST_BIRD_PATH` and `EGREFINE_TEST_BEAVER_PATH`; MySQL checks also
need imported BEAVER databases. Missing external datasets are reported as skips.
The MAC-SQL template test checks raw and rendered SHA-256 values independently
recorded from the pinned upstream source, so a third-party checkout is unnecessary.

No new model benchmark was launched for this synchronization. Historical PVLDB
results remain historical artifacts, not measurements of this updated runtime.
Research drafts, internal deployment scripts, credentials and run data are excluded
from the public distribution.
