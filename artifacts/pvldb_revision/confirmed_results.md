# Confirmed Results in the PVLDB Revision

All values below are copied from the submitted revision PDF. Accuracy and
delta values are percentages or percentage points as indicated by each table.
No entry is a target or placeholder.

## Table 1: Dr.Spider Schema-Abbreviation

| Method | C3 | DIN-SQL | MAC-SQL |
|---|---:|---:|---:|
| Clean Spider | 72.77 | 80.41 | 63.79 |
| Degraded | 70.87 | 76.73 | 59.59 |
| LLM-Direct | 72.34 | 75.85 | 57.17 |
| SNAILS-style | 67.33 | 73.33 | 58.15 |
| EGRefine | 73.43 | 79.21 | 60.22 |
| EGRefine - Degraded | +2.56 | +2.48 | +0.63 |
| Recovery rate | 134.7% | 67.4% | 15.0% |

## Table 2: Offline preprocessing cost

| Phase | LLM requests | Execution comparisons | Accumulated time |
|---|---:|---:|---:|
| Screening | 1,605 | 0 | 5.7 min |
| Candidate generation | 665 | 0 | 56.8 min |
| Verification | 112,449 | 112,449 | 34.2 h |
| VIEW synthesis | 0 | 0 | seconds |
| Total | 114,719 | 112,449 | 35.2 h |

## Table 3: Component ablation

| Variant | C3 | DIN-SQL | MAC-SQL |
|---|---:|---:|---:|
| EGRefine, M=2 | 73.43 | 79.21 | 60.22 |
| Without execution | 72.34 | 75.85 | 57.17 |
| Without conservative rule | 73.01 | 78.83 | 58.92 |
| Without screening | 71.08 | 77.32 | 58.78 |
| Without multi-algorithm verification | 73.36 | 76.34 | 59.48 |
| Degraded | 70.87 | 76.73 | 59.59 |

## Table 4: Threshold sensitivity

| tau | Commits | Delta C3 | Delta DIN-SQL | Delta MAC-SQL |
|---:|---:|---:|---:|---:|
| 0.01 | 104 | +0.70 | -0.25 | -0.81 |
| 0.03 | 89 | +0.81 | +1.15 | -0.11 |
| 0.05 | 81 | +2.56 | +2.48 | +0.63 |
| 0.10 | 54 | +0.70 | +0.94 | +0.56 |

## Table 5: Cross-verifier robustness

| Verifier set | Algorithm | ExAcc | Delta | Recovery |
|---|---|---:|---:|---:|
| C3+DIN | C3 | 73.43 | +2.56 | 134.7% |
| C3+DIN | DIN-SQL | 79.21 | +2.48 | 67.4% |
| C3+DIN | MAC-SQL | 60.22 | +0.63 | 15.0% |
| DIN+MAC | C3 | 75.34 | +4.47 | 235.3% |
| DIN+MAC | DIN-SQL | 79.87 | +3.14 | 85.2% |
| DIN+MAC | MAC-SQL | 63.17 | +3.58 | 85.2% |

## Table 6: Candidate-count and sample-row ablation

NoRef on this endpoint is 70.10 C3 / 76.45 DIN-SQL.

| Setting | Commits | LLM calls | C3 | DIN-SQL |
|---|---:|---:|---:|---:|
| k=1 | 53 | 58,359 | 72.24 | 78.79 |
| k=2 | 54 | 85,924 | 71.99 | 79.14 |
| k=3 | 80 | 114,719 | 71.96 | 79.50 |
| k=5 | 96 | 159,644 | 72.38 | 79.36 |
| N_s=5 | 80 | 108,609 | 72.38 | 79.39 |

## Table 7: Refinement/evaluation transfer matrix

| Refiner | Evaluator | C3 (Delta) | DIN-SQL (Delta) | MAC-SQL (Delta) |
|---|---|---:|---:|---:|
| 9B | 9B | 63.34 (+3.96) | 75.81 (+5.18) | 53.63 (+2.11) |
| 27B | 9B | 62.36 (+2.98) | 75.60 (+4.97) | 53.49 (+1.97) |
| 9B | 27B | 73.22 (+2.35) | 79.53 (+2.80) | 59.34 (-0.25) |
| 27B | 27B | 73.43 (+2.56) | 79.21 (+2.48) | 60.22 (+0.63) |

## Table 8: Serving-backbone transfer

| Backbone | Schema | C3 | DIN-SQL | MAC-SQL |
|---|---|---:|---:|---:|
| Qwen3.5-9B | NoRef | 59.38 | 70.63 | 51.52 |
| Qwen3.5-9B | Self-refined | 63.34 | 75.81 | 53.63 |
| Qwen3.5-9B | 27B-refined | 62.36 | 75.60 | 53.49 |
| Phi-4-14B | NoRef | 56.85 | 69.54 | 34.00 |
| Phi-4-14B | Self-refined | 56.22 | 69.93 | 32.32 |
| Phi-4-14B | 27B-refined | 57.59 | 72.38 | 34.38 |
| Gemma3-27B | NoRef | 69.33 | 75.67 | 34.77 |
| Gemma3-27B | 27B-refined | 71.15 | 76.94 | 35.02 |

## Table 9: BIRD verification necessity

| Method | C3 | DIN-SQL | MAC-SQL |
|---|---:|---:|---:|
| NoRef | 28.75 | 31.75 | 20.60 |
| LLM-Direct | 28.23 | 30.12 | 19.17 |
| EGRefine | 30.25 | 33.25 | 20.14 |
| EGRefine - Direct | +2.02 | +3.13 | +0.97 |

## Table 10: BIRD evidence/refinement ablation

| Configuration | C3 | DIN-SQL | MAC-SQL |
|---|---:|---:|---:|
| NoRef | 41.53 | 38.59 | 29.99 |
| Evidence | 54.24 | 62.13 | 46.74 |
| EGRefine | 41.72 | 39.24 | 31.23 |
| Evidence + EGRefine | 54.82 | 63.56 | 47.33 |
| Both - Evidence | +0.58 | +1.43 | +0.59 |

## Table 11: Real-world deployment

| Benchmark/refiner | Schema | C3 | DIN-SQL | MAC-SQL |
|---|---|---:|---:|---:|
| BIRD 27B | NoRef | 41.53 | 38.59 | 29.99 |
| BIRD 27B | EGRefine | 41.72 | 39.24 | 31.23 |
| BIRD 9B | NoRef | 28.75 | 31.75 | 20.60 |
| BIRD 9B | EGRefine | 30.25 | 33.25 | 20.14 |
| BEAVER 27B | NoRef | 13.33 | 13.33 | 6.67 |
| BEAVER 27B | EGRefine, 0 commits | 13.33 | 13.33 | 6.67 |
| BEAVER MiniMax-M2.7 | NoRef | 13.33 | 13.33 | 6.67 |
| BEAVER MiniMax-M2.7 | EGRefine | 13.33 | 13.33 | 11.11 |

## Table 12: Unseen-query validation

| Benchmark | Schema | C3/DIN ExAcc | Split-NoRef delta | Split-NoRef p | Canonical-Split p |
|---|---|---|---|---|---|
| ABBR 70/30 | NoRef | 69.68/75.32 | - | - | - |
| ABBR 70/30 | Split | 70.62/75.68 | +0.94/+0.36 | 0.302/0.629 | - |
| ABBR 70/30 | Canonical | 70.51/76.38 | - | - | 1.000/0.146 |
| ABBR 50/50 | NoRef | 71.96/75.87 | - | - | - |
| ABBR 50/50 | Split | 72.66/78.88 | +0.70/+3.01 | 0.343/<0.001 | - |
| ABBR 50/50 | Canonical | 74.06/78.67 | - | - | 0.029/0.711 |
| BIRD 50/50 | NoRef | 41.17/39.22 | - | - | - |
| BIRD 50/50 | Split | 40.91/39.22 | -0.26/+0.00 | 0.888/1.000 | - |
| BIRD 50/50 | Canonical | 41.56/38.96 | - | - | 0.590/0.860 |

## Table 13: Workload-size ablation

| Workload | Commits | Delta C3 | Delta DIN-SQL |
|---|---:|---:|---:|
| 715 (25%) | 34 | +1.05 | +1.61 |
| 1,433 (50%) | 70 | +1.58 | +2.91 |
| 2,136 (75%) | 66 | +1.54 | +2.03 |
| 2,853 (100%) | 73 | +2.32 | +3.08 |

## Table 14: Identifier replacement vs. descriptions

| Schema | Descriptions | C3 ExAcc | Delta vs. NoRef |
|---|---|---:|---:|
| NoRef | No | 70.83 | - |
| NoRef | Yes | 72.48 | +1.65 |
| EGRefine | No | 73.46 | +2.63 |
| EGRefine | Yes | 74.69 | +3.86 |

## Table 15: Query-level flips

| Cell | Algorithm | EGRefine C-to-W | EGRefine W-to-C | Ratio | Direct C-to-W | Direct W-to-C | Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| ABBR 27B | C3 | 24 | 97 | 4.04 | 102 | 144 | 1.41 |
| ABBR 27B | DIN | 13 | 84 | 6.46 | 137 | 112 | 0.82 |
| ABBR 27B | MAC | 49 | 67 | 1.37 | 169 | 100 | 0.59 |
| ABBR 9B | C3 | 67 | 180 | 2.69 | 160 | 177 | 1.11 |
| ABBR 9B | DIN | 47 | 195 | 4.15 | 145 | 192 | 1.32 |
| ABBR 9B | MAC | 85 | 145 | 1.71 | 166 | 184 | 1.11 |
| BIRD 27B | C3 | 60 | 63 | 1.05 | 89 | 96 | 1.08 |
| BIRD 27B | DIN | 35 | 45 | 1.29 | 54 | 55 | 1.02 |
| BIRD 27B | MAC | 44 | 63 | 1.43 | 52 | 80 | 1.54 |
| BIRD 9B | C3 | 74 | 97 | 1.31 | 81 | 73 | 0.90 |
| BIRD 9B | DIN | 52 | 75 | 1.44 | 81 | 56 | 0.69 |
| BIRD 9B | MAC | 63 | 56 | 0.89 | 69 | 47 | 0.68 |

## Table 16: Touching-query decomposition

| Cell | Algorithm | Touching queries | Touching delta | Full delta | Ratio |
|---|---|---:|---:|---:|---:|
| ABBR 27B | C3 | 633 | +11.53 | +2.56 | 4.50x |
| ABBR 27B | DIN | 633 | +11.22 | +2.48 | 4.52x |
| ABBR 27B | MAC | 633 | +2.84 | +0.63 | 4.51x |
| ABBR 9B | C3 | 1,126 | +10.04 | +3.96 | 2.53x |
| ABBR 9B | DIN | 1,126 | +13.14 | +5.19 | 2.53x |
| ABBR 9B | MAC | 1,126 | +5.33 | +2.10 | 2.54x |
| BIRD 27B | C3 | 126 | -1.59 | +0.19 | - |
| BIRD 27B | DIN | 126 | +6.35 | +0.65 | 9.77x |
| BIRD 27B | MAC | 126 | -3.17 | +1.24 | - |
| BIRD 9B | C3 | 320 | +7.19 | +1.64 | 4.39x |
| BIRD 9B | DIN | 320 | +5.62 | +1.64 | 3.43x |
| 27B-to-9B | DIN | 633 | +21.48 | +8.95 | 2.40x |

## Table 17: Candidate outcomes

| Outcome | ABBR | BIRD | BEAVER |
|---|---:|---:|---:|
| Verified columns | 350 | 111 | 45 |
| Committed | 81 (23.1%) | 28 (25.2%) | 0 |
| Original retained | 269 (76.9%) | 83 (74.8%) | 45 (100%) |
| Original already perfect | 99 (36.8%) | 10 (12.0%) | 0 |
| Name-insensitive | 149 (55.4%) | 53 (63.9%) | 43 (95.6%) |
| Lexical proxy | 28 (8.0%) | 12 (10.8%) | - |

## Table 18: Forced-target query conditioning

| Benchmark | Temperature | Targets | Novel candidates | Rescues | Reversals | C3 held-out all/touching | DIN held-out all/touching |
|---|---:|---:|---:|---:|---:|---|---|
| ABBR | 0.7 | 28 | 40 | 4 | 1 | 0.00/0.00 | 0.00/0.00 |
| ABBR | 0 | 28 | 33 | 3 | 0 | -0.34/-2.63 | 0.00/0.00 |
| BIRD | 0.7 | 10 | 15 | 0 | 0 | 0.00/0.00 | 0.00/0.00 |
| BIRD | 0 | 10 | 14 | 0 | 0 | 0.00/0.00 | 0.00/0.00 |

## Table 19: Primary composition flips

| Column/database | Q | Isolation | Original spillover | Selected movement | Composed |
|---|---:|---:|---:|---:|---:|
| Treatments.cot / dog_kennels_1 | 12 | +0.125 | +0.208 | 0 | -0.083 |
| Treatments.dot / dog_kennels_4 | 6 | +0.167 | +0.250 | -0.083 | -0.167 |
| Cartoon.wr / tvshow_4 | 6 | +0.167 | +0.167 | -0.083 | -0.083 |
| TV_Channel.cntry / tvshow_5 | 12 | +0.125 | +0.208 | 0 | -0.083 |
| circuits.lng / formula_1 | 8 | +0.063 | +0.125 | -0.063 | -0.125 |

## Table 20: Screening/refinement funnel

| Cell | DBs | Columns | Phase-1 output | Renames | Exclusion | Rename coverage | Local delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| ABBR 27B | 90 | 1,985 | 665 | 81 | 66.5% | 4.08% | 20.68 |
| ABBR 9B | 90 | 1,985 | 841 | 151 | 57.6% | 7.61% | 20.94 |
| BIRD 27B | 11 | 798 | 251 | 28 | 68.5% | 3.51% | 26.62 |
| BIRD+evidence 27B | 11 | 798 | 252 | 28 | 68.4% | 3.51% | 21.76 |
| BIRD 9B | 11 | 798 | 308 | 39 | 61.4% | 4.89% | 19.95 |
| BEAVER 27B | 4 | 1,439 | 106 | 0 | 92.6% | 0.00% | - |
| BEAVER MiniMax-M2.7 | 4 | 1,439 | 172 | 4 | 88.0% | 0.28% | - |

## Table 21: Refiner strength

| Arm | C3 | Delta C3 | DIN-SQL | Delta DIN-SQL |
|---|---:|---:|---:|---:|
| NoRef | 70.10 | - | 76.45 | - |
| Direct, DeepSeek-V4-Flash | 70.38 | +0.28 n.s. | 76.94 | +0.49 n.s. |
| Direct, DeepSeek-V4-Pro | 70.00 | -0.11 n.s. | 72.59 | -3.86, p<0.001 |
| EGRefine, Flash candidates | 72.73 | +2.63, p<0.001 | 78.86 | +2.41, p<0.001 |
| EGRefine, Pro candidates | 72.80 | +2.70, p<0.001 | 79.39 | +2.94, p<0.001 |
| EGRefine, Qwen3.5-27B | 72.03 | +1.93, p<0.001 | 78.90 | +2.45, p<0.001 |

## Additional reported checks

- Independent-authoring BIRD holdout: 96 queries, 48.96 EGRefine vs. 47.92
  NoRef, +1.04 pp, 95% CI [-3.1, +5.2].
- Role-preserving PK-to-FK comparison on 222 queries: C3 66.7 vs. 65.3;
  DIN-SQL 57.7 vs. 57.7; p=0.61/1.00.
- Composition audit: 109 commits total; 93 non-trivial; 49 replicate their
  isolation gain, with 30 positive, 14 neutral, and 5 negative under
  composition.
- BEAVER failure decomposition: 96.3% before complete schema assembly, 2.9%
  after assembly but with logic failure, and 0.8% truncation.
