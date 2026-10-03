# Beam Search: measured controlled runtime evaluation

Data: existing architecture dataset and official LightGBM handoffs. These cases are controlled/synthetic, not production generalization evidence.

Cases: REQ_000003. Paths/variant: 1. Repeats: 2. Baseline pair limit: 20000.

Time excludes ranking and exhaustive domain preflight; both are recorded separately in JSON. Exhaustive time includes full H8 generation, H9, and H10 for every pair. Beam time includes its input validation, all search stages, H9, and H10.

Quality locates the actual Beam winner in the exhaustive H9 pool, including invalid architectures in normalization; the reference winner/rank uses only H10 VALID architectures. Quality = common-pool Beam score / common-pool best VALID score. Scores from different pools are never compared. Missing comparisons remain absent.

Times are means of per-case medians. Counts are totals across cases. Speedup is the mean of per-case runtime ratios.

| K | B | Feasible rate | Time/case (s) | Branches | Pruned | H10 | VALID | Quality | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 100.0% | 0.006668 | 154 | 45 | 1 | 1 | 0.9748 | 208.0x |
| 5 | 2 | 100.0% | 0.008633 | 173 | 58 | 2 | 2 | 0.9758 | 160.7x |
| 5 | 4 | 100.0% | 0.013238 | 211 | 84 | 4 | 4 | 1.0000 | 104.8x |
| 5 | 8 | 100.0% | 0.022153 | 269 | 121 | 8 | 8 | 0.9758 | 62.6x |
| 5 | 16 | 100.0% | 0.041410 | 373 | 185 | 16 | 16 | 0.9999 | 33.5x |
| 5 | 32 | 100.0% | 0.076580 | 557 | 295 | 32 | 32 | 0.9602 | 18.1x |
| 5 | 64 | 100.0% | 0.128772 | 589 | 263 | 64 | 64 | 0.9999 | 10.8x |
| 5 | 1024 | 100.0% | 0.171802 | 615 | 237 | 90 | 90 | 0.9999 | 8.1x |

## Exhaustive reference runs

| Case | K | Status | Pairs | Time (s) | H10 | VALID | Best architecture |
|---|---:|---|---:|---:|---:|---:|---|
| REQ_000003 | 5 | COMPLETE | 900 | 1.3870071000419557 | 900 | 90 | ARCH_REQ_000003_d6d34b86f177b1cd |

## Common-pool quality and loss attribution

Search regret measures the best VALID missing from the retained pool. Normalization regret measures a different local H9 winner inside that retained pool. Their sum equals total common-pool regret.

| Case | K | B | VALID rank | Quality | Best survived | Best chosen | Search regret | Normalization regret |
|---|---:|---:|---:|---:|---|---|---:|---:|
| REQ_000003 | 5 | 1 | 11 | 0.974806 | False | False | 0.020944738 | 0.000000000 |
| REQ_000003 | 5 | 2 | 5 | 0.975803 | False | False | 0.020115951 | 0.000000000 |
| REQ_000003 | 5 | 4 | 1 | 1.000000 | True | True | 0.000000000 | 0.000000000 |
| REQ_000003 | 5 | 8 | 6 | 0.975790 | True | False | 0.000000000 | 0.020127040 |
| REQ_000003 | 5 | 16 | 2 | 0.999853 | True | False | 0.000000000 | 0.000122498 |
| REQ_000003 | 5 | 32 | 22 | 0.960247 | True | False | 0.000000000 | 0.033047879 |
| REQ_000003 | 5 | 64 | 2 | 0.999853 | True | False | 0.000000000 | 0.000122498 |
| REQ_000003 | 5 | 1024 | 2 | 0.999853 | True | False | 0.000000000 | 0.000122498 |

## Scope and limitations

`SKIPPED_PAIR_LIMIT` means no exhaustive quality/time comparison was executed for that domain. The pair limit never truncates a baseline and never limits Beam. A missing Beam solution does not prove global infeasibility. An increased width need not monotonically improve quality or feasibility. The physical catalog and per-variant path cap constrain both methods.
