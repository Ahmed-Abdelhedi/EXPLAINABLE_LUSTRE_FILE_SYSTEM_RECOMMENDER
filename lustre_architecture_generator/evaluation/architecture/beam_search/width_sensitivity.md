# Beam Search: measured controlled runtime evaluation

Data: existing architecture dataset and official LightGBM handoffs. These cases are controlled/synthetic, not production generalization evidence.

Cases: REQ_000003. Paths/variant: 1. Repeats: 2. Baseline pair limit: 20000.

Time excludes ranking and exhaustive domain preflight; both are recorded separately in JSON. Exhaustive time includes full H8 generation, H9, and H10 for every pair. Beam time includes its input validation, all search stages, H9, and H10.

Quality locates the actual Beam winner in the exhaustive H9 pool, including invalid architectures in normalization; the reference winner/rank uses only H10 VALID architectures. Quality = common-pool Beam score / common-pool best VALID score. Scores from different pools are never compared. Missing comparisons remain absent.

Times are means of per-case medians. Counts are totals across cases. Speedup is the mean of per-case runtime ratios.

| K | B | Feasible rate | Time/case (s) | Branches | Pruned | H10 | VALID | Quality | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 64 | 0.0% | 0.138898 | 663 | 470 | 0 | 0 | n/a | 10.1x |
| 5 | 128 | 0.0% | 0.347565 | 1111 | 790 | 0 | 0 | n/a | 4.1x |
| 5 | 256 | 100.0% | 0.595957 | 1371 | 886 | 14 | 14 | 0.9756 | 2.4x |
| 5 | 1024 | 100.0% | 1.191099 | 2015 | 810 | 90 | 90 | 0.9999 | 1.2x |

## Exhaustive reference runs

| Case | K | Status | Pairs | Time (s) | H10 | VALID | Best architecture |
|---|---:|---|---:|---:|---:|---:|---|
| REQ_000003 | 5 | COMPLETE | 900 | 1.4082164000719786 | 900 | 90 | ARCH_REQ_000003_d6d34b86f177b1cd |

## Scope and limitations

`SKIPPED_PAIR_LIMIT` means no exhaustive quality/time comparison was executed for that domain. The pair limit never truncates a baseline and never limits Beam. A missing Beam solution does not prove global infeasibility. An increased width need not monotonically improve quality or feasibility. The physical catalog and per-variant path cap constrain both methods.
