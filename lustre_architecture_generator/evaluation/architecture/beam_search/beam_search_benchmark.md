# Beam Search: measured controlled runtime evaluation

Data: existing architecture dataset and official LightGBM handoffs. These cases are controlled/synthetic, not production generalization evidence.

Cases: REQ_000001, REQ_000002, REQ_000003. Paths/variant: 1. Repeats: 2. Baseline pair limit: 20000.

Time excludes ranking and exhaustive domain preflight; both are recorded separately in JSON. Exhaustive time includes full H8 generation, H9, and H10 for every pair. Beam time includes its input validation, all search stages, H9, and H10.

Quality locates the actual Beam winner in the exhaustive H9 pool, including invalid architectures in normalization; the reference winner/rank uses only H10 VALID architectures. Quality = common-pool Beam score / common-pool best VALID score. Scores from different pools are never compared. Missing comparisons remain absent.

Times are means of per-case medians. Counts are totals across cases. Speedup is the mean of per-case runtime ratios.

| K | B | Feasible rate | Time/case (s) | Branches | Pruned | H10 | VALID | Quality | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 66.7% | 0.004224 | 72 | 55 | 2 | 2 | 0.9043 | 318.0x |
| 5 | 2 | 66.7% | 0.007558 | 129 | 95 | 4 | 4 | 0.9043 | 177.2x |
| 5 | 4 | 66.7% | 0.012294 | 243 | 175 | 8 | 8 | 0.9021 | 114.5x |
| 5 | 8 | 66.7% | 0.023475 | 417 | 290 | 16 | 16 | 0.9101 | 61.0x |
| 5 | 16 | 66.7% | 0.047564 | 729 | 492 | 30 | 30 | 0.9046 | 29.8x |
| 5 | 32 | 66.7% | 0.102970 | 1317 | 872 | 58 | 58 | 0.9046 | 13.7x |
| 10 | 1 | 66.7% | 0.004447 | 102 | 85 | 2 | 2 | 0.9436 | 1349.3x |
| 10 | 2 | 66.7% | 0.007948 | 174 | 140 | 4 | 4 | 0.9436 | 768.0x |
| 10 | 4 | 66.7% | 0.016896 | 318 | 250 | 8 | 8 | 0.9396 | 385.8x |
| 10 | 8 | 66.7% | 0.033612 | 606 | 470 | 16 | 16 | 0.9440 | 191.7x |
| 10 | 16 | 66.7% | 0.070595 | 1074 | 822 | 30 | 30 | 0.9382 | 91.1x |
| 10 | 32 | 66.7% | 0.146433 | 1938 | 1466 | 58 | 58 | 0.9382 | 42.3x |
| 20 | 1 | 66.7% | 0.006355 | 162 | 145 | 2 | 2 | 0.9261 | 5485.5x |
| 20 | 2 | 66.7% | 0.010965 | 264 | 230 | 4 | 4 | 0.9261 | 3211.8x |
| 20 | 4 | 66.7% | 0.021204 | 468 | 400 | 8 | 8 | 0.9208 | 1660.3x |
| 20 | 8 | 66.7% | 0.041445 | 876 | 740 | 16 | 16 | 0.9216 | 858.2x |
| 20 | 16 | 66.7% | 0.087863 | 1692 | 1422 | 30 | 30 | 0.9153 | 400.4x |
| 20 | 32 | 66.7% | 0.196823 | 3108 | 2606 | 58 | 58 | 0.9153 | 178.0x |
| 50 | 1 | 66.7% | 0.010904 | 342 | 325 | 2 | 2 | n/a | n/a |
| 50 | 2 | 66.7% | 0.019623 | 534 | 500 | 4 | 4 | n/a | n/a |
| 50 | 4 | 66.7% | 0.036293 | 918 | 850 | 8 | 8 | n/a | n/a |
| 50 | 8 | 66.7% | 0.071876 | 1686 | 1550 | 16 | 16 | n/a | n/a |
| 50 | 16 | 66.7% | 0.155353 | 3222 | 2952 | 30 | 30 | n/a | n/a |
| 50 | 32 | 66.7% | 0.391835 | 6294 | 5756 | 58 | 58 | n/a | n/a |

## Exhaustive reference runs

| Case | K | Status | Pairs | Time (s) | H10 | VALID | Best architecture |
|---|---:|---|---:|---:|---:|---:|---|
| REQ_000001 | 5 | COMPLETE | 720 | 1.1966893000062555 | 720 | 420 | ARCH_REQ_000001_1d38d46c2a0de81f |
| REQ_000001 | 10 | COMPLETE | 2520 | 4.521609599934891 | 2520 | 1460 | ARCH_REQ_000001_b90146e139bf3b22 |
| REQ_000001 | 20 | COMPLETE | 11520 | 29.583778299973346 | 11520 | 6384 | ARCH_REQ_000001_0939aa29ff9046a4 |
| REQ_000001 | 50 | SKIPPED_PAIR_LIMIT | 79200 | n/a | n/a | n/a | n/a |
| REQ_000002 | 5 | COMPLETE | 900 | 1.3800379999447614 | 900 | 780 | ARCH_REQ_000002_df283e6366b9efab |
| REQ_000002 | 10 | COMPLETE | 3600 | 6.317825000034645 | 3600 | 3360 | ARCH_REQ_000002_d7a95f563342ce56 |
| REQ_000002 | 20 | COMPLETE | 13680 | 36.608453900087625 | 13680 | 13200 | ARCH_REQ_000002_f2f6913af4a3dbe9 |
| REQ_000002 | 50 | SKIPPED_PAIR_LIMIT | 86400 | n/a | n/a | n/a | n/a |
| REQ_000003 | 5 | COMPLETE | 900 | 1.3858445000369102 | 900 | 90 | ARCH_REQ_000003_d6d34b86f177b1cd |
| REQ_000003 | 10 | COMPLETE | 3600 | 6.268390699988231 | 3600 | 180 | ARCH_REQ_000003_d6d34b86f177b1cd |
| REQ_000003 | 20 | COMPLETE | 13680 | 35.28386219998356 | 13680 | 360 | ARCH_REQ_000003_d6d34b86f177b1cd |
| REQ_000003 | 50 | SKIPPED_PAIR_LIMIT | 86400 | n/a | n/a | n/a | n/a |

## Scope and limitations

`SKIPPED_PAIR_LIMIT` means no exhaustive quality/time comparison was executed for that domain. The pair limit never truncates a baseline and never limits Beam. A missing Beam solution does not prove global infeasibility. An increased width need not monotonically improve quality or feasibility. The physical catalog and per-variant path cap constrain both methods.
