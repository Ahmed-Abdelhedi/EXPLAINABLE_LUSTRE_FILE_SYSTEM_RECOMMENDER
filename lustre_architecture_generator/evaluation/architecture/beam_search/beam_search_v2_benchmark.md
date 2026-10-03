# Beam Search: measured controlled runtime evaluation

Data: existing architecture dataset and official LightGBM handoffs. These cases are controlled/synthetic, not production generalization evidence.

Cases: REQ_000001, REQ_000002, REQ_000003. Paths/variant: 1. Repeats: 2. Baseline pair limit: 20000.

Time excludes ranking and exhaustive domain preflight; both are recorded separately in JSON. Exhaustive time includes full H8 generation, H9, and H10 for every pair. Beam time includes its input validation, all search stages, H9, and H10.

Quality locates the actual Beam winner in the exhaustive H9 pool, including invalid architectures in normalization; the reference winner/rank uses only H10 VALID architectures. Quality = common-pool Beam score / common-pool best VALID score. Scores from different pools are never compared. Missing comparisons remain absent.

Times are means of per-case medians. Counts are totals across cases. Speedup is the mean of per-case runtime ratios.

| K | B | Feasible rate | Time/case (s) | Branches | Pruned | H10 | VALID | Quality | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 8 | 100.0% | 0.029008 | 805 | 320 | 24 | 24 | 0.9883 | 46.8x |
| 5 | 16 | 100.0% | 0.050722 | 1117 | 512 | 48 | 48 | 0.9979 | 26.0x |
| 5 | 32 | 100.0% | 0.096340 | 1693 | 862 | 96 | 96 | 0.9847 | 13.7x |
| 10 | 8 | 100.0% | 0.041707 | 1377 | 545 | 24 | 24 | 0.9978 | 151.4x |
| 10 | 16 | 100.0% | 0.064520 | 1845 | 887 | 48 | 48 | 0.9897 | 93.1x |
| 10 | 32 | 100.0% | 0.125614 | 2709 | 1511 | 96 | 96 | 0.9967 | 48.1x |
| 20 | 8 | 100.0% | 0.067587 | 2408 | 899 | 24 | 24 | 0.9875 | 536.0x |
| 20 | 16 | 100.0% | 0.094405 | 3224 | 1571 | 48 | 48 | 0.9946 | 386.6x |
| 20 | 32 | 100.0% | 0.168664 | 4640 | 2735 | 96 | 96 | 0.9945 | 218.4x |

## Exhaustive reference runs

| Case | K | Status | Pairs | Time (s) | H10 | VALID | Best architecture |
|---|---:|---|---:|---:|---:|---:|---|
| REQ_000001 | 5 | COMPLETE | 720 | 1.1262338999658823 | 720 | 420 | ARCH_REQ_000001_1d38d46c2a0de81f |
| REQ_000001 | 10 | COMPLETE | 2520 | 4.083271999959834 | 2520 | 1460 | ARCH_REQ_000001_b90146e139bf3b22 |
| REQ_000001 | 20 | COMPLETE | 11520 | 26.97358779993374 | 11520 | 6384 | ARCH_REQ_000001_0939aa29ff9046a4 |
| REQ_000002 | 5 | COMPLETE | 900 | 1.322776599903591 | 900 | 780 | ARCH_REQ_000002_df283e6366b9efab |
| REQ_000002 | 10 | COMPLETE | 3600 | 6.158963400055654 | 3600 | 3360 | ARCH_REQ_000002_d7a95f563342ce56 |
| REQ_000002 | 20 | COMPLETE | 13680 | 34.88286519993562 | 13680 | 13200 | ARCH_REQ_000002_f2f6913af4a3dbe9 |
| REQ_000003 | 5 | COMPLETE | 900 | 1.3583886999404058 | 900 | 90 | ARCH_REQ_000003_d6d34b86f177b1cd |
| REQ_000003 | 10 | COMPLETE | 3600 | 6.187223499990068 | 3600 | 180 | ARCH_REQ_000003_d6d34b86f177b1cd |
| REQ_000003 | 20 | COMPLETE | 13680 | 36.15018240001518 | 13680 | 360 | ARCH_REQ_000003_d6d34b86f177b1cd |

## Common-pool quality and loss attribution

Search regret measures the best VALID missing from the retained pool. Normalization regret measures a different local H9 winner inside that retained pool. Their sum equals total common-pool regret.

| Case | K | B | VALID rank | Quality | Best survived | Best chosen | Search regret | Normalization regret |
|---|---:|---:|---:|---:|---|---|---:|---:|
| REQ_000001 | 5 | 8 | 10 | 0.992801 | False | False | 0.003736095 | 0.002059764 |
| REQ_000001 | 5 | 16 | 2 | 0.997442 | True | False | 0.000000000 | 0.002059764 |
| REQ_000001 | 5 | 32 | 2 | 0.997442 | False | False | 0.002059764 | 0.000000000 |
| REQ_000001 | 10 | 8 | 4 | 0.995545 | True | False | 0.000000000 | 0.003655433 |
| REQ_000001 | 10 | 16 | 4 | 0.995545 | True | False | 0.000000000 | 0.003655433 |
| REQ_000001 | 10 | 32 | 9 | 0.992628 | True | False | 0.000000000 | 0.006049042 |
| REQ_000001 | 20 | 8 | 3 | 0.989220 | True | False | 0.000000000 | 0.009000584 |
| REQ_000001 | 20 | 16 | 5 | 0.984729 | True | False | 0.000000000 | 0.012749673 |
| REQ_000001 | 20 | 32 | 5 | 0.984729 | True | False | 0.000000000 | 0.012749673 |
| REQ_000002 | 5 | 8 | 5 | 0.996316 | True | False | 0.000000000 | 0.003119406 |
| REQ_000002 | 5 | 16 | 5 | 0.996316 | True | False | 0.000000000 | 0.003119406 |
| REQ_000002 | 5 | 32 | 5 | 0.996316 | True | False | 0.000000000 | 0.003119406 |
| REQ_000002 | 10 | 8 | 3 | 0.997743 | False | False | 0.001630603 | 0.000276200 |
| REQ_000002 | 10 | 16 | 3 | 0.997743 | True | False | 0.000000000 | 0.001906803 |
| REQ_000002 | 10 | 32 | 3 | 0.997743 | True | False | 0.000000000 | 0.001906803 |
| REQ_000002 | 20 | 8 | 2 | 0.999047 | True | False | 0.000000000 | 0.000805921 |
| REQ_000002 | 20 | 16 | 2 | 0.999047 | True | False | 0.000000000 | 0.000805921 |
| REQ_000002 | 20 | 32 | 2 | 0.999047 | True | False | 0.000000000 | 0.000805921 |
| REQ_000003 | 5 | 8 | 6 | 0.975790 | True | False | 0.000000000 | 0.020127040 |
| REQ_000003 | 5 | 16 | 2 | 0.999853 | True | False | 0.000000000 | 0.000122498 |
| REQ_000003 | 5 | 32 | 22 | 0.960247 | True | False | 0.000000000 | 0.033047879 |
| REQ_000003 | 10 | 8 | 1 | 1.000000 | True | True | 0.000000000 | 0.000000000 |
| REQ_000003 | 10 | 16 | 11 | 0.975813 | True | False | 0.000000000 | 0.020161058 |
| REQ_000003 | 10 | 32 | 2 | 0.999833 | True | False | 0.000000000 | 0.000139048 |
| REQ_000003 | 20 | 8 | 36 | 0.974188 | False | False | 0.020103506 | 0.000873399 |
| REQ_000003 | 20 | 16 | 1 | 1.000000 | True | True | 0.000000000 | 0.000000000 |
| REQ_000003 | 20 | 32 | 2 | 0.999829 | True | False | 0.000000000 | 0.000139135 |

## BEFORE versus AFTER

Old times are the archived V1 campaign, not a simultaneous timing experiment.

| Case | K | B | Old valid | New valid | Old quality | New quality | Old time (s) | New time (s) |
|---|---:|---:|---|---|---:|---:|---:|---:|
| REQ_000001 | 5 | 8 | True | True | 0.997442 | 0.992801 | 0.027473 | 0.038814 |
| REQ_000001 | 5 | 16 | True | True | 0.986470 | 0.997442 | 0.054349 | 0.062508 |
| REQ_000001 | 5 | 32 | True | True | 0.986470 | 0.997442 | 0.118007 | 0.116180 |
| REQ_000001 | 10 | 8 | True | True | 0.992628 | 0.995545 | 0.039881 | 0.060386 |
| REQ_000001 | 10 | 16 | True | True | 0.981032 | 0.995545 | 0.081603 | 0.083981 |
| REQ_000001 | 10 | 32 | True | True | 0.981032 | 0.992628 | 0.160638 | 0.161346 |
| REQ_000001 | 20 | 8 | True | True | 0.956983 | 0.989220 | 0.049478 | 0.081304 |
| REQ_000001 | 20 | 16 | True | True | 0.944302 | 0.984729 | 0.102142 | 0.121063 |
| REQ_000001 | 20 | 32 | True | True | 0.944302 | 0.984729 | 0.226533 | 0.208598 |
| REQ_000002 | 5 | 8 | True | True | 0.822722 | 0.996316 | 0.027422 | 0.025040 |
| REQ_000002 | 5 | 16 | True | True | 0.822722 | 0.996316 | 0.055804 | 0.047393 |
| REQ_000002 | 5 | 32 | True | True | 0.822722 | 0.996316 | 0.119487 | 0.093560 |
| REQ_000002 | 10 | 8 | True | True | 0.895296 | 0.997743 | 0.040433 | 0.036014 |
| REQ_000002 | 10 | 16 | True | True | 0.895296 | 0.997743 | 0.087001 | 0.063126 |
| REQ_000002 | 10 | 32 | True | True | 0.895296 | 0.997743 | 0.180508 | 0.128082 |
| REQ_000002 | 20 | 8 | True | True | 0.886211 | 0.999047 | 0.044076 | 0.077796 |
| REQ_000002 | 20 | 16 | True | True | 0.886211 | 0.999047 | 0.093680 | 0.101089 |
| REQ_000002 | 20 | 32 | True | True | 0.886211 | 0.999047 | 0.209591 | 0.192569 |
| REQ_000003 | 5 | 8 | False | True | n/a | 0.975790 | 0.015530 | 0.023170 |
| REQ_000003 | 5 | 16 | False | True | n/a | 0.999853 | 0.032540 | 0.042265 |
| REQ_000003 | 5 | 32 | False | True | n/a | 0.960247 | 0.071417 | 0.079281 |
| REQ_000003 | 10 | 8 | False | True | n/a | 1.000000 | 0.020523 | 0.028722 |
| REQ_000003 | 10 | 16 | False | True | n/a | 0.975813 | 0.043182 | 0.046454 |
| REQ_000003 | 10 | 32 | False | True | n/a | 0.999833 | 0.098153 | 0.087415 |
| REQ_000003 | 20 | 8 | False | True | n/a | 0.974188 | 0.030782 | 0.043661 |
| REQ_000003 | 20 | 16 | False | True | n/a | 1.000000 | 0.067767 | 0.061064 |
| REQ_000003 | 20 | 32 | False | True | n/a | 0.999829 | 0.154346 | 0.104826 |

## Scope and limitations

`SKIPPED_PAIR_LIMIT` means no exhaustive quality/time comparison was executed for that domain. The pair limit never truncates a baseline and never limits Beam. A missing Beam solution does not prove global infeasibility. An increased width need not monotonically improve quality or feasibility. The physical catalog and per-variant path cap constrain both methods.
