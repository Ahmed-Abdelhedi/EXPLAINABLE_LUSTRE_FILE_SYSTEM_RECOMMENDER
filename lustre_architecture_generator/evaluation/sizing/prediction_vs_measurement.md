# S8 - Prediction vs Measurement

## Scope and interpretation rule

The sizing model estimates **required technical capacity**. Toubkal benchmarks measure **delivered performance on one concrete Lustre system**. Absolute benchmark maxima are therefore not fitted directly to sizing constants. S8 compares direction, relative scaling, variability, contention and evidence coverage.

## MDT client scaling

| Operation | Clients | Model relative scale | Observed relative scale | Scaling efficiency |
|---|---:|---:|---:|---:|
| file_create_ops_s | 1 | 1.00x | 1.00x | 100.00% |
| file_create_ops_s | 2 | 2.00x | 1.68x | 84.20% |
| file_create_ops_s | 4 | 4.00x | 2.86x | 71.58% |
| file_stat_ops_s | 1 | 1.00x | 1.00x | 100.00% |
| file_stat_ops_s | 2 | 2.00x | 1.13x | 56.28% |
| file_stat_ops_s | 4 | 4.00x | 1.55x | 38.72% |
| file_remove_ops_s | 1 | 1.00x | 1.00x | 100.00% |
| file_remove_ops_s | 2 | 2.00x | 1.52x | 75.85% |
| file_remove_ops_s | 4 | 4.00x | 2.79x | 69.71% |

The direction predicted by the client-count term is observed, but delivered metadata throughput scales sublinearly. This is evidence for monotonic client influence, not a reason to replace `base_iops_per_client` with MDTest peak rates.

## OST client scaling

| Metric | Clients | Ideal relative scale | Mean observed | Median observed | Mean efficiency | Median efficiency |
|---|---:|---:|---:|---:|---:|---:|
| write_mib_s | 1 | 1.00x | 1.00x | 1.00x | 100.00% | 100.00% |
| write_mib_s | 2 | 2.00x | 1.61x | 1.87x | 80.47% | 93.62% |
| write_mib_s | 4 | 4.00x | 2.85x | 3.25x | 71.23% | 81.15% |
| read_mib_s | 1 | 1.00x | 1.00x | 1.00x | 100.00% | 100.00% |
| read_mib_s | 2 | 2.00x | 2.16x | 2.14x | 107.82% | 106.99% |
| read_mib_s | 4 | 4.00x | 3.68x | 3.61x | 91.96% | 90.23% |

The 2-node and 4-node write means are affected by a very low first iteration, so medians are retained as a robust secondary statistic.

## Mixed workload impact (M5)

- Metadata create change vs M1: -2.81%.
- Metadata stat change vs M1: -31.78%.
- Metadata remove change vs M1: -2.08%.
- IOR write change vs same-size sequential control M6-A: -18.73%.
- IOR read change vs same-size sequential control M6-A: -0.95%.
- The M5 write result implies a local recovery/headroom factor of about 1.230x; current OST factor is 1.25x.

The configured OST bandwidth factor is now 1.25x. S8 evidence is local/system-specific and final calibration status is assigned separately in S9-B.

## Access-pattern evidence (M6)

- Shuffled vs sequential write change: -9.49%.
- Shuffled vs sequential read change: 0.39%.
- M6-C fully random overlapping I/O is rejected as a measurement because IOR 4.1.0+dev failed with SIGFPE / integer divide-by-zero.
- These are IOR/OST observations. They **do not validate MDT access multipliers**.

## Assumption evidence coverage after Toubkal

| Assumption | Evidence level | S8 status |
|---|---|---|
| `CAPACITY_FILL_RATIO_DEFAULT` | NOT_TESTED | PENDING_SENSITIVITY |
| `WORKLOAD_METADATA_FILE_COUNT_WEIGHT` | NOT_ISOLATED | PENDING_SENSITIVITY |
| `WORKLOAD_METADATA_SMALL_FILE_WEIGHT` | PARTIAL | PENDING_SENSITIVITY |
| `WORKLOAD_METADATA_CLIENT_WEIGHT` | DIRECT_DIRECTIONAL | PENDING_SENSITIVITY |
| `WORKLOAD_DATA_CAPACITY_WEIGHT` | NOT_TESTED | PENDING_SENSITIVITY |
| `WORKLOAD_DATA_BANDWIDTH_WEIGHT` | DIRECT_DIRECTIONAL | PENDING_SENSITIVITY |
| `WORKLOAD_DATA_LARGE_FILE_WEIGHT` | PARTIAL | PENDING_SENSITIVITY |
| `WORKLOAD_DOMINANCE_MARGIN` | PARTIAL | PENDING_SENSITIVITY |
| `MDT_BASE_IOPS_PER_CLIENT` | DIRECT_DIRECTIONAL_NOT_ABSOLUTE | PENDING_SENSITIVITY |
| `MDT_SMALL_FILE_MULTIPLIER` | PARTIAL | PENDING_SENSITIVITY |
| `MDT_MEDIUM_FILE_MULTIPLIER` | NOT_TESTED | PENDING_SENSITIVITY |
| `MDT_LARGE_FILE_MULTIPLIER` | NOT_TESTED | PENDING_SENSITIVITY |
| `MDT_RANDOM_ACCESS_MULTIPLIER` | NOT_TESTED | PENDING_SENSITIVITY |
| `MDT_MIXED_ACCESS_MULTIPLIER` | PARTIAL_NOT_ISOLATED | PENDING_SENSITIVITY |
| `MDT_SEQUENTIAL_ACCESS_MULTIPLIER` | NOT_TESTED | PENDING_SENSITIVITY |
| `MDT_HIGH_METADATA_MULTIPLIER` | NOT_ISOLATED | PENDING_SENSITIVITY |
| `MDT_MEDIUM_METADATA_MULTIPLIER` | NOT_ISOLATED | PENDING_SENSITIVITY |
| `MDT_LOW_METADATA_MULTIPLIER` | PARTIAL | PENDING_SENSITIVITY |
| `MDT_IOPS_SAFETY_FACTOR` | PARTIAL | PENDING_SENSITIVITY |
| `MDT_METADATA_BYTES_PER_FILE` | NOT_TESTED | PENDING_SENSITIVITY |
| `MDT_METADATA_CAPACITY_SAFETY_FACTOR` | NOT_TESTED | PENDING_SENSITIVITY |
| `OST_BANDWIDTH_SAFETY_FACTOR` | DIRECT_LOCAL | PENDING_SENSITIVITY |
| `OST_CAPACITY_SAFETY_FACTOR` | NOT_TESTED | PENDING_SENSITIVITY |

## S8 decision

1. Do not change the sizing configuration yet.
2. Preserve Toubkal as empirical evidence for trends, variability and mixed-workload contention.
3. Do not claim validation for constants whose required controlled benchmark was not run.
4. Continue to S9 sensitivity analysis; only then assign `SUPPORTED`, `CALIBRATED`, `POLICY_CHOICE` or `NEEDS_REVISION`.
