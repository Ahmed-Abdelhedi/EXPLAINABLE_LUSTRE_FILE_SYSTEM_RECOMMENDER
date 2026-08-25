# S9 - Lustre Sizing Sensitivity Analysis

Cases replayed: **1200**.

The historical 1200-case S9 fixture predates planning_horizon_years. For reproducibility only, the S9 analysis materialises an explicit 1.0-year horizon in an in-memory copy before calling the strict S10 analyzer. No production fallback exists.

Score-weight perturbations preserve the sum-to-one invariant by proportionally renormalizing the other weights in the same score group.

| Assumption | Path | Sensitivity | Max workload flips | Max metadata-pressure flips | Max p95 numeric impact |
|---|---|---|---:|---:|---:|
| `CAPACITY_FILL_RATIO_DEFAULT` | `capacity_planning.default_target_fill_ratio` | MEDIUM | 1.83% | 0.00% | 14.29% |
| `WORKLOAD_METADATA_FILE_COUNT_WEIGHT` | `score_weights.metadata.file_count` | MEDIUM | 7.50% | 3.33% | 0.00% |
| `WORKLOAD_METADATA_SMALL_FILE_WEIGHT` | `score_weights.metadata.small_file_factor` | MEDIUM | 4.83% | 1.17% | 0.00% |
| `WORKLOAD_METADATA_CLIENT_WEIGHT` | `score_weights.metadata.client_count` | LOW | 1.67% | 1.92% | 0.00% |
| `WORKLOAD_DATA_CAPACITY_WEIGHT` | `score_weights.data.capacity` | MEDIUM | 6.58% | 0.00% | 0.00% |
| `WORKLOAD_DATA_BANDWIDTH_WEIGHT` | `score_weights.data.bandwidth` | MEDIUM | 6.75% | 0.00% | 0.00% |
| `WORKLOAD_DATA_LARGE_FILE_WEIGHT` | `score_weights.data.large_file_factor` | MEDIUM | 5.33% | 0.00% | 0.00% |
| `WORKLOAD_DOMINANCE_MARGIN` | `workload_classification.dominance_margin` | MEDIUM | 9.33% | 0.00% | 0.00% |
| `MDT_BASE_IOPS_PER_CLIENT` | `mdt_estimation.base_iops_per_client` | HIGH | 0.00% | 0.00% | 20.00% |
| `MDT_SMALL_FILE_MULTIPLIER` | `mdt_estimation.small_file_multiplier` | HIGH | 0.00% | 0.00% | 20.00% |
| `MDT_MEDIUM_FILE_MULTIPLIER` | `mdt_estimation.medium_file_multiplier` | HIGH | 0.00% | 0.00% | 20.00% |
| `MDT_LARGE_FILE_MULTIPLIER` | `mdt_estimation.large_file_multiplier` | HIGH | 0.00% | 0.00% | 20.00% |
| `MDT_RANDOM_ACCESS_MULTIPLIER` | `mdt_estimation.random_access_multiplier` | HIGH | 0.00% | 0.00% | 20.00% |
| `MDT_MIXED_ACCESS_MULTIPLIER` | `mdt_estimation.mixed_access_multiplier` | HIGH | 0.00% | 0.00% | 20.00% |
| `MDT_SEQUENTIAL_ACCESS_MULTIPLIER` | `mdt_estimation.sequential_access_multiplier` | HIGH | 0.00% | 0.00% | 20.00% |
| `MDT_HIGH_METADATA_MULTIPLIER` | `mdt_estimation.high_metadata_multiplier` | MEDIUM | 0.00% | 0.00% | 20.00% |
| `MDT_MEDIUM_METADATA_MULTIPLIER` | `mdt_estimation.medium_metadata_multiplier` | HIGH | 0.00% | 0.00% | 20.00% |
| `MDT_LOW_METADATA_MULTIPLIER` | `mdt_estimation.low_metadata_multiplier` | HIGH | 0.00% | 0.00% | 20.00% |
| `MDT_IOPS_SAFETY_FACTOR` | `mdt_estimation.iops_safety_factor` | HIGH | 0.00% | 0.00% | 20.00% |
| `MDT_METADATA_BYTES_PER_FILE` | `mdt_estimation.metadata_bytes_per_file` | HIGH | 0.00% | 0.00% | 20.00% |
| `MDT_METADATA_CAPACITY_SAFETY_FACTOR` | `mdt_estimation.metadata_capacity_safety_factor` | HIGH | 0.00% | 0.00% | 20.00% |
| `OST_BANDWIDTH_SAFETY_FACTOR` | `ost_estimation.bandwidth_safety_factor` | HIGH | 0.00% | 0.00% | 20.00% |
| `OST_CAPACITY_SAFETY_FACTOR` | `ost_estimation.capacity_safety_factor` | HIGH | 0.00% | 0.00% | 20.00% |

## Interpretation

- HIGH means a tested perturbation causes at least 10% classification flips or at least 20% p95 numeric output change.
- MEDIUM means at least 2% classification flips or at least 10% p95 numeric output change.
- LOW means the tested perturbations remain below those thresholds.
- Sensitivity is not empirical validation. Final statuses must combine this artifact with S8 evidence and design semantics.
