# S9-B - Final Sizing Assumption Decisions

Configuration version: **2.0**.

## Decision rule

Calibrate only numerically identifiable quantities; retain non-identifiable heuristics as explicit POLICY_CHOICE values instead of overclaiming validation.

## Numerical calibration

- `OST_BANDWIDTH_SAFETY_FACTOR`: **1.20 -> 1.25**.
- M5 mixed-write change: **-18.73%**.
- Implied recovery headroom: **1.230x**.
- Config change relative to 1.20: **+4.17%**.
- The final 1.25 value is deliberately rounded upward rather than fitted to the exact Toubkal observation.

## Final status table

| Assumption | Initial | Final | Status | Sensitivity | Toubkal evidence |
|---|---:|---:|---|---|---|
| `CAPACITY_FILL_RATIO_DEFAULT` | 0.8 | 0.8 | **POLICY_CHOICE** | MEDIUM | NOT_TESTED |
| `WORKLOAD_METADATA_FILE_COUNT_WEIGHT` | 0.5 | 0.5 | **POLICY_CHOICE** | MEDIUM | NOT_ISOLATED |
| `WORKLOAD_METADATA_SMALL_FILE_WEIGHT` | 0.3 | 0.3 | **POLICY_CHOICE** | MEDIUM | PARTIAL |
| `WORKLOAD_METADATA_CLIENT_WEIGHT` | 0.2 | 0.2 | **POLICY_CHOICE** | LOW | DIRECT_DIRECTIONAL |
| `WORKLOAD_DATA_CAPACITY_WEIGHT` | 0.4 | 0.4 | **POLICY_CHOICE** | MEDIUM | NOT_TESTED |
| `WORKLOAD_DATA_BANDWIDTH_WEIGHT` | 0.4 | 0.4 | **POLICY_CHOICE** | MEDIUM | DIRECT_DIRECTIONAL |
| `WORKLOAD_DATA_LARGE_FILE_WEIGHT` | 0.2 | 0.2 | **POLICY_CHOICE** | MEDIUM | PARTIAL |
| `WORKLOAD_DOMINANCE_MARGIN` | 0.15 | 0.15 | **POLICY_CHOICE** | MEDIUM | PARTIAL |
| `MDT_BASE_IOPS_PER_CLIENT` | 100 | 100 | **POLICY_CHOICE** | HIGH | DIRECT_DIRECTIONAL_NOT_ABSOLUTE |
| `MDT_SMALL_FILE_MULTIPLIER` | 3.0 | 3.0 | **POLICY_CHOICE** | HIGH | PARTIAL |
| `MDT_MEDIUM_FILE_MULTIPLIER` | 1.5 | 1.5 | **POLICY_CHOICE** | HIGH | NOT_TESTED |
| `MDT_LARGE_FILE_MULTIPLIER` | 1.0 | 1.0 | **POLICY_CHOICE** | HIGH | NOT_TESTED |
| `MDT_RANDOM_ACCESS_MULTIPLIER` | 1.4 | 1.4 | **POLICY_CHOICE** | HIGH | NOT_TESTED |
| `MDT_MIXED_ACCESS_MULTIPLIER` | 1.15 | 1.15 | **POLICY_CHOICE** | HIGH | PARTIAL_NOT_ISOLATED |
| `MDT_SEQUENTIAL_ACCESS_MULTIPLIER` | 1.0 | 1.0 | **POLICY_CHOICE** | HIGH | NOT_TESTED |
| `MDT_HIGH_METADATA_MULTIPLIER` | 1.5 | 1.5 | **POLICY_CHOICE** | MEDIUM | NOT_ISOLATED |
| `MDT_MEDIUM_METADATA_MULTIPLIER` | 1.2 | 1.2 | **POLICY_CHOICE** | HIGH | NOT_ISOLATED |
| `MDT_LOW_METADATA_MULTIPLIER` | 1.0 | 1.0 | **POLICY_CHOICE** | HIGH | PARTIAL |
| `MDT_IOPS_SAFETY_FACTOR` | 1.25 | 1.25 | **POLICY_CHOICE** | HIGH | PARTIAL |
| `MDT_METADATA_BYTES_PER_FILE` | 4096 | 4096 | **POLICY_CHOICE** | HIGH | NOT_TESTED |
| `MDT_METADATA_CAPACITY_SAFETY_FACTOR` | 2.0 | 2.0 | **POLICY_CHOICE** | HIGH | NOT_TESTED |
| `OST_BANDWIDTH_SAFETY_FACTOR` | 1.2 | 1.25 | **CALIBRATED** | HIGH | DIRECT_LOCAL |
| `OST_CAPACITY_SAFETY_FACTOR` | 1.0 | 1.0 | **SUPPORTED** | HIGH | NOT_TESTED |

## Interpretation

- `SUPPORTED`: retained because the current design/evidence supports the value and semantics.
- `CALIBRATED`: numerical value changed using measured evidence plus a conservative engineering rule.
- `POLICY_CHOICE`: retained intentionally, but must not be described as empirically measured or universal.
- `NEEDS_REVISION`: reserved for assumptions contradicted by evidence or structurally unsuitable.

No assumption remains `TO_VALIDATE` after this stage.

## Important limitation

The Toubkal campaign validates trends and local contention behavior on one Lustre installation. It does not turn heuristic workload-demand coefficients into universal hardware-performance constants.
