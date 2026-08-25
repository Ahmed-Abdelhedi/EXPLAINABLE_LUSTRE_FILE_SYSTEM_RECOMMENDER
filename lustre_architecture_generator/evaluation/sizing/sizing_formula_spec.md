# Lustre Sizing Formula Specification

Version: 2.0  
Scope: workload analysis and MDT/OST sizing before drive selection.

## 1. Capacity planning

The planned usable capacity is:

`planned_usable_capacity_tib = requested_usable_capacity_tib * (1 + annual_growth_percent / 100)^planning_horizon_years / target_fill_ratio`

Current default `target_fill_ratio = 0.8`.

`planning_horizon_years` is a mandatory input field. Since the S10 freeze, the workload analyzer applies **no default horizon and no compatibility fallback**. Missing, boolean, zero, negative, or non-finite horizons are rejected so that planning intent must be resolved upstream.

Reference test:

- requested capacity = 100 TiB
- annual growth = 20 %
- planning horizon = 3 years
- target fill ratio = 0.8
- expected planned capacity = 216 TiB

## 2. Workload classification

Metadata score:

`metadata_score = 0.5*file_count_score + 0.3*small_file_factor + 0.2*client_count_score`

Data score:

`data_score = 0.4*capacity_score + 0.4*bandwidth_score + 0.2*large_file_factor`

Dominance:

- metadata-heavy when `metadata_score - data_score >= 0.15`
- data-heavy when `metadata_score - data_score <= -0.15`
- balanced otherwise

These weights and the `0.15` margin are finalized as explicit `POLICY_CHOICE` heuristics; sensitivity is documented in S9.

## 3. MDT IOPS sizing

Raw MDT IOPS proxy:

`raw_iops = base_iops_per_client * client_count * file_size_multiplier * access_multiplier * metadata_pressure_multiplier`

Required MDT IOPS:

`required_total_iops = ceil(raw_iops * iops_safety_factor)`

Current values:

- base IOPS/client = 100
- file size multipliers = 3.0 / 1.5 / 1.0
- access multipliers = 1.4 / 1.15 / 1.0
- metadata pressure multipliers = 1.5 / 1.2 / 1.0
- IOPS safety factor = 1.25

These MDT values remain explicit demand-model `POLICY_CHOICE` proxies. Toubkal supports some directions/trends but does not identify their absolute magnitudes.

## 4. MDT metadata-capacity sizing

`required_metadata_capacity_tib = file_count * metadata_bytes_per_file * metadata_capacity_safety_factor / 1024^4`

Current values:

- metadata bytes/file = 4096 bytes
- metadata capacity safety factor = 2.0

These are finalized engineering `POLICY_CHOICE` assumptions; the benchmark campaign did not measure per-file MDT space consumption.

## 5. OST throughput sizing

Read requirement:

`required_read_bandwidth = target_read_bandwidth * bandwidth_safety_factor`

Write requirement:

`required_write_bandwidth = target_write_bandwidth * bandwidth_safety_factor`

Total requirement:

`required_total_bandwidth = required_read_bandwidth + required_write_bandwidth`

Current bandwidth safety factor = 1.25. It was `CALIBRATED` from 1.20 using M5 mixed-workload write contention (implied recovery headroom about 1.230x) and conservative upward rounding.

The current repository uses legacy field names ending in `_gbps`, but the project convention currently interprets those values as GB/s. No mass rename is performed during sizing validation.

## 6. OST capacity sizing

`required_usable_capacity_tib = planned_usable_capacity_tib * capacity_safety_factor`

Current `capacity_safety_factor = 1.0` (`SUPPORTED` by the double-margin design review).

This is intentionally separate from RAID/raw-capacity calculations, which belong to later architecture generation.

## 7. Validation rule

No constant in this specification is considered empirically validated only because deterministic tests pass.

Deterministic tests verify implementation correctness.

Toubkal experiments must provide evidence for empirical validity using:

- MDTest for metadata behavior;
- IOR for read/write throughput;
- prediction-vs-measurement comparison;
- sensitivity analysis;
- repeated runs where possible.

S9-B has assigned one of the following final statuses to every registered assumption:

- `SUPPORTED`
- `CALIBRATED`
- `POLICY_CHOICE`
- `NEEDS_REVISION`

The authoritative per-assumption decisions are stored in `calibration_decisions.json` and `sizing_assumptions.json` registry version 2.0.

## 8. S10 freeze contract

The sizing layer is frozen with these operational rules:

- `architecture_rules.json` version = `2.0`;
- `planning_horizon_years` is mandatory and must be finite and strictly positive;
- there is no production fallback for missing planning horizon;
- all 23 registered assumptions have a final status;
- OST bandwidth safety factor = `1.25`;
- ranking, drive selection, RAID, target counts, servers and striping remain downstream concerns and are outside the frozen sizing scope.

Historical S8/S9 artifacts remain reproducible. The S9 sensitivity script explicitly materialises a one-year horizon only in an in-memory copy of the old 1200-case evaluation fixture; this does not alter the production input contract.
