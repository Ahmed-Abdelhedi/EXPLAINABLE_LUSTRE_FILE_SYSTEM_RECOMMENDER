# Deterministic MDT/OST Drive Filter Benchmark

Benchmark version: `1.0`
Scope: `deterministic_mdt_ost_drive_filter`
Throughput contract: Historical fields ending in _gbps carry GB/s values; catalogue MB/s is converted with x0.001.

## Purpose

This benchmark strengthens the negative-case evaluation of the deterministic drive filter.
It contains clearly feasible, clearly infeasible, and boundary requests for both MDT and OST.
The ML ranker is not involved: invalid candidates must be rejected before ranking.

## Summary

| Metric | Result |
|---|---:|
| Cases | 24 |
| Decision accuracy | 100.00% |
| False-feasible rate | 0.00% |
| False-infeasible rate | 0.00% |
| Rejection-reason accuracy | 100.00% |

## Accuracy by role

| Role | Cases | Accuracy |
|---|---:|---:|
| MDT | 12 | 100.00% |
| OST | 12 | 100.00% |

## Accuracy by benchmark group

| Group | Cases | Accuracy |
|---|---:|---:|
| boundary | 8 | 100.00% |
| clearly_feasible | 8 | 100.00% |
| clearly_infeasible | 8 | 100.00% |

## Case-level results

| Case | Role | Group | Expected | Predicted | Feasible drives | Decision | Reasons |
|---|---|---|---|---|---:|---|---|
| MDT_F01_FULL_MODERATE | MDT | clearly_feasible | feasible | feasible | 174 | PASS | not_mdt_eligible:59, raw_mdt_cost_exceeds_global_budget:2 |
| MDT_F02_FULL_CRITICAL | MDT | clearly_feasible | feasible | feasible | 3 | PASS | endurance_requirement_not_met:170, latency_requirement_not_met:41, not_mdt_eligible:59, raw_mdt_cost_exceeds_global_budget:1, reliability_requirement_not_met:67 |
| MDT_F03_SINGLE_ANCHOR | MDT | clearly_feasible | feasible | feasible | 1 | PASS | - |
| MDT_F04_FULL_HIGH_IOPS | MDT | clearly_feasible | feasible | feasible | 41 | PASS | endurance_requirement_not_met:111, latency_requirement_not_met:41, not_mdt_eligible:59, raw_mdt_cost_exceeds_global_budget:4, reliability_requirement_not_met:67 |
| MDT_I01_FULL_BUDGET | MDT | clearly_infeasible | infeasible | infeasible | 0 | PASS | not_mdt_eligible:59, raw_mdt_cost_exceeds_global_budget:176 |
| MDT_I02_FULL_POWER | MDT | clearly_infeasible | infeasible | infeasible | 0 | PASS | not_mdt_eligible:59, raw_mdt_power_exceeds_global_power_limit:176 |
| MDT_I03_SINGLE_RELIABILITY | MDT | clearly_infeasible | infeasible | infeasible | 0 | PASS | reliability_requirement_not_met:1 |
| MDT_I04_SINGLE_ENDURANCE | MDT | clearly_infeasible | infeasible | infeasible | 0 | PASS | endurance_requirement_not_met:1 |
| MDT_B01_EXACT_BUDGET | MDT | boundary | feasible | feasible | 1 | PASS | - |
| MDT_B02_JUST_BELOW_BUDGET | MDT | boundary | infeasible | infeasible | 0 | PASS | raw_mdt_cost_exceeds_global_budget:1 |
| MDT_B03_EXACT_POWER | MDT | boundary | feasible | feasible | 1 | PASS | - |
| MDT_B04_LATENCY_STEP | MDT | boundary | infeasible | infeasible | 0 | PASS | latency_requirement_not_met:1 |
| OST_F01_FULL_MODERATE | OST | clearly_feasible | feasible | feasible | 224 | PASS | raw_ost_cost_exceeds_global_budget:11 |
| OST_F02_FULL_LARGE_CRITICAL | OST | clearly_feasible | feasible | feasible | 150 | PASS | raw_ost_cost_exceeds_global_budget:38, reliability_requirement_not_met:67 |
| OST_F03_SINGLE_ANCHOR | OST | clearly_feasible | feasible | feasible | 1 | PASS | - |
| OST_F04_FULL_RANDOM_CRITICAL | OST | clearly_feasible | feasible | feasible | 167 | PASS | raw_ost_cost_exceeds_global_budget:9, reliability_requirement_not_met:67 |
| OST_I01_FULL_BUDGET | OST | clearly_infeasible | infeasible | infeasible | 0 | PASS | raw_ost_cost_exceeds_global_budget:235 |
| OST_I02_FULL_POWER | OST | clearly_infeasible | infeasible | infeasible | 0 | PASS | raw_ost_power_exceeds_global_power_limit:235 |
| OST_I03_SINGLE_RELIABILITY | OST | clearly_infeasible | infeasible | infeasible | 0 | PASS | reliability_requirement_not_met:1 |
| OST_I04_FULL_EXTREME_THROUGHPUT | OST | clearly_infeasible | infeasible | infeasible | 0 | PASS | raw_ost_cost_exceeds_global_budget:235 |
| OST_B01_EXACT_THROUGHPUT | OST | boundary | feasible | feasible | 1 | PASS | - |
| OST_B02_JUST_ABOVE_READ | OST | boundary | infeasible | infeasible | 0 | PASS | raw_ost_cost_exceeds_global_budget:1 |
| OST_B03_EXACT_CAPACITY | OST | boundary | feasible | feasible | 1 | PASS | - |
| OST_B04_JUST_ABOVE_CAPACITY | OST | boundary | infeasible | infeasible | 0 | PASS | raw_ost_cost_exceeds_global_budget:1 |

## Interpretation

Budget and power checks remain lower-bound filters at this drive-selection stage.
Final global budget and power validation belongs to the complete architecture layer.
Boundary OST cases explicitly verify that a 550 MB/s catalogue drive is treated as 0.55 GB/s under the frozen historical field contract.

## Stop condition

The filter benchmark is considered validated only if decision accuracy and rejection-reason accuracy are 100%, with zero false-feasible and zero false-infeasible cases.
