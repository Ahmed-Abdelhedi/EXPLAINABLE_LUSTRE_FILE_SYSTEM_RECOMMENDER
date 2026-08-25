# S10 Validation Results

Freeze status: **VALIDATED**

## Regression tests

Command:

```text
pytest lustre_architecture_generator/tests -q
```

Result:

```text
42 passed
```

## 1200-case end-to-end verification

The historical 1200-case fixture does not contain `planning_horizon_years`.
For this verification only, an explicit `planning_horizon_years = 1.0` was
materialised in a temporary copy before invoking the strict S10 analyzer.
No production fallback was used.

Validation results:

- Workload Analyzer: 1200 input / 1200 output, 0 structural errors, 0 monotonicity errors.
- Feature Calculator: 1200 cases, 0 structural errors, 0 recalculation errors, 0 determinism errors.
- Architecture Generator: 1200 cases, 0 structural errors, 0 recalculation errors, 0 determinism errors, 0 business-rule errors.
- OST safety-margin reclassification occurred for 169 cases and was accepted by the existing validator.

## Final contract checks

- `architecture_rules.json` version: `2.0`
- analyzer trace version: `3.0`
- explicit `planning_horizon_years`: required
- missing-horizon fallback: removed
- registered assumptions: 23/23 finalized
- `TO_VALIDATE`: 0
- OST bandwidth safety factor: `1.25`
- OST capacity factor: `1.0`
