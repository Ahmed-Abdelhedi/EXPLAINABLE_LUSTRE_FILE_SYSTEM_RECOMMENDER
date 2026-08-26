# OST Candidate Filter — Bandwidth Unit Contract Fix

## Status

**VALIDATED** on the frozen 1200-case architecture dataset.

## Contract issue

The sizing freeze explicitly records that historical fields ending in `_gbps` carry values in **GB/s**. The previous OST candidate generator converted catalogue throughput using `MB/s × 0.008`, which produces **Gb/s** and therefore compared incompatible units.

The corrected conversion is:

```text
legacy *_gbps value (semantic unit GB/s) = catalogue MB/s × 0.001
```

No sizing field is renamed in this stage; the semantic debt remains frozen and documented.

## Impact on 1200 cases

- Top-10 OST changed: **834/1200 (69.50%)**
- Top-1 OST changed: **466/1200 (38.83%)**
- Cases with at least one feasible candidate whose raw minimum drive count changes: **480/1200 (40.00%)**
- Feasible candidate minimum per case: **23 → 4**
- Cases without any feasible OST candidate after correction: **0**
- Top-1 media distribution: **{'SSD': 273, 'HDD': 927} → {'SSD': 389, 'HDD': 811}**

## Validation

The corrected generator and validator produce:

```text
Cas architecturaux       : 1200
Cas OST générés          : 1200
Cas sans candidat        : 0
Erreurs structure        : 0
Erreurs de recalcul      : 0
Erreurs de déterminisme  : 0
Erreurs métier           : 0
STATUT : VALIDÉ
```

The complete Python test suite contains **45 passing tests** after adding the unit-contract regression tests.

## Important downstream consequence

The existing OST training dataset and OST CatBoost model were produced with the previous conversion and are therefore **stale**. They must be regenerated/retrained before the OST ranker is used again.

The current inference code itself already applies hard filters to **all feasible candidates before ML**; it does not apply the deterministic Top-K before CatBoost. The standalone Top-K remains a deterministic teacher/reference used by the training workflow.
