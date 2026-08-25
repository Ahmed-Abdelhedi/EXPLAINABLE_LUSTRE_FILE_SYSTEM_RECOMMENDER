# S10 — Lustre Sizing Freeze

Status: **FROZEN**  
Config version: **2.0**  
Analyzer trace version: **3.0**

## Frozen contract

The sizing layer requires an explicit, finite, strictly positive `planning_horizon_years` in every input requirement. The production analyzer has no compatibility fallback. Missing planning intent must be fixed upstream.

## Final assumption state

- SUPPORTED: 1
- CALIBRATED: 1
- POLICY_CHOICE: 21
- NEEDS_REVISION: 0
- TO_VALIDATE: 0

The calibrated OST bandwidth safety factor is **1.25**. The OST capacity factor remains **1.0** to avoid double-counting capacity headroom already introduced by growth and target fill ratio.

## Empirical basis

Toubkal M1–M5 plus M6-A/M6-B provide the empirical reference. M6-C is explicitly excluded because IOR 4.1.0+dev terminated with SIGFPE before producing a valid measurement. Prediction-vs-measurement and 1200-case sensitivity analyses are retained under `evaluation/sizing/`.

## Historical S9 fixture

The old 1200-case dataset predates `planning_horizon_years`. For reproducibility of the already-completed sensitivity study only, `sensitivity_analysis.py` creates an in-memory copy and explicitly materialises `planning_horizon_years = 1.0` before calling the strict analyzer. This is evaluation-only and is not a production fallback.

## Boundary of the freeze

The frozen sizing stage outputs hardware-independent MDT/OST technical requirements. Drive ranking/selection, RAID, target counts, server topology and striping remain downstream and are not part of this freeze.

## Known semantic debt

Legacy fields ending in `_gbps` are still interpreted by the project convention as GB/s. Renaming them is deferred to a controlled contract migration to avoid breaking the validated pipeline.
