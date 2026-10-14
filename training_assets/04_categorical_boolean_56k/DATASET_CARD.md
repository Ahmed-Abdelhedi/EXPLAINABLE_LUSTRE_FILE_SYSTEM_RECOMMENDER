# Categorical / Boolean Semantic FINAL Dataset

Version: `categorical_boolean_semantic_final_v1_20260826`

Status: **FINAL_DATASET_QA_PASS**

## Scope

Controlled multilingual synthetic corpus for the single multitask XLM-R training run. Each example has two simultaneous semantic labels: one for `ha_required` and one for `access_type`.

## Size and splits

- Train: 40,000
- Validation: 6,000
- Test: 6,000
- Final holdout: 4,000
- Total: 56,000

Each split is exactly 40% English, 40% French, and 20% mixed EN/FR.

## Label ontology

HA: `HA_REQUIRED`, `HA_NOT_REQUIRED`, `HA_MENTION_NO_COMMITMENT`, `HA_NO_EVIDENCE`.

Access: `SEQUENTIAL`, `RANDOM`, `MIXED`, `NO_SUPPORTED_ACCESS_CLASS`.

The 4×4 joint label matrix is perfectly balanced in every split, and that balance is also preserved inside each language partition.

## Stress coverage

The corpus intentionally includes negation, double-meaning traps, historical HA mentions, importance-vs-requirement distinctions, downtime semantics, single-node-failure semantics, corrections, concessions, streaming, parallel-only traps, parallel-random, parallel-sequential, mixed phases, dominance statements, implicit offsets/contiguity, throughput-only and client-count-only hard negatives, code switching, typos, punctuation noise, informal prefixes, and multi-constraint sentences.

## Difficulty profile

- train: {'hard': 9257, 'stress': 26828, 'medium': 3439, 'easy': 476}

- validation: {'stress': 3973, 'hard': 1406, 'medium': 541, 'easy': 80}

- test: {'stress': 4013, 'hard': 1377, 'medium': 545, 'easy': 65}

- final_holdout: {'easy': 56, 'medium': 342, 'stress': 2696, 'hard': 906}

## Leakage QA

Cross-split exact normalized text overlap: 0. Cross-split masked-number skeleton overlap: 0. Template-id overlap: 0. Structure-fingerprint overlap: 0.

## Negative-class safety audit

Rows labeled `HA_NO_EVIDENCE` containing HA/failure cues: 0. Rows labeled `NO_SUPPORTED_ACCESS_CLASS` containing explicit access-order class cues: 0.

## Scientific note

This corpus is deliberately engineered and stress-oriented. It should not be described as a universal real-world benchmark. The final holdout is reserved from model training/calibration/inference.
