# Semantic Linker V3 — Global Stress Dataset

## Purpose
This dataset is a new, large stress-oriented fine-tuning corpus for the Requirement Extractor V2 Semantic Linker. It covers **all 11 quantitative fields plus `__UNRESOLVED__`**, not only `client_count`.

## Important design decision: X or Y
Alternative expressions such as `20 or 30 GB/s` are **not labelled ambiguous at the Semantic Linker stage**. The Semantic Linker must associate **both quantities** with the correct field. Each original alternative sentence therefore appears as a pair of training/evaluation samples sharing the same `group_id`, with `[Q]...[/Q]` placed on one candidate at a time. A downstream deterministic guard is responsible for deciding whether one candidate, both, or neither should survive.

## Sizes
- Training stress-only: **96,000** rows
- Training combined (V1 curated train + V3 stress): **98,100** rows
- Validation: **15,360** rows
- Untouched V3 test: **15,360** rows

## Stress families
- clean_natural
- paraphrase_synonym
- semantic_anchor_typo
- mixed_code_switch
- noisy_mixed_typos
- hpc_hard_negative
- multi_quantity_collision
- alternative_pair
- complex_role_temporal
- current_target_collision
- long_context
- short_telegraphic
- punctuation_case_noise
- quantity_words
- unit_surface_variation
- conversational_human_like

## Critical HPC distinctions included
The corpus includes collisions such as compute-node count vs Lustre-client count, network bandwidth vs Lustre throughput, archive/local capacity vs requested Lustre capacity, CPU/GPU/MPI/OST/OSS counts vs filesystem counts, current values vs targets, and several quantities in one message.

## Language
English, French, and mixed EN/FR code-switching are included. Mixed-language examples are intentionally natural and domain-specific rather than random word replacement.

## Typo generalization
Semantic-anchor typos use different mutation operators in train, validation, and test to reduce memorization of exact misspellings.

## Reproducibility / QA
See `semantic_linker_v3_QA_REPORT.json` and `semantic_linker_v3_metadata.json`. The QA checks cover schema consistency, quantity markers, field/role/dimension contracts, ID uniqueness, split leakage, group leakage, and the two-candidate invariant for alternative expressions.

## Scientific limitation
This is a large **synthetic + curated** corpus. No finite dataset can guarantee coverage of every possible user utterance or be proven to have zero unknown defects. The included QA report verifies the known structural and semantic invariants used by this project. For publication-grade robustness, retaining a future small human-written external test set is still recommended.

## Recommended training file
`semantic_linker_train_v3_combined.jsonl`

Use validation only for model selection/calibration. Use the V3 test once after freezing the model and thresholds.
