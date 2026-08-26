# Final CatBoost vs LightGBM comparison

## Objective

Run the controlled ranking comparison requested before freezing the official MDT/OST ranker.

The comparison uses the same final data, case-level splits, feature schemas, relevance labels,
seeds and metrics for both CatBoostRanker and LightGBM LGBMRanker.

## Final data required

Use only the `lustre_ranker_training_data_kaggle.zip` generated after the corrected OST unit
contract.

Expected uncompressed dataset SHA256:

- MDT: `a6dbcb1ae8c446f626a05d1f8393500a8ee77292770baf6e6ce10dc5824b273c`
- OST: `28380ba8e4ae5d988da834b5d74bce6bd3062d2a948cdece3710227ae51fd2b2`

The training script aborts if these hashes do not match during a final run.

## Models

For both MDT and OST:

- CatBoostRanker / YetiRankPairwise
- LightGBM LGBMRanker / lambdarank

Seeds:

`21, 42, 84, 126, 168`

Maximum boosting rounds / trees: 1800

Early stopping: 150 rounds, using validation only.

## Model selection rule

The official family is selected using validation results, not test results.
Test metrics are retained as independent final confirmation.

Primary comparison order:

1. validation NDCG@5;
2. validation Top-1 agreement;
3. validation NDCG@10;
4. multi-seed stability;
5. inference time / model footprint as secondary criteria.

The generated output contains role-level and family-level summaries.

## Kaggle

1. Enable a GPU accelerator.
2. Add the final `lustre_ranker_training_data_kaggle.zip` as notebook input.
3. Open/import `ranker_comparison_final_kaggle.ipynb`.
4. Run all cells.
5. Download `/kaggle/working/ranker_comparison_final_artifacts.zip`.

## Outputs

The ZIP contains:

- 5 CatBoost MDT models;
- 5 CatBoost OST models;
- 5 LightGBM MDT models;
- 5 LightGBM OST models;
- per-seed metrics;
- multi-seed summaries;
- predictions;
- feature importances and stability;
- role-level decision table;
- overall family table;
- immutable comparison contract;
- Markdown comparison report.
