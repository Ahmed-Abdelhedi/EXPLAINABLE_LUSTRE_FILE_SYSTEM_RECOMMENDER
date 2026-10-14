# Preference Layer 2 — Kaggle Fresh R3

This is the clean Kaggle package to use in a NEW Kaggle session.

Unique split filenames:
- preference_layer2_r3_train.jsonl
- preference_layer2_r3_validation.jsonl
- preference_layer2_r3_test.jsonl
- preference_layer2_r3_final_holdout.jsonl

The companion notebook does not trust a generic filename or metadata lookup.
It finds candidate roots containing the unique R3 TRAIN filename and accepts
only the root whose four JSONL SHA256 fingerprints all match
KAGGLE_DATASET_FINGERPRINTS.json.

Do not attach any older Layer-2 dataset to the new Kaggle session.
