# Step 3.1E — Preference Signal Detector V2.1 targeted data

## Why V2.1

Step 3.1D reproduced exactly 72 false negatives on V2 VALIDATION.

All 72 are:

```text
hard_long_context_final_choice
```

Language distribution:

```text
EN    48
mixed 24
FR     0
```

Most are extremely confidently classified as negative, so lowering the
threshold is not a safe solution.

## Design

V2.1 adds new examples without copying the 72 known validation messages.

Targeted training augmentation:

```text
2400 examples
1200 positive
1200 negative

EN    1600
mixed  800
```

Targeted validation augmentation:

```text
600 examples
300 positive
300 negative

EN    400
mixed 200
```

Positive family:

```text
v21_long_context_final_choice
```

Matched negative families:

```text
v21_long_context_rejected_choice
v21_long_context_third_party_choice
v21_long_context_capability_choice
v21_long_context_no_preference
```

The matched negatives are important: the model must learn discourse ownership
and final-choice semantics, not simply memorize long contexts containing words
such as "choose", "priority", or "final recommendation".

Targeted TRAIN and targeted VAL use disjoint template families and template IDs.

## Resulting combined datasets

The generator merges the targeted data with the existing V2 datasets.

```text
TRAIN V2.1 = 26,000 + 2,400 = 28,400
VAL   V2.1 =  5,200 +   600 =  5,800
```

Both remain 50/50 balanced:

```text
TRAIN: 14,200 / 14,200
VAL:    2,900 / 2,900
```

The old V2 validation, old V2 holdout and known 500 benchmark remain regression
sets only.

The fresh final holdout V3 must remain untouched until the V2.1 checkpoint and
final threshold are frozen.

## Install

Extract this package directly under the project root and merge
`preference_extractor`.

From:

```text
C:\Users\LENOVO\Desktop\internship\version2
```

run:

```powershell
python -m preference_extractor.training.generate_preference_signal_v2_1
```

Then:

```powershell
python -m preference_extractor.training.qa_preference_signal_v2_1
```

Do not start Kaggle training unless:

```text
all_checks_pass = true
```

## Output directory

```text
preference_extractor/training/data/data_layer1_v2_1/
```

Expected files include:

```text
preference_signal_targeted_train_v2_1.jsonl
preference_signal_targeted_val_v2_1.jsonl
preference_signal_train_v2_1.jsonl
preference_signal_val_v2_1.jsonl
preference_signal_regression_val_v2_5200.jsonl
preference_signal_regression_holdout_v2_1200.jsonl
preference_signal_legacy_test_v1_4000.jsonl
preference_signal_v2_1_metadata.json
preference_signal_v2_1_QA_REPORT.json
```
