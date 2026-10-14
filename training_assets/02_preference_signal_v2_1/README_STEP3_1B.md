# Step 3.1B — Preference Signal Detector V2

The V1 internal data has 20,000 TRAIN / 4,000 VAL / 4,000 TEST examples,
50/50 labels and 40% EN / 40% FR / 20% mixed. Its QA already proves zero
text/group overlap, but TRAIN/VAL/TEST all draw from the same 20 broad
synthetic families.

Step 3.1B keeps V1 and adds NEW hard discourse families with split-specific
template families.

## New hard families

Positive:
- conditional choice
- preference question / trade-off
- long-context final choice
- ranked telegraphic preference
- counterfactual trade-off

Negative:
- rejected third-party preference
- negated preference report
- system capability statement
- rejected/superseded draft
- lexical "priority" trap

The already-inspected 500 stress messages are never copied into training.

## Sizes

Hard addition:
- TRAIN: 6,000
- VAL: 1,200
- INDEPENDENT HOLDOUT: 1,200

Combined:
- TRAIN V2: 26,000
- VAL V2: 5,200
- INDEPENDENT HOLDOUT V2: 1,200

All are exactly 50/50 labels and 40% EN / 40% FR / 20% mixed.

## Add these files

- `preference_extractor/training/generate_preference_signal_v2.py`
- `preference_extractor/training/qa_preference_signal_v2.py`
- `preference_extractor/training/notebooks/layer1_v2.ipynb`
- `preference_extractor/evaluation/run_signal_detector_v2_comparison.py`

The ZIP also includes reproducible hard-only JSONL files under:
`preference_extractor/training/data/data_layer1_v2/`.

## 1. Generate combined V2 data

From repo root:

```powershell
python -m preference_extractor.training.generate_preference_signal_v2
```

Expected:
- `preference_signal_train_v2.jsonl` = 26,000
- `preference_signal_val_v2.jsonl` = 5,200
- `preference_signal_independent_holdout_v2.jsonl` = 1,200

## 2. QA

```powershell
python -m preference_extractor.training.qa_preference_signal_v2
```

All checks must pass, especially:
- train/val/holdout exact-text overlap = 0
- group overlap = 0
- hard template-family overlap = 0
- hard template-id overlap = 0
- known 500 benchmark exact-text overlap = 0
- 50/50 labels
- 40/40/20 language split

## 3. Kaggle

Upload:
- `preference_signal_train_v2.jsonl`
- `preference_signal_val_v2.jsonl`
- `preference_signal_independent_holdout_v2.jsonl`
- `preference_signal_legacy_test_v1_4000.jsonl`

Run `layer1_v2.ipynb` with GPU + Internet.

The notebook:
1. trains from `distilbert-base-multilingual-cased`;
2. selects checkpoint on V2 VALIDATION only;
3. calibrates threshold on V2 VALIDATION only;
4. freezes checkpoint + threshold;
5. evaluates V2 independent holdout FIRST;
6. only afterward runs old V1 TEST as regression;
7. exports `preference_signal_detector_v2.zip`.

## Scientific rule

Preserve the first:
`independent_holdout_metrics_FIRST_RUN.json`

If we inspect errors and modify the system afterward, later runs on that same
holdout become regression runs. We would create a new holdout for new
independent evidence.

Do NOT replace the production V1 artifact yet. Send the FIRST_RUN holdout
metrics back before promoting V2.
