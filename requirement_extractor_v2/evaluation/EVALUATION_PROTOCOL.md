# Requirement Extractor V2 — Evaluation Protocol

This document freezes the benchmark roles used from Step 1 onward.

## 1. Why this file exists

Earlier evaluation artifacts used inconsistent wording around the 300-message
benchmark and the 96-message benchmark. In particular, an artifact could report
`n_messages = 300` while a methodology note still said that the ablation used a
fixed 96-message benchmark.

From now on, benchmark identity is determined by the **dataset actually passed
to the runner**, not by a hard-coded sentence in a result file.

No historical metric is changed by this protocol. If an old result was computed
on the correct data but has the wrong label/note, only its metadata should be
corrected. If a result was computed on the wrong dataset, it must be rerun.

## 2. Official benchmark roles

### A. 300-message broad evaluation / ablation benchmark

Legacy filename:

`datasets/v2_independent_end_to_end_benchmark_v1.jsonl`

Official role from now on:

- broad component evaluation source;
- ablation benchmark when passed to `run_ablation_config.py`;
- source used to construct the actual LLM-fallback subset;
- robustness coverage across English, French and mixed-language messages.

Expected size:

`300 messages`

Important:

The legacy filename contains `independent_end_to_end`, but this file must not be
confused with the dedicated 96-message quantity E2E holdout below.

### B. 96-message quantity-only E2E holdout

Filename:

`datasets/quantity_e2e_holdout_v1.jsonl`

Official role:

- independent quantity-only end-to-end holdout on its **first preserved run**;
- safety evaluation for ambiguity and out-of-scope cases.

Expected size:

`96 messages`

The preserved first run remains the independent holdout result.

After implementation changes are made after inspecting failures from this
dataset, later runs on the same 96 messages are **regression results**, not new
independent holdout results.

For example, the Step 1 final run after relation, robustness and LLM-prompt
changes is a regression run even if it reaches 96/96.

## 3. LLM fallback evaluation

The LLM fallback is evaluated only on the subset that actually reaches it after:

`Scope -> Quantity Scanner -> Explicit Resolver -> Semantic Linker abstention`

The current subset is generated from the 300-message benchmark by:

`evaluation/build_actual_fallback_subset.py`

The generated dataset is:

`datasets/actual_fallback_subset_v1.jsonl`

Current Step 1 subset size:

- 23 quantity cases;
- 23 source messages;
- 3 expected RESOLVE cases;
- 20 expected ABSTAIN cases.

Model comparison must report at least:

- field accuracy on resolvable cases;
- role accuracy on resolvable cases;
- field+role pair accuracy;
- correct abstention rate;
- false-resolution rate;
- mean latency;
- p95 latency.

## 4. Step 1 final model decision

The retained LLM fallback model is:

`qwen2.5-coder:7b`

Reason:

- it recovered the 3/3 resolvable actual-fallback cases after prompt revision;
- it abstained correctly on 20/20 abstention cases;
- it produced 0 false resolutions.

`qwen2.5:3b` is faster but recovered 0/3 resolvable cases in the same subset, so
it is not the official fallback model for Step 1.

## 5. Metadata rules for every new result JSON

Every new result should record, directly or indirectly:

- `dataset_path`
- `dataset_name`
- `n_messages` or `n_cases`
- `evaluation_role`
- `run_kind`
- model name when an LLM is used
- a methodology note that matches the actual dataset

Recommended values:

### 300-message ablation run

`evaluation_role = "broad_ablation_benchmark"`

`run_kind = "evaluation"`

### First 96-message holdout run

`evaluation_role = "quantity_e2e_holdout"`

`run_kind = "independent_first_run"`

### Later 96-message runs after observed-failure fixes

`evaluation_role = "quantity_e2e_holdout"`

`run_kind = "regression_after_holdout_inspection"`

### Actual LLM fallback subset

`evaluation_role = "actual_llm_fallback_subset"`

`run_kind = "component_evaluation"`

## 6. Reporting rule

Do not write only:

`the benchmark contains 96 messages`

or:

`the benchmark contains 300 messages`

Instead write the benchmark role explicitly, for example:

- `The broad ablation benchmark contains 300 messages.`
- `The independent quantity-only E2E holdout contains 96 messages.`
- `After fixes informed by the holdout, the same 96-message set is used only
  for regression testing.`

This wording removes the previous ambiguity without changing valid historical
metric values.
