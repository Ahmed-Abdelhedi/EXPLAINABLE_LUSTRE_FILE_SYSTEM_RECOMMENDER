# Requirement Extractor V2 — Direct FGCS Evaluation Pack

Master benchmark: `datasets/v2_independent_end_to_end_benchmark_v1.jsonl`

- 300 cases
- robustness categories included directly
- English, French and mixed-language inputs
- short contextual answers, corrections, ambiguity and out-of-scope cases
- built after baseline freeze; do not train or tune thresholds on it

Category counts:
```json
{
  "normal": 30,
  "synonym": 30,
  "spelling_error": 30,
  "written_number": 25,
  "typoed_written_number": 25,
  "unit_variation": 30,
  "missing_unit": 20,
  "ambiguity": 20,
  "broad_signal": 20,
  "short_conversational_answer": 25,
  "correction": 20,
  "out_of_scope": 25
}
```

Copy this folder to `requirement_extractor_v2/evaluation/`, then run:

```powershell
python -m requirement_extractor_v2.evaluation.run_quantity_scanner_benchmark --dataset requirement_extractor_v2/evaluation/datasets/v2_independent_end_to_end_benchmark_v1.jsonl --output requirement_extractor_v2/evaluation/quantity_scanner_metrics.json

python -m requirement_extractor_v2.evaluation.run_scope_resolver_benchmark --dataset requirement_extractor_v2/evaluation/datasets/v2_independent_end_to_end_benchmark_v1.jsonl --output requirement_extractor_v2/evaluation/scope_resolver_metrics.json
```

Important: the current `VerifiedRequirementPipeline` states that ConversationScopeResolver is not integrated yet. Component benchmarks can run now; full Scope→Quantity→Explicit→Semantic→LLM→Verifier E2E should be reported after that wiring exists.
