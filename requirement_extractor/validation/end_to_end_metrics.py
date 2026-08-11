#!/usr/bin/env python3
"""
Compute detailed metrics for Requirement Pipeline End-to-End Validation v1.

The script consumes the JSON produced by:
    requirement_extractor.validation.run_end_to_end_validation

It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_RESULT = (
    Path(__file__).resolve().parent
    / "reports"
    / "end_to_end"
    / "results_end_to_end_v1.json"
)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "reports"
    / "end_to_end"
    / "metrics_end_to_end_v1.json"
)

OUTCOME_ORDER = [
    "READY_COHERENT",
    "READY_AMBIGUOUS",
    "BLOCKED_PLAUSIBILITY",
    "CLARIFICATION_REQUIRED",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute metrics for the end-to-end requirement pipeline benchmark."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULT,
        help="Path to results_end_to_end_v1.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write metrics_end_to_end_v1.json.",
    )
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None

    ordered = sorted(float(v) for v in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_summary(values: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values]

    if not vals:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    ordered = sorted(vals)
    n = len(ordered)

    if n % 2:
        median = ordered[n // 2]
    else:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0

    return {
        "count": n,
        "mean_ms": sum(ordered) / n,
        "median_ms": median,
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def compact_rate(passed: int, total: int) -> Dict[str, Any]:
    return {
        "passed": int(passed),
        "total": int(total),
        "rate": safe_div(passed, total),
    }


def get_expected_outcome(scenario: Mapping[str, Any]) -> Optional[str]:
    # The runner stores the expected final outcome at scenario level.
    return scenario.get("pipeline_outcome")


def get_actual_outcome(scenario: Mapping[str, Any]) -> Optional[str]:
    return scenario.get("actual", {}).get("pipeline_outcome")


def get_fallback_call_count(scenario: Mapping[str, Any]) -> int:
    instrumentation = (
        scenario.get("actual", {})
        .get("fallback_instrumentation", {})
    )
    try:
        return int(instrumentation.get("call_count_delta", 0) or 0)
    except (TypeError, ValueError):
        return 0


def get_fallback_log_delta(scenario: Mapping[str, Any]) -> int:
    instrumentation = (
        scenario.get("actual", {})
        .get("fallback_instrumentation", {})
    )
    try:
        return int(instrumentation.get("call_log_length_delta", 0) or 0)
    except (TypeError, ValueError):
        return 0


def get_llm_final_candidate_count(scenario: Mapping[str, Any]) -> int:
    try:
        return int(
            scenario.get("actual", {}).get("llm_fallback_candidate_count", 0)
            or 0
        )
    except (TypeError, ValueError):
        return 0


def get_plausibility(scenario: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    value = scenario.get("actual", {}).get("plausibility")
    return value if isinstance(value, Mapping) else None


def plausibility_llm_path_expected(scenario: Mapping[str, Any]) -> bool:
    plausibility = get_plausibility(scenario)
    return bool(
        plausibility
        and plausibility.get("llm_enrichment_path_expected")
    )


def plausibility_llm_output_observed(scenario: Mapping[str, Any]) -> bool:
    plausibility = get_plausibility(scenario)
    return bool(
        plausibility
        and plausibility.get("llm_output_observed")
    )


def raw_response_json_parseable(scenario: Mapping[str, Any]) -> Optional[bool]:
    plausibility = get_plausibility(scenario)

    if not plausibility:
        return None

    raw = plausibility.get("raw_response")

    if not isinstance(raw, str) or not raw.strip():
        return None

    try:
        json.loads(raw)
    except json.JSONDecodeError:
        return False

    return True


def group_success(
    scenarios: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)

    for scenario in scenarios:
        key = key_fn(scenario)
        grouped[str(key)].append(scenario)

    output: Dict[str, Any] = {}

    for key in sorted(grouped):
        items = grouped[key]
        passed = sum(
            bool(item.get("checks", {}).get("scenario_success"))
            for item in items
        )
        output[key] = {
            "scenario_count": len(items),
            "successful_scenarios": passed,
            "failed_scenarios": len(items) - passed,
            "success_rate": safe_div(passed, len(items)),
            "latency": latency_summary(
                [item.get("latency_ms", 0.0) for item in items]
            ),
        }

    return output


def group_latency(
    scenarios: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], Any],
) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = defaultdict(list)

    for scenario in scenarios:
        grouped[str(key_fn(scenario))].append(
            float(scenario.get("latency_ms", 0.0))
        )

    return {
        key: latency_summary(values)
        for key, values in sorted(grouped.items())
    }


def build_confusion_matrix(
    scenarios: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    labels = list(OUTCOME_ORDER)

    extras = sorted(
        {
            label
            for scenario in scenarios
            for label in (
                get_expected_outcome(scenario),
                get_actual_outcome(scenario),
            )
            if label is not None and label not in labels
        }
    )

    labels.extend(extras)

    matrix = {
        expected: {actual: 0 for actual in labels}
        for expected in labels
    }

    for scenario in scenarios:
        expected = get_expected_outcome(scenario)
        actual = get_actual_outcome(scenario)

        if expected is None or actual is None:
            continue

        if expected not in matrix:
            matrix[expected] = {label: 0 for label in labels}

        if actual not in matrix[expected]:
            for row in matrix.values():
                row.setdefault(actual, 0)
            if actual not in labels:
                labels.append(actual)

        matrix[expected][actual] += 1

    return {
        "labels": labels,
        "matrix": matrix,
    }


def main() -> int:
    args = parse_args()

    with args.results.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    scenarios = document.get("scenarios", [])

    if not isinstance(scenarios, list):
        raise ValueError("The results JSON does not contain a valid 'scenarios' list.")

    scenario_count = len(scenarios)
    total_turns = sum(int(s.get("turn_count", 0) or 0) for s in scenarios)

    scenario_success = sum(
        bool(s.get("checks", {}).get("scenario_success"))
        for s in scenarios
    )

    pipeline_outcome_matches = sum(
        bool(s.get("checks", {}).get("pipeline_outcome_match"))
        for s in scenarios
    )

    technical_errors = sum(
        bool(s.get("technical_error"))
        or any(bool(turn.get("technical_error")) for turn in s.get("turns", []))
        for s in scenarios
    )

    # ------------------------------------------------------------------
    # Extraction metrics
    # ------------------------------------------------------------------
    extraction_all_exact = sum(
        bool(s.get("checks", {}).get("extraction_all_fields_exact"))
        for s in scenarios
    )

    field_totals: Counter[str] = Counter()
    field_matches: Counter[str] = Counter()
    total_field_slots = 0
    correct_field_slots = 0
    non_null_expected_fields = 0
    non_null_correct_fields = 0
    missing_value_inference_scenarios = 0
    missing_value_inference_fields: Counter[str] = Counter()

    for scenario in scenarios:
        field_comparison = (
            scenario.get("checks", {})
            .get("field_comparison", {})
        )

        for field_name, matched in field_comparison.get(
            "field_matches",
            {},
        ).items():
            field_totals[field_name] += 1
            total_field_slots += 1

            if matched:
                field_matches[field_name] += 1
                correct_field_slots += 1

        non_null_expected_fields += int(
            field_comparison.get("non_null_expected_field_count", 0)
            or 0
        )
        non_null_correct_fields += int(
            field_comparison.get("non_null_correct_count", 0)
            or 0
        )

        inferred_fields = field_comparison.get(
            "missing_value_inference_fields",
            [],
        ) or []

        if inferred_fields:
            missing_value_inference_scenarios += 1

        for field_name in inferred_fields:
            missing_value_inference_fields[str(field_name)] += 1

    per_field_exactness = {
        field_name: {
            "correct": field_matches[field_name],
            "total": field_totals[field_name],
            "accuracy": safe_div(
                field_matches[field_name],
                field_totals[field_name],
            ),
        }
        for field_name in sorted(field_totals)
    }

    # ------------------------------------------------------------------
    # Clarification metrics
    # ------------------------------------------------------------------
    clarification_applicable = []
    clarification_covered = 0
    clarification_exact = 0

    for scenario in scenarios:
        expected_fields = (
            scenario.get("expected", {}).get("clarification_fields", [])
            or []
        )

        if not expected_fields:
            continue

        clarification_applicable.append(scenario)

        actual_pending = (
            scenario.get("actual", {}).get("pending_fields", [])
            or []
        )

        expected_set = set(expected_fields)
        actual_set = set(actual_pending)

        if expected_set.issubset(actual_set):
            clarification_covered += 1

        if expected_set == actual_set:
            clarification_exact += 1

    # ------------------------------------------------------------------
    # Plausibility metrics
    # ------------------------------------------------------------------
    plausibility_applicable = []
    plausibility_label_correct = 0
    plausibility_block_correct = 0
    issue_codes_exact = 0

    for scenario in scenarios:
        expected_status = scenario.get("expected", {}).get(
            "plausibility_status"
        )

        if expected_status is None:
            continue

        plausibility_applicable.append(scenario)
        checks = scenario.get("checks", {})

        plausibility_label_correct += bool(
            checks.get("plausibility_label_match")
        )
        plausibility_block_correct += bool(
            checks.get("should_block_match")
        )
        issue_codes_exact += bool(
            checks.get("issue_codes_exact")
        )

    # ------------------------------------------------------------------
    # LLM fallback instrumentation
    # ------------------------------------------------------------------
    fallback_calls = sum(get_fallback_call_count(s) for s in scenarios)
    fallback_log_delta = sum(get_fallback_log_delta(s) for s in scenarios)

    fallback_scenarios = [
        s for s in scenarios
        if get_fallback_call_count(s) > 0
    ]

    no_fallback_scenarios = [
        s for s in scenarios
        if get_fallback_call_count(s) == 0
    ]

    final_llm_candidates = sum(
        get_llm_final_candidate_count(s)
        for s in scenarios
    )

    fallback_calls_without_final_candidate = sum(
        get_fallback_call_count(s)
        for s in fallback_scenarios
        if get_llm_final_candidate_count(s) == 0
    )

    # ------------------------------------------------------------------
    # Plausibility LLM enrichment instrumentation
    # ------------------------------------------------------------------
    enrichment_path_scenarios = [
        s for s in scenarios
        if plausibility_llm_path_expected(s)
    ]

    enrichment_output_scenarios = [
        s for s in scenarios
        if plausibility_llm_output_observed(s)
    ]

    unexpected_enrichment_outputs = [
        s for s in scenarios
        if plausibility_llm_output_observed(s)
        and not plausibility_llm_path_expected(s)
    ]

    enrichment_errors = [
        s for s in scenarios
        if (
            get_plausibility(s)
            and get_plausibility(s).get("enrichment_error")
        )
    ]

    raw_json_applicable = []
    raw_json_parseable = 0

    for scenario in enrichment_output_scenarios:
        parseability = raw_response_json_parseable(scenario)

        if parseability is None:
            continue

        raw_json_applicable.append(scenario)
        raw_json_parseable += bool(parseability)

    # ------------------------------------------------------------------
    # Latency metrics
    # ------------------------------------------------------------------
    overall_latency = latency_summary(
        [s.get("latency_ms", 0.0) for s in scenarios]
    )

    llm_warning_scenarios = [
        s for s in scenarios
        if plausibility_llm_path_expected(s)
    ]

    no_llm_warning_scenarios = [
        s for s in scenarios
        if not plausibility_llm_path_expected(s)
    ]

    multi_turn_scenarios = [
        s for s in scenarios
        if int(s.get("turn_count", 0) or 0) > 1
    ]

    single_turn_scenarios = [
        s for s in scenarios
        if int(s.get("turn_count", 0) or 0) <= 1
    ]

    deterministic_fast_path = [
        s for s in scenarios
        if (
            get_fallback_call_count(s) == 0
            and not plausibility_llm_path_expected(s)
        )
    ]

    fallback_only_path = [
        s for s in scenarios
        if (
            get_fallback_call_count(s) > 0
            and not plausibility_llm_path_expected(s)
        )
    ]

    plausibility_llm_only_path = [
        s for s in scenarios
        if (
            get_fallback_call_count(s) == 0
            and plausibility_llm_path_expected(s)
        )
    ]

    both_llm_paths = [
        s for s in scenarios
        if (
            get_fallback_call_count(s) > 0
            and plausibility_llm_path_expected(s)
        )
    ]

    expected_outcome_counts = Counter(
        get_expected_outcome(s) for s in scenarios
    )
    actual_outcome_counts = Counter(
        get_actual_outcome(s) for s in scenarios
    )

    output = {
        "metrics_version": "1.0.0",
        "evaluation_scope": document.get("experiment", {}).get(
            "scope",
            "End-to-end requirement pipeline validation",
        ),
        "source_results": str(args.results),
        "run_metadata": document.get("run", {}),
        "summary": {
            "scenario_count": scenario_count,
            "turn_count": total_turns,
            "successful_scenarios": scenario_success,
            "failed_scenarios": scenario_count - scenario_success,
            "scenario_success_rate": safe_div(
                scenario_success,
                scenario_count,
            ),
            "technical_error_count": technical_errors,
            "technical_error_rate": safe_div(
                technical_errors,
                scenario_count,
            ),
        },
        "pipeline_outcome": {
            "accuracy": safe_div(
                pipeline_outcome_matches,
                scenario_count,
            ),
            "correct": pipeline_outcome_matches,
            "total": scenario_count,
            "expected_distribution": dict(
                sorted(expected_outcome_counts.items())
            ),
            "actual_distribution": dict(
                sorted(actual_outcome_counts.items())
            ),
            "confusion_matrix": build_confusion_matrix(scenarios),
        },
        "success_by": {
            "outcome": group_success(
                scenarios,
                lambda s: get_expected_outcome(s),
            ),
            "category": group_success(
                scenarios,
                lambda s: s.get("category"),
            ),
            "language": group_success(
                scenarios,
                lambda s: s.get("language"),
            ),
            "difficulty": group_success(
                scenarios,
                lambda s: s.get("difficulty"),
            ),
            "turn_count": group_success(
                scenarios,
                lambda s: s.get("turn_count"),
            ),
        },
        "extraction": {
            "scenario_all_fields_exact": compact_rate(
                extraction_all_exact,
                scenario_count,
            ),
            "field_slot_accuracy": {
                "correct": correct_field_slots,
                "total": total_field_slots,
                "accuracy": safe_div(
                    correct_field_slots,
                    total_field_slots,
                ),
            },
            "non_null_value_accuracy": {
                "correct": non_null_correct_fields,
                "total": non_null_expected_fields,
                "accuracy": safe_div(
                    non_null_correct_fields,
                    non_null_expected_fields,
                ),
            },
            "per_field_exactness": per_field_exactness,
            "no_missing_value_inference": {
                "scenarios_without_inference": (
                    scenario_count
                    - missing_value_inference_scenarios
                ),
                "scenario_count": scenario_count,
                "rate": safe_div(
                    scenario_count
                    - missing_value_inference_scenarios,
                    scenario_count,
                ),
                "inferred_field_counts": dict(
                    sorted(missing_value_inference_fields.items())
                ),
            },
        },
        "clarification": {
            "applicable_scenarios": len(clarification_applicable),
            "target_coverage": compact_rate(
                clarification_covered,
                len(clarification_applicable),
            ),
            "exact_pending_field_set": compact_rate(
                clarification_exact,
                len(clarification_applicable),
            ),
        },
        "plausibility": {
            "applicable_scenarios": len(plausibility_applicable),
            "status_accuracy": compact_rate(
                plausibility_label_correct,
                len(plausibility_applicable),
            ),
            "blocking_decision_accuracy": compact_rate(
                plausibility_block_correct,
                len(plausibility_applicable),
            ),
            "issue_code_exact_match": compact_rate(
                issue_codes_exact,
                len(plausibility_applicable),
            ),
        },
        "llm_fallback": {
            "total_calls_recorded": fallback_calls,
            "call_log_entries_delta": fallback_log_delta,
            "scenario_count_with_calls": len(fallback_scenarios),
            "scenario_rate_with_calls": safe_div(
                len(fallback_scenarios),
                scenario_count,
            ),
            "call_rate_per_turn": safe_div(
                fallback_calls,
                total_turns,
            ),
            "final_llm_candidate_count": final_llm_candidates,
            "final_candidate_yield_per_call": safe_div(
                final_llm_candidates,
                fallback_calls,
            ),
            "calls_in_scenarios_with_zero_final_llm_candidates": (
                fallback_calls_without_final_candidate
            ),
            "note": (
                "A zero final-candidate yield is a routing/performance "
                "observation, not a correctness failure by itself."
            ),
        },
        "plausibility_llm_enrichment": {
            "expected_warning_paths": len(enrichment_path_scenarios),
            "observed_outputs": len(enrichment_output_scenarios),
            "observation_rate_on_expected_paths": safe_div(
                sum(
                    plausibility_llm_output_observed(s)
                    for s in enrichment_path_scenarios
                ),
                len(enrichment_path_scenarios),
            ),
            "unexpected_output_count": len(
                unexpected_enrichment_outputs
            ),
            "enrichment_error_count": len(enrichment_errors),
            "raw_response_json_parseability": {
                "parseable": raw_json_parseable,
                "total_observed_with_raw_response": len(
                    raw_json_applicable
                ),
                "rate": safe_div(
                    raw_json_parseable,
                    len(raw_json_applicable),
                ),
                "note": (
                    "This measures only raw LLM response JSON syntax. "
                    "It does not measure final decision correctness, "
                    "because the deterministic guard remains authoritative."
                ),
            },
        },
        "latency": {
            "overall": overall_latency,
            "by_outcome": group_latency(
                scenarios,
                lambda s: get_actual_outcome(s),
            ),
            "by_category": group_latency(
                scenarios,
                lambda s: s.get("category"),
            ),
            "by_language": group_latency(
                scenarios,
                lambda s: s.get("language"),
            ),
            "by_difficulty": group_latency(
                scenarios,
                lambda s: s.get("difficulty"),
            ),
            "single_turn": latency_summary(
                [s.get("latency_ms", 0.0) for s in single_turn_scenarios]
            ),
            "multi_turn": latency_summary(
                [s.get("latency_ms", 0.0) for s in multi_turn_scenarios]
            ),
            "fallback_called": latency_summary(
                [s.get("latency_ms", 0.0) for s in fallback_scenarios]
            ),
            "fallback_not_called": latency_summary(
                [s.get("latency_ms", 0.0) for s in no_fallback_scenarios]
            ),
            "plausibility_llm_path": latency_summary(
                [s.get("latency_ms", 0.0) for s in llm_warning_scenarios]
            ),
            "no_plausibility_llm_path": latency_summary(
                [s.get("latency_ms", 0.0) for s in no_llm_warning_scenarios]
            ),
            "path_decomposition": {
                "deterministic_fast_path": latency_summary(
                    [
                        s.get("latency_ms", 0.0)
                        for s in deterministic_fast_path
                    ]
                ),
                "fallback_only": latency_summary(
                    [
                        s.get("latency_ms", 0.0)
                        for s in fallback_only_path
                    ]
                ),
                "plausibility_llm_only": latency_summary(
                    [
                        s.get("latency_ms", 0.0)
                        for s in plausibility_llm_only_path
                    ]
                ),
                "both_llm_paths": latency_summary(
                    [
                        s.get("latency_ms", 0.0)
                        for s in both_llm_paths
                    ]
                ),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            output,
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    print("=" * 92)
    print("END-TO-END REQUIREMENT PIPELINE METRICS")
    print("=" * 92)
    print(f"Results                    : {args.results}")
    print(f"Scenarios                  : {scenario_count}")
    print(
        f"Scenario success           : "
        f"{scenario_success}/{scenario_count} "
        f"({safe_div(scenario_success, scenario_count) * 100:.2f}%)"
    )
    print(
        f"Pipeline outcome accuracy  : "
        f"{pipeline_outcome_matches}/{scenario_count} "
        f"({safe_div(pipeline_outcome_matches, scenario_count) * 100:.2f}%)"
    )
    print(
        f"All-fields exact           : "
        f"{extraction_all_exact}/{scenario_count} "
        f"({safe_div(extraction_all_exact, scenario_count) * 100:.2f}%)"
    )
    print(
        f"Field-slot accuracy        : "
        f"{correct_field_slots}/{total_field_slots} "
        f"({safe_div(correct_field_slots, total_field_slots) * 100:.2f}%)"
    )
    print(
        f"Non-null value accuracy    : "
        f"{non_null_correct_fields}/{non_null_expected_fields} "
        f"({safe_div(non_null_correct_fields, non_null_expected_fields) * 100:.2f}%)"
    )
    print(
        f"No missing-value inference : "
        f"{scenario_count - missing_value_inference_scenarios}/{scenario_count} "
        f"({safe_div(scenario_count - missing_value_inference_scenarios, scenario_count) * 100:.2f}%)"
    )
    print(
        f"Clarification exact        : "
        f"{clarification_exact}/{len(clarification_applicable)} "
        f"({safe_div(clarification_exact, len(clarification_applicable)) * 100:.2f}%)"
    )
    print(
        f"Plausibility status        : "
        f"{plausibility_label_correct}/{len(plausibility_applicable)} "
        f"({safe_div(plausibility_label_correct, len(plausibility_applicable)) * 100:.2f}%)"
    )
    print(
        f"Plausibility block         : "
        f"{plausibility_block_correct}/{len(plausibility_applicable)} "
        f"({safe_div(plausibility_block_correct, len(plausibility_applicable)) * 100:.2f}%)"
    )
    print(
        f"Plausibility issue codes   : "
        f"{issue_codes_exact}/{len(plausibility_applicable)} "
        f"({safe_div(issue_codes_exact, len(plausibility_applicable)) * 100:.2f}%)"
    )
    print(f"Technical errors           : {technical_errors}")
    print(f"Fallback calls             : {fallback_calls}")
    print(f"Fallback scenarios         : {len(fallback_scenarios)}/{scenario_count}")
    print(f"Final LLM candidates       : {final_llm_candidates}")
    print(
        f"Plausibility LLM outputs   : "
        f"{len(enrichment_output_scenarios)}/{len(enrichment_path_scenarios)} expected paths"
    )
    print(
        f"Raw LLM JSON parseability  : "
        f"{raw_json_parseable}/{len(raw_json_applicable)}"
    )
    print(
        f"Latency mean / median      : "
        f"{overall_latency['mean_ms']:.3f} / "
        f"{overall_latency['median_ms']:.3f} ms"
    )
    print(
        f"Latency p95 / p99 / max    : "
        f"{overall_latency['p95_ms']:.3f} / "
        f"{overall_latency['p99_ms']:.3f} / "
        f"{overall_latency['max_ms']:.3f} ms"
    )
    print(f"Output                     : {args.output}")
    print("=" * 92)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())