#!/usr/bin/env python3
"""
Analyze functional errors and non-fatal diagnostics in the Requirement
Pipeline End-to-End Validation v1 result file.

Functional failures and performance/safety observations are deliberately
separated. In particular, an LLM fallback call that produces no final
candidate is not automatically treated as a correctness failure.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


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
    / "errors_end_to_end_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze errors in the end-to-end requirement pipeline benchmark."
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
        help="Path to write errors_end_to_end_v1.json.",
    )
    return parser.parse_args()


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def fallback_call_count(scenario: Mapping[str, Any]) -> int:
    value = (
        scenario.get("actual", {})
        .get("fallback_instrumentation", {})
        .get("call_count_delta", 0)
    )

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def fallback_log_delta(scenario: Mapping[str, Any]) -> int:
    value = (
        scenario.get("actual", {})
        .get("fallback_instrumentation", {})
        .get("call_log_length_delta", 0)
    )

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def final_llm_candidate_count(scenario: Mapping[str, Any]) -> int:
    value = scenario.get("actual", {}).get(
        "llm_fallback_candidate_count",
        0,
    )

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def add_error(
    errors: List[Dict[str, Any]],
    error_type: str,
    details: Mapping[str, Any],
) -> None:
    errors.append(
        {
            "type": error_type,
            "details": dict(details),
        }
    )


def add_observation(
    observations: List[Dict[str, Any]],
    observation_type: str,
    details: Mapping[str, Any],
) -> None:
    observations.append(
        {
            "type": observation_type,
            "details": dict(details),
        }
    )


def inspect_scenario(
    scenario: Mapping[str, Any],
) -> Dict[str, Any]:
    scenario_id = scenario.get("id")
    checks = scenario.get("checks", {})
    actual = scenario.get("actual", {})
    expected = scenario.get("expected", {})

    errors: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Technical errors
    # ------------------------------------------------------------------
    scenario_technical_error = scenario.get("technical_error")

    turn_technical_errors = [
        {
            "turn_index": turn.get("turn_index"),
            "error": turn.get("technical_error"),
        }
        for turn in scenario.get("turns", [])
        if turn.get("technical_error")
    ]

    if scenario_technical_error or turn_technical_errors:
        add_error(
            errors,
            "TECHNICAL_ERROR",
            {
                "scenario_error": scenario_technical_error,
                "turn_errors": turn_technical_errors,
            },
        )

    # ------------------------------------------------------------------
    # Pipeline outcome
    # ------------------------------------------------------------------
    if not checks.get("pipeline_outcome_match", False):
        add_error(
            errors,
            "WRONG_PIPELINE_OUTCOME",
            {
                "expected": scenario.get("pipeline_outcome"),
                "actual": actual.get("pipeline_outcome"),
            },
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    field_comparison = checks.get("field_comparison", {})

    if not checks.get("extraction_all_fields_exact", False):
        add_error(
            errors,
            "FIELD_MISMATCH",
            {
                "field_mismatches": field_comparison.get(
                    "field_mismatches",
                    {},
                ),
                "field_matches": field_comparison.get(
                    "field_matches",
                    {},
                ),
            },
        )

    inferred_fields = field_comparison.get(
        "missing_value_inference_fields",
        [],
    ) or []

    if inferred_fields:
        add_error(
            errors,
            "MISSING_VALUE_INFERRED",
            {
                "fields": inferred_fields,
            },
        )

    # ------------------------------------------------------------------
    # Clarification
    # ------------------------------------------------------------------
    expected_clarification = set(
        expected.get("clarification_fields", []) or []
    )

    if expected_clarification:
        actual_pending = set(
            actual.get("pending_fields", []) or []
        )

        if expected_clarification != actual_pending:
            add_error(
                errors,
                "CLARIFICATION_MISMATCH",
                {
                    "expected_fields": sorted(
                        expected_clarification
                    ),
                    "actual_pending_fields": sorted(
                        actual_pending
                    ),
                    "missing_expected_fields": sorted(
                        expected_clarification - actual_pending
                    ),
                    "unexpected_pending_fields": sorted(
                        actual_pending - expected_clarification
                    ),
                },
            )

    # ------------------------------------------------------------------
    # Plausibility
    # ------------------------------------------------------------------
    expected_plausibility_status = expected.get(
        "plausibility_status"
    )

    if expected_plausibility_status is not None:
        if not checks.get("plausibility_label_match", False):
            actual_plausibility = actual.get("plausibility") or {}
            add_error(
                errors,
                "PLAUSIBILITY_STATUS_MISMATCH",
                {
                    "expected": expected_plausibility_status,
                    "actual": actual_plausibility.get(
                        "predicted_label"
                    ),
                },
            )

        if not checks.get("should_block_match", False):
            actual_plausibility = actual.get("plausibility") or {}
            add_error(
                errors,
                "PLAUSIBILITY_BLOCK_MISMATCH",
                {
                    "expected": expected.get(
                        "should_block_recommendation"
                    ),
                    "actual": actual_plausibility.get(
                        "should_block_recommendation"
                    ),
                },
            )

        if not checks.get("issue_codes_exact", False):
            add_error(
                errors,
                "PLAUSIBILITY_ISSUE_MISMATCH",
                {
                    "expected_issue_codes": checks.get(
                        "expected_issue_codes",
                        [],
                    ),
                    "actual_issue_codes": checks.get(
                        "actual_issue_codes",
                        [],
                    ),
                },
            )

    # ------------------------------------------------------------------
    # Fallback routing consistency
    # ------------------------------------------------------------------
    calls = fallback_call_count(scenario)
    log_delta = fallback_log_delta(scenario)
    final_candidates = final_llm_candidate_count(scenario)

    if calls != log_delta:
        add_error(
            errors,
            "FALLBACK_ROUTING_ANOMALY",
            {
                "reason": "call_count_delta != call_log_length_delta",
                "call_count_delta": calls,
                "call_log_length_delta": log_delta,
            },
        )

    if final_candidates > 0 and calls <= 0:
        add_error(
            errors,
            "FALLBACK_ROUTING_ANOMALY",
            {
                "reason": (
                    "final LLM candidates exist but no fallback call "
                    "was recorded"
                ),
                "call_count_delta": calls,
                "final_llm_candidate_count": final_candidates,
            },
        )

    if calls > 0 and final_candidates == 0:
        add_observation(
            observations,
            "FALLBACK_CALL_WITHOUT_FINAL_CANDIDATE",
            {
                "fallback_calls": calls,
                "final_llm_candidate_count": 0,
                "latency_ms": scenario.get("latency_ms"),
                "note": (
                    "Non-fatal routing/performance observation. "
                    "This is not treated as a correctness error."
                ),
            },
        )

    # ------------------------------------------------------------------
    # Plausibility LLM routing and raw output diagnostics
    # ------------------------------------------------------------------
    plausibility = actual.get("plausibility")

    if isinstance(plausibility, Mapping):
        path_expected = bool(
            plausibility.get("llm_enrichment_path_expected")
        )
        output_observed = bool(
            plausibility.get("llm_output_observed")
        )

        if path_expected != output_observed:
            add_error(
                errors,
                "LLM_ENRICHMENT_MISMATCH",
                {
                    "llm_enrichment_path_expected": path_expected,
                    "llm_output_observed": output_observed,
                    "enrichment_error": plausibility.get(
                        "enrichment_error"
                    ),
                },
            )

        raw_response = plausibility.get("raw_response")

        if path_expected and output_observed and isinstance(
            raw_response,
            str,
        ):
            try:
                json.loads(raw_response)
            except json.JSONDecodeError as exc:
                add_observation(
                    observations,
                    "LLM_ENRICHMENT_RAW_JSON_INVALID",
                    {
                        "json_error": str(exc),
                        "raw_response_length": len(raw_response),
                        "note": (
                            "Non-fatal because the guarded deterministic "
                            "plausibility output remains authoritative."
                        ),
                    },
                )

        if plausibility.get("enrichment_error"):
            add_observation(
                observations,
                "LLM_ENRICHMENT_REPORTED_ERROR",
                {
                    "enrichment_error": plausibility.get(
                        "enrichment_error"
                    ),
                    "note": (
                        "Reported as a non-fatal observation unless it "
                        "also causes an expected/observed routing mismatch."
                    ),
                },
            )

    # ------------------------------------------------------------------
    # Catch unclassified runner failures
    # ------------------------------------------------------------------
    if (
        not checks.get("scenario_success", False)
        and not errors
    ):
        add_error(
            errors,
            "SCENARIO_SUCCESS_FALSE_UNCLASSIFIED",
            {
                "checks": checks,
            },
        )

    return {
        "id": scenario_id,
        "category": scenario.get("category"),
        "language": scenario.get("language"),
        "difficulty": scenario.get("difficulty"),
        "expected_pipeline_outcome": scenario.get(
            "pipeline_outcome"
        ),
        "actual_pipeline_outcome": actual.get(
            "pipeline_outcome"
        ),
        "scenario_success": bool(
            checks.get("scenario_success")
        ),
        "latency_ms": scenario.get("latency_ms"),
        "errors": errors,
        "observations": observations,
    }


def main() -> int:
    args = parse_args()

    with args.results.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    scenarios = document.get("scenarios", [])

    if not isinstance(scenarios, list):
        raise ValueError("The results JSON does not contain a valid 'scenarios' list.")

    analyses = [
        inspect_scenario(scenario)
        for scenario in scenarios
    ]

    failed_analyses = [
        item for item in analyses
        if item["errors"]
    ]

    observation_analyses = [
        item for item in analyses
        if item["observations"]
    ]

    error_type_counts: Counter[str] = Counter()
    observation_type_counts: Counter[str] = Counter()

    for item in analyses:
        for error in item["errors"]:
            error_type_counts[error["type"]] += 1

        for observation in item["observations"]:
            observation_type_counts[
                observation["type"]
            ] += 1

    total_errors = sum(error_type_counts.values())
    total_observations = sum(
        observation_type_counts.values()
    )

    output = {
        "analyzer_version": "1.0.0",
        "evaluation_scope": (
            "Requirement Pipeline End-to-End Validation v1"
        ),
        "source_results": str(args.results),
        "summary": {
            "scenario_count": len(scenarios),
            "successful_scenarios": sum(
                bool(
                    item.get("scenario_success")
                )
                for item in analyses
            ),
            "failed_scenarios": len(failed_analyses),
            "scenarios_with_detected_errors": len(
                failed_analyses
            ),
            "total_detected_errors": total_errors,
            "scenarios_with_non_fatal_observations": len(
                observation_analyses
            ),
            "total_non_fatal_observations": (
                total_observations
            ),
        },
        "error_type_counts": dict(
            sorted(error_type_counts.items())
        ),
        "observation_type_counts": dict(
            sorted(observation_type_counts.items())
        ),
        "failed_scenarios": failed_analyses,
        "non_fatal_observations": observation_analyses,
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
    print("END-TO-END REQUIREMENT PIPELINE ERROR ANALYZER")
    print("=" * 92)
    print(f"Results                       : {args.results}")
    print(f"Scenarios                     : {len(scenarios)}")
    print(
        f"Functional failures           : "
        f"{len(failed_analyses)}"
    )
    print(
        f"Detected functional errors    : "
        f"{total_errors}"
    )
    print(
        f"Non-fatal observations        : "
        f"{total_observations}"
    )

    if error_type_counts:
        print("Error types:")
        for error_type, count in sorted(
            error_type_counts.items()
        ):
            print(f"  - {error_type}: {count}")
    else:
        print("Error types                    : none")

    if observation_type_counts:
        print("Observation types:")
        for observation_type, count in sorted(
            observation_type_counts.items()
        ):
            print(
                f"  - {observation_type}: {count}"
            )
    else:
        print("Observation types              : none")

    print(f"Output                        : {args.output}")
    print("=" * 92)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())