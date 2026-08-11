#!/usr/bin/env python3
"""
run_ai_plausibility_full_agent_validation.py

Validation finale de l'AI Plausibility Agent complet :
- 150 scénarios du dataset dédié ;
- deterministic plausibility guard actif ;
- LLM enrichment actif ;
- aucune LLM fallback d'extraction ;
- COHERENT / AMBIGUOUS / INCOHERENT évalués ensemble.

Sortie par défaut :
    requirement_extractor/validation/reports/ai_plausibility/full_agent/
        results_ai_plausibility_full_agent.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


VALIDATION_DIR = Path(__file__).resolve().parent
REQUIREMENT_EXTRACTOR_DIR = VALIDATION_DIR.parent
REPOSITORY_ROOT = REQUIREMENT_EXTRACTOR_DIR.parent

DEFAULT_DATASET = (
    VALIDATION_DIR
    / "datasets"
    / "ai_plausibility_stress_dataset_v1.json"
)

DEFAULT_OUTPUT = (
    VALIDATION_DIR
    / "reports"
    / "ai_plausibility"
    / "full_agent"
    / "results_ai_plausibility_full_agent.json"
)

FIELD_UNITS: Dict[str, Optional[str]] = {
    "requested_usable_capacity_tib": "TiB",
    "client_count": None,
    "average_file_size_gb": "GB",
    "max_file_size_gb": "GB",
    "total_file_count": None,
    "read_write_ratio": "%",
    "access_type": None,
    "target_read_gbps": "GB/s",
    "target_write_gbps": "GB/s",
    "ha_required": None,
    "max_budget_usd": "USD",
    "max_power_w": "W",
    "annual_growth_percent": "%",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation finale de l'AI Plausibility Agent complet "
            "sur les 150 scénarios avec LLM enrichment activé."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite facultative pour un smoke test.",
    )

    parser.add_argument(
        "--label",
        choices=[
            "COHERENT",
            "AMBIGUOUS",
            "INCOHERENT",
        ],
        default=None,
        help="Filtre facultatif par label.",
    )

    parser.add_argument(
        "--ollama-model",
        default=None,
    )

    parser.add_argument(
        "--ollama-host",
        default=None,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--debug-agent",
        action="store_true",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
    )

    return parser.parse_args()


def configure_environment(
    args: argparse.Namespace,
) -> Dict[str, str]:
    load_dotenv(
        REPOSITORY_ROOT / ".env",
        override=False,
    )

    os.environ[
        "ENABLE_AI_PLAUSIBILITY_AGENT"
    ] = "true"

    os.environ[
        "ENABLE_LLM_FALLBACK"
    ] = "false"

    os.environ[
        "PLAUSIBILITY_AGENT_USE_LLM_ENRICHMENT"
    ] = "true"

    if args.ollama_model:
        os.environ[
            "PLAUSIBILITY_AGENT_MODEL"
        ] = args.ollama_model

    if args.ollama_host:
        os.environ[
            "OLLAMA_HOST"
        ] = args.ollama_host

    if args.temperature is not None:
        os.environ[
            "PLAUSIBILITY_AGENT_TEMPERATURE"
        ] = str(args.temperature)

    if args.debug_agent:
        os.environ[
            "PLAUSIBILITY_AGENT_DEBUG"
        ] = "true"

    return {
        "ENABLE_AI_PLAUSIBILITY_AGENT": os.getenv(
            "ENABLE_AI_PLAUSIBILITY_AGENT",
            "",
        ),
        "ENABLE_LLM_FALLBACK": os.getenv(
            "ENABLE_LLM_FALLBACK",
            "",
        ),
        "PLAUSIBILITY_AGENT_USE_LLM_ENRICHMENT": os.getenv(
            "PLAUSIBILITY_AGENT_USE_LLM_ENRICHMENT",
            "",
        ),
        "OLLAMA_HOST": os.getenv(
            "OLLAMA_HOST",
            "http://localhost:11434",
        ),
        "PLAUSIBILITY_AGENT_MODEL": os.getenv(
            "PLAUSIBILITY_AGENT_MODEL",
            os.getenv(
                "OLLAMA_MODEL",
                "qwen2.5:3b",
            ),
        ),
        "PLAUSIBILITY_AGENT_TEMPERATURE": os.getenv(
            "PLAUSIBILITY_AGENT_TEMPERATURE",
            "0",
        ),
        "PLAUSIBILITY_AGENT_DEBUG": os.getenv(
            "PLAUSIBILITY_AGENT_DEBUG",
            "false",
        ),
    }


def load_dataset(
    path: Path,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "La racine du dataset doit être un objet JSON."
        )

    scenarios = data.get("scenarios")

    if not isinstance(scenarios, list):
        raise ValueError(
            "La clé 'scenarios' doit être une liste."
        )

    return data


def select_scenarios(
    dataset: Dict[str, Any],
    label: Optional[str],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    selected = []

    for scenario in dataset["scenarios"]:
        if not isinstance(scenario, dict):
            continue

        if (
            label is not None
            and scenario.get("label") != label
        ):
            continue

        selected.append(scenario)

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit doit être strictement positif."
            )

        selected = selected[:limit]

    if not selected:
        raise ValueError(
            "Aucun scénario sélectionné."
        )

    return selected


def make_final_json(
    plain_fields: Dict[str, Any],
    final_field_value_class: Any,
) -> Dict[str, Any]:
    final_json: Dict[str, Any] = {}

    for field_name in FIELD_UNITS:
        value = plain_fields.get(
            field_name
        )

        if value is None:
            final_json[field_name] = None
            continue

        final_json[field_name] = (
            final_field_value_class(
                value=copy.deepcopy(value),
                unit=FIELD_UNITS[
                    field_name
                ],
                confidence=1.0,
                evidence=(
                    "AI plausibility full-agent "
                    "validation dataset"
                ),
                source=(
                    "AI_PLAUSIBILITY_DATASET"
                ),
            )
        )

    return final_json


def final_json_to_plain(
    final_json: Dict[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    for field_name in FIELD_UNITS:
        item = final_json.get(
            field_name
        )

        if item is None:
            result[field_name] = None
        else:
            result[field_name] = copy.deepcopy(
                getattr(
                    item,
                    "value",
                    None,
                )
            )

    return result


def enum_value(
    value: Any,
) -> Any:
    return getattr(
        value,
        "value",
        value,
    )


def serialize_issue(
    issue: Any,
) -> Dict[str, Any]:
    return {
        "issue_type": str(
            getattr(
                issue,
                "issue_type",
                "",
            )
        ),
        "field": enum_value(
            getattr(
                issue,
                "field",
                None,
            )
        ),
        "severity": str(
            getattr(
                issue,
                "severity",
                "",
            )
        ),
        "message": str(
            getattr(
                issue,
                "message",
                "",
            )
        ),
        "question": str(
            getattr(
                issue,
                "question",
                "",
            )
        ),
        "confidence": float(
            getattr(
                issue,
                "confidence",
                0.0,
            )
            or 0.0
        ),
        "suggested_correction": copy.deepcopy(
            getattr(
                issue,
                "suggested_correction",
                None,
            )
        ),
        "evidence_fields": copy.deepcopy(
            getattr(
                issue,
                "evidence_fields",
                {},
            )
            or {}
        ),
    }


def predicted_label_from_report(
    report: Any,
    status_class: Any,
) -> str:
    status = str(
        enum_value(
            getattr(
                report,
                "status",
                "",
            )
        )
    ).upper()

    if status == str(
        status_class.OK
    ).upper():
        return "COHERENT"

    if status == str(
        status_class.WARNING
    ).upper():
        return "AMBIGUOUS"

    if status in {
        str(
            status_class.NEEDS_CLARIFICATION
        ).upper(),
        str(
            status_class.BLOCKING
        ).upper(),
    }:
        return "INCOHERENT"

    return "__UNKNOWN__"


def report_should_block(
    report: Any,
    status_class: Any,
) -> bool:
    status = str(
        enum_value(
            getattr(
                report,
                "status",
                "",
            )
        )
    ).upper()

    return status in {
        str(
            status_class.NEEDS_CLARIFICATION
        ).upper(),
        str(
            status_class.BLOCKING
        ).upper(),
    }


def detect_no_inference_violations(
    before: Dict[str, Any],
    after: Dict[str, Any],
    expected: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    violations: List[
        Dict[str, Any]
    ] = []

    if before != after:
        violations.append(
            {
                "type": "INPUT_MUTATION",
                "message": (
                    "L'agent a modifié "
                    "les requirements."
                ),
            }
        )

    protected = set(
        expected.get(
            "must_not_infer",
            [],
        )
    )

    for issue in issues:
        field_name = issue.get(
            "field"
        )

        suggested = issue.get(
            "suggested_correction"
        )

        if (
            field_name in protected
            and before.get(
                field_name
            )
            is None
            and isinstance(
                suggested,
                dict,
            )
            and suggested.get(
                "value"
            )
            is not None
        ):
            violations.append(
                {
                    "type": (
                        "MISSING_VALUE_INFERENCE"
                    ),
                    "field": field_name,
                    "suggested_correction": (
                        suggested
                    ),
                }
            )

    return violations


def execute_scenario(
    agent: Any,
    status_class: Any,
    final_field_value_class: Any,
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    fields = copy.deepcopy(
        scenario.get(
            "fields",
            {},
        )
    )

    expected = copy.deepcopy(
        scenario.get(
            "expected",
            {},
        )
    )

    final_json = make_final_json(
        fields,
        final_field_value_class,
    )

    before = final_json_to_plain(
        final_json
    )

    start = time.perf_counter()

    try:
        report = agent.analyze(
            final_json
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        after = final_json_to_plain(
            final_json
        )

        issues = [
            serialize_issue(
                issue
            )
            for issue in (
                getattr(
                    report,
                    "issues",
                    None,
                )
                or []
            )
        ]

        raw_response = str(
            getattr(
                report,
                "raw_response",
                "",
            )
            or ""
        )

        no_inference_violations = (
            detect_no_inference_violations(
                before=before,
                after=after,
                expected=expected,
                issues=issues,
            )
        )

        raw_status = str(
            enum_value(
                getattr(
                    report,
                    "status",
                    "",
                )
            )
        )

        predicted_label = (
            predicted_label_from_report(
                report,
                status_class,
            )
        )

        llm_path_expected = (
            raw_status.upper()
            == str(
                status_class.WARNING
            ).upper()
        )

        enrichment_error = None

        if raw_response.startswith(
            "AI plausibility enrichment unavailable:"
        ):
            enrichment_error = {
                "type": (
                    "LLM_ENRICHMENT_UNAVAILABLE"
                ),
                "message": raw_response,
            }

        # Purely observational flag:
        # this does NOT claim an exact Ollama request count.
        if llm_path_expected:
            if raw_response.startswith(
                "{"
            ) and (
                '"decision_source": '
                '"DETERMINISTIC_PLAUSIBILITY_GUARD"'
                in raw_response
            ):
                llm_output_observed = False
            else:
                llm_output_observed = True
        else:
            llm_output_observed = False

        return {
            "raw_status": (
                raw_status
            ),
            "predicted_label": (
                predicted_label
            ),
            "should_block_recommendation": (
                report_should_block(
                    report,
                    status_class,
                )
            ),
            "issues": issues,
            "issue_count": len(
                issues
            ),
            "input_mutated": (
                before != after
            ),
            "input_after_analysis": (
                after
            ),
            "no_inference_violations": (
                no_inference_violations
            ),
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "technical_error": None,
            "enrichment_error": (
                enrichment_error
            ),
            "llm_enrichment_path_expected": (
                llm_path_expected
            ),
            "llm_output_observed": (
                llm_output_observed
            ),
            "raw_response": (
                raw_response
            ),
        }

    except Exception as exc:
        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return {
            "raw_status": None,
            "predicted_label": None,
            "should_block_recommendation": (
                None
            ),
            "issues": [],
            "issue_count": 0,
            "input_mutated": False,
            "input_after_analysis": (
                before
            ),
            "no_inference_violations": [],
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "technical_error": {
                "type": type(
                    exc
                ).__name__,
                "message": str(
                    exc
                ),
            },
            "enrichment_error": None,
            "llm_enrichment_path_expected": (
                False
            ),
            "llm_output_observed": (
                False
            ),
            "raw_response": "",
        }


def run(
    args: argparse.Namespace,
) -> Path:
    environment = configure_environment(
        args
    )

    from requirement_extractor.ai_plausibility_agent import (
        AIPlausibilityAgent,
        AIPlausibilityStatus,
    )
    from requirement_extractor.models import (
        FinalFieldValue,
    )

    dataset_path = (
        args.dataset.resolve()
    )

    output_path = (
        args.output.resolve()
    )

    dataset = load_dataset(
        dataset_path
    )

    scenarios = select_scenarios(
        dataset=dataset,
        label=args.label,
        limit=args.limit,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    agent = AIPlausibilityAgent()

    expected_label_counts = Counter(
        scenario.get(
            "label",
            "__MISSING__",
        )
        for scenario in scenarios
    )

    category_counts = Counter(
        scenario.get(
            "category",
            "__MISSING__",
        )
        for scenario in scenarios
    )

    started_at = datetime.now(
        timezone.utc
    )

    global_start = time.perf_counter()

    results = []

    technical_errors = 0
    enrichment_errors = 0
    status_matches = 0
    block_matches = 0
    mutations = 0
    no_inference_violations = 0

    print(
        "=" * 84
    )
    print(
        "AI PLAUSIBILITY - FULL AGENT VALIDATION"
    )
    print(
        "=" * 84
    )
    print(
        f"Dataset      : {dataset_path}"
    )
    print(
        f"Scenarios    : {len(scenarios)}"
    )
    print(
        f"Model        : "
        f"{environment['PLAUSIBILITY_AGENT_MODEL']}"
    )
    print(
        "LLM enrich   : true"
    )
    print(
        f"Output       : {output_path}"
    )
    print(
        "=" * 84
    )

    for index, scenario in enumerate(
        scenarios,
        start=1,
    ):
        actual = execute_scenario(
            agent=agent,
            status_class=(
                AIPlausibilityStatus
            ),
            final_field_value_class=(
                FinalFieldValue
            ),
            scenario=scenario,
        )

        expected = copy.deepcopy(
            scenario.get(
                "expected",
                {},
            )
        )

        expected_label = expected.get(
            "plausibility_status",
            scenario.get("label"),
        )

        expected_block = bool(
            expected.get(
                "should_block_recommendation",
                False,
            )
        )

        status_match = (
            actual.get(
                "predicted_label"
            )
            == expected_label
        )

        block_match = (
            actual.get(
                "should_block_recommendation"
            )
            == expected_block
        )

        if (
            actual.get(
                "technical_error"
            )
            is not None
        ):
            technical_errors += 1

        if (
            actual.get(
                "enrichment_error"
            )
            is not None
        ):
            enrichment_errors += 1

        if status_match:
            status_matches += 1

        if block_match:
            block_matches += 1

        if actual.get(
            "input_mutated"
        ):
            mutations += 1

        no_inference_violations += len(
            actual.get(
                "no_inference_violations",
                [],
            )
        )

        result = {
            "id": scenario.get(
                "id"
            ),
            "label": scenario.get(
                "label"
            ),
            "category": scenario.get(
                "category"
            ),
            "difficulty": scenario.get(
                "difficulty"
            ),
            "language": scenario.get(
                "language"
            ),
            "user_text": scenario.get(
                "user_text"
            ),
            "fields": copy.deepcopy(
                scenario.get(
                    "fields",
                    {},
                )
            ),
            "expected": expected,
            "actual": actual,
            "quick_checks": {
                "status_match": (
                    status_match
                ),
                "should_block_match": (
                    block_match
                ),
            },
        }

        results.append(
            result
        )

        print(
            f"[{index:03d}/{len(scenarios):03d}] "
            f"{scenario.get('id')} | "
            f"expected={expected_label:<10} | "
            f"actual={str(actual.get('predicted_label')):<10} | "
            f"status_ok={status_match} | "
            f"block_ok={block_match} | "
            f"lat={actual.get('latency_ms', 0.0):.3f} ms"
        )

        if (
            args.fail_fast
            and (
                actual.get(
                    "technical_error"
                )
                is not None
                or not status_match
                or not block_match
                or actual.get(
                    "input_mutated"
                )
                or actual.get(
                    "no_inference_violations"
                )
            )
        ):
            break

    duration = (
        time.perf_counter()
        - global_start
    )

    latencies = [
        float(
            item[
                "actual"
            ][
                "latency_ms"
            ]
        )
        for item in results
    ]

    llm_path_expected_count = sum(
        1
        for item in results
        if item[
            "actual"
        ].get(
            "llm_enrichment_path_expected"
        )
    )

    llm_output_observed_count = sum(
        1
        for item in results
        if item[
            "actual"
        ].get(
            "llm_output_observed"
        )
    )

    report = {
        "run": {
            "started_at_utc": (
                started_at.isoformat()
            ),
            "finished_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "environment": (
                environment
            ),
            "repository_root": str(
                REPOSITORY_ROOT
            ),
            "dataset_path": str(
                dataset_path
            ),
            "results_path": str(
                output_path
            ),
        },
        "experiment": {
            "name": (
                "AI Plausibility Full Agent "
                "Validation"
            ),
            "scope": (
                "All dedicated plausibility labels "
                "with LLM enrichment enabled"
            ),
            "architecture": (
                "Deterministic guarded decision layer "
                "+ controlled LLM warning enrichment"
            ),
            "label_filter": (
                args.label
            ),
            "expected_label_distribution": dict(
                sorted(
                    expected_label_counts.items()
                )
            ),
            "category_distribution": dict(
                sorted(
                    category_counts.items()
                )
            ),
        },
        "execution_summary": {
            "requested_scenarios": len(
                scenarios
            ),
            "executed_scenarios": len(
                results
            ),
            "technical_errors": (
                technical_errors
            ),
            "llm_enrichment_errors": (
                enrichment_errors
            ),
            "status_matches": (
                status_matches
            ),
            "status_match_rate": (
                status_matches
                / len(results)
                if results
                else 0.0
            ),
            "should_block_matches": (
                block_matches
            ),
            "should_block_match_rate": (
                block_matches
                / len(results)
                if results
                else 0.0
            ),
            "input_mutations": (
                mutations
            ),
            "no_inference_violation_count": (
                no_inference_violations
            ),
            "llm_enrichment_path_expected_count": (
                llm_path_expected_count
            ),
            "llm_output_observed_count": (
                llm_output_observed_count
            ),
            "total_duration_seconds": round(
                duration,
                3,
            ),
            "average_latency_ms": round(
                sum(latencies)
                / len(latencies)
                if latencies
                else 0.0,
                3,
            ),
            "max_latency_ms": round(
                max(latencies)
                if latencies
                else 0.0,
                3,
            ),
        },
        "scenarios": (
            results
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "=" * 84
    )
    print(
        "FULL AGENT VALIDATION COMPLETE"
    )
    print(
        "=" * 84
    )
    print(
        f"Executed              : "
        f"{len(results)}"
    )
    print(
        f"Status correct        : "
        f"{status_matches}/{len(results)}"
    )
    print(
        f"Block correct         : "
        f"{block_matches}/{len(results)}"
    )
    print(
        f"Technical errors      : "
        f"{technical_errors}"
    )
    print(
        f"LLM enrich errors     : "
        f"{enrichment_errors}"
    )
    print(
        f"Input mutations       : "
        f"{mutations}"
    )
    print(
        f"No-inference violations: "
        f"{no_inference_violations}"
    )
    print(
        f"LLM warning path      : "
        f"{llm_path_expected_count}"
    )
    print(
        f"Observed LLM outputs  : "
        f"{llm_output_observed_count}"
    )
    print(
        f"Average latency       : "
        f"{report['execution_summary']['average_latency_ms']} ms"
    )
    print(
        f"Results               : "
        f"{output_path}"
    )

    return output_path


def main() -> int:
    args = parse_args()

    try:
        run(
            args
        )
        return 0

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        print(
            f"VALIDATION ERROR: {exc}",
            file=sys.stderr,
        )
        return 2

    except KeyboardInterrupt:
        print(
            "\nExecution interrupted.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
