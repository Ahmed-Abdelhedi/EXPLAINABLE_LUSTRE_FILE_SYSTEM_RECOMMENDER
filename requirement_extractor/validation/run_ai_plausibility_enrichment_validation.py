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
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv


VALIDATION_DIR = Path(__file__).resolve().parent
REQUIREMENT_EXTRACTOR_DIR = VALIDATION_DIR.parent
REPOSITORY_ROOT = REQUIREMENT_EXTRACTOR_DIR.parent

DEFAULT_DATASET_PATH = (
    VALIDATION_DIR
    / "datasets"
    / "ai_plausibility_stress_dataset_v1.json"
)

DEFAULT_REPORTS_ROOT = (
    VALIDATION_DIR
    / "reports"
    / "ai_plausibility"
    / "enrichment"
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

TARGET_LABEL = "AMBIGUOUS"


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valide l'enrichissement LLM de l'AI Plausibility Agent "
            "uniquement sur les scénarios AMBIGUOUS. "
            "Chaque scénario est exécuté une fois sans LLM (baseline), "
            "puis une fois avec LLM, afin de vérifier la préservation "
            "de la décision et des issues."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Chemin du dataset AI Plausibility.",
    )

    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Dossier de sortie des résultats enrichment.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite facultative du nombre de scénarios AMBIGUOUS.",
    )

    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Filtre facultatif par catégorie, par exemple "
            "budget_requires_architecture."
        ),
    )

    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Surcharge facultative de OLLAMA_HOST.",
    )

    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Surcharge facultative de PLAUSIBILITY_AGENT_MODEL.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Surcharge de PLAUSIBILITY_AGENT_TEMPERATURE.",
    )

    parser.add_argument(
        "--debug-agent",
        action="store_true",
        help="Active PLAUSIBILITY_AGENT_DEBUG.",
    )

    parser.add_argument(
        "--omit-raw-response",
        action="store_true",
        help="N'enregistre pas les réponses brutes.",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Arrête au premier échec technique.",
    )

    return parser.parse_args()


# ============================================================
# Environment
# ============================================================

def configure_environment(
    args: argparse.Namespace,
) -> Dict[str, str]:
    load_dotenv(
        REPOSITORY_ROOT / ".env",
        override=False,
    )

    os.environ["ENABLE_AI_PLAUSIBILITY_AGENT"] = "true"
    os.environ["ENABLE_LLM_FALLBACK"] = "false"

    if args.ollama_host:
        os.environ["OLLAMA_HOST"] = args.ollama_host

    if args.ollama_model:
        os.environ[
            "PLAUSIBILITY_AGENT_MODEL"
        ] = args.ollama_model

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


# ============================================================
# Dataset
# ============================================================

def load_dataset(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        dataset = json.load(handle)

    if not isinstance(dataset, dict):
        raise ValueError(
            "La racine du dataset doit être un objet JSON."
        )

    scenarios = dataset.get("scenarios")

    if not isinstance(scenarios, list):
        raise ValueError(
            "Le dataset doit contenir une liste 'scenarios'."
        )

    return dataset


def select_ambiguous_scenarios(
    dataset: Dict[str, Any],
    category: Optional[str],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    scenarios = []

    for scenario in dataset["scenarios"]:
        if not isinstance(scenario, dict):
            continue

        if scenario.get("label") != TARGET_LABEL:
            continue

        expected = scenario.get("expected", {})

        if (
            expected.get("plausibility_status")
            != TARGET_LABEL
        ):
            raise ValueError(
                f"{scenario.get('id')} : label AMBIGUOUS "
                "mais expected.plausibility_status différent."
            )

        if category is not None:
            if scenario.get("category") != category:
                continue

        scenarios.append(scenario)

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit doit être strictement positif."
            )
        scenarios = scenarios[:limit]

    if not scenarios:
        raise ValueError(
            "Aucun scénario AMBIGUOUS sélectionné."
        )

    return scenarios


# ============================================================
# Structured input
# ============================================================

def make_final_json(
    plain_fields: Dict[str, Any],
    final_field_value_class: Any,
) -> Dict[str, Any]:
    final_json: Dict[str, Any] = {}

    for field_name in FIELD_UNITS:
        value = plain_fields.get(field_name)

        if value is None:
            final_json[field_name] = None
            continue

        final_json[field_name] = final_field_value_class(
            value=copy.deepcopy(value),
            unit=FIELD_UNITS[field_name],
            confidence=1.0,
            evidence=(
                "AI plausibility enrichment "
                "validation dataset"
            ),
            source="AI_PLAUSIBILITY_DATASET",
        )

    return final_json


def final_json_to_plain(
    final_json: Dict[str, Any],
) -> Dict[str, Any]:
    plain: Dict[str, Any] = {}

    for field_name in FIELD_UNITS:
        item = final_json.get(field_name)

        if item is None:
            plain[field_name] = None
            continue

        plain[field_name] = copy.deepcopy(
            getattr(item, "value", None)
        )

    return plain


# ============================================================
# Report serialization
# ============================================================

def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def serialize_issue(issue: Any) -> Dict[str, Any]:
    field_value = enum_value(
        getattr(issue, "field", None)
    )

    return {
        "issue_type": str(
            getattr(issue, "issue_type", "")
        ),
        "field": field_value,
        "severity": str(
            getattr(issue, "severity", "")
        ),
        "message": str(
            getattr(issue, "message", "")
        ),
        "question": str(
            getattr(issue, "question", "")
        ),
        "confidence": float(
            getattr(issue, "confidence", 0.0)
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
            getattr(report, "status", "")
        )
    ).upper()

    if status == str(status_class.OK).upper():
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
            getattr(report, "status", "")
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
    plain_before: Dict[str, Any],
    plain_after: Dict[str, Any],
    expected: Dict[str, Any],
    serialized_issues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []

    if plain_before != plain_after:
        violations.append(
            {
                "type": "INPUT_MUTATION",
                "message": (
                    "L'agent a modifié l'entrée "
                    "structurée."
                ),
            }
        )

    protected = set(
        expected.get(
            "must_not_infer",
            [],
        )
    )

    for issue in serialized_issues:
        field_name = issue.get("field")
        suggested = issue.get(
            "suggested_correction"
        )

        if (
            field_name in protected
            and plain_before.get(field_name)
            is None
            and isinstance(suggested, dict)
            and suggested.get("value")
            is not None
        ):
            violations.append(
                {
                    "type": (
                        "SUGGESTED_VALUE_FOR_"
                        "MISSING_FIELD"
                    ),
                    "field": field_name,
                    "suggested_correction": (
                        suggested
                    ),
                }
            )

    return violations


# ============================================================
# One agent execution
# ============================================================

def execute_once(
    agent: Any,
    status_class: Any,
    final_field_value_class: Any,
    scenario: Dict[str, Any],
    omit_raw_response: bool,
) -> Dict[str, Any]:
    fields = copy.deepcopy(
        scenario["fields"]
    )
    expected = copy.deepcopy(
        scenario["expected"]
    )

    final_json = make_final_json(
        plain_fields=fields,
        final_field_value_class=(
            final_field_value_class
        ),
    )

    before = final_json_to_plain(
        final_json
    )

    start = time.perf_counter()

    try:
        report = agent.analyze(final_json)
        latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

        after = final_json_to_plain(
            final_json
        )

        issues = [
            serialize_issue(issue)
            for issue in (
                getattr(
                    report,
                    "issues",
                    None,
                )
                or []
            )
        ]

        violations = (
            detect_no_inference_violations(
                plain_before=before,
                plain_after=after,
                expected=expected,
                serialized_issues=issues,
            )
        )

        raw_response = str(
            getattr(
                report,
                "raw_response",
                "",
            )
            or ""
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

        result = {
            "raw_status": str(
                enum_value(
                    getattr(
                        report,
                        "status",
                        "",
                    )
                )
            ),
            "predicted_label": (
                predicted_label_from_report(
                    report,
                    status_class,
                )
            ),
            "should_block_recommendation": (
                report_should_block(
                    report,
                    status_class,
                )
            ),
            "issues": issues,
            "issue_count": len(issues),
            "input_mutated": before != after,
            "input_after_analysis": after,
            "no_inference_violations": (
                violations
            ),
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "technical_error": None,
            "enrichment_error": (
                enrichment_error
            ),
        }

        if not omit_raw_response:
            result[
                "raw_response"
            ] = raw_response

        return result

    except Exception as exc:
        latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return {
            "raw_status": None,
            "predicted_label": None,
            "should_block_recommendation": None,
            "issues": [],
            "issue_count": 0,
            "input_mutated": False,
            "input_after_analysis": before,
            "no_inference_violations": [],
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "technical_error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "enrichment_error": None,
        }


# ============================================================
# Preservation comparison
# ============================================================

def identity_tuple(
    issue: Dict[str, Any],
) -> Tuple[str, str, str]:
    return (
        str(
            issue.get(
                "issue_type",
                "",
            )
        ),
        str(
            issue.get(
                "field",
                "",
            )
        ),
        str(
            issue.get(
                "severity",
                "",
            )
        ),
    )


def identity_counter(
    issues: List[Dict[str, Any]],
) -> Counter:
    return Counter(
        identity_tuple(issue)
        for issue in issues
    )


def pair_issues_by_identity(
    baseline: List[Dict[str, Any]],
    enriched: List[Dict[str, Any]],
) -> List[
    Tuple[
        Dict[str, Any],
        Optional[Dict[str, Any]],
    ]
]:
    unused = list(enriched)
    pairs = []

    for base_issue in baseline:
        identity = identity_tuple(
            base_issue
        )

        match_index = None

        for index, candidate in enumerate(
            unused
        ):
            if (
                identity_tuple(candidate)
                == identity
            ):
                match_index = index
                break

        if match_index is None:
            pairs.append(
                (base_issue, None)
            )
        else:
            pairs.append(
                (
                    base_issue,
                    unused.pop(
                        match_index
                    ),
                )
            )

    return pairs


def compare_baseline_and_enriched(
    baseline: Dict[str, Any],
    enriched: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_issues = baseline.get(
        "issues",
        [],
    )
    enriched_issues = enriched.get(
        "issues",
        [],
    )

    baseline_counter = (
        identity_counter(
            baseline_issues
        )
    )
    enriched_counter = (
        identity_counter(
            enriched_issues
        )
    )

    new_identities = list(
        (
            enriched_counter
            - baseline_counter
        ).elements()
    )

    deleted_identities = list(
        (
            baseline_counter
            - enriched_counter
        ).elements()
    )

    pairs = pair_issues_by_identity(
        baseline_issues,
        enriched_issues,
    )

    issue_text_changes = []

    immutable_attribute_changes = []

    for base_issue, enriched_issue in pairs:
        if enriched_issue is None:
            continue

        identity = identity_tuple(
            base_issue
        )

        message_changed = (
            base_issue.get("message")
            != enriched_issue.get("message")
        )

        question_changed = (
            base_issue.get("question")
            != enriched_issue.get("question")
        )

        issue_text_changes.append(
            {
                "identity": list(identity),
                "message_changed": (
                    message_changed
                ),
                "question_changed": (
                    question_changed
                ),
            }
        )

        # Ces attributs ne doivent jamais
        # être altérés par l'enrichissement.
        for attribute in (
            "confidence",
            "suggested_correction",
            "evidence_fields",
        ):
            if (
                base_issue.get(attribute)
                != enriched_issue.get(attribute)
            ):
                immutable_attribute_changes.append(
                    {
                        "identity": list(
                            identity
                        ),
                        "attribute": (
                            attribute
                        ),
                        "baseline": (
                            base_issue.get(
                                attribute
                            )
                        ),
                        "enriched": (
                            enriched_issue.get(
                                attribute
                            )
                        ),
                    }
                )

    status_preserved = (
        baseline.get("raw_status")
        == enriched.get("raw_status")
    )

    predicted_label_preserved = (
        baseline.get(
            "predicted_label"
        )
        == enriched.get(
            "predicted_label"
        )
    )

    should_block_preserved = (
        baseline.get(
            "should_block_recommendation"
        )
        == enriched.get(
            "should_block_recommendation"
        )
    )

    issue_identity_preserved = (
        baseline_counter
        == enriched_counter
    )

    immutable_attributes_preserved = (
        not immutable_attribute_changes
    )

    no_new_issue = not new_identities
    no_deleted_issue = (
        not deleted_identities
    )

    input_safe = (
        not enriched.get(
            "input_mutated",
            False,
        )
    )

    no_inference_safe = (
        len(
            enriched.get(
                "no_inference_violations",
                [],
            )
        )
        == 0
    )

    safety_preserved = all(
        (
            status_preserved,
            predicted_label_preserved,
            should_block_preserved,
            issue_identity_preserved,
            immutable_attributes_preserved,
            no_new_issue,
            no_deleted_issue,
            input_safe,
            no_inference_safe,
        )
    )

    return {
        "status_preserved": (
            status_preserved
        ),
        "predicted_label_preserved": (
            predicted_label_preserved
        ),
        "should_block_preserved": (
            should_block_preserved
        ),
        "issue_identity_preserved": (
            issue_identity_preserved
        ),
        "issue_count_preserved": (
            baseline.get("issue_count")
            == enriched.get("issue_count")
        ),
        "no_new_issue": (
            no_new_issue
        ),
        "no_deleted_issue": (
            no_deleted_issue
        ),
        "new_issue_identities": [
            list(item)
            for item in new_identities
        ],
        "deleted_issue_identities": [
            list(item)
            for item in deleted_identities
        ],
        "immutable_attributes_preserved": (
            immutable_attributes_preserved
        ),
        "immutable_attribute_changes": (
            immutable_attribute_changes
        ),
        "input_safe": input_safe,
        "no_inference_safe": (
            no_inference_safe
        ),
        "issue_text_changes": (
            issue_text_changes
        ),
        "message_changed_count": sum(
            1
            for item in issue_text_changes
            if item["message_changed"]
        ),
        "question_changed_count": sum(
            1
            for item in issue_text_changes
            if item["question_changed"]
        ),
        "safety_preserved": (
            safety_preserved
        ),
    }


# ============================================================
# Validation run
# ============================================================

def make_agents(
    agent_class: Any,
) -> Tuple[Any, Any]:
    # Baseline : aucune utilisation du LLM.
    os.environ[
        "PLAUSIBILITY_AGENT_USE_LLM_ENRICHMENT"
    ] = "false"
    baseline_agent = agent_class()

    # Enriched : LLM actif uniquement sur WARNING.
    os.environ[
        "PLAUSIBILITY_AGENT_USE_LLM_ENRICHMENT"
    ] = "true"
    enriched_agent = agent_class()

    return (
        baseline_agent,
        enriched_agent,
    )


def run_validation(
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

    dataset = load_dataset(
        dataset_path
    )

    scenarios = (
        select_ambiguous_scenarios(
            dataset=dataset,
            category=args.category,
            limit=args.limit,
        )
    )

    reports_root = (
        args.reports_root.resolve()
    )
    reports_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        reports_root
        / "results_ai_plausibility_enrichment.json"
    )

    baseline_agent, enriched_agent = (
        make_agents(
            AIPlausibilityAgent
        )
    )

    started = datetime.now(
        timezone.utc
    )
    global_start = time.perf_counter()

    results = []

    baseline_technical_errors = 0
    enriched_technical_errors = 0
    enrichment_errors = 0
    preservation_failures = 0

    category_counts = Counter(
        scenario.get(
            "category",
            "unknown",
        )
        for scenario in scenarios
    )

    print("=" * 82)
    print(
        "AI PLAUSIBILITY - LLM ENRICHMENT VALIDATION"
    )
    print("=" * 82)
    print(
        f"Dataset      : {dataset_path}"
    )
    print(
        f"Scenarios    : {len(scenarios)} AMBIGUOUS"
    )
    print(
        f"Model        : "
        f"{environment['PLAUSIBILITY_AGENT_MODEL']}"
    )
    print(
        f"Ollama       : "
        f"{environment['OLLAMA_HOST']}"
    )
    print(
        f"Temperature  : "
        f"{environment['PLAUSIBILITY_AGENT_TEMPERATURE']}"
    )
    print(
        f"Report       : {results_path}"
    )
    print("=" * 82)

    for position, scenario in enumerate(
        scenarios,
        start=1,
    ):
        scenario_id = scenario["id"]

        print(
            f"[{position:02d}/{len(scenarios):02d}] "
            f"{scenario_id} | "
            f"{scenario.get('category')}"
        )

        baseline = execute_once(
            agent=baseline_agent,
            status_class=(
                AIPlausibilityStatus
            ),
            final_field_value_class=(
                FinalFieldValue
            ),
            scenario=scenario,
            omit_raw_response=(
                args.omit_raw_response
            ),
        )

        enriched = execute_once(
            agent=enriched_agent,
            status_class=(
                AIPlausibilityStatus
            ),
            final_field_value_class=(
                FinalFieldValue
            ),
            scenario=scenario,
            omit_raw_response=(
                args.omit_raw_response
            ),
        )

        if (
            baseline[
                "technical_error"
            ]
            is not None
        ):
            baseline_technical_errors += 1

        if (
            enriched[
                "technical_error"
            ]
            is not None
        ):
            enriched_technical_errors += 1

        if (
            enriched[
                "enrichment_error"
            ]
            is not None
        ):
            enrichment_errors += 1

        if (
            baseline[
                "technical_error"
            ]
            is None
            and enriched[
                "technical_error"
            ]
            is None
        ):
            preservation = (
                compare_baseline_and_enriched(
                    baseline=baseline,
                    enriched=enriched,
                )
            )
        else:
            preservation = {
                "status_preserved": False,
                "predicted_label_preserved": False,
                "should_block_preserved": False,
                "issue_identity_preserved": False,
                "issue_count_preserved": False,
                "no_new_issue": False,
                "no_deleted_issue": False,
                "new_issue_identities": [],
                "deleted_issue_identities": [],
                "immutable_attributes_preserved": False,
                "immutable_attribute_changes": [],
                "input_safe": False,
                "no_inference_safe": False,
                "issue_text_changes": [],
                "message_changed_count": 0,
                "question_changed_count": 0,
                "safety_preserved": False,
            }

        if not preservation[
            "safety_preserved"
        ]:
            preservation_failures += 1

        result = {
            "id": scenario_id,
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
            "expected": copy.deepcopy(
                scenario.get(
                    "expected",
                    {},
                )
            ),
            "baseline": baseline,
            "enriched": enriched,
            "preservation": (
                preservation
            ),
        }

        results.append(result)

        print(
            "  baseline="
            f"{baseline.get('raw_status')} "
            f"{baseline.get('latency_ms'):.3f} ms | "
            "enriched="
            f"{enriched.get('raw_status')} "
            f"{enriched.get('latency_ms'):.3f} ms | "
            "safe="
            f"{preservation['safety_preserved']} | "
            "msgΔ="
            f"{preservation['message_changed_count']} | "
            "qΔ="
            f"{preservation['question_changed_count']}"
        )

        if (
            args.fail_fast
            and (
                baseline[
                    "technical_error"
                ]
                is not None
                or enriched[
                    "technical_error"
                ]
                is not None
                or not preservation[
                    "safety_preserved"
                ]
            )
        ):
            break

    total_duration = (
        time.perf_counter()
        - global_start
    )

    baseline_latencies = [
        float(
            item["baseline"][
                "latency_ms"
            ]
        )
        for item in results
    ]

    enriched_latencies = [
        float(
            item["enriched"][
                "latency_ms"
            ]
        )
        for item in results
    ]

    message_changed_scenarios = sum(
        1
        for item in results
        if item["preservation"][
            "message_changed_count"
        ]
        > 0
    )

    question_changed_scenarios = sum(
        1
        for item in results
        if item["preservation"][
            "question_changed_count"
        ]
        > 0
    )

    final_report = {
        "run": {
            "started_at_utc": (
                started.isoformat()
            ),
            "finished_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "environment": {
                **environment,
                (
                    "PLAUSIBILITY_AGENT_"
                    "USE_LLM_ENRICHMENT"
                ): "true",
            },
            "repository_root": str(
                REPOSITORY_ROOT
            ),
            "dataset_path": str(
                dataset_path
            ),
            "results_path": str(
                results_path
            ),
        },
        "experiment": {
            "name": (
                "AI Plausibility LLM "
                "Enrichment Safety Validation"
            ),
            "scope": (
                "AMBIGUOUS scenarios only"
            ),
            "baseline": (
                "Deterministic guarded "
                "plausibility checks with "
                "LLM enrichment disabled"
            ),
            "treatment": (
                "Same deterministic decision "
                "with LLM enrichment enabled; "
                "only message/question may change"
            ),
            "target_label": (
                TARGET_LABEL
            ),
            "selected_categories": dict(
                sorted(
                    category_counts.items()
                )
            ),
        },
        "execution_summary": {
            "requested_scenarios": (
                len(scenarios)
            ),
            "executed_scenarios": (
                len(results)
            ),
            "baseline_technical_errors": (
                baseline_technical_errors
            ),
            "enriched_technical_errors": (
                enriched_technical_errors
            ),
            "llm_enrichment_errors": (
                enrichment_errors
            ),
            "preservation_failures": (
                preservation_failures
            ),
            "safe_preservation_count": (
                len(results)
                - preservation_failures
            ),
            "message_changed_scenarios": (
                message_changed_scenarios
            ),
            "question_changed_scenarios": (
                question_changed_scenarios
            ),
            "total_duration_seconds": round(
                total_duration,
                3,
            ),
            "average_baseline_latency_ms": round(
                (
                    sum(
                        baseline_latencies
                    )
                    / len(
                        baseline_latencies
                    )
                )
                if baseline_latencies
                else 0.0,
                3,
            ),
            "average_enriched_latency_ms": round(
                (
                    sum(
                        enriched_latencies
                    )
                    / len(
                        enriched_latencies
                    )
                )
                if enriched_latencies
                else 0.0,
                3,
            ),
        },
        "scenarios": results,
    }

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            final_report,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 82)
    print("ENRICHMENT VALIDATION COMPLETE")
    print("=" * 82)
    print(
        f"Executed              : "
        f"{len(results)}"
    )
    print(
        f"Baseline tech errors  : "
        f"{baseline_technical_errors}"
    )
    print(
        f"Enriched tech errors  : "
        f"{enriched_technical_errors}"
    )
    print(
        f"LLM enrichment errors : "
        f"{enrichment_errors}"
    )
    print(
        f"Preservation failures : "
        f"{preservation_failures}"
    )
    print(
        f"Safe preservation     : "
        f"{len(results) - preservation_failures}"
        f"/{len(results)}"
    )
    print(
        f"Message changed       : "
        f"{message_changed_scenarios}"
        f"/{len(results)}"
    )
    print(
        f"Question changed      : "
        f"{question_changed_scenarios}"
        f"/{len(results)}"
    )
    print(
        f"Average baseline      : "
        f"{final_report['execution_summary']['average_baseline_latency_ms']}"
        " ms"
    )
    print(
        f"Average enriched      : "
        f"{final_report['execution_summary']['average_enriched_latency_ms']}"
        " ms"
    )
    print(
        f"Results               : "
        f"{results_path}"
    )

    return results_path


def main() -> int:
    args = parse_args()

    try:
        run_validation(args)
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
    raise SystemExit(main())
