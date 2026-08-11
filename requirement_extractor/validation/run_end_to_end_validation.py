#!/usr/bin/env python3
"""
run_end_to_end_validation.py

Validation End-to-End du pipeline Requirement Processing :

    User text
        -> RequirementChatbot
        -> HybridExtractor
        -> optional LLM fallback
        -> StateGuard
        -> AI Plausibility Agent
        -> optional controlled LLM enrichment
        -> final pipeline outcome

Le runner part du texte utilisateur brut et utilise l'API publique
RequirementChatbot.process_user_message().

Il ne modifie aucune décision du pipeline. Un proxy transparent autour du
PlausibilityAgent sert uniquement à enregistrer le dernier rapport de
plausibilité pour l'évaluation des warnings non bloquants.

Entrée par défaut :
    requirement_extractor/validation/datasets/end_to_end_stress_dataset_v1.json

Sortie par défaut :
    requirement_extractor/validation/reports/end_to_end/
        results_end_to_end_v1.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


VALIDATION_DIR = Path(__file__).resolve().parent
REQUIREMENT_EXTRACTOR_DIR = VALIDATION_DIR.parent
REPOSITORY_ROOT = REQUIREMENT_EXTRACTOR_DIR.parent

DEFAULT_DATASET = (
    VALIDATION_DIR
    / "datasets"
    / "end_to_end_stress_dataset_v1.json"
)

DEFAULT_OUTPUT = (
    VALIDATION_DIR
    / "reports"
    / "end_to_end"
    / "results_end_to_end_v1.json"
)

EXPECTED_FIELDS = (
    "requested_usable_capacity_tib",
    "client_count",
    "average_file_size_gb",
    "max_file_size_gb",
    "total_file_count",
    "read_write_ratio",
    "access_type",
    "target_read_gbps",
    "target_write_gbps",
    "ha_required",
    "max_budget_usd",
    "max_power_w",
    "annual_growth_percent",
)

VALID_OUTCOMES = {
    "READY_COHERENT",
    "READY_AMBIGUOUS",
    "BLOCKED_PLAUSIBILITY",
    "CLARIFICATION_REQUIRED",
}

PLAUSIBILITY_LABELS = {
    "OK": "COHERENT",
    "WARNING": "AMBIGUOUS",
    "NEEDS_CLARIFICATION": "INCOHERENT",
    "BLOCKING": "INCOHERENT",
}


# ============================================================
# Generic helpers
# ============================================================

def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(enum_value(value)).strip()


def normalize_upper(value: Any) -> str:
    return normalize_text(value).upper()


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_json_value(value: Any) -> Any:
    """
    Convertit récursivement les Enums/objets simples vers une forme JSON.
    """
    value = enum_value(value)

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(enum_value(key)): safe_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            safe_json_value(item)
            for item in value
        ]

    if hasattr(value, "to_dict"):
        try:
            data = value.to_dict()
            if isinstance(data, dict):
                return safe_json_value(data)
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                key: safe_json_value(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        except Exception:
            pass

    return str(value)


def values_equal(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None

    if (
        isinstance(expected, bool)
        or isinstance(actual, bool)
    ):
        return expected is actual

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(
            float(expected),
            float(actual),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return False
        return all(
            values_equal(
                expected[key],
                actual[key],
            )
            for key in expected
        )

    return str(expected).strip().lower() == str(actual).strip().lower()


def state_to_dict(state: Any) -> Dict[str, Any]:
    if hasattr(state, "to_dict"):
        data = state.to_dict()

        if isinstance(data, dict):
            return data

    raise TypeError(
        "RequirementState doit fournir to_dict() -> dict."
    )


def final_json_to_plain(
    final_json: Any,
) -> Dict[str, Any]:
    if not isinstance(final_json, dict):
        return {
            field: None
            for field in EXPECTED_FIELDS
        }

    output: Dict[str, Any] = {}

    for field in EXPECTED_FIELDS:
        item = final_json.get(field)

        if item is None:
            output[field] = None
            continue

        if isinstance(item, dict) and "value" in item:
            output[field] = safe_json_value(
                item.get("value")
            )
            continue

        if hasattr(item, "value"):
            output[field] = safe_json_value(
                getattr(item, "value")
            )
            continue

        output[field] = safe_json_value(
            item
        )

    return output


def state_plain_fields(
    state: Any,
    state_data: Dict[str, Any],
) -> Dict[str, Any]:
    plain = state_data.get(
        "plain_final_json"
    )

    if isinstance(plain, dict):
        return {
            field: safe_json_value(
                plain.get(field)
            )
            for field in EXPECTED_FIELDS
        }

    return final_json_to_plain(
        getattr(
            state,
            "final_json",
            None,
        )
    )


def issue_field(issue: Any) -> Optional[str]:
    if isinstance(issue, dict):
        field = issue.get("field")
    else:
        field = getattr(
            issue,
            "field",
            None,
        )

    if field is None:
        return None

    return normalize_text(field)


# ============================================================
# Environment
# ============================================================

def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv(
        REPOSITORY_ROOT / ".env",
        override=False,
    )


def configure_environment(
    args: argparse.Namespace,
) -> Dict[str, str]:
    load_dotenv_if_available()

    # End-to-end = fallback extraction ON + plausibility ON.
    os.environ[
        "ENABLE_LLM_FALLBACK"
    ] = "true"

    os.environ[
        "ENABLE_AI_PLAUSIBILITY_AGENT"
    ] = "true"

    os.environ[
        "PLAUSIBILITY_AGENT_USE_LLM_ENRICHMENT"
    ] = "true"

    # OLLAMA_MODEL belongs to the extraction fallback.
    if args.fallback_model:
        os.environ[
            "OLLAMA_MODEL"
        ] = args.fallback_model

    # Plausibility uses its dedicated model variable.
    if args.plausibility_model:
        os.environ[
            "PLAUSIBILITY_AGENT_MODEL"
        ] = args.plausibility_model

    if args.ollama_host:
        os.environ[
            "OLLAMA_HOST"
        ] = args.ollama_host

    os.environ[
        "PLAUSIBILITY_AGENT_TEMPERATURE"
    ] = str(args.plausibility_temperature)

    os.environ[
        "PLAUSIBILITY_AGENT_DEBUG"
    ] = (
        "true"
        if args.debug_agent
        else "false"
    )

    return {
        "ENABLE_LLM_FALLBACK": os.getenv(
            "ENABLE_LLM_FALLBACK",
            "",
        ),
        "ENABLE_AI_PLAUSIBILITY_AGENT": os.getenv(
            "ENABLE_AI_PLAUSIBILITY_AGENT",
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
        "OLLAMA_MODEL": os.getenv(
            "OLLAMA_MODEL",
            "qwen2.5-coder:7b",
        ),
        "PLAUSIBILITY_AGENT_MODEL": os.getenv(
            "PLAUSIBILITY_AGENT_MODEL",
            "qwen2.5:3b",
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

    scenarios = data.get(
        "scenarios"
    )

    if not isinstance(scenarios, list):
        raise ValueError(
            "La clé 'scenarios' doit être une liste."
        )

    return data


def select_scenarios(
    data: Dict[str, Any],
    outcome: Optional[str],
    category: Optional[str],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    selected = []

    for scenario in data["scenarios"]:
        if not isinstance(scenario, dict):
            continue

        if (
            outcome is not None
            and scenario.get(
                "pipeline_outcome"
            )
            != outcome
        ):
            continue

        if (
            category is not None
            and scenario.get(
                "category"
            )
            != category
        ):
            continue

        selected.append(
            scenario
        )

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


# ============================================================
# Plausibility capture
# ============================================================

class CapturingPlausibilityAgent:
    """
    Proxy transparent.

    Il délègue analyze() au vrai AIPlausibilityAgent et mémorise uniquement
    le rapport retourné. Il ne change ni le statut, ni les issues, ni le texte.
    """

    def __init__(
        self,
        wrapped: Any,
    ) -> None:
        self.wrapped = wrapped
        self.last_report = None
        self.call_count = 0
        self.turn_call_count = 0

    def reset_turn_capture(self) -> None:
        self.last_report = None
        self.turn_call_count = 0

    def analyze(
        self,
        final_json: Any,
    ) -> Any:
        report = self.wrapped.analyze(
            final_json
        )

        self.last_report = report
        self.call_count += 1
        self.turn_call_count += 1

        return report

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        return getattr(
            self.wrapped,
            name,
        )


def serialize_plausibility_issue(
    issue: Any,
) -> Dict[str, Any]:
    return {
        "issue_type": normalize_text(
            getattr(
                issue,
                "issue_type",
                "",
            )
        ),
        "field": normalize_text(
            getattr(
                issue,
                "field",
                "",
            )
        ),
        "severity": normalize_text(
            getattr(
                issue,
                "severity",
                "",
            )
        ),
        "message": normalize_text(
            getattr(
                issue,
                "message",
                "",
            )
        ),
        "question": normalize_text(
            getattr(
                issue,
                "question",
                "",
            )
        ),
        "confidence": safe_json_value(
            getattr(
                issue,
                "confidence",
                None,
            )
        ),
        "suggested_correction": safe_json_value(
            getattr(
                issue,
                "suggested_correction",
                None,
            )
        ),
        "evidence_fields": safe_json_value(
            getattr(
                issue,
                "evidence_fields",
                {},
            )
            or {}
        ),
    }


def serialize_plausibility_report(
    report: Any,
) -> Optional[Dict[str, Any]]:
    if report is None:
        return None

    raw_status = normalize_upper(
        getattr(
            report,
            "status",
            "",
        )
    )

    issues = [
        serialize_plausibility_issue(
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

    raw_response = normalize_text(
        getattr(
            report,
            "raw_response",
            "",
        )
    )

    llm_warning_path = (
        raw_status == "WARNING"
    )

    if llm_warning_path:
        llm_output_observed = not (
            raw_response.startswith("{")
            and (
                '"decision_source": '
                '"DETERMINISTIC_PLAUSIBILITY_GUARD"'
                in raw_response
            )
        )
    else:
        llm_output_observed = False

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

    return {
        "raw_status": raw_status,
        "predicted_label": (
            PLAUSIBILITY_LABELS.get(
                raw_status
            )
        ),
        "should_block_recommendation": (
            raw_status
            in {
                "BLOCKING",
                "NEEDS_CLARIFICATION",
            }
        ),
        "issues": issues,
        "issue_codes": [
            item["issue_type"]
            for item in issues
        ],
        "issue_count": len(
            issues
        ),
        "llm_enrichment_path_expected": (
            llm_warning_path
        ),
        "llm_output_observed": (
            llm_output_observed
        ),
        "enrichment_error": (
            enrichment_error
        ),
        "raw_response": (
            raw_response
        ),
    }


# ============================================================
# LLM fallback instrumentation
# ============================================================

def iter_child_objects(
    root: Any,
    max_depth: int = 3,
) -> Iterable[Any]:
    """
    Petite introspection bornée utilisée uniquement par le runner.

    Elle permet de retrouver le LLMFallbackExtractor même si l'attribut
    exact de HybridExtractor change légèrement.
    """
    visited: Set[int] = set()
    queue: List[Tuple[Any, int]] = [
        (root, 0)
    ]

    while queue:
        obj, depth = queue.pop(0)

        if obj is None:
            continue

        obj_id = id(obj)

        if obj_id in visited:
            continue

        visited.add(
            obj_id
        )
        yield obj

        if depth >= max_depth:
            continue

        try:
            values = vars(obj).values()
        except Exception:
            continue

        for value in values:
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                    bytes,
                    bytearray,
                    dict,
                    list,
                    tuple,
                    set,
                    Path,
                    type(None),
                ),
            ):
                continue

            queue.append(
                (
                    value,
                    depth + 1,
                )
            )


def find_fallback_instruments(
    extractor: Any,
) -> List[Any]:
    instruments = []

    for obj in iter_child_objects(
        extractor
    ):
        class_name = type(
            obj
        ).__name__.lower()

        if (
            hasattr(
                obj,
                "call_count",
            )
            and (
                "fallback" in class_name
                or hasattr(
                    obj,
                    "call_log",
                )
            )
        ):
            instruments.append(
                obj
            )

    # Deduplicate by identity.
    unique = []
    seen = set()

    for item in instruments:
        if id(item) in seen:
            continue

        seen.add(
            id(item)
        )
        unique.append(
            item
        )

    return unique


def fallback_snapshot(
    extractor: Any,
) -> Dict[str, Any]:
    instruments = find_fallback_instruments(
        extractor
    )

    total_calls = 0
    logs = []

    for item in instruments:
        try:
            total_calls += int(
                getattr(
                    item,
                    "call_count",
                    0,
                )
                or 0
            )
        except Exception:
            pass

        call_log = getattr(
            item,
            "call_log",
            None,
        )

        if isinstance(
            call_log,
            list,
        ):
            logs.extend(
                copy.deepcopy(
                    call_log
                )
            )

    return {
        "instrument_count": len(
            instruments
        ),
        "call_count": total_calls,
        "call_log_length": len(
            logs
        ),
        "call_log": safe_json_value(
            logs
        ),
    }


def fallback_delta(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "instrument_count": after.get(
            "instrument_count",
            0,
        ),
        "call_count_delta": max(
            0,
            int(
                after.get(
                    "call_count",
                    0,
                )
            )
            - int(
                before.get(
                    "call_count",
                    0,
                )
            ),
        ),
        "call_log_length_delta": max(
            0,
            int(
                after.get(
                    "call_log_length",
                    0,
                )
            )
            - int(
                before.get(
                    "call_log_length",
                    0,
                )
            ),
        ),
    }


# ============================================================
# State / extraction diagnostics
# ============================================================

def extracted_candidate_source_counts(
    state_data: Dict[str, Any],
) -> Dict[str, int]:
    counts = Counter()

    for candidate in as_list(
        state_data.get(
            "extracted_candidates"
        )
    ):
        if not isinstance(
            candidate,
            dict,
        ):
            continue

        source = normalize_upper(
            candidate.get(
                "source",
                "__UNKNOWN__",
            )
        )

        counts[
            source
        ] += 1

    return dict(
        sorted(
            counts.items()
        )
    )


def compare_fields(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
) -> Dict[str, Any]:
    matches: Dict[str, bool] = {}
    mismatches: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for field in EXPECTED_FIELDS:
        expected_value = expected.get(
            field
        )
        actual_value = actual.get(
            field
        )

        ok = values_equal(
            expected_value,
            actual_value,
        )

        matches[
            field
        ] = ok

        if not ok:
            mismatches[
                field
            ] = {
                "expected": (
                    expected_value
                ),
                "actual": (
                    actual_value
                ),
            }

    non_null_expected_fields = [
        field
        for field in EXPECTED_FIELDS
        if expected.get(
            field
        )
        is not None
    ]

    non_null_correct = sum(
        1
        for field in non_null_expected_fields
        if matches[
            field
        ]
    )

    missing_expected_null_fields = [
        field
        for field in EXPECTED_FIELDS
        if (
            expected.get(
                field
            )
            is None
            and actual.get(
                field
            )
            is not None
        )
    ]

    return {
        "all_fields_exact": all(
            matches.values()
        ),
        "field_matches": matches,
        "field_mismatches": (
            mismatches
        ),
        "non_null_expected_field_count": len(
            non_null_expected_fields
        ),
        "non_null_correct_count": (
            non_null_correct
        ),
        "non_null_value_accuracy": (
            non_null_correct
            / len(
                non_null_expected_fields
            )
            if non_null_expected_fields
            else 1.0
        ),
        "missing_value_inference_fields": (
            missing_expected_null_fields
        ),
        "no_missing_value_inference": (
            len(
                missing_expected_null_fields
            )
            == 0
        ),
    }


def active_question_field(
    state: Any,
) -> Optional[str]:
    issues = (
        getattr(
            state,
            "issues",
            None,
        )
        or []
    )

    questions = (
        getattr(
            state,
            "questions",
            None,
        )
        or []
    )

    if not issues or not questions:
        return None

    return issue_field(
        issues[0]
    )


def pending_fields_from_state(
    state_data: Dict[str, Any],
) -> List[str]:
    output = set()

    for key in (
        "missing_fields",
        "conflicting_fields",
        "invalid_fields",
        "unsupported_fields",
    ):
        for field in as_list(
            state_data.get(
                key
            )
        ):
            output.add(
                normalize_text(
                    field
                )
            )

    return sorted(
        item
        for item in output
        if item
    )


# ============================================================
# Outcome inference
# ============================================================

def infer_pipeline_outcome(
    state_data: Dict[str, Any],
    plausibility: Optional[
        Dict[str, Any]
    ],
) -> str:
    if plausibility is not None:
        raw_status = normalize_upper(
            plausibility.get(
                "raw_status"
            )
        )

        if raw_status in {
            "BLOCKING",
            "NEEDS_CLARIFICATION",
        }:
            return (
                "BLOCKED_PLAUSIBILITY"
            )

        if raw_status == "WARNING":
            return "READY_AMBIGUOUS"

        if raw_status == "OK":
            return "READY_COHERENT"

    state_status = normalize_upper(
        state_data.get(
            "status"
        )
    )

    if (
        state_status
        == "NEEDS_CLARIFICATION"
    ):
        return (
            "CLARIFICATION_REQUIRED"
        )

    return "__UNKNOWN__"


def expected_issue_codes(
    scenario: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    expected = scenario.get(
        "expected",
        {},
    )

    blocking = [
        normalize_upper(
            item
        )
        for item in as_list(
            expected.get(
                "blocking_issue_codes"
            )
        )
    ]

    warnings = [
        normalize_upper(
            item
        )
        for item in as_list(
            expected.get(
                "warning_issue_codes"
            )
        )
    ]

    return (
        blocking,
        warnings,
    )


def actual_issue_codes(
    plausibility: Optional[
        Dict[str, Any]
    ],
) -> List[str]:
    if plausibility is None:
        return []

    return [
        normalize_upper(
            item
        )
        for item in as_list(
            plausibility.get(
                "issue_codes"
            )
        )
    ]


# ============================================================
# Scenario execution
# ============================================================

def execute_scenario(
    chatbot_class: Any,
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    scenario_id = scenario[
        "id"
    ]

    bot = chatbot_class()

    capturing_agent = (
        CapturingPlausibilityAgent(
            bot.plausibility_agent
        )
    )

    bot.plausibility_agent = (
        capturing_agent
    )

    turns = as_list(
        scenario.get(
            "turns"
        )
    )

    expected_fields = copy.deepcopy(
        scenario.get(
            "expected_final_requirements",
            {},
        )
    )

    expected = copy.deepcopy(
        scenario.get(
            "expected",
            {},
        )
    )

    fallback_before_scenario = (
        fallback_snapshot(
            bot.extractor
        )
    )

    turn_results = []
    final_state = None
    final_plausibility = None
    scenario_start = time.perf_counter()

    try:
        for turn_index, user_text in enumerate(
            turns,
            start=1,
        ):
            if not isinstance(
                user_text,
                str,
            ):
                raise TypeError(
                    f"{scenario_id}: "
                    f"turn {turn_index} "
                    "must be a string."
                )

            capturing_agent.reset_turn_capture()

            fallback_before_turn = (
                fallback_snapshot(
                    bot.extractor
                )
            )

            turn_start = time.perf_counter()

            state = (
                bot.process_user_message(
                    user_text
                )
            )

            turn_latency_ms = (
                time.perf_counter()
                - turn_start
            ) * 1000.0

            fallback_after_turn = (
                fallback_snapshot(
                    bot.extractor
                )
            )

            state_data = (
                state_to_dict(
                    state
                )
            )

            plain_fields = (
                state_plain_fields(
                    state,
                    state_data,
                )
            )

            plausibility = (
                serialize_plausibility_report(
                    capturing_agent.last_report
                )
            )

            turn_results.append(
                {
                    "turn_index": (
                        turn_index
                    ),
                    "user_text": (
                        user_text
                    ),
                    "latency_ms": round(
                        turn_latency_ms,
                        3,
                    ),
                    "state": {
                        "status": (
                            state_data.get(
                                "status"
                            )
                        ),
                        "stage": (
                            state_data.get(
                                "stage"
                            )
                        ),
                        "plain_final_json": (
                            plain_fields
                        ),
                        "missing_fields": (
                            state_data.get(
                                "missing_fields",
                                [],
                            )
                        ),
                        "conflicting_fields": (
                            state_data.get(
                                "conflicting_fields",
                                [],
                            )
                        ),
                        "invalid_fields": (
                            state_data.get(
                                "invalid_fields",
                                [],
                            )
                        ),
                        "unsupported_fields": (
                            state_data.get(
                                "unsupported_fields",
                                [],
                            )
                        ),
                        "questions": (
                            state_data.get(
                                "questions",
                                [],
                            )
                        ),
                        "question_field": (
                            active_question_field(
                                state
                            )
                        ),
                        "calculation_result": (
                            state_data.get(
                                "calculation_result",
                                {},
                            )
                        ),
                    },
                    "plausibility": (
                        plausibility
                    ),
                    "fallback_instrumentation": (
                        fallback_delta(
                            fallback_before_turn,
                            fallback_after_turn,
                        )
                    ),
                    "technical_error": None,
                }
            )

            final_state = state
            final_plausibility = (
                plausibility
            )

        if final_state is None:
            raise ValueError(
                f"{scenario_id}: no turns executed."
            )

        latency_ms = (
            time.perf_counter()
            - scenario_start
        ) * 1000.0

        final_state_data = (
            state_to_dict(
                final_state
            )
        )

        final_plain_fields = (
            state_plain_fields(
                final_state,
                final_state_data,
            )
        )

        field_comparison = (
            compare_fields(
                expected_fields,
                final_plain_fields,
            )
        )

        actual_outcome = (
            infer_pipeline_outcome(
                final_state_data,
                final_plausibility,
            )
        )

        expected_outcome = (
            scenario.get(
                "pipeline_outcome"
            )
        )

        expected_blocking, expected_warnings = (
            expected_issue_codes(
                scenario
            )
        )

        actual_codes = (
            actual_issue_codes(
                final_plausibility
            )
        )

        expected_plausibility_label = (
            expected.get(
                "plausibility_status"
            )
        )

        actual_plausibility_label = (
            final_plausibility.get(
                "predicted_label"
            )
            if final_plausibility
            is not None
            else None
        )

        expected_should_block = (
            expected.get(
                "should_block_recommendation"
            )
        )

        actual_should_block = (
            final_plausibility.get(
                "should_block_recommendation"
            )
            if final_plausibility
            is not None
            else None
        )

        expected_clarification_fields = set(
            normalize_text(
                item
            )
            for item in as_list(
                expected.get(
                    "clarification_fields"
                )
            )
        )

        actual_pending_fields = set(
            pending_fields_from_state(
                final_state_data
            )
        )

        clarification_fields_covered = (
            expected_clarification_fields
            .issubset(
                actual_pending_fields
            )
        )

        fallback_after_scenario = (
            fallback_snapshot(
                bot.extractor
            )
        )

        fallback_summary = (
            fallback_delta(
                fallback_before_scenario,
                fallback_after_scenario,
            )
        )

        candidate_source_counts = (
            extracted_candidate_source_counts(
                final_state_data
            )
        )

        llm_fallback_candidate_count = sum(
            count
            for source, count
            in candidate_source_counts.items()
            if "LLM_FALLBACK" in source
        )

        issue_codes_exact = (
            Counter(
                expected_blocking
                + expected_warnings
            )
            == Counter(
                actual_codes
            )
        )

        outcome_match = (
            actual_outcome
            == expected_outcome
        )

        plausibility_label_match = (
            expected_plausibility_label
            == actual_plausibility_label
        )

        if (
            expected_outcome
            == "CLARIFICATION_REQUIRED"
        ):
            # Plausibility must not be forced when extraction is incomplete.
            plausibility_label_match = (
                actual_plausibility_label
                is None
            )

        should_block_match = (
            expected_should_block
            == actual_should_block
        )

        if (
            expected_outcome
            == "CLARIFICATION_REQUIRED"
        ):
            should_block_match = (
                actual_should_block
                is None
            )

        scenario_success = all(
            (
                outcome_match,
                field_comparison[
                    "all_fields_exact"
                ],
                field_comparison[
                    "no_missing_value_inference"
                ],
                plausibility_label_match,
                should_block_match,
                issue_codes_exact,
                (
                    clarification_fields_covered
                    if expected_outcome
                    == "CLARIFICATION_REQUIRED"
                    else True
                ),
            )
        )

        return {
            "id": scenario_id,
            "pipeline_outcome": (
                expected_outcome
            ),
            "category": scenario.get(
                "category"
            ),
            "language": scenario.get(
                "language"
            ),
            "difficulty": scenario.get(
                "difficulty"
            ),
            "turn_count": len(
                turns
            ),
            "turns": turn_results,
            "expected_final_requirements": (
                expected_fields
            ),
            "expected": expected,
            "actual": {
                "pipeline_outcome": (
                    actual_outcome
                ),
                "final_state_status": (
                    final_state_data.get(
                        "status"
                    )
                ),
                "final_state_stage": (
                    final_state_data.get(
                        "stage"
                    )
                ),
                "final_requirements": (
                    final_plain_fields
                ),
                "pending_fields": sorted(
                    actual_pending_fields
                ),
                "active_question_field": (
                    active_question_field(
                        final_state
                    )
                ),
                "questions": (
                    final_state_data.get(
                        "questions",
                        [],
                    )
                ),
                "calculation_result": (
                    final_state_data.get(
                        "calculation_result",
                        {},
                    )
                ),
                "plausibility": (
                    final_plausibility
                ),
                "candidate_source_counts": (
                    candidate_source_counts
                ),
                "llm_fallback_candidate_count": (
                    llm_fallback_candidate_count
                ),
                "fallback_instrumentation": (
                    fallback_summary
                ),
            },
            "checks": {
                "pipeline_outcome_match": (
                    outcome_match
                ),
                "extraction_all_fields_exact": (
                    field_comparison[
                        "all_fields_exact"
                    ]
                ),
                "field_comparison": (
                    field_comparison
                ),
                "plausibility_label_match": (
                    plausibility_label_match
                ),
                "should_block_match": (
                    should_block_match
                ),
                "issue_codes_exact": (
                    issue_codes_exact
                ),
                "expected_issue_codes": (
                    expected_blocking
                    + expected_warnings
                ),
                "actual_issue_codes": (
                    actual_codes
                ),
                "clarification_fields_covered": (
                    clarification_fields_covered
                ),
                "expected_clarification_fields": sorted(
                    expected_clarification_fields
                ),
                "actual_pending_fields": sorted(
                    actual_pending_fields
                ),
                "no_missing_value_inference": (
                    field_comparison[
                        "no_missing_value_inference"
                    ]
                ),
                "scenario_success": (
                    scenario_success
                ),
            },
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "technical_error": None,
        }

    except Exception as exc:
        latency_ms = (
            time.perf_counter()
            - scenario_start
        ) * 1000.0

        return {
            "id": scenario_id,
            "pipeline_outcome": (
                scenario.get(
                    "pipeline_outcome"
                )
            ),
            "category": scenario.get(
                "category"
            ),
            "language": scenario.get(
                "language"
            ),
            "difficulty": scenario.get(
                "difficulty"
            ),
            "turn_count": len(
                turns
            ),
            "turns": turn_results,
            "expected_final_requirements": (
                expected_fields
            ),
            "expected": expected,
            "actual": None,
            "checks": {
                "scenario_success": (
                    False
                ),
            },
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
        }


# ============================================================
# Run
# ============================================================

def run_validation(
    args: argparse.Namespace,
) -> Path:
    environment = configure_environment(
        args
    )

    # Late import: all env vars must be set before components are built.
    from requirement_extractor.requirement_chatbot import (
        RequirementChatbot,
    )

    dataset_path = (
        args.dataset.resolve()
    )

    output_path = (
        args.output.resolve()
    )

    data = load_dataset(
        dataset_path
    )

    scenarios = select_scenarios(
        data=data,
        outcome=args.outcome,
        category=args.category,
        limit=args.limit,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_started = datetime.now(
        timezone.utc
    )

    global_start = time.perf_counter()

    results = []
    technical_errors = 0
    successful_scenarios = 0
    fallback_calls_recorded = 0
    fallback_candidates = 0
    plausibility_llm_warning_paths = 0
    plausibility_llm_outputs = 0

    expected_distribution = Counter(
        scenario.get(
            "pipeline_outcome",
            "__MISSING__",
        )
        for scenario in scenarios
    )

    print("=" * 92)
    print(
        "END-TO-END REQUIREMENT PIPELINE VALIDATION"
    )
    print("=" * 92)
    print(
        f"Dataset            : {dataset_path}"
    )
    print(
        f"Scenarios          : {len(scenarios)}"
    )
    print(
        f"Fallback model     : "
        f"{environment['OLLAMA_MODEL']}"
    )
    print(
        f"Plausibility model : "
        f"{environment['PLAUSIBILITY_AGENT_MODEL']}"
    )
    print(
        f"Ollama             : "
        f"{environment['OLLAMA_HOST']}"
    )
    print(
        f"Output             : {output_path}"
    )
    print("=" * 92)

    for position, scenario in enumerate(
        scenarios,
        start=1,
    ):
        result = execute_scenario(
            chatbot_class=(
                RequirementChatbot
            ),
            scenario=scenario,
        )

        results.append(
            result
        )

        if (
            result.get(
                "technical_error"
            )
            is not None
        ):
            technical_errors += 1

        if result.get(
            "checks",
            {},
        ).get(
            "scenario_success",
            False,
        ):
            successful_scenarios += 1

        actual = result.get(
            "actual"
        )

        if isinstance(
            actual,
            dict,
        ):
            fallback_info = actual.get(
                "fallback_instrumentation",
                {},
            )

            fallback_calls_recorded += int(
                fallback_info.get(
                    "call_count_delta",
                    0,
                )
                or 0
            )

            fallback_candidates += int(
                actual.get(
                    "llm_fallback_candidate_count",
                    0,
                )
                or 0
            )

            plausibility = actual.get(
                "plausibility"
            )

            if isinstance(
                plausibility,
                dict,
            ):
                plausibility_llm_warning_paths += int(
                    bool(
                        plausibility.get(
                            "llm_enrichment_path_expected"
                        )
                    )
                )

                plausibility_llm_outputs += int(
                    bool(
                        plausibility.get(
                            "llm_output_observed"
                        )
                    )
                )

        status = (
            "PASS"
            if result.get(
                "checks",
                {},
            ).get(
                "scenario_success",
                False,
            )
            else "FAIL"
        )

        actual_outcome = (
            result.get(
                "actual",
                {},
            ).get(
                "pipeline_outcome"
            )
            if isinstance(
                result.get(
                    "actual"
                ),
                dict,
            )
            else "__ERROR__"
        )

        print(
            f"[{position:02d}/{len(scenarios):02d}] "
            f"{scenario.get('id')} | "
            f"expected={scenario.get('pipeline_outcome'):<24} | "
            f"actual={str(actual_outcome):<24} | "
            f"{status} | "
            f"{result.get('latency_ms', 0.0):.1f} ms"
        )

        if (
            args.fail_fast
            and status == "FAIL"
        ):
            break

    total_duration_seconds = (
        time.perf_counter()
        - global_start
    )

    executed = len(
        results
    )

    latencies = [
        float(
            result.get(
                "latency_ms",
                0.0,
            )
            or 0.0
        )
        for result in results
    ]

    result_object = {
        "run": {
            "started_at_utc": (
                run_started.isoformat()
            ),
            "finished_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
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
            "environment": (
                environment
            ),
        },
        "experiment": {
            "name": (
                "Requirement Pipeline End-to-End "
                "Validation v1"
            ),
            "scope": (
                "Raw user text -> extraction -> "
                "optional LLM fallback -> StateGuard -> "
                "AI Plausibility Agent -> optional warning "
                "LLM enrichment -> final pipeline outcome"
            ),
            "architecture_generation_in_scope": (
                False
            ),
            "expected_outcome_distribution": dict(
                sorted(
                    expected_distribution.items()
                )
            ),
            "filters": {
                "outcome": (
                    args.outcome
                ),
                "category": (
                    args.category
                ),
                "limit": (
                    args.limit
                ),
            },
        },
        "execution_summary": {
            "requested_scenarios": len(
                scenarios
            ),
            "executed_scenarios": (
                executed
            ),
            "successful_scenarios": (
                successful_scenarios
            ),
            "failed_scenarios": (
                executed
                - successful_scenarios
            ),
            "scenario_success_rate": (
                successful_scenarios
                / executed
                if executed
                else 0.0
            ),
            "technical_errors": (
                technical_errors
            ),
            "fallback_calls_recorded": (
                fallback_calls_recorded
            ),
            "llm_fallback_candidates_in_final_state": (
                fallback_candidates
            ),
            "plausibility_llm_warning_paths": (
                plausibility_llm_warning_paths
            ),
            "plausibility_llm_outputs_observed": (
                plausibility_llm_outputs
            ),
            "total_duration_seconds": round(
                total_duration_seconds,
                3,
            ),
            "average_scenario_latency_ms": round(
                (
                    sum(
                        latencies
                    )
                    / len(
                        latencies
                    )
                )
                if latencies
                else 0.0,
                3,
            ),
            "max_scenario_latency_ms": round(
                max(
                    latencies
                )
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
            result_object,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 92)
    print(
        "END-TO-END VALIDATION COMPLETE"
    )
    print("=" * 92)
    print(
        f"Executed scenarios       : {executed}"
    )
    print(
        f"Successful scenarios     : "
        f"{successful_scenarios}/{executed}"
    )
    print(
        f"Technical errors         : "
        f"{technical_errors}"
    )
    print(
        f"Fallback calls recorded  : "
        f"{fallback_calls_recorded}"
    )
    print(
        f"LLM fallback candidates  : "
        f"{fallback_candidates}"
    )
    print(
        f"Plausibility LLM paths   : "
        f"{plausibility_llm_warning_paths}"
    )
    print(
        f"Observed LLM enrichments : "
        f"{plausibility_llm_outputs}"
    )
    print(
        f"Average latency          : "
        f"{result_object['execution_summary']['average_scenario_latency_ms']} ms"
    )
    print(
        f"Results                  : {output_path}"
    )
    print("=" * 92)

    return output_path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation End-to-End du pipeline "
            "Requirement Extractor + AI Plausibility."
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
        "--fallback-model",
        default="qwen2.5-coder:7b",
        help=(
            "Modèle Ollama utilisé par le LLM fallback."
        ),
    )

    parser.add_argument(
        "--plausibility-model",
        default="qwen2.5:3b",
        help=(
            "Modèle Ollama utilisé uniquement pour "
            "l'enrichissement des warnings."
        ),
    )

    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
    )

    parser.add_argument(
        "--plausibility-temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--outcome",
        choices=sorted(
            VALID_OUTCOMES
        ),
        default=None,
        help=(
            "Exécute seulement un type de résultat attendu."
        ),
    )

    parser.add_argument(
        "--category",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
    )

    parser.add_argument(
        "--debug-agent",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        run_validation(
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
