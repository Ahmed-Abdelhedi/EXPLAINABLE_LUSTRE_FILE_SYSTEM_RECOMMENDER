#!/usr/bin/env python3
"""
plausibility_full_agent_metrics.py

Calcule les métriques de validation finale de l'AI Plausibility Agent complet (guard déterministe + LLM enrichment contrôlé).

Entrée attendue :
    requirement_extractor/validation/reports/ai_plausibility/full_agent/
        results_ai_plausibility_full_agent.json

Sortie par défaut :
    requirement_extractor/validation/reports/ai_plausibility/full_agent/
        metrics_ai_plausibility_full_agent.json

Le script ne dépend d'aucune librairie externe.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LABELS: Tuple[str, ...] = (
    "COHERENT",
    "AMBIGUOUS",
    "INCOHERENT",
)

BLOCKING_SEVERITIES = {
    "BLOCKING",
    "ERROR",
    "CRITICAL",
}

WARNING_SEVERITIES = {
    "WARNING",
    "WARN",
}


# ============================================================
# Generic helpers
# ============================================================

def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def prf(
    tp: int,
    fp: int,
    fn: int,
) -> Dict[str, Any]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def percentile(
    values: Sequence[float],
    probability: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    fraction = position - lower_index

    return (
        ordered[lower_index]
        + (
            ordered[upper_index]
            - ordered[lower_index]
        )
        * fraction
    )


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return statistics.fmean(values)


def normalize_severity(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def normalize_code(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def normalize_field(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def scenario_actual(
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    value = scenario.get("actual", {})
    return value if isinstance(value, dict) else {}


def scenario_technical_error(
    scenario: Dict[str, Any],
) -> Any:
    return scenario_actual(
        scenario
    ).get("technical_error")


def scenario_enrichment_error(
    scenario: Dict[str, Any],
) -> Any:
    return scenario_actual(
        scenario
    ).get("enrichment_error")


def scenario_latency_ms(
    scenario: Dict[str, Any],
) -> Optional[float]:
    value = scenario_actual(
        scenario
    ).get("latency_ms")

    if isinstance(value, (int, float)):
        return float(value)

    return None


# ============================================================
# Loading / validation
# ============================================================

def load_results(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de résultats introuvable : {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "Le fichier de résultats doit contenir "
            "un objet JSON racine."
        )

    scenarios = data.get("scenarios")

    if not isinstance(scenarios, list):
        raise ValueError(
            "La clé 'scenarios' doit contenir une liste."
        )

    return data


# ============================================================
# Expected / actual extraction
# ============================================================

def expected_label(scenario: Dict[str, Any]) -> str:
    expected = scenario.get("expected", {})

    value = expected.get(
        "plausibility_status",
        scenario.get("label"),
    )

    return normalize_code(value)


def predicted_label(scenario: Dict[str, Any]) -> str:
    if scenario_technical_error(scenario) not in {
        None,
        "",
        False,
    }:
        return "__TECHNICAL_ERROR__"

    actual = scenario.get("actual", {})
    return normalize_code(
        actual.get("predicted_label")
    )


def expected_should_block(
    scenario: Dict[str, Any],
) -> bool:
    expected = scenario.get("expected", {})
    return bool(
        expected.get(
            "should_block_recommendation",
            False,
        )
    )


def actual_should_block(
    scenario: Dict[str, Any],
) -> bool:
    actual = scenario.get("actual", {})
    return bool(
        actual.get(
            "should_block_recommendation",
            False,
        )
    )


def expected_issue_objects(
    scenario: Dict[str, Any],
    kind: str,
) -> List[Dict[str, Any]]:
    expected = scenario.get("expected", {})

    if kind == "blocking":
        return [
            item
            for item in as_list(
                expected.get("blocking_issues")
            )
            if isinstance(item, dict)
        ]

    if kind == "warning":
        return [
            item
            for item in as_list(
                expected.get("warnings")
            )
            if isinstance(item, dict)
        ]

    raise ValueError(
        f"Type d'issue non supporté : {kind}"
    )


def actual_issue_objects(
    scenario: Dict[str, Any],
    kind: str,
) -> List[Dict[str, Any]]:
    if scenario_technical_error(scenario) not in {
        None,
        "",
        False,
    }:
        return []

    actual = scenario.get("actual", {})
    issues = [
        item
        for item in as_list(actual.get("issues"))
        if isinstance(item, dict)
    ]

    selected: List[Dict[str, Any]] = []

    for issue in issues:
        severity = normalize_severity(
            issue.get("severity")
        )

        if (
            kind == "blocking"
            and severity in BLOCKING_SEVERITIES
        ):
            selected.append(issue)

        elif (
            kind == "warning"
            and severity in WARNING_SEVERITIES
        ):
            selected.append(issue)

    return selected


def expected_issue_codes(
    scenario: Dict[str, Any],
    kind: str,
) -> List[str]:
    return [
        normalize_code(item.get("code"))
        for item in expected_issue_objects(
            scenario,
            kind,
        )
        if normalize_code(item.get("code"))
    ]


def actual_issue_codes(
    scenario: Dict[str, Any],
    kind: str,
) -> List[str]:
    return [
        normalize_code(
            item.get(
                "issue_type",
                item.get("code"),
            )
        )
        for item in actual_issue_objects(
            scenario,
            kind,
        )
        if normalize_code(
            item.get(
                "issue_type",
                item.get("code"),
            )
        )
    ]


# ============================================================
# Classification metrics
# ============================================================

def classification_metrics(
    scenarios: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    per_label: Dict[str, Any] = {}

    confusion: Dict[str, Dict[str, int]] = {
        label: {
            predicted: 0
            for predicted in (
                list(LABELS)
                + ["__TECHNICAL_ERROR__", "__OTHER__"]
            )
        }
        for label in LABELS
    }

    correct = 0

    for scenario in scenarios:
        expected = expected_label(scenario)
        predicted = predicted_label(scenario)

        if predicted not in LABELS and (
            predicted != "__TECHNICAL_ERROR__"
        ):
            predicted_bucket = "__OTHER__"
        else:
            predicted_bucket = predicted

        if expected in confusion:
            confusion[expected][predicted_bucket] += 1

        if expected == predicted:
            correct += 1

    for label in LABELS:
        tp = sum(
            1
            for scenario in scenarios
            if expected_label(scenario) == label
            and predicted_label(scenario) == label
        )

        fp = sum(
            1
            for scenario in scenarios
            if expected_label(scenario) != label
            and predicted_label(scenario) == label
        )

        fn = sum(
            1
            for scenario in scenarios
            if expected_label(scenario) == label
            and predicted_label(scenario) != label
        )

        metrics = prf(tp, fp, fn)
        metrics["support"] = sum(
            1
            for scenario in scenarios
            if expected_label(scenario) == label
        )
        per_label[label] = metrics

    macro_precision = mean(
        [
            per_label[label]["precision"]
            for label in LABELS
        ]
    )
    macro_recall = mean(
        [
            per_label[label]["recall"]
            for label in LABELS
        ]
    )
    macro_f1 = mean(
        [
            per_label[label]["f1"]
            for label in LABELS
        ]
    )

    return {
        "plausibility_status_accuracy": safe_divide(
            correct,
            len(scenarios),
        ),
        "correct": correct,
        "total": len(scenarios),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


# ============================================================
# Issue metrics
# ============================================================

def compare_code_multisets(
    expected_codes: Sequence[str],
    actual_codes: Sequence[str],
) -> Tuple[int, int, int]:
    """
    Compare des codes avec multiplicité.

    Ex. :
      expected = [A, A, B]
      actual   = [A, B, C]

      TP=2, FP=1, FN=1
    """

    expected_counter = Counter(expected_codes)
    actual_counter = Counter(actual_codes)

    codes = (
        set(expected_counter)
        | set(actual_counter)
    )

    tp = sum(
        min(
            expected_counter[code],
            actual_counter[code],
        )
        for code in codes
    )

    fp = sum(
        max(
            actual_counter[code]
            - expected_counter[code],
            0,
        )
        for code in codes
    )

    fn = sum(
        max(
            expected_counter[code]
            - actual_counter[code],
            0,
        )
        for code in codes
    )

    return tp, fp, fn


def issue_detection_metrics(
    scenarios: Sequence[Dict[str, Any]],
    kind: str,
) -> Dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    exact_count = 0
    scenario_count = len(scenarios)

    per_code_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
    )

    for scenario in scenarios:
        expected_codes = expected_issue_codes(
            scenario,
            kind,
        )
        actual_codes = actual_issue_codes(
            scenario,
            kind,
        )

        tp, fp, fn = compare_code_multisets(
            expected_codes,
            actual_codes,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if Counter(expected_codes) == Counter(actual_codes):
            exact_count += 1

        expected_counter = Counter(expected_codes)
        actual_counter = Counter(actual_codes)

        for code in (
            set(expected_counter)
            | set(actual_counter)
        ):
            per_code_counts[code]["tp"] += min(
                expected_counter[code],
                actual_counter[code],
            )
            per_code_counts[code]["fp"] += max(
                actual_counter[code]
                - expected_counter[code],
                0,
            )
            per_code_counts[code]["fn"] += max(
                expected_counter[code]
                - actual_counter[code],
                0,
            )

    overall = prf(
        total_tp,
        total_fp,
        total_fn,
    )

    per_code = {
        code: prf(
            counts["tp"],
            counts["fp"],
            counts["fn"],
        )
        for code, counts
        in sorted(per_code_counts.items())
    }

    overall.update(
        {
            "scenario_exact_rate": safe_divide(
                exact_count,
                scenario_count,
            ),
            "scenario_exact_count": exact_count,
            "scenario_total": scenario_count,
            "per_code": per_code,
        }
    )

    return overall


# ============================================================
# Issue field reference accuracy
# ============================================================

def expected_fields(
    issue: Dict[str, Any],
) -> List[str]:
    return [
        normalize_field(field)
        for field in as_list(issue.get("fields"))
        if normalize_field(field)
    ]


def actual_referenced_fields(
    issue: Dict[str, Any],
) -> List[str]:
    fields = set()

    direct_field = normalize_field(
        issue.get("field")
    )
    if direct_field:
        fields.add(direct_field)

    for value in as_list(
        issue.get("referenced_fields")
    ):
        normalized = normalize_field(value)
        if normalized:
            fields.add(normalized)

    evidence = issue.get("evidence_fields")

    if isinstance(evidence, dict):
        # Les clés correspondant à des champs sont aussi
        # des références explicites.
        for key in evidence.keys():
            if key not in {
                "missing_fields",
                "ratio_sum",
                "capacity_gb",
                "estimated_dataset_volume_tib",
                "estimated_volume_to_capacity_ratio",
                "blocking_ratio",
                "total_throughput_gbps",
                "throughput_per_tib_gbps",
                "clients_per_tib",
                "budget_per_tib_usd",
                "power_per_tib_w",
            }:
                normalized = normalize_field(key)
                if normalized:
                    fields.add(normalized)

        # Cas INSUFFICIENT_INFORMATION :
        # la liste complète des champs manquants est dans
        # evidence_fields.missing_fields.
        for value in as_list(
            evidence.get("missing_fields")
        ):
            normalized = normalize_field(value)
            if normalized:
                fields.add(normalized)

    return sorted(fields)


def pair_issues_by_code(
    expected_issues: Sequence[Dict[str, Any]],
    actual_issues: Sequence[Dict[str, Any]],
) -> List[
    Tuple[
        Dict[str, Any],
        Optional[Dict[str, Any]],
    ]
]:
    """
    Associe chaque issue attendue à une issue réelle de même code.

    Si plusieurs candidates partagent le même code, on choisit celle
    avec le meilleur recouvrement de champs.
    """

    unused_actual = list(actual_issues)
    pairs = []

    for expected_issue in expected_issues:
        code = normalize_code(
            expected_issue.get("code")
        )
        expected_field_set = set(
            expected_fields(expected_issue)
        )

        candidate_indexes = [
            index
            for index, actual_issue
            in enumerate(unused_actual)
            if normalize_code(
                actual_issue.get(
                    "issue_type",
                    actual_issue.get("code"),
                )
            )
            == code
        ]

        if not candidate_indexes:
            pairs.append(
                (expected_issue, None)
            )
            continue

        def overlap_score(index: int) -> Tuple[int, int]:
            actual_field_set = set(
                actual_referenced_fields(
                    unused_actual[index]
                )
            )

            overlap = len(
                expected_field_set
                & actual_field_set
            )

            extra = len(
                actual_field_set
                - expected_field_set
            )

            return overlap, -extra

        best_index = max(
            candidate_indexes,
            key=overlap_score,
        )

        actual_issue = unused_actual.pop(best_index)
        pairs.append(
            (expected_issue, actual_issue)
        )

    return pairs


def issue_field_reference_metrics(
    scenarios: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    expected_issue_total = 0
    matched_code_total = 0
    complete_reference_total = 0
    exact_reference_total = 0

    field_tp = 0
    field_fp = 0
    field_fn = 0

    for scenario in scenarios:
        expected_all = (
            expected_issue_objects(
                scenario,
                "blocking",
            )
            + expected_issue_objects(
                scenario,
                "warning",
            )
        )

        actual_all = (
            actual_issue_objects(
                scenario,
                "blocking",
            )
            + actual_issue_objects(
                scenario,
                "warning",
            )
        )

        pairs = pair_issues_by_code(
            expected_all,
            actual_all,
        )

        for expected_issue, actual_issue in pairs:
            expected_issue_total += 1

            expected_set = set(
                expected_fields(expected_issue)
            )

            if actual_issue is None:
                field_fn += len(expected_set)
                continue

            matched_code_total += 1

            actual_set = set(
                actual_referenced_fields(actual_issue)
            )

            if expected_set.issubset(actual_set):
                complete_reference_total += 1

            if expected_set == actual_set:
                exact_reference_total += 1

            field_tp += len(
                expected_set & actual_set
            )
            field_fp += len(
                actual_set - expected_set
            )
            field_fn += len(
                expected_set - actual_set
            )

    field_prf = prf(
        field_tp,
        field_fp,
        field_fn,
    )

    return {
        # Métrique recommandée :
        # une issue est correcte si tous les champs attendus
        # sont effectivement référencés.
        "issue_field_reference_accuracy": safe_divide(
            complete_reference_total,
            expected_issue_total,
        ),
        "complete_reference_count": (
            complete_reference_total
        ),
        "exact_reference_rate": safe_divide(
            exact_reference_total,
            expected_issue_total,
        ),
        "exact_reference_count": exact_reference_total,
        "code_matched_issue_count": matched_code_total,
        "expected_issue_count": expected_issue_total,
        "field_micro_precision": field_prf["precision"],
        "field_micro_recall": field_prf["recall"],
        "field_micro_f1": field_prf["f1"],
        "field_tp": field_tp,
        "field_fp": field_fp,
        "field_fn": field_fn,
    }


# ============================================================
# Safety metrics
# ============================================================

def safety_metrics(
    scenarios: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    mutations = 0
    inference_violation_scenarios = 0
    inference_violation_count = 0
    technical_errors = 0

    false_blocking_count = 0
    expected_non_blocking_count = 0

    block_correct = 0

    for scenario in scenarios:
        actual = scenario.get("actual", {})

        if scenario_technical_error(scenario) not in {
            None,
            "",
            False,
        }:
            technical_errors += 1

        if bool(actual.get("input_mutated", False)):
            mutations += 1

        violations = as_list(
            actual.get("no_inference_violations")
        )

        if violations:
            inference_violation_scenarios += 1
            inference_violation_count += len(violations)

        expected_block = expected_should_block(
            scenario
        )
        predicted_block = actual_should_block(
            scenario
        )

        if expected_block == predicted_block:
            block_correct += 1

        if not expected_block:
            expected_non_blocking_count += 1

            if predicted_block:
                false_blocking_count += 1

    total = len(scenarios)

    return {
        "should_block_accuracy": safe_divide(
            block_correct,
            total,
        ),
        "false_blocking_rate": safe_divide(
            false_blocking_count,
            expected_non_blocking_count,
        ),
        "false_blocking_count": false_blocking_count,
        "expected_non_blocking_count": (
            expected_non_blocking_count
        ),
        "input_mutation_rate": safe_divide(
            mutations,
            total,
        ),
        "input_mutation_count": mutations,
        "no_inference_violation_rate": safe_divide(
            total - inference_violation_scenarios,
            total,
        ),
        "inference_violation_scenario_count": (
            inference_violation_scenarios
        ),
        "inference_violation_count": (
            inference_violation_count
        ),
        "technical_error_rate": safe_divide(
            technical_errors,
            total,
        ),
        "technical_error_count": technical_errors,
    }


# ============================================================
# Latency metrics
# ============================================================

def latency_metrics(
    scenarios: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    values = [
        value
        for scenario in scenarios
        if (
            value := scenario_latency_ms(
                scenario
            )
        )
        is not None
    ]

    if not values:
        return {
            "count": 0,
            "average_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "min_latency_ms": 0.0,
        }

    return {
        "count": len(values),
        "average_latency_ms": mean(values),
        "median_latency_ms": statistics.median(values),
        "p95_latency_ms": percentile(
            values,
            0.95,
        ),
        "p99_latency_ms": percentile(
            values,
            0.99,
        ),
        "max_latency_ms": max(values),
        "min_latency_ms": min(values),
    }


# ============================================================
# Breakdown metrics
# ============================================================

def subgroup_summary(
    scenarios: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    correct = sum(
        1
        for scenario in scenarios
        if expected_label(scenario)
        == predicted_label(scenario)
    )

    block_correct = sum(
        1
        for scenario in scenarios
        if expected_should_block(scenario)
        == actual_should_block(scenario)
    )

    mutations = sum(
        1
        for scenario in scenarios
        if bool(
            scenario.get(
                "actual",
                {},
            ).get(
                "input_mutated",
                False,
            )
        )
    )

    inference_safe = sum(
        1
        for scenario in scenarios
        if not as_list(
            scenario.get(
                "actual",
                {},
            ).get(
                "no_inference_violations"
            )
        )
    )

    technical_errors = sum(
        1
        for scenario in scenarios
        if scenario_technical_error(scenario) not in {
            None,
            "",
            False,
        }
    )

    latencies = [
        value
        for scenario in scenarios
        if (
            value := scenario_latency_ms(
                scenario
            )
        )
        is not None
    ]

    expected_distribution = Counter(
        expected_label(scenario)
        for scenario in scenarios
    )

    predicted_distribution = Counter(
        predicted_label(scenario)
        for scenario in scenarios
    )

    return {
        "count": len(scenarios),
        "accuracy": safe_divide(
            correct,
            len(scenarios),
        ),
        "should_block_accuracy": safe_divide(
            block_correct,
            len(scenarios),
        ),
        "input_mutation_rate": safe_divide(
            mutations,
            len(scenarios),
        ),
        "no_inference_violation_rate": safe_divide(
            inference_safe,
            len(scenarios),
        ),
        "technical_error_count": technical_errors,
        "average_latency_ms": mean(latencies),
        "expected_label_distribution": dict(
            sorted(expected_distribution.items())
        ),
        "predicted_label_distribution": dict(
            sorted(predicted_distribution.items())
        ),
    }


def breakdown(
    scenarios: Sequence[Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for scenario in scenarios:
        value = scenario.get(key)

        if value is None:
            value = "__MISSING__"

        groups[str(value)].append(scenario)

    return {
        group: subgroup_summary(items)
        for group, items
        in sorted(groups.items())
    }


# ============================================================
# LLM routing metrics
# ============================================================

def llm_routing_metrics(
    scenarios: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    expected_warning_path = 0
    reported_warning_path = 0
    observed_llm_outputs = 0
    unexpected_llm_path = 0
    missing_expected_llm_path = 0
    enrichment_errors = 0

    by_label: Dict[str, Dict[str, int]] = {
        label: {
            "scenario_count": 0,
            "expected_llm_path": 0,
            "reported_llm_path": 0,
            "observed_llm_output": 0,
            "enrichment_errors": 0,
        }
        for label in LABELS
    }

    for scenario in scenarios:
        label = expected_label(
            scenario
        )
        actual = scenario_actual(
            scenario
        )

        expected_path = (
            label == "AMBIGUOUS"
        )

        reported_path = bool(
            actual.get(
                "llm_enrichment_path_expected",
                False,
            )
        )

        observed_output = bool(
            actual.get(
                "llm_output_observed",
                False,
            )
        )

        enrichment_error = (
            scenario_enrichment_error(
                scenario
            )
            not in {
                None,
                "",
                False,
            }
        )

        if expected_path:
            expected_warning_path += 1

        if reported_path:
            reported_warning_path += 1

        if observed_output:
            observed_llm_outputs += 1

        if (
            reported_path
            and not expected_path
        ):
            unexpected_llm_path += 1

        if (
            expected_path
            and not reported_path
        ):
            missing_expected_llm_path += 1

        if enrichment_error:
            enrichment_errors += 1

        if label in by_label:
            item = by_label[label]
            item["scenario_count"] += 1
            item["expected_llm_path"] += int(
                expected_path
            )
            item["reported_llm_path"] += int(
                reported_path
            )
            item["observed_llm_output"] += int(
                observed_output
            )
            item["enrichment_errors"] += int(
                enrichment_error
            )

    total = len(scenarios)

    routing_correct = sum(
        1
        for scenario in scenarios
        if (
            bool(
                scenario_actual(
                    scenario
                ).get(
                    "llm_enrichment_path_expected",
                    False,
                )
            )
            == (
                expected_label(
                    scenario
                )
                == "AMBIGUOUS"
            )
        )
    )

    return {
        "routing_accuracy": safe_divide(
            routing_correct,
            total,
        ),
        "expected_warning_path_count": (
            expected_warning_path
        ),
        "reported_warning_path_count": (
            reported_warning_path
        ),
        "observed_llm_output_count": (
            observed_llm_outputs
        ),
        "unexpected_llm_path_count": (
            unexpected_llm_path
        ),
        "unexpected_llm_path_rate": safe_divide(
            unexpected_llm_path,
            total,
        ),
        "missing_expected_llm_path_count": (
            missing_expected_llm_path
        ),
        "missing_expected_llm_path_rate": safe_divide(
            missing_expected_llm_path,
            expected_warning_path,
        ),
        "llm_enrichment_error_count": (
            enrichment_errors
        ),
        "llm_enrichment_error_rate": safe_divide(
            enrichment_errors,
            expected_warning_path,
        ),
        "by_label": by_label,
        "interpretation_note": (
            "Le compteur observed_llm_output est un indicateur "
            "observationnel du runner, pas un compteur réseau exact "
            "des appels Ollama."
        ),
    }


# ============================================================
# Global report
# ============================================================

def compute_metrics(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    scenarios = [
        scenario
        for scenario in data.get("scenarios", [])
        if isinstance(scenario, dict)
    ]

    classification = classification_metrics(
        scenarios
    )

    blocking = issue_detection_metrics(
        scenarios,
        "blocking",
    )

    warnings = issue_detection_metrics(
        scenarios,
        "warning",
    )

    field_refs = issue_field_reference_metrics(
        scenarios
    )

    safety = safety_metrics(scenarios)
    latency = latency_metrics(scenarios)
    routing = llm_routing_metrics(
        scenarios
    )

    execution_summary = data.get(
        "execution_summary",
        {},
    )

    run = data.get("run", {})

    return {
        "metric_version": "1.0.0",
        "evaluation_scope": (
            "AI Plausibility full-agent benchmark with LLM enrichment enabled"
        ),
        "source_results_file": run.get(
            "results_path"
        ),
        "dataset_path": run.get(
            "dataset_path"
        ),
        "environment": run.get(
            "environment",
            {},
        ),
        "experiment_metadata": {
            "name": data.get(
                "experiment",
                {},
            ).get("name"),
            "scope": data.get(
                "experiment",
                {},
            ).get("scope"),
            "architecture": data.get(
                "experiment",
                {},
            ).get("architecture"),
            "expected_label_distribution": data.get(
                "experiment",
                {},
            ).get(
                "expected_label_distribution"
            ),
            "category_distribution": data.get(
                "experiment",
                {},
            ).get(
                "category_distribution"
            ),
        },
        "execution": {
            "requested_scenarios": (
                execution_summary.get(
                    "requested_scenarios"
                )
            ),
            "executed_scenarios": len(scenarios),
            "reported_technical_errors": (
                execution_summary.get(
                    "technical_errors"
                )
            ),
            "reported_total_duration_seconds": (
                execution_summary.get(
                    "total_duration_seconds"
                )
            ),
        },
        "classification": classification,
        "blocking_issues": blocking,
        "warnings": warnings,
        "issue_field_references": field_refs,
        "safety": safety,
        "llm_routing": routing,
        "latency": latency,
        "breakdowns": {
            "by_label": breakdown(
                scenarios,
                "label",
            ),
            "by_category": breakdown(
                scenarios,
                "category",
            ),
            "by_language": breakdown(
                scenarios,
                "language",
            ),
            "by_difficulty": breakdown(
                scenarios,
                "difficulty",
            ),
        },
    }


# ============================================================
# Console rendering
# ============================================================

def pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def print_summary(
    metrics: Dict[str, Any],
) -> None:
    classification = metrics["classification"]
    blocking = metrics["blocking_issues"]
    warnings = metrics["warnings"]
    field_refs = metrics["issue_field_references"]
    safety = metrics["safety"]
    routing = metrics["llm_routing"]
    latency = metrics["latency"]

    print()
    print("=" * 72)
    print("AI PLAUSIBILITY FULL AGENT - METRICS")
    print("=" * 72)

    print(
        "Scenarios              : "
        f"{classification['total']}"
    )
    print(
        "Status accuracy        : "
        f"{pct(classification['plausibility_status_accuracy'])}"
    )
    print(
        "Macro precision        : "
        f"{pct(classification['macro_precision'])}"
    )
    print(
        "Macro recall           : "
        f"{pct(classification['macro_recall'])}"
    )
    print(
        "Macro F1               : "
        f"{pct(classification['macro_f1'])}"
    )

    print()
    print("--- Blocking issues ---")
    print(
        "Precision              : "
        f"{pct(blocking['precision'])}"
    )
    print(
        "Recall                 : "
        f"{pct(blocking['recall'])}"
    )
    print(
        "F1                     : "
        f"{pct(blocking['f1'])}"
    )
    print(
        "Scenario exact         : "
        f"{pct(blocking['scenario_exact_rate'])}"
    )

    print()
    print("--- Warnings ---")
    print(
        "Precision              : "
        f"{pct(warnings['precision'])}"
    )
    print(
        "Recall                 : "
        f"{pct(warnings['recall'])}"
    )
    print(
        "F1                     : "
        f"{pct(warnings['f1'])}"
    )
    print(
        "Scenario exact         : "
        f"{pct(warnings['scenario_exact_rate'])}"
    )

    print()
    print("--- Safety ---")
    print(
        "False blocking rate    : "
        f"{pct(safety['false_blocking_rate'])}"
    )
    print(
        "No-inference safe rate : "
        f"{pct(safety['no_inference_violation_rate'])}"
    )
    print(
        "Input mutation rate    : "
        f"{pct(safety['input_mutation_rate'])}"
    )
    print(
        "Technical error rate   : "
        f"{pct(safety['technical_error_rate'])}"
    )

    print()
    print("--- Field references ---")
    print(
        "Reference accuracy     : "
        f"{pct(field_refs['issue_field_reference_accuracy'])}"
    )
    print(
        "Field micro F1         : "
        f"{pct(field_refs['field_micro_f1'])}"
    )

    print()
    print("--- LLM routing ---")
    print(
        "Routing accuracy        : "
        f"{pct(routing['routing_accuracy'])}"
    )
    print(
        "Expected warning path   : "
        f"{routing['expected_warning_path_count']}"
    )
    print(
        "Reported warning path   : "
        f"{routing['reported_warning_path_count']}"
    )
    print(
        "Observed LLM outputs    : "
        f"{routing['observed_llm_output_count']}"
    )
    print(
        "Unexpected LLM path     : "
        f"{routing['unexpected_llm_path_count']}"
    )
    print(
        "Missing expected path   : "
        f"{routing['missing_expected_llm_path_count']}"
    )
    print(
        "LLM enrichment errors  : "
        f"{routing['llm_enrichment_error_count']}"
    )

    print()
    print("--- Latency ---")
    print(
        "Average                : "
        f"{latency['average_latency_ms']:.3f} ms"
    )
    print(
        "Median                 : "
        f"{latency['median_latency_ms']:.3f} ms"
    )
    print(
        "P95                    : "
        f"{latency['p95_latency_ms']:.3f} ms"
    )
    print(
        "P99                    : "
        f"{latency['p99_latency_ms']:.3f} ms"
    )
    print(
        "Max                    : "
        f"{latency['max_latency_ms']:.3f} ms"
    )

    print()
    print("--- Per label ---")
    for label in LABELS:
        item = classification["per_label"][label]
        print(
            f"{label:12s} "
            f"P={pct(item['precision'])} "
            f"R={pct(item['recall'])} "
            f"F1={pct(item['f1'])} "
            f"n={item['support']}"
        )

    print("=" * 72)
    print()


# ============================================================
# CLI
# ============================================================

def repository_root_from_script() -> Path:
    """
    Le fichier est normalement placé dans :
        requirement_extractor/validation/plausibility_metrics.py

    parents[2] correspond alors à la racine du repository.
    """
    return Path(__file__).resolve().parents[2]


def default_results_path() -> Path:
    root = repository_root_from_script()

    return (
        root
        / "requirement_extractor"
        / "validation"
        / "reports"
        / "ai_plausibility"
        / "full_agent"
        / "results_ai_plausibility_full_agent.json"
    )


def default_output_path() -> Path:
    root = repository_root_from_script()

    return (
        root
        / "requirement_extractor"
        / "validation"
        / "reports"
        / "ai_plausibility"
        / "full_agent"
        / "metrics_ai_plausibility_full_agent.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcule les métriques du benchmark "
            "AI Plausibility."
        )
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Chemin vers results_ai_plausibility_full_agent.json. "
            "Par défaut, utilise le dossier reports/"
            "ai_plausibility du repository."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Chemin du JSON de métriques. "
            "Par défaut : metrics_ai_plausibility_full_agent.json."
        ),
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(
            "Calcule et affiche les métriques "
            "sans écrire de fichier JSON."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_path = (
        args.results.resolve()
        if args.results is not None
        else default_results_path()
    )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else default_output_path()
    )

    data = load_results(results_path)

    metrics = compute_metrics(data)

    print_summary(metrics)

    if not args.no_write:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                metrics,
                handle,
                ensure_ascii=False,
                indent=2,
            )

        print(
            "Metrics JSON written to:"
        )
        print(output_path)


if __name__ == "__main__":
    main()
