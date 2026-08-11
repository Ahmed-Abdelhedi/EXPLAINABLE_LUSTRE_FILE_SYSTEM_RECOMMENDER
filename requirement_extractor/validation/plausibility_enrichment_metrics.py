#!/usr/bin/env python3
"""
plausibility_enrichment_metrics.py

Calcule les métriques du benchmark dédié au LLM enrichment
de l'AI Plausibility Agent.

Entrée par défaut :
    requirement_extractor/validation/reports/ai_plausibility/enrichment/
        results_ai_plausibility_enrichment.json

Sortie par défaut :
    requirement_extractor/validation/reports/ai_plausibility/enrichment/
        metrics_ai_plausibility_enrichment.json

Objectifs principaux :
- vérifier que le LLM ne modifie jamais la décision ;
- vérifier qu'il ne crée/supprime/modifie aucune issue ;
- vérifier la préservation des champs, sévérités et preuves ;
- vérifier la préservation des nombres et unités métier ;
- mesurer le taux de reformulation réellement appliquée ;
- mesurer le coût en latence ;
- fournir des breakdowns par catégorie, langue et difficulté.

Aucune dépendance externe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# Generic helpers
# ============================================================

def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return statistics.fmean(values)


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


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_upper(value: Any) -> str:
    return normalize_text(value).upper()


def normalize_severity(value: Any) -> str:
    return normalize_upper(value)


def pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


# ============================================================
# Loading
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
            "La racine du fichier doit être un objet JSON."
        )

    scenarios = data.get("scenarios")

    if not isinstance(scenarios, list):
        raise ValueError(
            "La clé 'scenarios' doit contenir une liste."
        )

    return data


# ============================================================
# Issue matching
# ============================================================

def issue_identity(
    issue: Dict[str, Any],
) -> Tuple[str, str, str]:
    return (
        normalize_upper(
            issue.get("issue_type")
        ),
        normalize_text(
            issue.get("field")
        ),
        normalize_severity(
            issue.get("severity")
        ),
    )


def identity_counter(
    issues: Sequence[Dict[str, Any]],
) -> Counter:
    return Counter(
        issue_identity(issue)
        for issue in issues
        if isinstance(issue, dict)
    )


def issue_type_counter(
    issues: Sequence[Dict[str, Any]],
) -> Counter:
    return Counter(
        normalize_upper(
            issue.get("issue_type")
        )
        for issue in issues
        if isinstance(issue, dict)
    )


def field_counter(
    issues: Sequence[Dict[str, Any]],
) -> Counter:
    return Counter(
        normalize_text(
            issue.get("field")
        )
        for issue in issues
        if isinstance(issue, dict)
    )


def severity_counter(
    issues: Sequence[Dict[str, Any]],
) -> Counter:
    return Counter(
        normalize_severity(
            issue.get("severity")
        )
        for issue in issues
        if isinstance(issue, dict)
    )


def pair_by_identity(
    baseline_issues: Sequence[Dict[str, Any]],
    enriched_issues: Sequence[Dict[str, Any]],
) -> List[
    Tuple[
        Dict[str, Any],
        Optional[Dict[str, Any]],
    ]
]:
    """
    Associe chaque issue baseline à une issue enriched de même identité
    (issue_type, field, severity).

    Cette identité représente précisément les attributs qui ne doivent
    jamais être modifiés par le LLM.
    """
    unused = list(enriched_issues)
    pairs = []

    for baseline_issue in baseline_issues:
        identity = issue_identity(
            baseline_issue
        )

        match_index = None

        for index, candidate in enumerate(
            unused
        ):
            if (
                issue_identity(candidate)
                == identity
            ):
                match_index = index
                break

        if match_index is None:
            pairs.append(
                (baseline_issue, None)
            )
            continue

        pairs.append(
            (
                baseline_issue,
                unused.pop(match_index),
            )
        )

    return pairs


# ============================================================
# Semantic fact preservation
# ============================================================

def extract_numeric_facts(
    text: str,
) -> List[Decimal]:
    """
    Extrait les nombres du texte.

    10.00 et 10,00 sont considérés comme la même valeur.
    En revanche 1500 et 1536 sont différents.
    """
    if not text:
        return []

    tokens = re.findall(
        r"(?<![A-Za-z0-9_])[-+]?\d+(?:[.,]\d+)?",
        text,
    )

    values: List[Decimal] = []

    for token in tokens:
        normalized = token.replace(",", ".")

        try:
            values.append(
                Decimal(normalized)
            )
        except InvalidOperation:
            continue

    return sorted(values)


def extract_protected_units(
    text: str,
) -> List[str]:
    """
    Normalise les unités métier dont une modification peut
    changer le sens du warning.

    Exemples équivalents :
      clients/TiB == clients par TiB
      W/TiB == W par TiB

    Exemples NON équivalents :
      GB/s != Gbps
      gigabytes/s != gigabits/s
    """
    if not text:
        return []

    protected: List[str] = []

    patterns = (
        (
            r"\bUSD\s*(?:/|par)\s*TiB\b",
            "USD_PER_TIB",
        ),
        (
            r"\bW\s*(?:/|par)\s*TiB\b",
            "W_PER_TIB",
        ),
        (
            r"\bclients?\s*(?:/|par)\s*TiB\b",
            "CLIENTS_PER_TIB",
        ),
        (
            r"\bGB\s*/\s*s\b",
            "GIGABYTES_PER_SECOND",
        ),
        (
            r"\bGBps\b",
            "GIGABYTES_PER_SECOND",
        ),
        (
            r"\bgigabytes?\s*(?:/|par)\s*(?:second|seconde|s)\b",
            "GIGABYTES_PER_SECOND",
        ),
        (
            r"\bGbps\b",
            "GIGABITS_PER_SECOND",
        ),
        (
            r"\bGb\s*/\s*s\b",
            "GIGABITS_PER_SECOND",
        ),
        (
            r"\bgigabits?\s*(?:/|par)\s*(?:second|seconde|s)\b",
            "GIGABITS_PER_SECOND",
        ),
        (
            r"\bTiB\b",
            "TIB",
        ),
        (
            r"\bUSD\b",
            "USD",
        ),
    )

    for pattern, canonical in patterns:
        count = len(
            re.findall(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        )

        protected.extend(
            [canonical] * count
        )

    return sorted(protected)


def text_numeric_facts_preserved(
    baseline: str,
    enriched: str,
) -> bool:
    return (
        extract_numeric_facts(baseline)
        == extract_numeric_facts(enriched)
    )


def text_units_preserved(
    baseline: str,
    enriched: str,
) -> bool:
    return (
        extract_protected_units(baseline)
        == extract_protected_units(enriched)
    )


def issue_semantic_preservation(
    baseline_issue: Dict[str, Any],
    enriched_issue: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_message = normalize_text(
        baseline_issue.get("message")
    )
    enriched_message = normalize_text(
        enriched_issue.get("message")
    )

    baseline_question = normalize_text(
        baseline_issue.get("question")
    )
    enriched_question = normalize_text(
        enriched_issue.get("question")
    )

    message_numeric = (
        text_numeric_facts_preserved(
            baseline_message,
            enriched_message,
        )
    )

    question_numeric = (
        text_numeric_facts_preserved(
            baseline_question,
            enriched_question,
        )
    )

    message_units = (
        text_units_preserved(
            baseline_message,
            enriched_message,
        )
    )

    question_units = (
        text_units_preserved(
            baseline_question,
            enriched_question,
        )
    )

    return {
        "numeric_facts_preserved": (
            message_numeric
            and question_numeric
        ),
        "units_preserved": (
            message_units
            and question_units
        ),
        "message_numeric_facts_preserved": (
            message_numeric
        ),
        "question_numeric_facts_preserved": (
            question_numeric
        ),
        "message_units_preserved": (
            message_units
        ),
        "question_units_preserved": (
            question_units
        ),
        "message_changed": (
            baseline_message
            != enriched_message
        ),
        "question_changed": (
            baseline_question
            != enriched_question
        ),
    }


# ============================================================
# Scenario analysis
# ============================================================

def analyze_scenario(
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    baseline = scenario.get(
        "baseline",
        {},
    )
    enriched = scenario.get(
        "enriched",
        {},
    )

    baseline_issues = [
        issue
        for issue in as_list(
            baseline.get("issues")
        )
        if isinstance(issue, dict)
    ]

    enriched_issues = [
        issue
        for issue in as_list(
            enriched.get("issues")
        )
        if isinstance(issue, dict)
    ]

    baseline_identity = (
        identity_counter(
            baseline_issues
        )
    )
    enriched_identity = (
        identity_counter(
            enriched_issues
        )
    )

    new_issues = list(
        (
            enriched_identity
            - baseline_identity
        ).elements()
    )

    deleted_issues = list(
        (
            baseline_identity
            - enriched_identity
        ).elements()
    )

    pairs = pair_by_identity(
        baseline_issues,
        enriched_issues,
    )

    semantic_results = []

    for baseline_issue, enriched_issue in pairs:
        if enriched_issue is None:
            continue

        semantic_results.append(
            issue_semantic_preservation(
                baseline_issue,
                enriched_issue,
            )
        )

    numeric_preserved = (
        bool(semantic_results)
        and all(
            item[
                "numeric_facts_preserved"
            ]
            for item in semantic_results
        )
    )

    units_preserved = (
        bool(semantic_results)
        and all(
            item[
                "units_preserved"
            ]
            for item in semantic_results
        )
    )

    message_changed = any(
        item["message_changed"]
        for item in semantic_results
    )

    question_changed = any(
        item["question_changed"]
        for item in semantic_results
    )

    immutable_attributes_preserved = True

    for baseline_issue, enriched_issue in pairs:
        if enriched_issue is None:
            immutable_attributes_preserved = False
            continue

        for attribute in (
            "confidence",
            "suggested_correction",
            "evidence_fields",
        ):
            if (
                baseline_issue.get(attribute)
                != enriched_issue.get(attribute)
            ):
                immutable_attributes_preserved = False

    technical_error = bool(
        baseline.get("technical_error")
        or enriched.get("technical_error")
    )

    enrichment_error = bool(
        enriched.get("enrichment_error")
    )

    input_mutated = bool(
        enriched.get(
            "input_mutated",
            False,
        )
    )

    no_inference_violations = as_list(
        enriched.get(
            "no_inference_violations"
        )
    )

    status_preserved = (
        baseline.get("raw_status")
        == enriched.get("raw_status")
    )

    label_preserved = (
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

    issue_type_preserved = (
        issue_type_counter(
            baseline_issues
        )
        == issue_type_counter(
            enriched_issues
        )
    )

    field_preserved = (
        field_counter(
            baseline_issues
        )
        == field_counter(
            enriched_issues
        )
    )

    severity_preserved = (
        severity_counter(
            baseline_issues
        )
        == severity_counter(
            enriched_issues
        )
    )

    issue_identity_preserved = (
        baseline_identity
        == enriched_identity
    )

    issue_count_preserved = (
        len(baseline_issues)
        == len(enriched_issues)
    )

    no_new_issue = (
        len(new_issues) == 0
    )

    no_deleted_issue = (
        len(deleted_issues) == 0
    )

    decision_preserved = all(
        (
            status_preserved,
            label_preserved,
            should_block_preserved,
        )
    )

    structural_safety = all(
        (
            decision_preserved,
            issue_type_preserved,
            field_preserved,
            severity_preserved,
            issue_identity_preserved,
            issue_count_preserved,
            no_new_issue,
            no_deleted_issue,
            immutable_attributes_preserved,
            not input_mutated,
            not no_inference_violations,
            not technical_error,
        )
    )

    semantic_safety = all(
        (
            numeric_preserved,
            units_preserved,
        )
    )

    full_safety = (
        structural_safety
        and semantic_safety
    )

    return {
        "id": scenario.get("id"),
        "category": scenario.get(
            "category"
        ),
        "language": scenario.get(
            "language"
        ),
        "difficulty": scenario.get(
            "difficulty"
        ),
        "status_preserved": (
            status_preserved
        ),
        "predicted_label_preserved": (
            label_preserved
        ),
        "should_block_preserved": (
            should_block_preserved
        ),
        "decision_preserved": (
            decision_preserved
        ),
        "issue_type_preserved": (
            issue_type_preserved
        ),
        "field_preserved": (
            field_preserved
        ),
        "severity_preserved": (
            severity_preserved
        ),
        "issue_identity_preserved": (
            issue_identity_preserved
        ),
        "issue_count_preserved": (
            issue_count_preserved
        ),
        "immutable_attributes_preserved": (
            immutable_attributes_preserved
        ),
        "new_issue_created": (
            not no_new_issue
        ),
        "issue_deleted": (
            not no_deleted_issue
        ),
        "new_issue_identities": [
            list(identity)
            for identity in new_issues
        ],
        "deleted_issue_identities": [
            list(identity)
            for identity in deleted_issues
        ],
        "input_mutated": input_mutated,
        "no_inference_violation": (
            len(
                no_inference_violations
            )
            == 0
        ),
        "numeric_facts_preserved": (
            numeric_preserved
        ),
        "units_preserved": (
            units_preserved
        ),
        "message_changed": (
            message_changed
        ),
        "question_changed": (
            question_changed
        ),
        "effective_enrichment": (
            message_changed
            or question_changed
        ),
        "technical_error": (
            technical_error
        ),
        "enrichment_error": (
            enrichment_error
        ),
        "structural_safety": (
            structural_safety
        ),
        "semantic_safety": (
            semantic_safety
        ),
        "full_safety": (
            full_safety
        ),
        "baseline_latency_ms": float(
            baseline.get(
                "latency_ms",
                0.0,
            )
            or 0.0
        ),
        "enriched_latency_ms": float(
            enriched.get(
                "latency_ms",
                0.0,
            )
            or 0.0
        ),
    }


# ============================================================
# Aggregate metrics
# ============================================================

BOOLEAN_METRIC_KEYS = (
    "status_preserved",
    "predicted_label_preserved",
    "should_block_preserved",
    "decision_preserved",
    "issue_type_preserved",
    "field_preserved",
    "severity_preserved",
    "issue_identity_preserved",
    "issue_count_preserved",
    "immutable_attributes_preserved",
    "no_inference_violation",
    "numeric_facts_preserved",
    "units_preserved",
    "message_changed",
    "question_changed",
    "effective_enrichment",
    "structural_safety",
    "semantic_safety",
    "full_safety",
)


def aggregate_boolean_rates(
    analyses: Sequence[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    total = len(analyses)

    result = {}

    for key in BOOLEAN_METRIC_KEYS:
        count = sum(
            1
            for item in analyses
            if bool(item.get(key))
        )

        result[f"{key}_count"] = count
        result[f"{key}_rate"] = safe_divide(
            count,
            total,
        )

    new_issue_count = sum(
        1
        for item in analyses
        if item.get(
            "new_issue_created"
        )
    )

    issue_deleted_count = sum(
        1
        for item in analyses
        if item.get(
            "issue_deleted"
        )
    )

    mutation_count = sum(
        1
        for item in analyses
        if item.get(
            "input_mutated"
        )
    )

    technical_error_count = sum(
        1
        for item in analyses
        if item.get(
            "technical_error"
        )
    )

    enrichment_error_count = sum(
        1
        for item in analyses
        if item.get(
            "enrichment_error"
        )
    )

    result.update(
        {
            "new_issue_creation_count": (
                new_issue_count
            ),
            "new_issue_creation_rate": (
                safe_divide(
                    new_issue_count,
                    total,
                )
            ),
            "issue_deletion_count": (
                issue_deleted_count
            ),
            "issue_deletion_rate": (
                safe_divide(
                    issue_deleted_count,
                    total,
                )
            ),
            "input_mutation_count": (
                mutation_count
            ),
            "input_mutation_rate": (
                safe_divide(
                    mutation_count,
                    total,
                )
            ),
            "technical_error_count": (
                technical_error_count
            ),
            "technical_error_rate": (
                safe_divide(
                    technical_error_count,
                    total,
                )
            ),
            "llm_enrichment_error_count": (
                enrichment_error_count
            ),
            "llm_enrichment_error_rate": (
                safe_divide(
                    enrichment_error_count,
                    total,
                )
            ),
        }
    )

    return result


def latency_summary(
    analyses: Sequence[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    baseline = [
        float(
            item[
                "baseline_latency_ms"
            ]
        )
        for item in analyses
    ]

    enriched = [
        float(
            item[
                "enriched_latency_ms"
            ]
        )
        for item in analyses
    ]

    overhead = [
        enriched_value
        - baseline_value
        for baseline_value, enriched_value
        in zip(
            baseline,
            enriched,
        )
    ]

    def summarize(
        values: Sequence[float],
    ) -> Dict[str, float]:
        if not values:
            return {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }

        return {
            "mean_ms": mean(values),
            "median_ms": statistics.median(
                values
            ),
            "p95_ms": percentile(
                values,
                0.95,
            ),
            "p99_ms": percentile(
                values,
                0.99,
            ),
            "min_ms": min(values),
            "max_ms": max(values),
        }

    baseline_summary = summarize(
        baseline
    )
    enriched_summary = summarize(
        enriched
    )
    overhead_summary = summarize(
        overhead
    )

    speed_ratio = safe_divide(
        enriched_summary[
            "mean_ms"
        ],
        baseline_summary[
            "mean_ms"
        ],
    )

    return {
        "baseline": baseline_summary,
        "enriched": enriched_summary,
        "overhead": overhead_summary,
        "mean_latency_multiplier": (
            speed_ratio
        ),
    }


# ============================================================
# Breakdowns
# ============================================================

def subgroup_summary(
    analyses: Sequence[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    rates = aggregate_boolean_rates(
        analyses
    )

    return {
        "count": len(analyses),
        "decision_preservation_rate": (
            rates[
                "decision_preserved_rate"
            ]
        ),
        "issue_identity_preservation_rate": (
            rates[
                "issue_identity_preserved_rate"
            ]
        ),
        "numeric_fact_preservation_rate": (
            rates[
                "numeric_facts_preserved_rate"
            ]
        ),
        "unit_preservation_rate": (
            rates[
                "units_preserved_rate"
            ]
        ),
        "full_safety_rate": (
            rates[
                "full_safety_rate"
            ]
        ),
        "message_enrichment_rate": (
            rates[
                "message_changed_rate"
            ]
        ),
        "question_enrichment_rate": (
            rates[
                "question_changed_rate"
            ]
        ),
        "effective_enrichment_rate": (
            rates[
                "effective_enrichment_rate"
            ]
        ),
        "technical_error_rate": (
            rates[
                "technical_error_rate"
            ]
        ),
        "llm_enrichment_error_rate": (
            rates[
                "llm_enrichment_error_rate"
            ]
        ),
        "average_baseline_latency_ms": (
            mean(
                [
                    item[
                        "baseline_latency_ms"
                    ]
                    for item in analyses
                ]
            )
        ),
        "average_enriched_latency_ms": (
            mean(
                [
                    item[
                        "enriched_latency_ms"
                    ]
                    for item in analyses
                ]
            )
        ),
    }


def breakdown(
    analyses: Sequence[
        Dict[str, Any]
    ],
    key: str,
) -> Dict[str, Any]:
    groups: Dict[
        str,
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for item in analyses:
        value = item.get(key)

        if value is None:
            value = "__MISSING__"

        groups[str(value)].append(
            item
        )

    return {
        group: subgroup_summary(items)
        for group, items
        in sorted(
            groups.items()
        )
    }


# ============================================================
# Full metrics object
# ============================================================

def compute_metrics(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    scenarios = [
        scenario
        for scenario in data.get(
            "scenarios",
            [],
        )
        if isinstance(
            scenario,
            dict,
        )
    ]

    analyses = [
        analyze_scenario(
            scenario
        )
        for scenario in scenarios
    ]

    rates = aggregate_boolean_rates(
        analyses
    )

    latency = latency_summary(
        analyses
    )

    run = data.get(
        "run",
        {},
    )

    experiment = data.get(
        "experiment",
        {},
    )

    execution_summary = data.get(
        "execution_summary",
        {},
    )

    return {
        "metric_version": "1.0.0",
        "evaluation_scope": (
            "AI Plausibility LLM "
            "Enrichment Safety Validation"
        ),
        "source_results_file": (
            run.get("results_path")
        ),
        "dataset_path": (
            run.get("dataset_path")
        ),
        "environment": (
            run.get(
                "environment",
                {},
            )
        ),
        "experiment": {
            "name": (
                experiment.get("name")
            ),
            "scope": (
                experiment.get("scope")
            ),
            "target_label": (
                experiment.get(
                    "target_label"
                )
            ),
            "selected_categories": (
                experiment.get(
                    "selected_categories",
                    {},
                )
            ),
        },
        "execution": {
            "requested_scenarios": (
                execution_summary.get(
                    "requested_scenarios"
                )
            ),
            "executed_scenarios": (
                len(analyses)
            ),
            "reported_total_duration_seconds": (
                execution_summary.get(
                    "total_duration_seconds"
                )
            ),
        },
        "safety": {
            "status_preservation_rate": (
                rates[
                    "status_preserved_rate"
                ]
            ),
            "predicted_label_preservation_rate": (
                rates[
                    "predicted_label_preserved_rate"
                ]
            ),
            "should_block_preservation_rate": (
                rates[
                    "should_block_preserved_rate"
                ]
            ),
            "decision_preservation_rate": (
                rates[
                    "decision_preserved_rate"
                ]
            ),
            "issue_type_preservation_rate": (
                rates[
                    "issue_type_preserved_rate"
                ]
            ),
            "field_preservation_rate": (
                rates[
                    "field_preserved_rate"
                ]
            ),
            "severity_preservation_rate": (
                rates[
                    "severity_preserved_rate"
                ]
            ),
            "issue_identity_preservation_rate": (
                rates[
                    "issue_identity_preserved_rate"
                ]
            ),
            "issue_count_preservation_rate": (
                rates[
                    "issue_count_preserved_rate"
                ]
            ),
            "immutable_attributes_preservation_rate": (
                rates[
                    "immutable_attributes_preserved_rate"
                ]
            ),
            "new_issue_creation_rate": (
                rates[
                    "new_issue_creation_rate"
                ]
            ),
            "issue_deletion_rate": (
                rates[
                    "issue_deletion_rate"
                ]
            ),
            "input_mutation_rate": (
                rates[
                    "input_mutation_rate"
                ]
            ),
            "no_inference_violation_rate": (
                rates[
                    "no_inference_violation_rate"
                ]
            ),
            "technical_error_rate": (
                rates[
                    "technical_error_rate"
                ]
            ),
            "llm_enrichment_error_rate": (
                rates[
                    "llm_enrichment_error_rate"
                ]
            ),
            "structural_safety_rate": (
                rates[
                    "structural_safety_rate"
                ]
            ),
            "full_safety_rate": (
                rates[
                    "full_safety_rate"
                ]
            ),
        },
        "semantic_preservation": {
            "numeric_fact_preservation_rate": (
                rates[
                    "numeric_facts_preserved_rate"
                ]
            ),
            "unit_preservation_rate": (
                rates[
                    "units_preserved_rate"
                ]
            ),
            "semantic_safety_rate": (
                rates[
                    "semantic_safety_rate"
                ]
            ),
        },
        "enrichment_utility": {
            "message_enrichment_rate": (
                rates[
                    "message_changed_rate"
                ]
            ),
            "question_enrichment_rate": (
                rates[
                    "question_changed_rate"
                ]
            ),
            "effective_enrichment_rate": (
                rates[
                    "effective_enrichment_rate"
                ]
            ),
            "message_changed_count": (
                rates[
                    "message_changed_count"
                ]
            ),
            "question_changed_count": (
                rates[
                    "question_changed_count"
                ]
            ),
            "effective_enrichment_count": (
                rates[
                    "effective_enrichment_count"
                ]
            ),
            "safe_fallback_count": (
                len(analyses)
                - rates[
                    "effective_enrichment_count"
                ]
            ),
            "safe_fallback_rate": safe_divide(
                (
                    len(analyses)
                    - rates[
                        "effective_enrichment_count"
                    ]
                ),
                len(analyses),
            ),
        },
        "latency": latency,
        "breakdowns": {
            "by_category": breakdown(
                analyses,
                "category",
            ),
            "by_language": breakdown(
                analyses,
                "language",
            ),
            "by_difficulty": breakdown(
                analyses,
                "difficulty",
            ),
        },
        "scenario_diagnostics": (
            analyses
        ),
    }


# ============================================================
# Console
# ============================================================

def print_summary(
    metrics: Dict[str, Any],
) -> None:
    safety = metrics["safety"]
    semantic = metrics[
        "semantic_preservation"
    ]
    utility = metrics[
        "enrichment_utility"
    ]
    latency = metrics["latency"]
    execution = metrics["execution"]

    print()
    print("=" * 78)
    print(
        "AI PLAUSIBILITY - LLM ENRICHMENT METRICS"
    )
    print("=" * 78)

    print(
        "Scenarios                 : "
        f"{execution['executed_scenarios']}"
    )

    print()
    print("--- Decision / structure safety ---")
    print(
        "Decision preservation     : "
        f"{pct(safety['decision_preservation_rate'])}"
    )
    print(
        "Issue identity            : "
        f"{pct(safety['issue_identity_preservation_rate'])}"
    )
    print(
        "Field preservation        : "
        f"{pct(safety['field_preservation_rate'])}"
    )
    print(
        "Severity preservation     : "
        f"{pct(safety['severity_preservation_rate'])}"
    )
    print(
        "Immutable attributes      : "
        f"{pct(safety['immutable_attributes_preservation_rate'])}"
    )
    print(
        "New issue creation        : "
        f"{pct(safety['new_issue_creation_rate'])}"
    )
    print(
        "Issue deletion            : "
        f"{pct(safety['issue_deletion_rate'])}"
    )
    print(
        "Input mutation            : "
        f"{pct(safety['input_mutation_rate'])}"
    )
    print(
        "No-inference safe         : "
        f"{pct(safety['no_inference_violation_rate'])}"
    )
    print(
        "Technical errors          : "
        f"{pct(safety['technical_error_rate'])}"
    )
    print(
        "LLM enrichment errors     : "
        f"{pct(safety['llm_enrichment_error_rate'])}"
    )

    print()
    print("--- Semantic safety ---")
    print(
        "Numeric fact preservation : "
        f"{pct(semantic['numeric_fact_preservation_rate'])}"
    )
    print(
        "Unit preservation         : "
        f"{pct(semantic['unit_preservation_rate'])}"
    )
    print(
        "Full safety               : "
        f"{pct(safety['full_safety_rate'])}"
    )

    print()
    print("--- Enrichment utility ---")
    print(
        "Message changed           : "
        f"{utility['message_changed_count']}/"
        f"{execution['executed_scenarios']} "
        f"({pct(utility['message_enrichment_rate'])})"
    )
    print(
        "Question changed          : "
        f"{utility['question_changed_count']}/"
        f"{execution['executed_scenarios']} "
        f"({pct(utility['question_enrichment_rate'])})"
    )
    print(
        "Any effective enrichment  : "
        f"{utility['effective_enrichment_count']}/"
        f"{execution['executed_scenarios']} "
        f"({pct(utility['effective_enrichment_rate'])})"
    )
    print(
        "Safe fallback             : "
        f"{utility['safe_fallback_count']}/"
        f"{execution['executed_scenarios']} "
        f"({pct(utility['safe_fallback_rate'])})"
    )

    print()
    print("--- Latency ---")
    print(
        "Baseline mean             : "
        f"{latency['baseline']['mean_ms']:.3f} ms"
    )
    print(
        "Enriched mean             : "
        f"{latency['enriched']['mean_ms']:.3f} ms"
    )
    print(
        "Enriched median           : "
        f"{latency['enriched']['median_ms']:.3f} ms"
    )
    print(
        "Enriched P95              : "
        f"{latency['enriched']['p95_ms']:.3f} ms"
    )
    print(
        "Enriched P99              : "
        f"{latency['enriched']['p99_ms']:.3f} ms"
    )
    print(
        "Mean overhead             : "
        f"{latency['overhead']['mean_ms']:.3f} ms"
    )

    print()
    print("--- By category ---")

    for category, item in (
        metrics["breakdowns"][
            "by_category"
        ].items()
    ):
        print(
            f"{category:40s} "
            f"safe={pct(item['full_safety_rate'])} "
            f"msg={pct(item['message_enrichment_rate'])} "
            f"eff={pct(item['effective_enrichment_rate'])} "
            f"lat={item['average_enriched_latency_ms']:.1f} ms"
        )

    print("=" * 78)
    print()


# ============================================================
# CLI paths
# ============================================================

def repository_root_from_script() -> Path:
    return Path(
        __file__
    ).resolve().parents[2]


def default_results_path() -> Path:
    root = (
        repository_root_from_script()
    )

    return (
        root
        / "requirement_extractor"
        / "validation"
        / "reports"
        / "ai_plausibility"
        / "enrichment"
        / "results_ai_plausibility_enrichment.json"
    )


def default_output_path() -> Path:
    root = (
        repository_root_from_script()
    )

    return (
        root
        / "requirement_extractor"
        / "validation"
        / "reports"
        / "ai_plausibility"
        / "enrichment"
        / "metrics_ai_plausibility_enrichment.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcule les métriques du "
            "LLM enrichment de l'AI "
            "Plausibility Agent."
        )
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help=(
            "Chemin vers "
            "results_ai_plausibility_enrichment.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Chemin vers "
            "metrics_ai_plausibility_enrichment.json"
        ),
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(
            "Affiche seulement les métriques."
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

    data = load_results(
        results_path
    )

    metrics = compute_metrics(
        data
    )

    print_summary(
        metrics
    )

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
