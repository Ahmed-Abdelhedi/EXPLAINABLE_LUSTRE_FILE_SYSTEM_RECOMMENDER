"""Validation métier du workload_analyzer Lustre.

Ce script vérifie :
- la structure du dataset généré ;
- la conservation des contraintes utilisateur ;
- les formules de capacité ;
- la cohérence des classifications ;
- la monotonie des scores MDT/OST ;
- le déterminisme du calcul.

Exécution depuis lustre_architecture_generator :
    python .\src\validate_workload_analysis.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workload_analyzer import (  # noqa: E402
    WorkloadAnalysisError,
    analyze_workload,
    load_json,
    validate_config,
)


DEFAULT_INPUT = BASE_DIR / "data" / "use_cases_lustre_1200_v4.json"
DEFAULT_CONFIG = BASE_DIR / "config" / "architecture_rules.json"
DEFAULT_OUTPUT = BASE_DIR / "output" / "workload_analysis_dataset.json"


class ValidationError(AssertionError):
    """Erreur de validation métier."""


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def assert_close(
    actual: float,
    expected: float,
    *,
    tolerance: float = 1e-6,
    message: str,
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValidationError(
            f"{message} Valeur obtenue={actual}, attendue={expected}."
        )


def validate_dataset_structure(
    source_cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    """Vérifie la structure et les invariants de tous les cas."""

    errors: list[str] = []

    if len(source_cases) != len(results):
        errors.append(
            f"Nombre de cas différent : entrée={len(source_cases)}, "
            f"sortie={len(results)}."
        )

    source_by_id = {
        case.get("case_id"): case
        for case in source_cases
        if isinstance(case, dict)
    }

    result_ids: list[str] = []
    dominance_margin = float(
        config["workload_classification"]["dominance_margin"]
    )

    required_top_level = {
        "case_id",
        "source_requirement",
        "capacity_planning",
        "normalized_factors",
        "scores",
        "workload_type",
        "metadata_indicators",
        "data_indicators",
        "constraints",
        "preferences",
        "trace",
    }

    forbidden_fields = {
        "drive_id",
        "drive_name",
        "drive_count",
        "raid_level",
        "stripe_count",
        "stripe_size",
        "mdt_count",
        "ost_count",
        "oss_count",
    }

    for index, result in enumerate(results, start=1):
        case_id = result.get("case_id", f"<index {index}>")

        missing = required_top_level - set(result)
        if missing:
            errors.append(
                f"{case_id}: champs principaux manquants : {sorted(missing)}."
            )
            continue

        result_ids.append(case_id)

        forbidden_present = forbidden_fields & set(result)
        if forbidden_present:
            errors.append(
                f"{case_id}: champs interdits présents : "
                f"{sorted(forbidden_present)}."
            )

        scores = result["scores"]
        metadata_score = float(scores["metadata_score"])
        data_score = float(scores["data_score"])
        difference = float(scores["score_difference"])

        if not 0.0 <= metadata_score <= 1.0:
            errors.append(
                f"{case_id}: metadata_score hors [0,1] : {metadata_score}."
            )

        if not 0.0 <= data_score <= 1.0:
            errors.append(
                f"{case_id}: data_score hors [0,1] : {data_score}."
            )

        if not math.isclose(
            difference,
            metadata_score - data_score,
            abs_tol=2e-6,
        ):
            errors.append(
                f"{case_id}: score_difference incohérent."
            )

        expected_type = "balanced"
        if difference >= dominance_margin:
            expected_type = "metadata_heavy"
        elif difference <= -dominance_margin:
            expected_type = "data_heavy"

        if result["workload_type"] != expected_type:
            errors.append(
                f"{case_id}: workload_type={result['workload_type']}, "
                f"attendu={expected_type}."
            )

        planning = result["capacity_planning"]
        requested = float(planning["requested_usable_capacity_tib"])
        growth_percent = float(planning["annual_growth_percent"])
        fill_ratio = float(planning["target_fill_ratio"])
        planned = float(planning["planned_usable_capacity_tib"])

        expected_growth_factor = 1.0 + growth_percent / 100.0
        expected_planned = (
            requested * expected_growth_factor / fill_ratio
        )

        if not math.isclose(
            float(planning["growth_factor"]),
            expected_growth_factor,
            abs_tol=1e-6,
        ):
            errors.append(f"{case_id}: growth_factor incorrect.")

        if not math.isclose(
            planned,
            expected_planned,
            abs_tol=1e-5,
        ):
            errors.append(
                f"{case_id}: planned_usable_capacity_tib incorrect."
            )

        if planned < requested:
            errors.append(
                f"{case_id}: capacité planifiée inférieure à la demande."
            )

        source = source_by_id.get(case_id)
        if source is None:
            errors.append(
                f"{case_id}: identifiant absent du dataset d'entrée."
            )
            continue

        constraints = result["constraints"]
        preferences = result["preferences"]

        if bool(constraints["ha_required"]) != bool(source["ha_required"]):
            errors.append(f"{case_id}: ha_required non conservé.")

        for field in ("max_budget_usd", "max_power_w"):
            if not math.isclose(
                float(constraints[field]),
                float(source[field]),
                abs_tol=1e-6,
            ):
                errors.append(f"{case_id}: {field} non conservé.")

        preference_fields = (
            "performance_priority",
            "cost_priority",
            "power_priority",
            "reliability_priority",
        )

        for field in preference_fields:
            if not math.isclose(
                float(preferences[field]),
                float(source[field]),
                abs_tol=1e-6,
            ):
                errors.append(f"{case_id}: {field} non conservé.")

        preference_sum = sum(
            float(preferences[field])
            for field in preference_fields
        )
        if not math.isclose(preference_sum, 1.0, abs_tol=0.01):
            errors.append(
                f"{case_id}: somme des priorités différente de 1."
            )

    if len(result_ids) != len(set(result_ids)):
        errors.append("Des case_id dupliqués existent dans la sortie.")

    return errors


def make_reference_case() -> dict[str, Any]:
    """Cas synthétique utilisé pour les tests de monotonie."""

    return {
        "case_id": "TEST_BASE",
        "requested_usable_capacity_tib": 5000,
        "client_count": 500,
        "average_file_size_gb": 5.0,
        "max_file_size_gb": 50.0,
        "total_file_count": 5_000_000,
        "read_write_ratio": {
            "read_percent": 70,
            "write_percent": 30,
        },
        "access_type": "mixed",
        "target_read_gbps": 200,
        "target_write_gbps": 100,
        "ha_required": True,
        "max_budget_usd": 500000,
        "max_power_w": 30000,
        "annual_growth_percent": 20,
        "performance_priority": 0.4,
        "cost_priority": 0.2,
        "power_priority": 0.15,
        "reliability_priority": 0.25,
    }


def analyze_variant(
    base: dict[str, Any],
    config: dict[str, Any],
    **changes: Any,
) -> dict[str, Any]:
    variant = copy.deepcopy(base)
    variant.update(changes)

    if "average_file_size_gb" in changes:
        variant["max_file_size_gb"] = max(
            float(variant["max_file_size_gb"]),
            float(variant["average_file_size_gb"]),
        )

    return analyze_workload(variant, config)


def validate_monotonicity(config: dict[str, Any]) -> list[str]:
    """Vérifie que les besoins croissants ne réduisent pas les scores."""

    errors: list[str] = []
    base = make_reference_case()
    baseline = analyze_workload(base, config)

    baseline_metadata = float(baseline["scores"]["metadata_score"])
    baseline_data = float(baseline["scores"]["data_score"])
    baseline_planned = float(
        baseline["capacity_planning"]["planned_usable_capacity_tib"]
    )

    checks = [
        (
            "augmentation du nombre de fichiers",
            analyze_variant(
                base,
                config,
                total_file_count=50_000_000,
            ),
            "metadata_score",
            baseline_metadata,
        ),
        (
            "augmentation du nombre de clients",
            analyze_variant(
                base,
                config,
                client_count=5000,
            ),
            "metadata_score",
            baseline_metadata,
        ),
        (
            "réduction de la taille moyenne des fichiers",
            analyze_variant(
                base,
                config,
                average_file_size_gb=0.1,
            ),
            "metadata_score",
            baseline_metadata,
        ),
        (
            "augmentation de la capacité",
            analyze_variant(
                base,
                config,
                requested_usable_capacity_tib=50000,
            ),
            "data_score",
            baseline_data,
        ),
        (
            "augmentation du débit de lecture",
            analyze_variant(
                base,
                config,
                target_read_gbps=1500,
            ),
            "data_score",
            baseline_data,
        ),
        (
            "augmentation du débit d'écriture",
            analyze_variant(
                base,
                config,
                target_write_gbps=900,
            ),
            "data_score",
            baseline_data,
        ),
        (
            "augmentation de la taille moyenne des fichiers",
            analyze_variant(
                base,
                config,
                average_file_size_gb=30.0,
            ),
            "data_score",
            baseline_data,
        ),
        (
            "augmentation de la croissance",
            analyze_variant(
                base,
                config,
                annual_growth_percent=60,
            ),
            "data_score",
            baseline_data,
        ),
    ]

    for description, result, score_name, baseline_score in checks:
        actual = float(result["scores"][score_name])
        if actual + 1e-9 < baseline_score:
            errors.append(
                f"Monotonie violée : {description} réduit {score_name} "
                f"de {baseline_score:.6f} à {actual:.6f}."
            )

    higher_growth = analyze_variant(
        base,
        config,
        annual_growth_percent=60,
    )
    higher_growth_planned = float(
        higher_growth["capacity_planning"][
            "planned_usable_capacity_tib"
        ]
    )

    if higher_growth_planned <= baseline_planned:
        errors.append(
            "Monotonie violée : augmenter la croissance n'augmente pas "
            "la capacité planifiée."
        )

    first = analyze_workload(base, config)
    second = analyze_workload(copy.deepcopy(base), config)
    if first != second:
        errors.append(
            "Déterminisme violé : le même cas produit deux sorties différentes."
        )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valide le dataset d'analyse des workloads Lustre."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_json(args.config)
    source_cases = load_json(args.input)
    results = load_json(args.output)

    if not isinstance(config, dict):
        raise ValidationError(
            "architecture_rules.json doit contenir un objet JSON."
        )
    if not isinstance(source_cases, list):
        raise ValidationError(
            "Le dataset d'entrée doit contenir une liste JSON."
        )
    if not isinstance(results, list):
        raise ValidationError(
            "Le dataset de sortie doit contenir une liste JSON."
        )

    validate_config(config)

    structure_errors = validate_dataset_structure(
        source_cases,
        results,
        config,
    )
    monotonicity_errors = validate_monotonicity(config)
    errors = structure_errors + monotonicity_errors

    print("Validation du workload analyzer")
    print("--------------------------------")
    print(f"Cas d'entrée       : {len(source_cases)}")
    print(f"Cas de sortie      : {len(results)}")
    print(f"Erreurs structure  : {len(structure_errors)}")
    print(f"Erreurs monotonie  : {len(monotonicity_errors)}")

    if errors:
        print("\nSTATUT : ÉCHEC")
        for error in errors[:50]:
            print(f"  - {error}")

        if len(errors) > 50:
            print(
                f"  ... {len(errors) - 50} erreur(s) supplémentaire(s)."
            )

        raise SystemExit(1)

    print("\nSTATUT : VALIDÉ")
    print("Tous les invariants et tests de monotonie sont respectés.")


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, WorkloadAnalysisError) as exc:
        print(f"\nErreur de validation : {exc}")
        raise SystemExit(1) from exc
