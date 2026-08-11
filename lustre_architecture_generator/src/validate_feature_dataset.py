"""Validation métier de workload_features_dataset.json.

Ce script compare chaque sortie du Feature Calculator avec :
- workload_analysis_dataset.json ;
- architecture_rules.json ;
- les règles exécutées par feature_calculator.py.

Exécution depuis le dossier lustre_architecture_generator :

    python .\src\validate_feature_dataset.py
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

from feature_calculator import (  # noqa: E402
    FeatureCalculationError,
    calculate_features,
    load_json,
    validate_config,
)


DEFAULT_INPUT = BASE_DIR / "output" / "workload_analysis_dataset.json"
DEFAULT_CONFIG = BASE_DIR / "config" / "architecture_rules.json"
DEFAULT_OUTPUT = BASE_DIR / "output" / "workload_features_dataset.json"


class DatasetValidationError(AssertionError):
    """Erreur détectée dans le dataset de features."""


FORBIDDEN_CONFIGURATION_KEYS = {
    "drive",
    "drive_id",
    "drive_name",
    "drive_count",
    "raid",
    "raid_level",
    "protection_group",
    "mdt_count",
    "ost_count",
    "oss_count",
    "stripe_count",
    "stripe_size",
    "stripe_size_mb",
    "server_model",
}


def find_forbidden_keys(
    value: Any,
    path: str = "$",
) -> list[str]:
    """Cherche récursivement les décisions interdites à l'Étape 1."""

    findings: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"

            if key.lower() in FORBIDDEN_CONFIGURATION_KEYS:
                findings.append(child_path)

            findings.extend(find_forbidden_keys(child, child_path))

    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(
                find_forbidden_keys(child, f"{path}[{index}]")
            )

    return findings


def find_non_finite_numbers(
    value: Any,
    path: str = "$",
) -> list[str]:
    """Détecte NaN et les valeurs infinies."""

    findings: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            findings.extend(
                find_non_finite_numbers(child, f"{path}.{key}")
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(
                find_non_finite_numbers(child, f"{path}[{index}]")
            )

    elif isinstance(value, float) and not math.isfinite(value):
        findings.append(path)

    return findings


def values_close(
    actual: Any,
    expected: Any,
    *,
    absolute_tolerance: float = 1e-9,
) -> bool:
    """Compare deux structures en tolérant les arrondis flottants."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        if set(actual) != set(expected):
            return False

        return all(
            values_close(
                actual[key],
                expected[key],
                absolute_tolerance=absolute_tolerance,
            )
            for key in expected
        )

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False

        return all(
            values_close(
                actual_item,
                expected_item,
                absolute_tolerance=absolute_tolerance,
            )
            for actual_item, expected_item in zip(actual, expected)
        )

    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        )

    return actual == expected


def first_difference(
    actual: Any,
    expected: Any,
    path: str = "$",
) -> str | None:
    """Retourne le premier emplacement différent entre deux structures."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return (
                f"{path}: type obtenu={type(actual).__name__}, "
                "type attendu=dict"
            )

        actual_keys = set(actual)
        expected_keys = set(expected)

        missing = expected_keys - actual_keys
        if missing:
            return f"{path}: clés manquantes={sorted(missing)}"

        extra = actual_keys - expected_keys
        if extra:
            return f"{path}: clés supplémentaires={sorted(extra)}"

        for key in expected:
            difference = first_difference(
                actual[key],
                expected[key],
                f"{path}.{key}",
            )
            if difference is not None:
                return difference

        return None

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return (
                f"{path}: type obtenu={type(actual).__name__}, "
                "type attendu=list"
            )

        if len(actual) != len(expected):
            return (
                f"{path}: longueur obtenue={len(actual)}, "
                f"attendue={len(expected)}"
            )

        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            difference = first_difference(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference

        return None

    if not values_close(actual, expected):
        return (
            f"{path}: valeur obtenue={actual!r}, "
            f"attendue={expected!r}"
        )

    return None


def validate_case_invariants(
    case: dict[str, Any],
) -> list[str]:
    """Vérifie les relations numériques internes d'un cas."""

    errors: list[str] = []
    case_id = str(case.get("case_id", "<case_id absent>"))

    scores = case["scores"]
    metadata_score = float(scores["metadata_score"])
    data_score = float(scores["data_score"])
    difference = float(scores["score_difference"])

    if not 0.0 <= metadata_score <= 1.0:
        errors.append(
            f"{case_id}: metadata_score hors [0,1]."
        )

    if not 0.0 <= data_score <= 1.0:
        errors.append(
            f"{case_id}: data_score hors [0,1]."
        )

    if not math.isclose(
        difference,
        metadata_score - data_score,
        abs_tol=2e-6,
    ):
        errors.append(
            f"{case_id}: score_difference incohérent."
        )

    mdt = case["mdt_features"]
    ost = case["ost_features"]

    requested = float(ost["requested_usable_capacity_tib"])
    planned = float(ost["planned_usable_capacity_tib"])
    read_gbps = float(ost["target_read_gbps"])
    write_gbps = float(ost["target_write_gbps"])
    total_gbps = float(ost["total_bandwidth_gbps"])
    file_count = int(mdt["file_count"])
    client_count = int(mdt["client_count"])

    if planned < requested:
        errors.append(
            f"{case_id}: capacité planifiée inférieure à la capacité demandée."
        )

    if not math.isclose(
        total_gbps,
        read_gbps + write_gbps,
        abs_tol=1e-6,
    ):
        errors.append(
            f"{case_id}: total_bandwidth_gbps incorrect."
        )

    expected_bandwidth_per_tib = (
        total_gbps / planned if planned > 0 else 0.0
    )
    if not math.isclose(
        float(ost["bandwidth_per_planned_tib"]),
        expected_bandwidth_per_tib,
        abs_tol=1e-8,
    ):
        errors.append(
            f"{case_id}: bandwidth_per_planned_tib incorrect."
        )

    expected_bandwidth_per_million = (
        total_gbps / (file_count / 1_000_000)
        if file_count > 0
        else 0.0
    )
    if not math.isclose(
        float(ost["bandwidth_per_million_files"]),
        expected_bandwidth_per_million,
        abs_tol=1e-6,
    ):
        errors.append(
            f"{case_id}: bandwidth_per_million_files incorrect."
        )

    expected_files_per_tib = (
        file_count / planned if planned > 0 else 0.0
    )
    if not math.isclose(
        float(mdt["files_per_planned_tib"]),
        expected_files_per_tib,
        abs_tol=1e-5,
    ):
        errors.append(
            f"{case_id}: files_per_planned_tib incorrect."
        )

    expected_clients_per_tib = (
        client_count / planned if planned > 0 else 0.0
    )
    if not math.isclose(
        float(mdt["clients_per_planned_tib"]),
        expected_clients_per_tib,
        abs_tol=1e-8,
    ):
        errors.append(
            f"{case_id}: clients_per_planned_tib incorrect."
        )

    for field in (
        "small_file_factor",
        "file_count_score",
        "client_count_score",
    ):
        value = float(mdt[field])
        if not 0.0 <= value <= 1.0:
            errors.append(
                f"{case_id}: mdt_features.{field} hors [0,1]."
            )

    for field in (
        "large_file_factor",
        "capacity_score",
        "bandwidth_score",
    ):
        value = float(ost[field])
        if not 0.0 <= value <= 1.0:
            errors.append(
                f"{case_id}: ost_features.{field} hors [0,1]."
            )

    ratio = case["read_write_ratio"]
    ratio_sum = (
        float(ratio["read_percent"])
        + float(ratio["write_percent"])
    )
    if not math.isclose(ratio_sum, 100.0, abs_tol=1e-6):
        errors.append(
            f"{case_id}: read_percent + write_percent != 100."
        )

    preferences = case["preferences"]
    priority_fields = (
        "performance_priority",
        "cost_priority",
        "power_priority",
        "reliability_priority",
    )
    priority_sum = sum(
        float(preferences[field])
        for field in priority_fields
    )
    if not math.isclose(priority_sum, 1.0, abs_tol=0.01):
        errors.append(
            f"{case_id}: somme des priorités différente de 1."
        )

    non_finite = find_non_finite_numbers(case)
    for path in non_finite:
        errors.append(
            f"{case_id}: valeur non finie détectée dans {path}."
        )

    forbidden = find_forbidden_keys(case)
    for path in forbidden:
        errors.append(
            f"{case_id}: décision de configuration interdite dans {path}."
        )

    return errors


def validate_dataset(
    input_cases: list[dict[str, Any]],
    output_cases: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Valide structure, recalcul exact et déterminisme."""

    structural_errors: list[str] = []
    recalculation_errors: list[str] = []
    determinism_errors: list[str] = []

    if len(input_cases) != len(output_cases):
        structural_errors.append(
            f"Nombre de cas différent : entrée={len(input_cases)}, "
            f"sortie={len(output_cases)}."
        )

    output_by_id: dict[str, dict[str, Any]] = {}

    for case in output_cases:
        if not isinstance(case, dict):
            structural_errors.append(
                "Une entrée de sortie n'est pas un objet JSON."
            )
            continue

        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            structural_errors.append(
                "Une sortie possède un case_id absent ou invalide."
            )
            continue

        if case_id in output_by_id:
            structural_errors.append(
                f"case_id dupliqué dans la sortie : {case_id}."
            )
            continue

        output_by_id[case_id] = case
        structural_errors.extend(validate_case_invariants(case))

    seen_input_ids: set[str] = set()

    for input_case in input_cases:
        if not isinstance(input_case, dict):
            structural_errors.append(
                "Une entrée du dataset d'analyse n'est pas un objet JSON."
            )
            continue

        case_id = input_case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            structural_errors.append(
                "Une entrée possède un case_id absent ou invalide."
            )
            continue

        if case_id in seen_input_ids:
            structural_errors.append(
                f"case_id dupliqué dans l'entrée : {case_id}."
            )
            continue

        seen_input_ids.add(case_id)

        actual = output_by_id.get(case_id)
        if actual is None:
            structural_errors.append(
                f"{case_id}: sortie manquante."
            )
            continue

        try:
            expected = calculate_features(input_case, config)
        except FeatureCalculationError as exc:
            recalculation_errors.append(
                f"{case_id}: impossible de recalculer les features : {exc}"
            )
            continue

        difference = first_difference(actual, expected)
        if difference is not None:
            recalculation_errors.append(
                f"{case_id}: sortie différente du recalcul — {difference}"
            )

        first = calculate_features(copy.deepcopy(input_case), config)
        second = calculate_features(copy.deepcopy(input_case), config)

        if not values_close(first, second):
            determinism_errors.append(
                f"{case_id}: calcul non déterministe."
            )

    extra_ids = set(output_by_id) - seen_input_ids
    for case_id in sorted(extra_ids):
        structural_errors.append(
            f"{case_id}: sortie sans entrée correspondante."
        )

    return (
        structural_errors,
        recalculation_errors,
        determinism_errors,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valide workload_features_dataset.json contre "
            "le workload analysé et les règles métier."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Dataset produit par workload_analyzer.py.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Fichier architecture_rules.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Dataset produit par feature_calculator.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_json(args.config)
    input_cases = load_json(args.input)
    output_cases = load_json(args.output)

    if not isinstance(config, dict):
        raise DatasetValidationError(
            "architecture_rules.json doit contenir un objet JSON."
        )

    if not isinstance(input_cases, list):
        raise DatasetValidationError(
            "Le dataset d'analyse doit contenir une liste JSON."
        )

    if not isinstance(output_cases, list):
        raise DatasetValidationError(
            "Le dataset de features doit contenir une liste JSON."
        )

    validate_config(config)

    (
        structural_errors,
        recalculation_errors,
        determinism_errors,
    ) = validate_dataset(
        input_cases,
        output_cases,
        config,
    )

    all_errors = (
        structural_errors
        + recalculation_errors
        + determinism_errors
    )

    print("Validation du Feature Calculator")
    print("--------------------------------")
    print(f"Cas analysés            : {len(input_cases)}")
    print(f"Cas de features         : {len(output_cases)}")
    print(f"Erreurs structure       : {len(structural_errors)}")
    print(f"Erreurs de recalcul     : {len(recalculation_errors)}")
    print(f"Erreurs de déterminisme : {len(determinism_errors)}")

    if all_errors:
        print("\nSTATUT : ÉCHEC")

        for error in all_errors[:50]:
            print(f"  - {error}")

        if len(all_errors) > 50:
            print(
                f"  ... {len(all_errors) - 50} "
                "erreur(s) supplémentaire(s)."
            )

        raise SystemExit(1)

    print("\nSTATUT : VALIDÉ")
    print(
        "Les 1 200 sorties sont cohérentes avec les règles, "
        "les entrées et les calculs attendus."
    )


if __name__ == "__main__":
    try:
        main()
    except (
        DatasetValidationError,
        FeatureCalculationError,
        FileNotFoundError,
    ) as exc:
        print(f"\nErreur de validation : {exc}")
        raise SystemExit(1) from exc
