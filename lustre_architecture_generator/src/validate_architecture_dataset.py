"""Validation finale du Lustre Architecture Generator.

Ce script compare :
- workload_features_dataset.json ;
- architecture_rules.json ;
- lustre_architecture_dataset.json ;
- les résultats recalculés par architecture_generator.py.

Exécution depuis le dossier lustre_architecture_generator :

    python .\src\validate_architecture_dataset.py
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from architecture_generator import (  # noqa: E402
    ArchitectureGenerationError,
    generate_architecture_case,
    load_json,
    validate_config,
)


DEFAULT_INPUT = BASE_DIR / "output" / "workload_features_dataset.json"
DEFAULT_CONFIG = BASE_DIR / "config" / "architecture_rules.json"
DEFAULT_OUTPUT = BASE_DIR / "output" / "lustre_architecture_dataset.json"


class ArchitectureDatasetValidationError(AssertionError):
    """Erreur détectée pendant la validation finale."""


FORBIDDEN_CONFIGURATION_KEYS = {
    "drive",
    "drive_id",
    "drive_name",
    "drive_count",
    "raid",
    "raid_level",
    "protection_group",
    "target_count",
    "mdt_count",
    "ost_count",
    "oss_count",
    "mds_count",
    "stripe_count",
    "stripe_size",
    "stripe_size_mb",
    "server_model",
    "server_count",
}


def values_close(
    actual: Any,
    expected: Any,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Compare récursivement deux structures JSON."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        if set(actual) != set(expected):
            return False

        return all(
            values_close(
                actual[key],
                expected[key],
                tolerance=tolerance,
            )
            for key in expected
        )

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if len(actual) != len(expected):
            return False

        return all(
            values_close(
                actual_item,
                expected_item,
                tolerance=tolerance,
            )
            for actual_item, expected_item in zip(actual, expected)
        )

    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=tolerance,
        )

    return actual == expected


def first_difference(
    actual: Any,
    expected: Any,
    path: str = "$",
) -> str | None:
    """Retourne la première différence entre deux objets JSON."""

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


def find_forbidden_keys(
    value: Any,
    path: str = "$",
) -> list[str]:
    """Cherche récursivement les décisions interdites à cette étape."""

    findings: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"

            if (
                key.lower() in FORBIDDEN_CONFIGURATION_KEYS
                and path != "$.trace"
            ):
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
    """Détecte les valeurs NaN et infinies."""

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


def expected_bandwidth_level(
    total_bandwidth_gbps: float,
    config: dict[str, Any],
) -> str:
    """Classe une bande passante avec les seuils configurés."""

    thresholds = config["bandwidth_thresholds_gbps"]

    if total_bandwidth_gbps < float(thresholds["medium"]):
        return "low"
    if total_bandwidth_gbps < float(thresholds["high"]):
        return "medium"
    if total_bandwidth_gbps < float(thresholds["very_high"]):
        return "high"
    return "very_high"


def validate_output_case(
    case: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    """Vérifie les invariants métier d'une architecture."""

    errors: list[str] = []
    case_id = str(case.get("case_id", "<case_id absent>"))

    required_sections = {
        "case_id",
        "source_requirement",
        "workload_analysis",
        "role_analysis",
        "MDT_requirement",
        "OST_requirement",
        "constraints",
        "preferences",
        "trace",
    }

    missing = required_sections - set(case)
    if missing:
        errors.append(
            f"{case_id}: sections manquantes={sorted(missing)}."
        )
        return errors

    source = case["source_requirement"]
    capacity = source["capacity_planning"]
    characteristics = source["data_characteristics"]
    io_profile = source["io_profile"]
    workload = case["workload_analysis"]
    role = case["role_analysis"]
    mdt = case["MDT_requirement"]
    ost = case["OST_requirement"]
    trace = case["trace"]

    if trace.get("architecture_generator_version") != "1.1":
        errors.append(
            f"{case_id}: architecture_generator_version doit être 1.1."
        )

    required_source_sections = {
        "capacity_planning",
        "data_characteristics",
        "io_profile",
        "constraints",
        "preferences",
    }
    missing_source = required_source_sections - set(source)
    if missing_source:
        errors.append(
            f"{case_id}: source_requirement incomplet : "
            f"{sorted(missing_source)}."
        )

    requested = float(capacity["requested_usable_capacity_tib"])
    growth_percent = float(capacity["annual_growth_percent"])
    growth_factor = float(capacity["growth_factor"])
    fill_ratio = float(capacity["target_fill_ratio"])
    planned = float(capacity["planned_usable_capacity_tib"])

    expected_growth = 1.0 + growth_percent / 100.0
    expected_planned = requested * expected_growth / fill_ratio

    if not math.isclose(
        growth_factor,
        expected_growth,
        abs_tol=1e-9,
    ):
        errors.append(f"{case_id}: growth_factor incorrect.")

    if not math.isclose(
        planned,
        expected_planned,
        abs_tol=1e-6,
    ):
        errors.append(
            f"{case_id}: planned_usable_capacity_tib incorrect."
        )

    metadata_score = float(workload["metadata_score"])
    data_score = float(workload["data_score"])
    score_difference = float(workload["score_difference"])

    if not math.isclose(
        score_difference,
        metadata_score - data_score,
        abs_tol=2e-6,
    ):
        errors.append(f"{case_id}: score_difference incorrect.")

    ratio = io_profile["read_write_ratio"]
    read_percent = float(ratio["read_percent"])
    write_percent = float(ratio["write_percent"])

    if not math.isclose(
        read_percent + write_percent,
        100.0,
        abs_tol=1e-6,
    ):
        errors.append(
            f"{case_id}: read_percent + write_percent != 100."
        )

    required_total_iops = int(mdt["required_total_iops"])
    required_read_iops = int(mdt["required_read_iops"])
    required_write_iops = int(mdt["required_write_iops"])

    if required_read_iops + required_write_iops != required_total_iops:
        errors.append(
            f"{case_id}: somme des IOPS MDT incorrecte."
        )

    mdt_trace = trace["mdt_estimation"]
    expected_total_iops = math.ceil(
        float(mdt_trace["base_iops_per_client"])
        * int(mdt_trace["client_count"])
        * float(mdt_trace["file_size_multiplier"])
        * float(mdt_trace["access_multiplier"])
        * float(mdt_trace["metadata_pressure_multiplier"])
        * float(mdt_trace["iops_safety_factor"])
    )

    if required_total_iops != expected_total_iops:
        errors.append(
            f"{case_id}: formule required_total_iops incorrecte."
        )

    expected_read_iops = round(
        required_total_iops * read_percent / 100.0
    )
    if required_read_iops != expected_read_iops:
        errors.append(
            f"{case_id}: répartition read/write des IOPS incorrecte."
        )

    expected_metadata_capacity = (
        int(characteristics["total_file_count"])
        * float(mdt_trace["metadata_bytes_per_file"])
        * float(mdt_trace["metadata_capacity_safety_factor"])
        / (1024.0 ** 4)
    )

    if not math.isclose(
        float(mdt["required_metadata_capacity_tib"]),
        round(expected_metadata_capacity, 9),
        abs_tol=1e-12,
    ):
        errors.append(
            f"{case_id}: capacité metadata MDT incorrecte."
        )

    ost_trace = trace["ost_estimation"]
    bandwidth_factor = float(
        ost_trace["bandwidth_safety_factor"]
    )
    capacity_factor = float(
        ost_trace["capacity_safety_factor"]
    )

    expected_read_bandwidth = (
        float(io_profile["target_read_gbps"])
        * bandwidth_factor
    )
    expected_write_bandwidth = (
        float(io_profile["target_write_gbps"])
        * bandwidth_factor
    )
    expected_total_bandwidth = (
        expected_read_bandwidth
        + expected_write_bandwidth
    )
    expected_usable_capacity = planned * capacity_factor

    if not math.isclose(
        float(ost["required_read_bandwidth_gbps"]),
        round(expected_read_bandwidth, 6),
        abs_tol=1e-9,
    ):
        errors.append(
            f"{case_id}: bande passante OST en lecture incorrecte."
        )

    if not math.isclose(
        float(ost["required_write_bandwidth_gbps"]),
        round(expected_write_bandwidth, 6),
        abs_tol=1e-9,
    ):
        errors.append(
            f"{case_id}: bande passante OST en écriture incorrecte."
        )

    if not math.isclose(
        float(ost["required_total_bandwidth_gbps"]),
        round(expected_total_bandwidth, 6),
        abs_tol=1e-9,
    ):
        errors.append(
            f"{case_id}: bande passante OST totale incorrecte."
        )

    if not math.isclose(
        float(ost["required_usable_capacity_tib"]),
        round(expected_usable_capacity, 6),
        abs_tol=1e-9,
    ):
        errors.append(
            f"{case_id}: capacité OST requise incorrecte."
        )

    planned_level = expected_bandwidth_level(
        float(ost["required_total_bandwidth_gbps"]),
        config,
    )
    if ost["throughput_requirement"] != planned_level:
        errors.append(
            f"{case_id}: throughput_requirement="
            f"{ost['throughput_requirement']}, attendu={planned_level}."
        )

    input_total_bandwidth = (
        float(io_profile["target_read_gbps"])
        + float(io_profile["target_write_gbps"])
    )
    input_level = expected_bandwidth_level(
        input_total_bandwidth,
        config,
    )
    if ost["input_bandwidth_level"] != input_level:
        errors.append(
            f"{case_id}: input_bandwidth_level incorrect."
        )

    absolute_importance = role["absolute_importance"]

    if mdt["priority"] != absolute_importance["mdt_importance"]:
        errors.append(f"{case_id}: priorité MDT incorrecte.")

    if ost["priority"] != absolute_importance["ost_importance"]:
        errors.append(f"{case_id}: priorité OST incorrecte.")

    if mdt.get("priority_basis") != "normalized_metadata_intensity":
        errors.append(f"{case_id}: priority_basis MDT incorrect.")

    if ost.get("priority_basis") != "normalized_data_intensity":
        errors.append(f"{case_id}: priority_basis OST incorrect.")

    if case["constraints"] != source["constraints"]:
        errors.append(
            f"{case_id}: contraintes source/finale différentes."
        )

    if case["preferences"] != source["preferences"]:
        errors.append(
            f"{case_id}: préférences source/finale différentes."
        )

    ha_required = bool(case["constraints"]["ha_required"])

    if (
        bool(mdt["ha_required"]) != ha_required
        or bool(ost["ha_required"]) != ha_required
    ):
        errors.append(f"{case_id}: propagation HA incorrecte.")

    expected_mdt_reliability = (
        str(config["reliability_rules"]["ha_required_level"])
        if ha_required
        else (
            "high"
            if mdt["priority"] in {"high", "critical"}
            else str(config["reliability_rules"]["default_level"])
        )
    )
    expected_ost_reliability = (
        str(config["reliability_rules"]["ha_required_level"])
        if ha_required
        else (
            "high"
            if ost["priority"] in {"high", "critical"}
            else str(config["reliability_rules"]["default_level"])
        )
    )

    if mdt["reliability_requirement"] != expected_mdt_reliability:
        errors.append(
            f"{case_id}: fiabilité MDT incorrecte."
        )

    if ost["reliability_requirement"] != expected_ost_reliability:
        errors.append(
            f"{case_id}: fiabilité OST incorrecte."
        )

    forbidden = find_forbidden_keys(case)
    for path in forbidden:
        errors.append(
            f"{case_id}: décision interdite trouvée dans {path}."
        )

    non_finite = find_non_finite_numbers(case)
    for path in non_finite:
        errors.append(
            f"{case_id}: valeur non finie trouvée dans {path}."
        )

    return errors


def validate_dataset(
    input_cases: list[dict[str, Any]],
    output_cases: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Valide structure, recalcul, déterminisme et règles métier."""

    structure_errors: list[str] = []
    recalculation_errors: list[str] = []
    determinism_errors: list[str] = []
    business_errors: list[str] = []

    if len(input_cases) != len(output_cases):
        structure_errors.append(
            f"Nombre de cas différent : entrée={len(input_cases)}, "
            f"sortie={len(output_cases)}."
        )

    input_by_id: dict[str, dict[str, Any]] = {}
    output_by_id: dict[str, dict[str, Any]] = {}

    for case in input_cases:
        if not isinstance(case, dict):
            structure_errors.append(
                "Une entrée de features n'est pas un objet JSON."
            )
            continue

        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            structure_errors.append(
                "Une entrée possède un case_id invalide."
            )
            continue

        if case_id in input_by_id:
            structure_errors.append(
                f"case_id dupliqué dans les features : {case_id}."
            )
            continue

        input_by_id[case_id] = case

    for case in output_cases:
        if not isinstance(case, dict):
            structure_errors.append(
                "Une architecture n'est pas un objet JSON."
            )
            continue

        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            structure_errors.append(
                "Une architecture possède un case_id invalide."
            )
            continue

        if case_id in output_by_id:
            structure_errors.append(
                f"case_id dupliqué dans les architectures : {case_id}."
            )
            continue

        output_by_id[case_id] = case
        business_errors.extend(validate_output_case(case, config))

    missing_outputs = set(input_by_id) - set(output_by_id)
    extra_outputs = set(output_by_id) - set(input_by_id)

    for case_id in sorted(missing_outputs):
        structure_errors.append(
            f"{case_id}: architecture manquante."
        )

    for case_id in sorted(extra_outputs):
        structure_errors.append(
            f"{case_id}: architecture sans features correspondantes."
        )

    for case_id, input_case in input_by_id.items():
        actual = output_by_id.get(case_id)
        if actual is None:
            continue

        try:
            expected = generate_architecture_case(
                copy.deepcopy(input_case),
                config,
            )
        except ArchitectureGenerationError as exc:
            recalculation_errors.append(
                f"{case_id}: recalcul impossible : {exc}"
            )
            continue

        difference = first_difference(actual, expected)
        if difference is not None:
            recalculation_errors.append(
                f"{case_id}: sortie différente du recalcul — {difference}"
            )

        first = generate_architecture_case(
            copy.deepcopy(input_case),
            config,
        )
        second = generate_architecture_case(
            copy.deepcopy(input_case),
            config,
        )

        if not values_close(first, second):
            determinism_errors.append(
                f"{case_id}: génération non déterministe."
            )

    return (
        structure_errors,
        recalculation_errors,
        determinism_errors,
        business_errors,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valide lustre_architecture_dataset.json contre les "
            "features, la configuration et le générateur."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Dataset produit par feature_calculator.py.",
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
        help="Dataset produit par architecture_generator.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_json(args.config)
    input_cases = load_json(args.input)
    output_cases = load_json(args.output)

    if not isinstance(config, dict):
        raise ArchitectureDatasetValidationError(
            "architecture_rules.json doit contenir un objet JSON."
        )

    if not isinstance(input_cases, list):
        raise ArchitectureDatasetValidationError(
            "Le dataset de features doit contenir une liste JSON."
        )

    if not isinstance(output_cases, list):
        raise ArchitectureDatasetValidationError(
            "Le dataset architectural doit contenir une liste JSON."
        )

    validate_config(config)

    (
        structure_errors,
        recalculation_errors,
        determinism_errors,
        business_errors,
    ) = validate_dataset(
        input_cases,
        output_cases,
        config,
    )

    all_errors = (
        structure_errors
        + recalculation_errors
        + determinism_errors
        + business_errors
    )

    changed_throughput_count = sum(
        1
        for case in output_cases
        if (
            case["OST_requirement"]["input_bandwidth_level"]
            != case["OST_requirement"]["throughput_requirement"]
        )
    )

    print("Validation du Lustre Architecture Generator")
    print("-------------------------------------------")
    print(f"Cas de features          : {len(input_cases)}")
    print(f"Architectures générées   : {len(output_cases)}")
    print(f"Erreurs structure        : {len(structure_errors)}")
    print(f"Erreurs de recalcul      : {len(recalculation_errors)}")
    print(f"Erreurs de déterminisme  : {len(determinism_errors)}")
    print(f"Erreurs métier           : {len(business_errors)}")
    print(
        "Classes modifiées après marge OST : "
        f"{changed_throughput_count}"
    )

    if all_errors:
        print("\nSTATUT : ÉCHEC")

        for error in all_errors[:60]:
            print(f"  - {error}")

        if len(all_errors) > 60:
            print(
                f"  ... {len(all_errors) - 60} "
                "erreur(s) supplémentaire(s)."
            )

        raise SystemExit(1)

    print("\nSTATUT : VALIDÉ")
    print(
        "Les architectures MDT/OST sont cohérentes avec les "
        "features, les règles et les formules déterministes."
    )


if __name__ == "__main__":
    try:
        main()
    except (
        ArchitectureDatasetValidationError,
        ArchitectureGenerationError,
        FileNotFoundError,
    ) as exc:
        print(f"\nErreur de validation : {exc}")
        raise SystemExit(1) from exc
