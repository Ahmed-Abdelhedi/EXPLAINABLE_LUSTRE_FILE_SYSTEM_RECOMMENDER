r"""Validation du dataset de candidats MDT.

Entrées par défaut :
- output/lustre_architecture_dataset.json
- data/catalogue_drives_ready_final.json
- output/mdt_drive_candidates_dataset.json

Le validateur contrôle :
- structure et traçabilité ;
- conservation des exigences, contraintes et préférences ;
- recalcul complet de tous les candidats ;
- formules de capacité, IOPS, coût et puissance ;
- compatibilité latence, endurance et fiabilité ;
- classement Top-K ;
- absence de décisions RAID, targets, serveurs ou striping ;
- déterminisme du générateur.

Exécution depuis la racine du projet :

    python .\src\validate_mdt_candidates.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mdt_candidate_generator as generator  # noqa: E402


DEFAULT_ARCHITECTURES = (
    BASE_DIR / "output" / "lustre_architecture_dataset.json"
)
DEFAULT_CATALOG = (
    BASE_DIR / "data" / "catalogue_drives_ready_final.json"
)
DEFAULT_CANDIDATES = (
    BASE_DIR / "output" / "mdt_drive_candidates_dataset.json"
)


FORBIDDEN_DECISION_KEYS = {
    "raid",
    "raid_type",
    "raid_level",
    "selected_raid",
    "stripe_count",
    "stripe_size",
    "mdt_count",
    "target_count",
    "mds_count",
    "mds_server",
    "mds_servers",
    "final_drive_count",
    "selected_drive_count",
}


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close(
    actual: float,
    expected: float,
    *,
    tolerance: float = 1e-6,
) -> bool:
    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def find_non_finite(
    value: Any,
    path: str = "$",
) -> list[str]:
    errors: list[str] = []

    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path}: valeur non finie.")

    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(
                find_non_finite(
                    child,
                    f"{path}.{key}",
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(
                find_non_finite(
                    child,
                    f"{path}[{index}]",
                )
            )

    return errors


def find_forbidden_keys(
    value: Any,
    path: str = "$",
) -> list[str]:
    errors: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()

            if normalized in FORBIDDEN_DECISION_KEYS:
                errors.append(
                    f"{path}.{key}: décision prématurée interdite."
                )

            errors.extend(
                find_forbidden_keys(
                    child,
                    f"{path}.{key}",
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(
                find_forbidden_keys(
                    child,
                    f"{path}[{index}]",
                )
            )

    return errors


def compare_exact(
    actual: Any,
    expected: Any,
    path: str,
    errors: list[str],
) -> None:
    if actual != expected:
        errors.append(
            f"{path}: différence avec la régénération déterministe."
        )


def validate_candidate_formula(
    case_id: str,
    candidate: dict[str, Any],
    drive: dict[str, Any],
    requirement: dict[str, Any],
    constraints: dict[str, Any],
    preferences: dict[str, Any],
    recalculation_errors: list[str],
    business_errors: list[str],
) -> None:
    path = (
        f"{case_id}.candidates["
        f"{candidate.get('rank', '?')}]"
    )

    expected, rejection_reasons = generator.evaluate_drive(
        drive,
        requirement,
        constraints,
        preferences,
    )

    if rejection_reasons:
        business_errors.append(
            f"{path}: candidat retourné alors qu'il devrait être rejeté : "
            f"{rejection_reasons}."
        )

    for key, expected_value in expected.items():
        actual_value = candidate.get(key)

        if actual_value != expected_value:
            recalculation_errors.append(
                f"{path}.{key}: obtenu={actual_value!r}, "
                f"attendu={expected_value!r}."
            )

    required_capacity = float(
        requirement["required_metadata_capacity_tib"]
    )
    required_read = float(requirement["required_read_iops"])
    required_write = float(requirement["required_write_iops"])

    count_capacity = generator.ceil_ratio(
        required_capacity,
        float(drive["capacity_tib"]),
    )
    count_read = generator.ceil_ratio(
        required_read,
        float(drive["random_read_iops_4k"]),
    )
    count_write = generator.ceil_ratio(
        required_write,
        float(drive["random_write_iops_4k"]),
    )

    expected_count = max(
        1,
        count_capacity,
        count_read,
        count_write,
    )

    independent_checks = {
        "count_by_capacity": count_capacity,
        "count_by_read_iops": count_read,
        "count_by_write_iops": count_write,
        "raw_minimum_drive_count": expected_count,
        "raw_provided_capacity_tib": round(
            expected_count * float(drive["capacity_tib"]),
            6,
        ),
        "raw_provided_read_iops": int(
            expected_count
            * float(drive["random_read_iops_4k"])
        ),
        "raw_provided_write_iops": int(
            expected_count
            * float(drive["random_write_iops_4k"])
        ),
        "raw_drive_cost_usd": round(
            expected_count * float(drive["price_en_dollars"]),
            2,
        ),
        "raw_drive_power_w": round(
            expected_count
            * float(drive["power_consumption_en_w"]),
            3,
        ),
    }

    for key, expected_value in independent_checks.items():
        if candidate.get(key) != expected_value:
            recalculation_errors.append(
                f"{path}.{key}: formule indépendante incohérente ; "
                f"obtenu={candidate.get(key)!r}, "
                f"attendu={expected_value!r}."
            )

    if candidate.get("pre_raid_feasible") is not True:
        business_errors.append(
            f"{path}: tout candidat Top-K doit être pré-RAID faisable."
        )

    if not bool(drive["mdt_eligible"]):
        business_errors.append(
            f"{path}: disque non éligible au MDT."
        )

    if not generator.latency_is_compatible(
        str(drive["latency_class"]),
        str(requirement["latency_requirement"]),
    ):
        business_errors.append(
            f"{path}: latence incompatible."
        )

    minimum_dwpd = generator.ENDURANCE_MIN_DWPD[
        str(requirement["endurance_requirement"])
    ]
    if float(drive["endurance_dwpd_numeric"]) < minimum_dwpd:
        business_errors.append(
            f"{path}: endurance insuffisante."
        )

    minimum_mtbf = generator.RELIABILITY_MIN_MTBF_HOURS[
        str(requirement["reliability_requirement"])
    ]
    if float(drive["mtbf_hours"]) < minimum_mtbf:
        business_errors.append(
            f"{path}: fiabilité insuffisante."
        )

    if float(candidate["raw_drive_cost_usd"]) > float(
        constraints["max_budget_usd"]
    ):
        business_errors.append(
            f"{path}: coût brut supérieur au budget global."
        )

    if float(candidate["raw_drive_power_w"]) > float(
        constraints["max_power_w"]
    ):
        business_errors.append(
            f"{path}: puissance brute supérieure à la limite globale."
        )


def validate_case(
    architecture: dict[str, Any],
    output_case: dict[str, Any],
    catalog: list[dict[str, Any]],
    catalog_by_id: dict[str, dict[str, Any]],
    top_k: int,
    structure_errors: list[str],
    recalculation_errors: list[str],
    business_errors: list[str],
) -> None:
    case_id = architecture.get("case_id", "<missing>")

    required_top_level = {
        "case_id",
        "MDT_requirement",
        "constraints",
        "preferences",
        "candidate_summary",
        "candidates",
        "trace",
    }

    missing = required_top_level - set(output_case)
    if missing:
        structure_errors.append(
            f"{case_id}: champs absents={sorted(missing)}."
        )
        return

    if output_case["case_id"] != case_id:
        structure_errors.append(
            f"{case_id}: case_id non conservé."
        )

    compare_exact(
        output_case["MDT_requirement"],
        architecture["MDT_requirement"],
        f"{case_id}.MDT_requirement",
        structure_errors,
    )
    compare_exact(
        output_case["constraints"],
        architecture["constraints"],
        f"{case_id}.constraints",
        structure_errors,
    )
    compare_exact(
        output_case["preferences"],
        architecture["preferences"],
        f"{case_id}.preferences",
        structure_errors,
    )

    candidates = output_case["candidates"]
    summary = output_case["candidate_summary"]
    trace = output_case["trace"]

    if not isinstance(candidates, list):
        structure_errors.append(
            f"{case_id}.candidates: liste requise."
        )
        return

    if len(candidates) > top_k:
        business_errors.append(
            f"{case_id}: plus de {top_k} candidats retournés."
        )

    if summary.get("top_k_requested") != top_k:
        structure_errors.append(
            f"{case_id}.candidate_summary.top_k_requested incohérent."
        )

    if summary.get("top_k_returned") != len(candidates):
        structure_errors.append(
            f"{case_id}: top_k_returned ne correspond pas "
            "au nombre de candidats."
        )

    if summary.get("catalog_drive_count") != len(catalog):
        structure_errors.append(
            f"{case_id}: catalog_drive_count incohérent."
        )

    eligible_count = sum(
        bool(drive["mdt_eligible"])
        for drive in catalog
    )
    if summary.get("mdt_eligible_drive_count") != eligible_count:
        structure_errors.append(
            f"{case_id}: mdt_eligible_drive_count incohérent."
        )

    all_expected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for drive in catalog:
        if not bool(drive["mdt_eligible"]):
            rejection_counts.update(["not_mdt_eligible"])
            continue

        expected_candidate, reasons = generator.evaluate_drive(
            drive,
            architecture["MDT_requirement"],
            architecture["constraints"],
            architecture["preferences"],
        )

        if reasons:
            rejection_counts.update(reasons)
        else:
            all_expected.append(expected_candidate)

    all_expected.sort(key=generator.candidate_sort_key)
    expected_top = all_expected[:top_k]

    for rank, candidate in enumerate(expected_top, start=1):
        candidate["rank"] = rank

    if summary.get("pre_raid_feasible_count") != len(all_expected):
        recalculation_errors.append(
            f"{case_id}: pre_raid_feasible_count incohérent."
        )

    expected_rejected = len(catalog) - len(all_expected)
    if summary.get("rejected_count") != expected_rejected:
        recalculation_errors.append(
            f"{case_id}: rejected_count incohérent."
        )

    expected_rejection_counts = dict(
        sorted(rejection_counts.items())
    )
    if summary.get("rejection_counts") != expected_rejection_counts:
        recalculation_errors.append(
            f"{case_id}: rejection_counts incohérent."
        )

    if candidates != expected_top:
        recalculation_errors.append(
            f"{case_id}: Top-K ou ordre des candidats incorrect."
        )

    ranks = [candidate.get("rank") for candidate in candidates]
    expected_ranks = list(range(1, len(candidates) + 1))
    if ranks != expected_ranks:
        business_errors.append(
            f"{case_id}: rangs incorrects={ranks}."
        )

    drive_ids = [
        candidate.get("drive_id")
        for candidate in candidates
    ]
    if len(drive_ids) != len(set(drive_ids)):
        business_errors.append(
            f"{case_id}: drive_id dupliqué dans le Top-K."
        )

    for candidate in candidates:
        drive_id = candidate.get("drive_id")

        if drive_id not in catalog_by_id:
            structure_errors.append(
                f"{case_id}: drive_id inconnu={drive_id!r}."
            )
            continue

        validate_candidate_formula(
            case_id,
            candidate,
            catalog_by_id[drive_id],
            architecture["MDT_requirement"],
            architecture["constraints"],
            architecture["preferences"],
            recalculation_errors,
            business_errors,
        )

    expected_trace = {
        "mdt_candidate_generator_version":
            generator.GENERATOR_VERSION,
        "candidate_stage":
            "pre_raid_drive_model_ranking",
        "raw_minimum_drive_count_is_lower_bound": True,
        "global_budget_and_power_are_not_allocated_per_role":
            True,
        "raid_not_selected": True,
        "final_drive_count_not_selected": True,
        "ha_preserved_for_beam_search": bool(
            architecture["constraints"]["ha_required"]
        ),
    }

    if trace != expected_trace:
        structure_errors.append(
            f"{case_id}: trace incorrecte."
        )



def safe_generate_case(
    architecture: dict[str, Any],
    catalog: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    case_id = architecture["case_id"]
    requirement = architecture["MDT_requirement"]
    constraints = architecture["constraints"]
    preferences = architecture["preferences"]

    feasible: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for drive in catalog:
        if not bool(drive["mdt_eligible"]):
            rejection_counts.update(["not_mdt_eligible"])
            continue

        candidate, reasons = generator.evaluate_drive(
            drive,
            requirement,
            constraints,
            preferences,
        )

        if reasons:
            rejection_counts.update(reasons)
        else:
            feasible.append(candidate)

    feasible.sort(key=generator.candidate_sort_key)
    selected = feasible[:top_k]

    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank

    return {
        "case_id": case_id,
        "MDT_requirement": requirement,
        "constraints": constraints,
        "preferences": preferences,
        "candidate_summary": {
            "catalog_drive_count": len(catalog),
            "mdt_eligible_drive_count": sum(
                bool(drive["mdt_eligible"])
                for drive in catalog
            ),
            "pre_raid_feasible_count": len(feasible),
            "rejected_count": len(catalog) - len(feasible),
            "top_k_requested": top_k,
            "top_k_returned": len(selected),
            "rejection_counts": dict(
                sorted(rejection_counts.items())
            ),
        },
        "candidates": selected,
        "trace": {
            "mdt_candidate_generator_version":
                generator.GENERATOR_VERSION,
            "candidate_stage":
                "pre_raid_drive_model_ranking",
            "raw_minimum_drive_count_is_lower_bound": True,
            "global_budget_and_power_are_not_allocated_per_role":
                True,
            "raid_not_selected": True,
            "final_drive_count_not_selected": True,
            "ha_preserved_for_beam_search": bool(
                constraints["ha_required"]
            ),
        },
    }


def safe_generate_dataset(
    architectures: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    return [
        safe_generate_case(case, catalog, top_k)
        for case in architectures
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valide le dataset de candidats MDT."
    )
    parser.add_argument(
        "--architectures",
        type=Path,
        default=DEFAULT_ARCHITECTURES,
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    architectures = load_json(args.architectures)
    catalog = load_json(args.catalog)
    outputs = load_json(args.candidates)

    if not isinstance(architectures, list):
        raise ValueError(
            "Le dataset architectural doit être une liste JSON."
        )
    if not isinstance(catalog, list):
        raise ValueError(
            "Le catalogue doit être une liste JSON."
        )
    if not isinstance(outputs, list):
        raise ValueError(
            "Le dataset MDT doit être une liste JSON."
        )

    eligible_catalog = [
        drive for drive in catalog
        if bool(drive.get("mdt_eligible"))
    ]
    generator.validate_catalog(eligible_catalog)

    structure_errors: list[str] = []
    recalculation_errors: list[str] = []
    determinism_errors: list[str] = []
    business_errors: list[str] = []

    if len(architectures) != len(outputs):
        structure_errors.append(
            "Le nombre de cas d'entrée et de sortie diffère : "
            f"{len(architectures)} != {len(outputs)}."
        )

    architecture_ids = [
        case.get("case_id")
        for case in architectures
        if isinstance(case, dict)
    ]
    output_ids = [
        case.get("case_id")
        for case in outputs
        if isinstance(case, dict)
    ]

    if len(architecture_ids) != len(set(architecture_ids)):
        structure_errors.append(
            "case_id dupliqué dans le dataset architectural."
        )

    if len(output_ids) != len(set(output_ids)):
        structure_errors.append(
            "case_id dupliqué dans le dataset MDT."
        )

    output_by_id = {
        case["case_id"]: case
        for case in outputs
        if isinstance(case, dict)
        and isinstance(case.get("case_id"), str)
    }

    catalog_by_id = {
        drive["drive_id"]: drive
        for drive in catalog
    }

    for architecture in architectures:
        case_id = architecture.get("case_id")

        if case_id not in output_by_id:
            structure_errors.append(
                f"{case_id}: sortie absente."
            )
            continue

        validate_case(
            architecture,
            output_by_id[case_id],
            catalog,
            catalog_by_id,
            args.top_k,
            structure_errors,
            recalculation_errors,
            business_errors,
        )

    regenerated_once = safe_generate_dataset(
        copy.deepcopy(architectures),
        copy.deepcopy(catalog),
        args.top_k,
    )
    regenerated_twice = safe_generate_dataset(
        copy.deepcopy(architectures),
        copy.deepcopy(catalog),
        args.top_k,
    )

    if regenerated_once != regenerated_twice:
        determinism_errors.append(
            "Deux exécutions identiques produisent des résultats différents."
        )

    if regenerated_once != outputs:
        recalculation_errors.append(
            "Le dataset complet diffère de la régénération."
        )

    structure_errors.extend(
        find_non_finite(outputs)
    )
    business_errors.extend(
        find_forbidden_keys(
            [
                {
                    "case_id": case.get("case_id"),
                    "candidates": case.get("candidates"),
                }
                for case in outputs
            ]
        )
    )

    cases_without_candidates = sum(
        not case.get("candidates")
        for case in outputs
        if isinstance(case, dict)
    )

    feasible_counts = [
        int(case["candidate_summary"]["pre_raid_feasible_count"])
        for case in outputs
        if isinstance(case, dict)
        and isinstance(case.get("candidate_summary"), dict)
    ]

    print("Validation du MDT Candidate Generator")
    print("-------------------------------------")
    print(f"Cas architecturaux       : {len(architectures)}")
    print(f"Cas MDT générés          : {len(outputs)}")
    print(f"Cas sans candidat        : {cases_without_candidates}")
    print(
        "Candidats faisables min  : "
        f"{min(feasible_counts) if feasible_counts else 0}"
    )
    print(
        "Candidats faisables max  : "
        f"{max(feasible_counts) if feasible_counts else 0}"
    )
    print(f"Erreurs structure        : {len(structure_errors)}")
    print(
        "Erreurs de recalcul      : "
        f"{len(recalculation_errors)}"
    )
    print(
        "Erreurs de déterminisme  : "
        f"{len(determinism_errors)}"
    )
    print(f"Erreurs métier           : {len(business_errors)}")

    all_errors = (
        structure_errors
        + recalculation_errors
        + determinism_errors
        + business_errors
    )

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
        "Les candidats MDT sont cohérents avec les exigences, "
        "le catalogue, les contraintes et les préférences."
    )


if __name__ == "__main__":
    main()
