from __future__ import annotations

import math
from typing import Any

from .architecture_state import build_complete_state_from_choices
from .full_architecture_generator import (
    architecture_id,
    enumerate_role_options,
)
from .full_architecture_validator import validate_complete_architecture


COVERAGE_SCHEMA_VERSION = "1.0"
COVERAGE_POLICY_ID = "H10B_FEASIBILITY_COVERAGE_V1"


class FeasibilityCoverageError(RuntimeError):
    """Erreur de l'analyse H10-B de couverture de faisabilité."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FeasibilityCoverageError(f"{field}: objet JSON requis.")
    return value


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise FeasibilityCoverageError(f"{field}: liste JSON requise.")
    return value


def _finite(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise FeasibilityCoverageError(f"{field}: nombre requis.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise FeasibilityCoverageError(f"{field}: nombre requis.") from error

    if not math.isfinite(number):
        raise FeasibilityCoverageError(f"{field}: nombre fini requis.")

    if minimum is not None and number < minimum:
        raise FeasibilityCoverageError(
            f"{field}: valeur >= {minimum} requise."
        )

    return number


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise FeasibilityCoverageError(f"{field}: entier > 0 requis.")

    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise FeasibilityCoverageError(
            f"{field}: entier > 0 requis."
        ) from error

    if number <= 0:
        raise FeasibilityCoverageError(f"{field}: entier > 0 requis.")

    return number


def role_option_cost_power(
    option: dict[str, Any],
) -> dict[str, float]:
    """
    Calcule la contribution additive coût/puissance d'une option MDT ou OST.

    H5 porte le coût/puissance des drives protégés et H6 porte la borne
    composants hardware. H7/H10 additionnent exactement ces deux termes par
    rôle.
    """

    protection = _mapping(
        option.get("protection"),
        field="option.protection",
    )
    path = _mapping(
        option.get("hardware_path"),
        field="option.hardware_path",
    )

    drive_cost = _finite(
        protection.get("protected_drive_cost_usd"),
        field="protection.protected_drive_cost_usd",
        minimum=0.0,
    )
    hardware_cost = _finite(
        path.get("component_cost_lower_bound_usd"),
        field="path.component_cost_lower_bound_usd",
        minimum=0.0,
    )
    drive_power = _finite(
        protection.get("protected_drive_power_w"),
        field="protection.protected_drive_power_w",
        minimum=0.0,
    )
    hardware_power = _finite(
        path.get("component_power_lower_bound_w"),
        field="path.component_power_lower_bound_w",
        minimum=0.0,
    )

    return {
        "cost_usd": drive_cost + hardware_cost,
        "power_w": drive_power + hardware_power,
    }

def classify_no_feasible_pair(
    *,
    minimum_total_cost_usd: float,
    maximum_budget_usd: float,
    minimum_total_power_w: float,
    maximum_power_w: float,
    joint_pair_exists: bool,
) -> str:
    budget_impossible = minimum_total_cost_usd > maximum_budget_usd + 1e-12
    power_impossible = minimum_total_power_w > maximum_power_w + 1e-12

    if joint_pair_exists:
        return "FEASIBLE_PAIR_EXISTS"

    if budget_impossible and power_impossible:
        return "BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED"

    if budget_impossible:
        return "BUDGET_LOWER_BOUND_EXCEEDS"

    if power_impossible:
        return "POWER_LOWER_BOUND_EXCEEDS"

    return "JOINT_BUDGET_POWER_CONFLICT"


def find_first_cost_power_feasible_pair(
    *,
    mdt_options: list[dict[str, Any]],
    ost_options: list[dict[str, Any]],
    maximum_budget_usd: float,
    maximum_power_w: float,
) -> dict[str, Any]:
    """
    Cherche la première paire MDT×OST respectant coût ET puissance.

    L'ordre est celui des options H8. Aucun score H9 et aucun Beam Search ne
    sont utilisés.
    """

    if not mdt_options:
        raise FeasibilityCoverageError("Aucune option MDT.")
    if not ost_options:
        raise FeasibilityCoverageError("Aucune option OST.")

    max_budget = _finite(
        maximum_budget_usd,
        field="maximum_budget_usd",
        minimum=0.0,
    )
    max_power = _finite(
        maximum_power_w,
        field="maximum_power_w",
        minimum=0.0,
    )

    mdt_resources = [
        role_option_cost_power(option)
        for option in mdt_options
    ]
    ost_resources = [
        role_option_cost_power(option)
        for option in ost_options
    ]

    minimum_total_cost = (
        min(item["cost_usd"] for item in mdt_resources)
        + min(item["cost_usd"] for item in ost_resources)
    )
    minimum_total_power = (
        min(item["power_w"] for item in mdt_resources)
        + min(item["power_w"] for item in ost_resources)
    )

    pairs_examined = 0

    for mdt_index, (mdt_option, mdt_resource) in enumerate(
        zip(mdt_options, mdt_resources, strict=True),
        start=1,
    ):
        for ost_index, (ost_option, ost_resource) in enumerate(
            zip(ost_options, ost_resources, strict=True),
            start=1,
        ):
            pairs_examined += 1
            total_cost = (
                mdt_resource["cost_usd"]
                + ost_resource["cost_usd"]
            )
            total_power = (
                mdt_resource["power_w"]
                + ost_resource["power_w"]
            )

            if (
                total_cost <= max_budget + 1e-12
                and total_power <= max_power + 1e-12
            ):
                return {
                    "found": True,
                    "mdt_option": mdt_option,
                    "ost_option": ost_option,
                    "mdt_option_index": mdt_index,
                    "ost_option_index": ost_index,
                    "total_cost_usd": total_cost,
                    "total_power_w": total_power,
                    "pairs_examined": pairs_examined,
                    "potential_pair_count": (
                        len(mdt_options) * len(ost_options)
                    ),
                    "minimum_total_cost_usd": minimum_total_cost,
                    "minimum_total_power_w": minimum_total_power,
                    "classification": "FEASIBLE_PAIR_EXISTS",
                }

    classification = classify_no_feasible_pair(
        minimum_total_cost_usd=minimum_total_cost,
        maximum_budget_usd=max_budget,
        minimum_total_power_w=minimum_total_power,
        maximum_power_w=max_power,
        joint_pair_exists=False,
    )

    return {
        "found": False,
        "mdt_option": None,
        "ost_option": None,
        "mdt_option_index": None,
        "ost_option_index": None,
        "total_cost_usd": None,
        "total_power_w": None,
        "pairs_examined": pairs_examined,
        "potential_pair_count": len(mdt_options) * len(ost_options),
        "minimum_total_cost_usd": minimum_total_cost,
        "minimum_total_power_w": minimum_total_power,
        "classification": classification,
    }


def unresolved_case_ids_from_h10(
    baseline_result: dict[str, Any],
) -> list[str]:
    cases = _list(
        baseline_result.get("cases"),
        field="baseline_result.cases",
    )

    result: list[str] = []

    for index, row in enumerate(cases):
        row_map = _mapping(
            row,
            field=f"baseline_result.cases[{index}]",
        )

        if row_map.get("status") != "OK":
            continue

        if row_map.get("has_valid_architecture") is False:
            case_id = row_map.get("case_id")

            if not isinstance(case_id, str) or not case_id.strip():
                raise FeasibilityCoverageError(
                    f"baseline_result.cases[{index}].case_id invalide."
                )

            result.append(case_id.strip())

    return result


def analyze_case_feasibility_domain(
    *,
    handoff: dict[str, Any],
    hardware_catalog: dict[str, Any],
    max_paths_per_variant: int = 2,
) -> dict[str, Any]:
    """
    Étend H8 en supprimant le cap `max_role_options`, tout en conservant
    `top_k` du handoff et un cap explicite de paths H6.

    Si une paire satisfait les bornes additives budget/power, H10 valide
    ensuite l'architecture exacte pour confirmer la récupération.
    """

    path_limit = _positive_int(
        max_paths_per_variant,
        field="max_paths_per_variant",
    )

    requirements = _mapping(
        handoff.get("requirements"),
        field="handoff.requirements",
    )
    constraints = _mapping(
        requirements.get("constraints"),
        field="handoff.requirements.constraints",
    )

    maximum_budget = _finite(
        constraints.get("max_budget_usd"),
        field="constraints.max_budget_usd",
        minimum=0.0,
    )
    maximum_power = _finite(
        constraints.get("max_power_w"),
        field="constraints.max_power_w",
        minimum=0.0,
    )

    mdt_options = enumerate_role_options(
        handoff=handoff,
        hardware_catalog=hardware_catalog,
        role="MDT",
        max_paths_per_variant=path_limit,
        max_role_options=None,
    )
    ost_options = enumerate_role_options(
        handoff=handoff,
        hardware_catalog=hardware_catalog,
        role="OST",
        max_paths_per_variant=path_limit,
        max_role_options=None,
    )

    pair = find_first_cost_power_feasible_pair(
        mdt_options=mdt_options,
        ost_options=ost_options,
        maximum_budget_usd=maximum_budget,
        maximum_power_w=maximum_power,
    )

    recovered_validation = None

    if pair["found"]:
        mdt_option = pair["mdt_option"]
        ost_option = pair["ost_option"]

        state = build_complete_state_from_choices(
            handoff=handoff,
            mdt_candidate=mdt_option["candidate"],
            ost_candidate=ost_option["candidate"],
            mdt_protection=mdt_option["protection"],
            ost_protection=ost_option["protection"],
            mdt_path=mdt_option["hardware_path"],
            ost_path=ost_option["hardware_path"],
        )

        arch_id = architecture_id(state)
        architecture = {
            "schema_version": "1.0",
            "architecture_id": arch_id,
            "case_id": handoff["case_id"],
            "generation_index": 1,
            "mdt_option_index": pair["mdt_option_index"],
            "ost_option_index": pair["ost_option_index"],
            "generation_semantics": {
                "beam_search_applied": False,
                "architecture_score_applied": False,
                "hard_compatibility_precedes_generation": True,
                "cartesian_pairing": True,
                "role_instances_are_isolated": True,
                "h10b_recovered_candidate": True,
            },
            "state": state,
        }

        recovered_validation = validate_complete_architecture(
            architecture=architecture,
            handoff=handoff,
            hardware_catalog=hardware_catalog,
        )

        if recovered_validation.get("valid") is not True:
            raise FeasibilityCoverageError(
                "Une paire passant les bornes additives coût/puissance a "
                "échoué H10. Cela indique une incohérence inter-couches à "
                "corriger avant le freeze."
            )

    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "policy_id": COVERAGE_POLICY_ID,
        "case_id": handoff.get("case_id"),
        "domain": {
            "top_k_is_fixed_by_handoff": True,
            "max_paths_per_variant": path_limit,
            "max_role_options_removed": True,
            "beam_search_applied": False,
            "architecture_scoring_required": False,
        },
        "option_counts": {
            "mdt": len(mdt_options),
            "ost": len(ost_options),
            "potential_pairs": len(mdt_options) * len(ost_options),
        },
        "limits": {
            "maximum_budget_usd": maximum_budget,
            "maximum_power_w": maximum_power,
        },
        "search": {
            key: value
            for key, value in pair.items()
            if key not in {"mdt_option", "ost_option"}
        },
        "recovered_valid_architecture": bool(
            recovered_validation
            and recovered_validation.get("valid") is True
        ),
        "recovered_architecture_id": (
            recovered_validation.get("architecture_id")
            if recovered_validation
            else None
        ),
        "h10_decision": (
            recovered_validation.get("decision")
            if recovered_validation
            else None
        ),
        "h10_violation_codes": (
            [
                item["code"]
                for item in recovered_validation.get("violations", [])
            ]
            if recovered_validation
            else []
        ),
        "coverage_interpretation": (
            "RECOVERED_BY_REMOVING_ROLE_OPTION_CAP"
            if recovered_validation
            else "UNRESOLVED_WITH_FULL_ROLE_DOMAIN_AT_CURRENT_PATH_LIMIT"
        ),
    }

FULL_PATH_COVERAGE_SCHEMA_VERSION = "1.0"
FULL_PATH_COVERAGE_POLICY_ID = "H10C_FULL_HARDWARE_PATH_COVERAGE_V1"


def hardware_path_upper_bound(
    hardware_catalog: dict[str, Any],
) -> int:
    """
    Calcule une borne supérieure sûre du nombre de chemins que H6 peut
    énumérer pour un couple candidate/protection.

    H6 parcourt :
        server × controller × network × HA × (DIRECT + enclosures)

    Certaines combinaisons seront éliminées avant `evaluate_hardware_path`,
    mais cette borne garantit que `find_compatible_hardware_paths` ne peut
    plus être tronqué par `max_paths`.
    """

    catalog = _mapping(
        hardware_catalog,
        field="hardware_catalog",
    )

    counts: dict[str, int] = {}

    for key in (
        "servers",
        "controllers",
        "networks",
        "ha_profiles",
        "enclosures",
    ):
        values = _list(
            catalog.get(key),
            field=f"hardware_catalog.{key}",
        )

        if not values:
            raise FeasibilityCoverageError(
                f"hardware_catalog.{key}: liste non vide requise."
            )

        counts[key] = len(values)

    return (
        counts["servers"]
        * counts["controllers"]
        * counts["networks"]
        * counts["ha_profiles"]
        * (counts["enclosures"] + 1)
    )


def pareto_frontier_options(
    options: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Conserve uniquement les options non dominées en coût et puissance.

    Une option A domine B si :
      cost(A) <= cost(B)
      power(A) <= power(B)
    avec au moins une inégalité stricte.

    Supprimer une option dominée ne peut pas supprimer une solution
    coût+puissance faisable : l'option dominante peut toujours la remplacer.

    Le tri est déterministe : coût, puissance, index d'origine.
    """

    if not options:
        raise FeasibilityCoverageError(
            "Aucune option pour la frontière de Pareto."
        )

    decorated: list[
        tuple[float, float, int, dict[str, Any]]
    ] = []

    for index, option in enumerate(
        options,
        start=1,
    ):
        resources = role_option_cost_power(
            option
        )

        decorated.append(
            (
                resources["cost_usd"],
                resources["power_w"],
                index,
                option,
            )
        )

    decorated.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )

    frontier: list[dict[str, Any]] = []
    best_power = math.inf

    for _, power, _, option in decorated:
        if power < best_power - 1e-12:
            frontier.append(option)
            best_power = power

    return frontier


def unresolved_case_ids_from_h10b(
    coverage_result: dict[str, Any],
) -> list[str]:
    """
    Retourne uniquement les cas qui n'ont pas été récupérés par H10-B.
    """

    rows = _list(
        coverage_result.get("cases"),
        field="coverage_result.cases",
    )

    result: list[str] = []

    for index, row in enumerate(rows):
        row_map = _mapping(
            row,
            field=f"coverage_result.cases[{index}]",
        )

        if row_map.get("status") != "OK":
            continue

        if row_map.get(
            "recovered_valid_architecture"
        ) is False:
            case_id = row_map.get("case_id")

            if (
                not isinstance(case_id, str)
                or not case_id.strip()
            ):
                raise FeasibilityCoverageError(
                    f"coverage_result.cases[{index}].case_id invalide."
                )

            result.append(case_id.strip())

    return result


def _confirm_pair_with_h10(
    *,
    handoff: dict[str, Any],
    hardware_catalog: dict[str, Any],
    mdt_option: dict[str, Any],
    ost_option: dict[str, Any],
    mdt_option_index: int,
    ost_option_index: int,
    marker: str,
) -> dict[str, Any]:
    state = build_complete_state_from_choices(
        handoff=handoff,
        mdt_candidate=mdt_option["candidate"],
        ost_candidate=ost_option["candidate"],
        mdt_protection=mdt_option["protection"],
        ost_protection=ost_option["protection"],
        mdt_path=mdt_option["hardware_path"],
        ost_path=ost_option["hardware_path"],
    )

    arch_id = architecture_id(state)

    architecture = {
        "schema_version": "1.0",
        "architecture_id": arch_id,
        "case_id": handoff["case_id"],
        "generation_index": 1,
        "mdt_option_index": mdt_option_index,
        "ost_option_index": ost_option_index,
        "generation_semantics": {
            "beam_search_applied": False,
            "architecture_score_applied": False,
            "hard_compatibility_precedes_generation": True,
            "cartesian_pairing": True,
            "role_instances_are_isolated": True,
            marker: True,
        },
        "state": state,
    }

    validation = validate_complete_architecture(
        architecture=architecture,
        handoff=handoff,
        hardware_catalog=hardware_catalog,
    )

    if validation.get("valid") is not True:
        codes = [
            item.get("code")
            for item in validation.get(
                "violations",
                [],
            )
            if isinstance(item, dict)
        ]

        raise FeasibilityCoverageError(
            "Une paire coût/puissance faisable a échoué H10 "
            f"(violations={codes}). Corriger l'incohérence avant freeze."
        )

    return validation


def analyze_case_full_path_domain(
    *,
    handoff: dict[str, Any],
    hardware_catalog: dict[str, Any],
) -> dict[str, Any]:
    """
    H10-C : supprime aussi le cap H6 `max_paths_per_variant`.

    Le Top-K du handoff reste inchangé. Le catalogue de référence reste
    inchangé. `max_role_options` est déjà supprimé comme en H10-B.

    Le domaine hardware path est exhaustif pour le catalogue courant car
    `hardware_path_upper_bound()` est supérieur ou égal au nombre maximal de
    chemins que H6 peut générer pour une variante.
    """

    requirements = _mapping(
        handoff.get("requirements"),
        field="handoff.requirements",
    )
    constraints = _mapping(
        requirements.get("constraints"),
        field="handoff.requirements.constraints",
    )

    maximum_budget = _finite(
        constraints.get("max_budget_usd"),
        field="constraints.max_budget_usd",
        minimum=0.0,
    )
    maximum_power = _finite(
        constraints.get("max_power_w"),
        field="constraints.max_power_w",
        minimum=0.0,
    )

    path_cap = hardware_path_upper_bound(
        hardware_catalog
    )

    mdt_options = enumerate_role_options(
        handoff=handoff,
        hardware_catalog=hardware_catalog,
        role="MDT",
        max_paths_per_variant=path_cap,
        max_role_options=None,
    )
    ost_options = enumerate_role_options(
        handoff=handoff,
        hardware_catalog=hardware_catalog,
        role="OST",
        max_paths_per_variant=path_cap,
        max_role_options=None,
    )

    if not mdt_options:
        raise FeasibilityCoverageError(
            "H10-C: aucune option MDT dans le domaine complet."
        )
    if not ost_options:
        raise FeasibilityCoverageError(
            "H10-C: aucune option OST dans le domaine complet."
        )

    mdt_frontier = pareto_frontier_options(
        mdt_options
    )
    ost_frontier = pareto_frontier_options(
        ost_options
    )

    pair = find_first_cost_power_feasible_pair(
        mdt_options=mdt_frontier,
        ost_options=ost_frontier,
        maximum_budget_usd=maximum_budget,
        maximum_power_w=maximum_power,
    )

    validation = None

    if pair["found"]:
        validation = _confirm_pair_with_h10(
            handoff=handoff,
            hardware_catalog=hardware_catalog,
            mdt_option=pair["mdt_option"],
            ost_option=pair["ost_option"],
            mdt_option_index=pair[
                "mdt_option_index"
            ],
            ost_option_index=pair[
                "ost_option_index"
            ],
            marker="h10c_full_path_recovered_candidate",
        )

    def path_index(
        option: dict[str, Any] | None,
    ) -> int | None:
        if not isinstance(option, dict):
            return None

        provenance = option.get("provenance")

        if not isinstance(provenance, dict):
            return None

        value = provenance.get(
            "hardware_path_index"
        )

        if isinstance(value, bool):
            return None

        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None

        return parsed if parsed > 0 else None

    return {
        "schema_version": (
            FULL_PATH_COVERAGE_SCHEMA_VERSION
        ),
        "policy_id": (
            FULL_PATH_COVERAGE_POLICY_ID
        ),
        "case_id": handoff.get("case_id"),
        "domain": {
            "top_k_is_fixed_by_handoff": True,
            "max_role_options_removed": True,
            "hardware_path_domain_exhaustive_for_current_catalog": True,
            "hardware_path_upper_bound": path_cap,
            "beam_search_applied": False,
            "architecture_scoring_required": False,
        },
        "option_counts": {
            "mdt_raw": len(mdt_options),
            "ost_raw": len(ost_options),
            "mdt_pareto": len(mdt_frontier),
            "ost_pareto": len(ost_frontier),
            "raw_potential_pairs": (
                len(mdt_options)
                * len(ost_options)
            ),
            "pareto_potential_pairs": (
                len(mdt_frontier)
                * len(ost_frontier)
            ),
        },
        "limits": {
            "maximum_budget_usd": maximum_budget,
            "maximum_power_w": maximum_power,
        },
        "search": {
            key: value
            for key, value in pair.items()
            if key not in {
                "mdt_option",
                "ost_option",
            }
        },
        "selected_path_indexes": {
            "mdt_hardware_path_index": (
                path_index(
                    pair.get(
                        "mdt_option"
                    )
                )
            ),
            "ost_hardware_path_index": (
                path_index(
                    pair.get(
                        "ost_option"
                    )
                )
            ),
        },
        "recovered_valid_architecture": bool(
            validation
            and validation.get("valid") is True
        ),
        "recovered_architecture_id": (
            validation.get("architecture_id")
            if validation
            else None
        ),
        "h10_decision": (
            validation.get("decision")
            if validation
            else None
        ),
        "coverage_interpretation": (
            "RECOVERED_BY_FULL_HARDWARE_PATH_EXPANSION"
            if validation
            else (
                "NO_FEASIBLE_PAIR_WITHIN_CURRENT_TOPK_AND_"
                "FULL_REFERENCE_HARDWARE_PATH_DOMAIN"
            )
        ),
        "global_infeasibility_claimed": False,
        "remaining_uncertainty": (
            "candidate Top-K truncation and reference-catalog scope"
            if not validation
            else None
        ),
    }
