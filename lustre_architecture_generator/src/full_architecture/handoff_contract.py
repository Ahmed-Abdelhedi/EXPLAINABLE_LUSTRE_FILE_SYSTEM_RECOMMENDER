from __future__ import annotations

import copy
import math
import re
from typing import Any


SCHEMA_VERSION = "1.0"
STAGE = "ranking_to_full_architecture"

FORBIDDEN_FINAL_HARDWARE_KEYS = {
    "raid",
    "raid_level",
    "protection_profile",
    "final_physical_drive_count",
    "physical_drive_count",
    "mdt_count",
    "ost_count",
    "mds_count",
    "oss_count",
    "server_model",
    "server_count",
    "controller_id",
    "controller_count",
    "enclosure_id",
    "enclosure_count",
    "network_id",
    "network_fabric",
    "bom",
}


class ArchitectureHandoffError(RuntimeError):
    """Erreur du contrat Ranking -> Full Architecture."""


def _finite(value: Any, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ArchitectureHandoffError(f"{field}: nombre requis.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ArchitectureHandoffError(f"{field}: nombre requis.") from error

    if not math.isfinite(number):
        raise ArchitectureHandoffError(f"{field}: nombre fini requis.")

    if minimum is not None and number < minimum:
        raise ArchitectureHandoffError(
            f"{field}: valeur >= {minimum} requise, obtenu={number}."
        )

    return number


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ArchitectureHandoffError(f"{field}: entier > 0 requis.")

    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ArchitectureHandoffError(f"{field}: entier > 0 requis.") from error

    if number <= 0:
        raise ArchitectureHandoffError(f"{field}: entier > 0 requis.")

    return number


def _mapping(parent: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ArchitectureHandoffError(f"{context}.{key}: objet requis.")
    return value


def _list(parent: dict[str, Any], key: str, context: str) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise ArchitectureHandoffError(f"{context}.{key}: liste requise.")
    return value


def _string(parent: dict[str, Any], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArchitectureHandoffError(f"{context}.{key}: chaîne non vide requise.")
    return value.strip()


def _optional_number(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return None


def _optional_pcie_generation(value: Any) -> int | None:
    """
    Normalise la génération PCIe provenant du catalogue drive.

    Valeurs acceptées :
    - 5
    - 5.0
    - "5"
    - "GEN5"
    - "PCIe Gen5"
    - "PCIE5"

    Cette fonction est volontairement placée dans le contrat H1 afin que
    l'information de compatibilité ne soit pas perdue avant H6.
    """

    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None

        generation = int(value)
        return generation if generation > 0 else None

    if isinstance(value, str):
        text = value.strip().upper()

        if not text:
            return None

        match = re.search(r"([1-9][0-9]*)", text)

        if match is None:
            return None

        generation = int(match.group(1))
        return generation if generation > 0 else None

    return None


def build_catalog_lookup(
    catalog: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}

    for index, drive in enumerate(catalog):
        if not isinstance(drive, dict):
            raise ArchitectureHandoffError(
                f"catalog[{index}]: objet JSON requis."
            )

        drive_id = drive.get("drive_id")
        if not isinstance(drive_id, str) or not drive_id.strip():
            raise ArchitectureHandoffError(
                f"catalog[{index}].drive_id invalide."
            )

        if drive_id in lookup:
            raise ArchitectureHandoffError(
                f"drive_id dupliqué dans le catalogue: {drive_id}"
            )

        lookup[drive_id] = drive

    if not lookup:
        raise ArchitectureHandoffError("Catalogue de drives vide.")

    return lookup


def _common_package(
    candidate: dict[str, Any],
    drive: dict[str, Any],
    role: str,
    selection_rank: int,
) -> dict[str, Any]:
    drive_id = _string(candidate, "drive_id", f"{role}.candidate")

    if drive_id != str(drive.get("drive_id")):
        raise ArchitectureHandoffError(
            f"{role}: incohérence drive_id ranking/catalogue."
        )

    reasons = candidate.get(
        "selection_reasons",
        ["ml_top_k"],
    )
    if not isinstance(reasons, list):
        raise ArchitectureHandoffError(
            f"{role}.selection_reasons: liste requise."
        )

    return {
        "role": role,
        "selection_rank": selection_rank,
        "identity": {
            "drive_id": drive_id,
            "drive_name": candidate.get("drive_name", drive.get("name")),
            "manufacturer": candidate.get(
                "manufacturer",
                drive.get("manufacturer"),
            ),
            "series": candidate.get("series", drive.get("series")),
            "media_type": candidate.get(
                "media_type",
                drive.get("media_type"),
            ),
            "catalog_id": drive.get("catalog_id"),
            "model_number": drive.get("model_number"),
        },
        "hardware_interface": {
            "protocol": drive.get("protocol"),
            "form_factor": drive.get("drive_form_factor_standard"),
            "pcie_gen_required": _optional_pcie_generation(
                drive.get("pcie_gen_required")
            ),
            "pcie_lanes_required": _optional_number(
                drive.get("pcie_lanes_required")
            ),
        },
        "reliability": {
            "endurance_dwpd": _optional_number(
                drive.get("endurance_dwpd_numeric")
            ),
            "mtbf_hours": _optional_number(drive.get("mtbf_hours")),
            "warranty_years": _optional_number(
                drive.get("warranty_years")
            ),
            "workload_rating_tb_per_year": _optional_number(
                drive.get("workload_rating_tb_per_year")
            ),
            "latency_class": drive.get("latency_class"),
            "quality_status": drive.get("quality_status"),
        },
        "ranking": {
            "ml_rank": _positive_int(
                candidate.get("ml_rank"),
                f"{role}.ml_rank",
            ),
            "ml_score": _finite(
                candidate.get("ml_score"),
                f"{role}.ml_score",
            ),
            "selection_reasons": [str(reason) for reason in reasons],
        },
    }


def build_mdt_candidate_package(
    candidate: dict[str, Any],
    drive: dict[str, Any],
    requirement: dict[str, Any],
    constraints: dict[str, Any],
    selection_rank: int,
) -> dict[str, Any]:
    package = _common_package(
        candidate,
        drive,
        "MDT",
        selection_rank,
    )

    required_capacity = _finite(
        requirement.get("required_metadata_capacity_tib"),
        "MDT.required_metadata_capacity_tib",
        0.0,
    )
    required_read = _finite(
        requirement.get("required_read_iops"),
        "MDT.required_read_iops",
        0.0,
    )
    required_write = _finite(
        requirement.get("required_write_iops"),
        "MDT.required_write_iops",
        0.0,
    )

    provided_capacity = _finite(
        candidate.get("raw_provided_capacity_tib"),
        "MDT.raw_provided_capacity_tib",
        0.0,
    )
    provided_read = _finite(
        candidate.get("raw_provided_read_iops"),
        "MDT.raw_provided_read_iops",
        0.0,
    )
    provided_write = _finite(
        candidate.get("raw_provided_write_iops"),
        "MDT.raw_provided_write_iops",
        0.0,
    )
    raw_cost = _finite(
        candidate.get("raw_drive_cost_usd"),
        "MDT.raw_drive_cost_usd",
        0.0,
    )
    raw_power = _finite(
        candidate.get("raw_drive_power_w"),
        "MDT.raw_drive_power_w",
        0.0,
    )
    max_budget = _finite(
        constraints.get("max_budget_usd"),
        "constraints.max_budget_usd",
        0.0,
    )
    max_power = _finite(
        constraints.get("max_power_w"),
        "constraints.max_power_w",
        0.0,
    )

    package["pre_raid"] = {
        "minimum_drive_count": _positive_int(
            candidate.get("raw_minimum_drive_count"),
            "MDT.raw_minimum_drive_count",
        ),
        "is_lower_bound_only": True,
        "provided_capacity_tib": provided_capacity,
        "provided_read_iops": provided_read,
        "provided_write_iops": provided_write,
        "drive_level_cost_usd": raw_cost,
        "drive_level_power_w": raw_power,
    }

    package["deterministic_filter_evidence"] = {
        "status": "feasible",
        "capacity": {
            "required": required_capacity,
            "provided": provided_capacity,
            "satisfied": provided_capacity >= required_capacity,
        },
        "read_iops": {
            "required": required_read,
            "provided": provided_read,
            "satisfied": provided_read >= required_read,
        },
        "write_iops": {
            "required": required_write,
            "provided": provided_write,
            "satisfied": provided_write >= required_write,
        },
        "latency_requirement": requirement.get("latency_requirement"),
        "endurance_requirement": requirement.get("endurance_requirement"),
        "reliability_requirement": requirement.get(
            "reliability_requirement"
        ),
        "raw_budget_lower_bound": {
            "maximum": max_budget,
            "used": raw_cost,
            "satisfied": raw_cost <= max_budget,
        },
        "raw_power_lower_bound": {
            "maximum": max_power,
            "used": raw_power,
            "satisfied": raw_power <= max_power,
        },
    }

    return package


def build_ost_candidate_package(
    candidate: dict[str, Any],
    drive: dict[str, Any],
    requirement: dict[str, Any],
    constraints: dict[str, Any],
    selection_rank: int,
) -> dict[str, Any]:
    package = _common_package(
        candidate,
        drive,
        "OST",
        selection_rank,
    )

    required_capacity = _finite(
        requirement.get("required_usable_capacity_tib"),
        "OST.required_usable_capacity_tib",
        0.0,
    )
    required_read = _finite(
        requirement.get("required_read_bandwidth_gbps"),
        "OST.required_read_bandwidth_gbps",
        0.0,
    )
    required_write = _finite(
        requirement.get("required_write_bandwidth_gbps"),
        "OST.required_write_bandwidth_gbps",
        0.0,
    )
    required_total = _finite(
        requirement.get("required_total_bandwidth_gbps"),
        "OST.required_total_bandwidth_gbps",
        0.0,
    )

    provided_capacity = _finite(
        candidate.get("raw_provided_capacity_tib"),
        "OST.raw_provided_capacity_tib",
        0.0,
    )
    provided_read = _finite(
        candidate.get("raw_provided_read_bandwidth_gbps"),
        "OST.raw_provided_read_bandwidth_gbps",
        0.0,
    )
    provided_write = _finite(
        candidate.get("raw_provided_write_bandwidth_gbps"),
        "OST.raw_provided_write_bandwidth_gbps",
        0.0,
    )
    provided_total = _finite(
        candidate.get("raw_provided_total_bandwidth_gbps"),
        "OST.raw_provided_total_bandwidth_gbps",
        0.0,
    )
    raw_cost = _finite(
        candidate.get("raw_drive_cost_usd"),
        "OST.raw_drive_cost_usd",
        0.0,
    )
    raw_power = _finite(
        candidate.get("raw_drive_power_w"),
        "OST.raw_drive_power_w",
        0.0,
    )
    max_budget = _finite(
        constraints.get("max_budget_usd"),
        "constraints.max_budget_usd",
        0.0,
    )
    max_power = _finite(
        constraints.get("max_power_w"),
        "constraints.max_power_w",
        0.0,
    )

    package["pre_raid"] = {
        "minimum_drive_count": _positive_int(
            candidate.get("raw_minimum_drive_count"),
            "OST.raw_minimum_drive_count",
        ),
        "is_lower_bound_only": True,
        "provided_capacity_tib": provided_capacity,
        "provided_read_bandwidth_gb_s": provided_read,
        "provided_write_bandwidth_gb_s": provided_write,
        "provided_total_bandwidth_gb_s": provided_total,
        "drive_level_cost_usd": raw_cost,
        "drive_level_power_w": raw_power,
    }

    package["deterministic_filter_evidence"] = {
        "status": "feasible",
        "capacity": {
            "required": required_capacity,
            "provided": provided_capacity,
            "satisfied": provided_capacity >= required_capacity,
        },
        "read_bandwidth": {
            "unit": "GB/s",
            "required": required_read,
            "provided": provided_read,
            "satisfied": provided_read >= required_read,
        },
        "write_bandwidth": {
            "unit": "GB/s",
            "required": required_write,
            "provided": provided_write,
            "satisfied": provided_write >= required_write,
        },
        "total_bandwidth": {
            "unit": "GB/s",
            "required": required_total,
            "provided": provided_total,
            "satisfied": provided_total >= required_total,
        },
        "reliability_requirement": requirement.get(
            "reliability_requirement"
        ),
        "raw_budget_lower_bound": {
            "maximum": max_budget,
            "used": raw_cost,
            "satisfied": raw_cost <= max_budget,
        },
        "raw_power_lower_bound": {
            "maximum": max_power,
            "used": raw_power,
            "satisfied": raw_power <= max_power,
        },
    }

    return package


def assemble_architecture_handoff(
    architecture: dict[str, Any],
    catalog: list[dict[str, Any]],
    mdt_ranking_result: dict[str, Any],
    ost_ranking_result: dict[str, Any],
    diversified_ost_result: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k doit être > 0.")

    case_id = _string(architecture, "case_id", "architecture")

    for label, result in (
        ("MDT", mdt_ranking_result),
        ("OST", ost_ranking_result),
        ("OST diversified", diversified_ost_result),
    ):
        if str(result.get("case_id", "")) != case_id:
            raise ArchitectureHandoffError(
                f"{label}: case_id incohérent."
            )

    mdt_requirement = _mapping(
        architecture,
        "MDT_requirement",
        case_id,
    )
    ost_requirement = _mapping(
        architecture,
        "OST_requirement",
        case_id,
    )
    constraints = _mapping(
        architecture,
        "constraints",
        case_id,
    )
    preferences = _mapping(
        architecture,
        "preferences",
        case_id,
    )

    mdt_ranked = _list(
        mdt_ranking_result,
        "ranked_candidates",
        "mdt_ranking_result",
    )
    ost_selected = _list(
        diversified_ost_result,
        "diversified_candidates",
        "diversified_ost_result",
    )

    if not mdt_ranked or not ost_selected:
        raise ArchitectureHandoffError(
            f"{case_id}: candidats MDT/OST absents."
        )

    lookup = build_catalog_lookup(catalog)

    mdt_packages: list[dict[str, Any]] = []
    for selection_rank, candidate_any in enumerate(
        mdt_ranked[:top_k],
        start=1,
    ):
        if not isinstance(candidate_any, dict):
            raise ArchitectureHandoffError("Candidat MDT invalide.")

        drive_id = _string(
            candidate_any,
            "drive_id",
            "MDT candidate",
        )
        drive = lookup.get(drive_id)
        if drive is None:
            raise ArchitectureHandoffError(
                f"Drive MDT absent du catalogue: {drive_id}"
            )

        mdt_packages.append(
            build_mdt_candidate_package(
                candidate_any,
                drive,
                mdt_requirement,
                constraints,
                selection_rank,
            )
        )

    ost_packages: list[dict[str, Any]] = []
    for selection_rank, candidate_any in enumerate(
        ost_selected[:top_k],
        start=1,
    ):
        if not isinstance(candidate_any, dict):
            raise ArchitectureHandoffError("Candidat OST invalide.")

        drive_id = _string(
            candidate_any,
            "drive_id",
            "OST candidate",
        )
        drive = lookup.get(drive_id)
        if drive is None:
            raise ArchitectureHandoffError(
                f"Drive OST absent du catalogue: {drive_id}"
            )

        ost_packages.append(
            build_ost_candidate_package(
                candidate_any,
                drive,
                ost_requirement,
                constraints,
                selection_rank,
            )
        )

    handoff = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "case_id": case_id,
        "requested_top_k": top_k,
        "actual_top_k": {
            "mdt": len(mdt_packages),
            "ost": len(ost_packages),
        },
        "unit_contract": {
            "capacity": "TiB",
            "mdt_iops": "IOPS",
            "throughput_rate": "GB/s",
            "catalog_sequential_rate": "MB/s",
            "legacy_note": (
                "Les champs historiques suffixés '_gbps' restent inchangés "
                "dans le sizing/ranking frozen; leurs valeurs sont interprétées "
                "en GB/s dans le downstream."
            ),
        },
        "requirements": {
            "MDT_requirement": copy.deepcopy(mdt_requirement),
            "OST_requirement": copy.deepcopy(ost_requirement),
            "constraints": copy.deepcopy(constraints),
            "preferences": copy.deepcopy(preferences),
            "workload_analysis": copy.deepcopy(
                architecture.get("workload_analysis", {})
            ),
            "role_analysis": copy.deepcopy(
                architecture.get("role_analysis", {})
            ),
        },
        "ranking_provenance": {
            "mdt": {
                "model_family": mdt_ranking_result.get("model_family"),
                "model_type": mdt_ranking_result.get("model_type"),
                "model_seed": mdt_ranking_result.get("model_seed"),
                "feature_count": mdt_ranking_result.get("feature_count"),
                "feasible_candidate_count": mdt_ranking_result.get(
                    "feasible_candidate_count"
                ),
            },
            "ost": {
                "model_family": ost_ranking_result.get("model_family"),
                "model_type": ost_ranking_result.get("model_type"),
                "model_seed": ost_ranking_result.get("model_seed"),
                "feature_count": ost_ranking_result.get("feature_count"),
                "feasible_candidate_count": ost_ranking_result.get(
                    "feasible_candidate_count"
                ),
            },
            "ost_diversification": {
                "global_top_count": diversified_ost_result.get(
                    "global_top_count"
                ),
                "diversification_pool_size": diversified_ost_result.get(
                    "diversification_pool_size"
                ),
                "maximum_specialized_ml_rank": diversified_ost_result.get(
                    "maximum_specialized_ml_rank"
                ),
                "media_distribution": copy.deepcopy(
                    diversified_ost_result.get("media_distribution", {})
                ),
            },
        },
        "contract_invariants": {
            "pre_raid_counts_are_lower_bounds": True,
            "hard_feasibility_precedes_ml": True,
            "no_final_hardware_selected": True,
            "beam_search_not_applied": True,
        },
        "mdt_candidates": mdt_packages,
        "ost_candidates": ost_packages,
    }

    assert_valid_architecture_handoff(handoff)
    return handoff


def _find_forbidden(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_FINAL_HARDWARE_KEYS:
                findings.append(child_path)
            findings.extend(_find_forbidden(child, child_path))

    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(
                _find_forbidden(child, f"{path}[{index}]")
            )

    return findings


def _validate_candidates(
    candidates: Any,
    role: str,
    top_k: int,
) -> list[str]:
    errors: list[str] = []

    if not isinstance(candidates, list) or not candidates:
        return [f"{role}: liste de candidats non vide requise."]

    if len(candidates) > top_k:
        errors.append(
            f"{role}: nombre de candidats > requested_top_k."
        )

    seen: set[str] = set()

    for expected_rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            errors.append(f"{role}[{expected_rank}]: objet requis.")
            continue

        if candidate.get("role") != role:
            errors.append(f"{role}[{expected_rank}]: role incorrect.")

        if candidate.get("selection_rank") != expected_rank:
            errors.append(
                f"{role}[{expected_rank}]: selection_rank incorrect."
            )

        identity = candidate.get("identity")
        if not isinstance(identity, dict):
            errors.append(f"{role}[{expected_rank}]: identity absente.")
            continue

        drive_id = identity.get("drive_id")
        if not isinstance(drive_id, str) or not drive_id:
            errors.append(f"{role}[{expected_rank}]: drive_id invalide.")
        elif drive_id in seen:
            errors.append(f"{role}: drive_id dupliqué={drive_id}.")
        else:
            seen.add(drive_id)

        pre_raid = candidate.get("pre_raid")
        if not isinstance(pre_raid, dict):
            errors.append(f"{role}[{expected_rank}]: pre_raid absent.")
            continue

        if pre_raid.get("is_lower_bound_only") is not True:
            errors.append(
                f"{role}[{expected_rank}]: pre_raid non marqué lower-bound."
            )

        count = pre_raid.get("minimum_drive_count")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):
            errors.append(
                f"{role}[{expected_rank}]: minimum_drive_count invalide."
            )

        evidence = candidate.get("deterministic_filter_evidence")
        if not isinstance(evidence, dict):
            errors.append(f"{role}[{expected_rank}]: evidence absente.")
            continue

        if evidence.get("status") != "feasible":
            errors.append(f"{role}[{expected_rank}]: status != feasible.")

        check_names = (
            ["capacity", "read_iops", "write_iops"]
            if role == "MDT"
            else [
                "capacity",
                "read_bandwidth",
                "write_bandwidth",
                "total_bandwidth",
            ]
        )

        for check_name in check_names + [
            "raw_budget_lower_bound",
            "raw_power_lower_bound",
        ]:
            check = evidence.get(check_name)
            if not isinstance(check, dict):
                errors.append(
                    f"{role}[{expected_rank}]: {check_name} absent."
                )
            elif check.get("satisfied") is not True:
                errors.append(
                    f"{role}[{expected_rank}]: {check_name} non satisfait."
                )

    return errors


def validate_architecture_handoff(
    handoff: dict[str, Any],
) -> list[str]:
    if not isinstance(handoff, dict):
        return ["Le handoff doit être un objet JSON."]

    required = {
        "schema_version",
        "stage",
        "case_id",
        "requested_top_k",
        "actual_top_k",
        "unit_contract",
        "requirements",
        "ranking_provenance",
        "contract_invariants",
        "mdt_candidates",
        "ost_candidates",
    }

    missing = required - set(handoff)
    if missing:
        return [f"Champs manquants: {sorted(missing)}."]

    errors: list[str] = []

    if handoff.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version incorrecte.")

    if handoff.get("stage") != STAGE:
        errors.append("stage incorrect.")

    try:
        top_k = _positive_int(
            handoff.get("requested_top_k"),
            "requested_top_k",
        )
    except ArchitectureHandoffError as error:
        return [str(error)]

    unit_contract = handoff.get("unit_contract")
    if (
        not isinstance(unit_contract, dict)
        or unit_contract.get("throughput_rate") != "GB/s"
    ):
        errors.append("throughput_rate doit être explicitement GB/s.")

    invariants = handoff.get("contract_invariants")
    if not isinstance(invariants, dict):
        errors.append("contract_invariants invalide.")
    else:
        for key in (
            "pre_raid_counts_are_lower_bounds",
            "hard_feasibility_precedes_ml",
            "no_final_hardware_selected",
            "beam_search_not_applied",
        ):
            if invariants.get(key) is not True:
                errors.append(f"Invariant non respecté: {key}.")

    mdt = handoff.get("mdt_candidates")
    ost = handoff.get("ost_candidates")

    errors.extend(_validate_candidates(mdt, "MDT", top_k))
    errors.extend(_validate_candidates(ost, "OST", top_k))

    if isinstance(handoff.get("actual_top_k"), dict):
        actual = handoff["actual_top_k"]
        if isinstance(mdt, list) and actual.get("mdt") != len(mdt):
            errors.append("actual_top_k.mdt incohérent.")
        if isinstance(ost, list) and actual.get("ost") != len(ost):
            errors.append("actual_top_k.ost incohérent.")
    else:
        errors.append("actual_top_k invalide.")

    forbidden = _find_forbidden(handoff)
    if forbidden:
        errors.append(
            "Décisions hardware finales interdites dans le handoff: "
            + ", ".join(forbidden[:10])
        )

    return errors


def assert_valid_architecture_handoff(
    handoff: dict[str, Any],
) -> None:
    errors = validate_architecture_handoff(handoff)
    if errors:
        raise ArchitectureHandoffError(
            "Handoff Ranking -> Full Architecture invalide:\n- "
            + "\n- ".join(errors)
        )
