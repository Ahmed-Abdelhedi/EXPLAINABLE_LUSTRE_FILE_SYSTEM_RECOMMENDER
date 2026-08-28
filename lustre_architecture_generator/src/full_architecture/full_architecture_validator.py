from __future__ import annotations

import copy
import math
from typing import Any

from .architecture_state import validate_full_architecture_state
from .compatibility_rules import evaluate_hardware_path
from .protection_arithmetic import (
    calculate_mdt_protection,
    calculate_ost_protection,
)


VALIDATOR_SCHEMA_VERSION = "1.0"
VALIDATOR_POLICY_ID = "FULL_ARCH_VALIDATOR_V1"


class FullArchitectureValidationError(RuntimeError):
    """Erreur d'exécution du validateur déterministe complet H10."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FullArchitectureValidationError(f"{field}: objet JSON requis.")
    return value


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise FullArchitectureValidationError(f"{field}: liste JSON requise.")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FullArchitectureValidationError(f"{field}: chaîne non vide requise.")
    return value.strip()


def _finite(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise FullArchitectureValidationError(f"{field}: nombre requis.")

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise FullArchitectureValidationError(f"{field}: nombre requis.") from error

    if not math.isfinite(number):
        raise FullArchitectureValidationError(f"{field}: nombre fini requis.")

    if minimum is not None and number < minimum:
        raise FullArchitectureValidationError(
            f"{field}: valeur >= {minimum} requise."
        )

    return number


def _numbers_equal(left: Any, right: Any) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(a) or not math.isfinite(b):
        return False

    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _index_by_id(
    values: list[dict[str, Any]],
    *,
    section: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for index, value in enumerate(values):
        item = _mapping(value, field=f"{section}[{index}]")
        item_id = _string(item.get("id"), field=f"{section}[{index}].id")

        if item_id in result:
            raise FullArchitectureValidationError(
                f"{section}: id dupliqué={item_id!r}."
            )

        result[item_id] = item

    return result


def _candidate_index(
    handoff: dict[str, Any],
    *,
    role: str,
) -> dict[str, dict[str, Any]]:
    key = "mdt_candidates" if role == "MDT" else "ost_candidates"
    candidates = _list(handoff.get(key), field=f"handoff.{key}")
    result: dict[str, dict[str, Any]] = {}

    for index, candidate in enumerate(candidates):
        item = _mapping(candidate, field=f"handoff.{key}[{index}]")
        identity = _mapping(
            item.get("identity"),
            field=f"handoff.{key}[{index}].identity",
        )
        drive_id = _string(
            identity.get("drive_id"),
            field=f"handoff.{key}[{index}].identity.drive_id",
        )
        result[drive_id] = item

    return result


def _violation(
    violations: list[dict[str, Any]],
    *,
    code: str,
    category: str,
    message: str,
    expected: Any = None,
    actual: Any = None,
) -> None:
    violations.append(
        {
            "code": code,
            "category": category,
            "message": message,
            "expected": expected,
            "actual": actual,
        }
    )


def _check_equal(
    violations: list[dict[str, Any]],
    *,
    code: str,
    category: str,
    field: str,
    expected: Any,
    actual: Any,
    numeric: bool = False,
) -> None:
    equal = (
        _numbers_equal(expected, actual)
        if numeric
        else expected == actual
    )

    if not equal:
        _violation(
            violations,
            code=code,
            category=category,
            message=f"Incohérence sur {field}.",
            expected=expected,
            actual=actual,
        )


def _selected_candidate(
    *,
    selected: dict[str, Any],
    role: str,
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = "mdt_drive" if role == "MDT" else "ost_drive"
    drive = _mapping(selected.get(key), field=f"selected.{key}")
    drive_id = _string(drive.get("drive_id"), field=f"selected.{key}.drive_id")

    candidate = candidates.get(drive_id)
    if candidate is None:
        raise FullArchitectureValidationError(
            f"{role}: drive sélectionné {drive_id!r} absent du handoff."
        )

    return candidate


def _protection_profile(
    *,
    selected_protection: dict[str, Any],
    hardware_catalog: dict[str, Any],
) -> dict[str, Any]:
    profile_id = _string(
        selected_protection.get("protection_profile_id"),
        field="selected_protection.protection_profile_id",
    )
    profiles = _index_by_id(
        _list(
            hardware_catalog.get("protection_profiles"),
            field="hardware_catalog.protection_profiles",
        ),
        section="hardware_catalog.protection_profiles",
    )
    profile = profiles.get(profile_id)

    if profile is None:
        raise FullArchitectureValidationError(
            f"Profil de protection inconnu={profile_id!r}."
        )

    return profile


def _component(
    *,
    hardware_catalog: dict[str, Any],
    section: str,
    component_id: Any,
    allow_none: bool = False,
) -> dict[str, Any] | None:
    if component_id is None and allow_none:
        return None

    expected_id = _string(component_id, field=f"selected_path.{section}_id")
    values = _index_by_id(
        _list(hardware_catalog.get(section), field=f"hardware_catalog.{section}"),
        section=f"hardware_catalog.{section}",
    )
    component = values.get(expected_id)

    if component is None:
        raise FullArchitectureValidationError(
            f"Composant {section} inconnu={expected_id!r}."
        )

    return component


def _recompute_role(
    *,
    state: dict[str, Any],
    handoff: dict[str, Any],
    hardware_catalog: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    selected = _mapping(state.get("selected"), field="state.selected")
    requirements = _mapping(handoff.get("requirements"), field="handoff.requirements")
    constraints = _mapping(
        requirements.get("constraints"),
        field="handoff.requirements.constraints",
    )

    candidates = _candidate_index(handoff, role=role)
    candidate = _selected_candidate(
        selected=selected,
        role=role,
        candidates=candidates,
    )

    protection_key = "mdt_protection" if role == "MDT" else "ost_protection"
    path_key = "mdt_hardware_path" if role == "MDT" else "ost_hardware_path"
    requirement_key = "MDT_requirement" if role == "MDT" else "OST_requirement"

    selected_protection = _mapping(
        selected.get(protection_key),
        field=f"state.selected.{protection_key}",
    )
    selected_path = _mapping(
        selected.get(path_key),
        field=f"state.selected.{path_key}",
    )
    requirement = _mapping(
        requirements.get(requirement_key),
        field=f"handoff.requirements.{requirement_key}",
    )
    profile = _protection_profile(
        selected_protection=selected_protection,
        hardware_catalog=hardware_catalog,
    )

    if role == "MDT":
        protection = calculate_mdt_protection(
            candidate=candidate,
            protection_profile=profile,
            requirement=requirement,
        )
    else:
        protection = calculate_ost_protection(
            candidate=candidate,
            protection_profile=profile,
            requirement=requirement,
        )

    server = _component(
        hardware_catalog=hardware_catalog,
        section="servers",
        component_id=selected_path.get("server_id"),
    )
    controller = _component(
        hardware_catalog=hardware_catalog,
        section="controllers",
        component_id=selected_path.get("controller_id"),
    )
    network = _component(
        hardware_catalog=hardware_catalog,
        section="networks",
        component_id=selected_path.get("network_id"),
    )
    ha_profile = _component(
        hardware_catalog=hardware_catalog,
        section="ha_profiles",
        component_id=selected_path.get("ha_profile_id"),
    )

    attachment_mode = str(selected_path.get("attachment_mode", "")).upper()
    enclosure_id = selected_path.get("enclosure_id")

    if attachment_mode == "DIRECT":
        enclosure = None
    else:
        enclosure = _component(
            hardware_catalog=hardware_catalog,
            section="enclosures",
            component_id=enclosure_id,
            allow_none=False,
        )

    path = evaluate_hardware_path(
        candidate=candidate,
        protection_result=protection,
        role=role,
        server=_mapping(server, field="server"),
        controller=_mapping(controller, field="controller"),
        network=_mapping(network, field="network"),
        ha_profile=_mapping(ha_profile, field="ha_profile"),
        ha_required=bool(constraints.get("ha_required", False)),
        enclosure=enclosure,
    )

    return {
        "candidate": candidate,
        "selected_protection": selected_protection,
        "selected_path": selected_path,
        "recomputed_protection": protection,
        "recomputed_path": path,
    }


def _compare_protection(
    *,
    role: str,
    selected: dict[str, Any],
    recomputed: dict[str, Any],
    violations: list[dict[str, Any]],
) -> None:
    scalar_fields = (
        "role",
        "drive_id",
        "protection_profile_id",
        "raid_level",
        "raw_minimum_drive_count",
        "group_count",
        "group_size",
        "data_drives_per_group",
        "parity_drives_per_group",
        "mirror_copies",
        "physical_drive_count",
        "fault_tolerance_drives_per_group",
    )

    for field in scalar_fields:
        _check_equal(
            violations,
            code=f"{role.lower()}_protection_mismatch",
            category="protection",
            field=f"{role}.{field}",
            expected=recomputed.get(field),
            actual=selected.get(field),
        )

    for section in ("provided", "requirements", "per_drive"):
        recomputed_section = _mapping(
            recomputed.get(section),
            field=f"recomputed.{role}.{section}",
        )
        selected_section = _mapping(
            selected.get(section),
            field=f"selected.{role}.{section}",
        )

        for field, expected in recomputed_section.items():
            _check_equal(
                violations,
                code=f"{role.lower()}_protection_numeric_mismatch",
                category="protection",
                field=f"{role}.{section}.{field}",
                expected=expected,
                actual=selected_section.get(field),
                numeric=True,
            )

    selected_satisfied = _mapping(
        selected.get("satisfied"),
        field=f"selected.{role}.satisfied",
    )
    recomputed_satisfied = _mapping(
        recomputed.get("satisfied"),
        field=f"recomputed.{role}.satisfied",
    )

    for field, expected in recomputed_satisfied.items():
        _check_equal(
            violations,
            code=f"{role.lower()}_satisfaction_mismatch",
            category="requirements",
            field=f"{role}.satisfied.{field}",
            expected=expected,
            actual=selected_satisfied.get(field),
        )

    for field in ("protected_drive_cost_usd", "protected_drive_power_w"):
        _check_equal(
            violations,
            code=f"{role.lower()}_protection_cost_power_mismatch",
            category="cost_power",
            field=f"{role}.{field}",
            expected=recomputed.get(field),
            actual=selected.get(field),
            numeric=True,
        )


def _compare_path(
    *,
    role: str,
    selected: dict[str, Any],
    recomputed: dict[str, Any],
    violations: list[dict[str, Any]],
) -> None:
    id_fields = (
        "compatible",
        "role",
        "attachment_mode",
        "drive_id",
        "protection_profile_id",
        "server_id",
        "controller_id",
        "enclosure_id",
        "network_id",
        "ha_profile_id",
    )

    for field in id_fields:
        _check_equal(
            violations,
            code=f"{role.lower()}_hardware_path_mismatch",
            category="hardware_compatibility",
            field=f"{role}.path.{field}",
            expected=recomputed.get(field),
            actual=selected.get(field),
        )

    if recomputed.get("compatible") is not True:
        _violation(
            violations,
            code=f"{role.lower()}_hardware_path_incompatible",
            category="hardware_compatibility",
            message=f"Le chemin hardware {role} recomputé est incompatible.",
            expected=[],
            actual=recomputed.get("violations"),
        )

    selected_resources = _mapping(
        selected.get("minimum_resources"),
        field=f"selected.{role}.minimum_resources",
    )
    recomputed_resources = _mapping(
        recomputed.get("minimum_resources"),
        field=f"recomputed.{role}.minimum_resources",
    )

    for field, expected in recomputed_resources.items():
        _check_equal(
            violations,
            code=f"{role.lower()}_resource_mismatch",
            category="hardware_resources",
            field=f"{role}.minimum_resources.{field}",
            expected=expected,
            actual=selected_resources.get(field),
        )

    for field in (
        "component_cost_lower_bound_usd",
        "component_power_lower_bound_w",
    ):
        _check_equal(
            violations,
            code=f"{role.lower()}_hardware_cost_power_mismatch",
            category="cost_power",
            field=f"{role}.path.{field}",
            expected=recomputed.get(field),
            actual=selected.get(field),
            numeric=True,
        )


def _validate_state_aggregates(
    *,
    state: dict[str, Any],
    mdt: dict[str, Any],
    ost: dict[str, Any],
    violations: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = _mapping(state.get("counts"), field="state.counts")
    performance = _mapping(state.get("performance"), field="state.performance")
    cost_power = _mapping(state.get("cost_power"), field="state.cost_power")

    mdt_p = mdt["recomputed_protection"]
    ost_p = ost["recomputed_protection"]
    mdt_path = mdt["recomputed_path"]
    ost_path = ost["recomputed_path"]
    mdt_r = mdt_path["minimum_resources"]
    ost_r = ost_path["minimum_resources"]

    expected_counts = {
        "mdt_physical_drives": mdt_p["physical_drive_count"],
        "ost_physical_drives": ost_p["physical_drive_count"],
        "mdt_count": mdt_p["group_count"],
        "ost_count": ost_p["group_count"],
        "mds_count": mdt_r["server_count"],
        "oss_count": ost_r["server_count"],
        "mdt_controller_count": mdt_r["controller_count"],
        "ost_controller_count": ost_r["controller_count"],
        "mdt_enclosure_count": mdt_r["enclosure_count"],
        "ost_enclosure_count": ost_r["enclosure_count"],
        "mdt_network_adapter_count": mdt_r["network_adapter_count"],
        "ost_network_adapter_count": ost_r["network_adapter_count"],
    }

    for field, expected in expected_counts.items():
        _check_equal(
            violations,
            code="state_count_mismatch",
            category="state_aggregation",
            field=f"state.counts.{field}",
            expected=expected,
            actual=counts.get(field),
        )

    expected_performance = {
        "metadata_capacity_tib": mdt_p["provided"]["usable_capacity_tib"],
        "mdt_read_iops": mdt_p["provided"]["read_iops"],
        "mdt_write_iops": mdt_p["provided"]["write_iops"],
        "ost_usable_capacity_tib": ost_p["provided"]["usable_capacity_tib"],
        "ost_read_bandwidth_gb_s": ost_p["provided"]["read_bandwidth_gb_s"],
        "ost_write_bandwidth_gb_s": ost_p["provided"]["write_bandwidth_gb_s"],
        "ost_total_bandwidth_gb_s": ost_p["provided"]["total_bandwidth_gb_s"],
    }

    for field, expected in expected_performance.items():
        _check_equal(
            violations,
            code="state_performance_mismatch",
            category="state_aggregation",
            field=f"state.performance.{field}",
            expected=expected,
            actual=performance.get(field),
            numeric=True,
        )

    expected_cost_power = {
        "mdt_drive_cost_usd": mdt_p["protected_drive_cost_usd"],
        "ost_drive_cost_usd": ost_p["protected_drive_cost_usd"],
        "mdt_hardware_cost_usd": mdt_path["component_cost_lower_bound_usd"],
        "ost_hardware_cost_usd": ost_path["component_cost_lower_bound_usd"],
        "mdt_drive_power_w": mdt_p["protected_drive_power_w"],
        "ost_drive_power_w": ost_p["protected_drive_power_w"],
        "mdt_hardware_power_w": mdt_path["component_power_lower_bound_w"],
        "ost_hardware_power_w": ost_path["component_power_lower_bound_w"],
    }
    expected_cost_power["total_cost_usd"] = sum(
        expected_cost_power[field]
        for field in (
            "mdt_drive_cost_usd",
            "ost_drive_cost_usd",
            "mdt_hardware_cost_usd",
            "ost_hardware_cost_usd",
        )
    )
    expected_cost_power["total_power_w"] = sum(
        expected_cost_power[field]
        for field in (
            "mdt_drive_power_w",
            "ost_drive_power_w",
            "mdt_hardware_power_w",
            "ost_hardware_power_w",
        )
    )

    for field, expected in expected_cost_power.items():
        _check_equal(
            violations,
            code="state_cost_power_mismatch",
            category="state_aggregation",
            field=f"state.cost_power.{field}",
            expected=expected,
            actual=cost_power.get(field),
            numeric=True,
        )

    return {
        "counts": expected_counts,
        "performance": expected_performance,
        "cost_power": expected_cost_power,
    }


def _validate_hard_requirements(
    *,
    handoff: dict[str, Any],
    mdt: dict[str, Any],
    ost: dict[str, Any],
    aggregates: dict[str, Any],
    violations: list[dict[str, Any]],
) -> dict[str, Any]:
    requirements = _mapping(handoff.get("requirements"), field="handoff.requirements")
    constraints = _mapping(
        requirements.get("constraints"),
        field="handoff.requirements.constraints",
    )

    mdt_satisfied = _mapping(
        mdt["recomputed_protection"].get("satisfied"),
        field="recomputed.MDT.satisfied",
    )
    ost_satisfied = _mapping(
        ost["recomputed_protection"].get("satisfied"),
        field="recomputed.OST.satisfied",
    )

    for field, satisfied in mdt_satisfied.items():
        if satisfied is not True:
            _violation(
                violations,
                code="mdt_requirement_unsatisfied",
                category="requirements",
                message=f"Exigence MDT non satisfaite: {field}.",
                expected=True,
                actual=satisfied,
            )

    for field, satisfied in ost_satisfied.items():
        if satisfied is not True:
            _violation(
                violations,
                code="ost_requirement_unsatisfied",
                category="requirements",
                message=f"Exigence OST non satisfaite: {field}.",
                expected=True,
                actual=satisfied,
            )

    total_cost = _finite(
        aggregates["cost_power"]["total_cost_usd"],
        field="recomputed.total_cost_usd",
        minimum=0.0,
    )
    total_power = _finite(
        aggregates["cost_power"]["total_power_w"],
        field="recomputed.total_power_w",
        minimum=0.0,
    )
    max_budget = _finite(
        constraints.get("max_budget_usd"),
        field="constraints.max_budget_usd",
        minimum=0.0,
    )
    max_power = _finite(
        constraints.get("max_power_w"),
        field="constraints.max_power_w",
        minimum=0.0,
    )

    budget_ok = total_cost <= max_budget + 1e-12
    power_ok = total_power <= max_power + 1e-12

    if not budget_ok:
        _violation(
            violations,
            code="budget_exceeded",
            category="hard_constraints",
            message="Le coût total recomputé dépasse le budget maximal.",
            expected=max_budget,
            actual=total_cost,
        )

    if not power_ok:
        _violation(
            violations,
            code="power_exceeded",
            category="hard_constraints",
            message="La puissance totale recomputée dépasse la limite.",
            expected=max_power,
            actual=total_power,
        )

    ha_required = bool(constraints.get("ha_required", False))
    mdt_ha_ok = (
        not ha_required
        or str(mdt["recomputed_path"].get("ha_profile_id", "")) != "HA_NONE"
    )
    ost_ha_ok = (
        not ha_required
        or str(ost["recomputed_path"].get("ha_profile_id", "")) != "HA_NONE"
    )

    if not mdt_ha_ok:
        _violation(
            violations,
            code="mdt_ha_requirement_unsatisfied",
            category="hard_constraints",
            message="HA requise mais absente sur le chemin MDT.",
            expected=True,
            actual=False,
        )

    if not ost_ha_ok:
        _violation(
            violations,
            code="ost_ha_requirement_unsatisfied",
            category="hard_constraints",
            message="HA requise mais absente sur le chemin OST.",
            expected=True,
            actual=False,
        )

    return {
        "mdt_requirements_satisfied": all(
            value is True for value in mdt_satisfied.values()
        ),
        "ost_requirements_satisfied": all(
            value is True for value in ost_satisfied.values()
        ),
        "budget": {
            "maximum_usd": max_budget,
            "recomputed_total_usd": total_cost,
            "satisfied": budget_ok,
        },
        "power": {
            "maximum_w": max_power,
            "recomputed_total_w": total_power,
            "satisfied": power_ok,
        },
        "ha": {
            "required": ha_required,
            "mdt_satisfied": mdt_ha_ok,
            "ost_satisfied": ost_ha_ok,
        },
    }


def validate_complete_architecture(
    *,
    architecture: dict[str, Any],
    handoff: dict[str, Any],
    hardware_catalog: dict[str, Any],
) -> dict[str, Any]:
    """
    Valide indépendamment une architecture complète H8.

    H10 recompute H5 et H6 à partir des choix stockés, puis compare les
    agrégats H7 et applique les contraintes dures. Aucun score H9 n'est requis.
    """

    architecture_id = _string(
        architecture.get("architecture_id"),
        field="architecture.architecture_id",
    )
    case_id = _string(architecture.get("case_id"), field="architecture.case_id")
    handoff_case = _string(handoff.get("case_id"), field="handoff.case_id")

    if case_id != handoff_case:
        raise FullArchitectureValidationError(
            "architecture.case_id et handoff.case_id sont incohérents."
        )

    state = _mapping(architecture.get("state"), field="architecture.state")
    validate_full_architecture_state(state)

    if state.get("stage") != "COMPLETE":
        raise FullArchitectureValidationError(
            f"{architecture_id}: H10 exige un ArchitectureState COMPLETE."
        )

    if str(state.get("case_id")) != case_id:
        raise FullArchitectureValidationError(
            f"{architecture_id}: state.case_id incohérent."
        )

    violations: list[dict[str, Any]] = []

    try:
        mdt = _recompute_role(
            state=state,
            handoff=handoff,
            hardware_catalog=hardware_catalog,
            role="MDT",
        )
        ost = _recompute_role(
            state=state,
            handoff=handoff,
            hardware_catalog=hardware_catalog,
            role="OST",
        )
    except (FullArchitectureValidationError, KeyError, ValueError, TypeError) as error:
        _violation(
            violations,
            code="architecture_recompute_failed",
            category="integrity",
            message=(
                "Impossible de recomputer les choix H5/H6 depuis "
                "l'architecture sélectionnée."
            ),
            expected="recomputable selected architecture",
            actual=f"{type(error).__name__}: {error}",
        )

        validated_state = copy.deepcopy(state)
        validated_state["validation"]["is_complete"] = True
        validated_state["validation"]["is_valid"] = False
        validated_state["validation"]["status"] = "INVALID"
        validated_state["validation"]["violations"] = [
            item["code"] for item in violations
        ]
        validated_state["validation"]["full_validator_policy_id"] = (
            VALIDATOR_POLICY_ID
        )
        validate_full_architecture_state(validated_state)

        return {
            "schema_version": VALIDATOR_SCHEMA_VERSION,
            "validator_policy_id": VALIDATOR_POLICY_ID,
            "architecture_id": architecture_id,
            "case_id": case_id,
            "valid": False,
            "decision": "INVALID",
            "violation_count": len(violations),
            "violation_categories": {"integrity": 1},
            "violations": violations,
            "hard_requirements": None,
            "recomputed": None,
            "validated_state": validated_state,
            "scoring_required": False,
            "beam_search_applied": False,
        }

    _compare_protection(
        role="MDT",
        selected=mdt["selected_protection"],
        recomputed=mdt["recomputed_protection"],
        violations=violations,
    )
    _compare_protection(
        role="OST",
        selected=ost["selected_protection"],
        recomputed=ost["recomputed_protection"],
        violations=violations,
    )
    _compare_path(
        role="MDT",
        selected=mdt["selected_path"],
        recomputed=mdt["recomputed_path"],
        violations=violations,
    )
    _compare_path(
        role="OST",
        selected=ost["selected_path"],
        recomputed=ost["recomputed_path"],
        violations=violations,
    )

    aggregates = _validate_state_aggregates(
        state=state,
        mdt=mdt,
        ost=ost,
        violations=violations,
    )
    hard_requirements = _validate_hard_requirements(
        handoff=handoff,
        mdt=mdt,
        ost=ost,
        aggregates=aggregates,
        violations=violations,
    )

    valid = len(violations) == 0
    validated_state = copy.deepcopy(state)
    validated_state["validation"]["is_complete"] = True
    validated_state["validation"]["is_valid"] = valid
    validated_state["validation"]["status"] = "VALIDATED" if valid else "INVALID"
    validated_state["validation"]["violations"] = [
        item["code"] for item in violations
    ]
    validated_state["validation"]["full_validator_policy_id"] = (
        VALIDATOR_POLICY_ID
    )

    # architecture_state.py est étendu en H10 pour accepter ces statuts terminaux.
    validate_full_architecture_state(validated_state)

    categories: dict[str, int] = {}
    for item in violations:
        category = str(item["category"])
        categories[category] = categories.get(category, 0) + 1

    return {
        "schema_version": VALIDATOR_SCHEMA_VERSION,
        "validator_policy_id": VALIDATOR_POLICY_ID,
        "architecture_id": architecture_id,
        "case_id": case_id,
        "valid": valid,
        "decision": "VALID" if valid else "INVALID",
        "violation_count": len(violations),
        "violation_categories": categories,
        "violations": violations,
        "hard_requirements": hard_requirements,
        "recomputed": {
            "mdt_protection": copy.deepcopy(mdt["recomputed_protection"]),
            "ost_protection": copy.deepcopy(ost["recomputed_protection"]),
            "mdt_hardware_path": copy.deepcopy(mdt["recomputed_path"]),
            "ost_hardware_path": copy.deepcopy(ost["recomputed_path"]),
            "aggregates": aggregates,
        },
        "validated_state": validated_state,
        "scoring_required": False,
        "beam_search_applied": False,
    }


def validate_generated_architectures(
    *,
    generation_result: dict[str, Any],
    handoff: dict[str, Any],
    hardware_catalog: dict[str, Any],
) -> dict[str, Any]:
    case_id = _string(
        generation_result.get("case_id"),
        field="generation_result.case_id",
    )
    architectures = _list(
        generation_result.get("architectures"),
        field="generation_result.architectures",
    )

    if not architectures:
        raise FullArchitectureValidationError(
            f"{case_id}: aucune architecture à valider."
        )

    decisions = [
        validate_complete_architecture(
            architecture=_mapping(
                architecture,
                field=f"architectures[{index}]",
            ),
            handoff=handoff,
            hardware_catalog=hardware_catalog,
        )
        for index, architecture in enumerate(architectures)
    ]

    valid_decisions = [item for item in decisions if item["valid"]]
    invalid_decisions = [item for item in decisions if not item["valid"]]

    violation_code_counts: dict[str, int] = {}
    for decision in invalid_decisions:
        for violation in decision["violations"]:
            code = str(violation["code"])
            violation_code_counts[code] = violation_code_counts.get(code, 0) + 1

    return {
        "schema_version": VALIDATOR_SCHEMA_VERSION,
        "stage": "full_architecture_validation",
        "validator_policy_id": VALIDATOR_POLICY_ID,
        "case_id": case_id,
        "summary": {
            "architecture_count": len(decisions),
            "valid_architecture_count": len(valid_decisions),
            "invalid_architecture_count": len(invalid_decisions),
            "has_valid_architecture": bool(valid_decisions),
            "first_valid_architecture_id": (
                valid_decisions[0]["architecture_id"]
                if valid_decisions
                else None
            ),
            "violation_code_counts": violation_code_counts,
        },
        "architectures": decisions,
        "scoring_required": False,
        "beam_search_applied": False,
    }


def assert_full_validation_result_valid(result: dict[str, Any]) -> None:
    if result.get("schema_version") != VALIDATOR_SCHEMA_VERSION:
        raise FullArchitectureValidationError(
            "schema_version H10 incorrecte."
        )

    if result.get("stage") != "full_architecture_validation":
        raise FullArchitectureValidationError("stage H10 incorrect.")

    decisions = _list(
        result.get("architectures"),
        field="result.architectures",
    )

    if not decisions:
        raise FullArchitectureValidationError(
            "H10: aucune décision architecture."
        )

    seen: set[str] = set()

    for index, decision in enumerate(decisions):
        item = _mapping(decision, field=f"result.architectures[{index}]")
        architecture_id = _string(
            item.get("architecture_id"),
            field=f"result.architectures[{index}].architecture_id",
        )
        if architecture_id in seen:
            raise FullArchitectureValidationError(
                f"H10: architecture_id dupliqué={architecture_id!r}."
            )
        seen.add(architecture_id)

        expected_decision = "VALID" if item.get("valid") is True else "INVALID"
        if item.get("decision") != expected_decision:
            raise FullArchitectureValidationError(
                "H10: décision textuelle incohérente."
            )

        violations = _list(
            item.get("violations"),
            field=f"result.architectures[{index}].violations",
        )
        if item.get("valid") is True and violations:
            raise FullArchitectureValidationError(
                "H10: architecture VALID avec violations."
            )
        if item.get("valid") is False and not violations:
            raise FullArchitectureValidationError(
                "H10: architecture INVALID sans violation."
            )

        if item.get("beam_search_applied") is not False:
            raise FullArchitectureValidationError(
                "H10 ne doit pas appliquer Beam Search."
            )
