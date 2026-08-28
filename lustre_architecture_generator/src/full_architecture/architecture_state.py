from __future__ import annotations

import copy
import math
from typing import Any


STATE_SCHEMA_VERSION = "1.1"
STAGE_ORDER = (
    "EMPTY",
    "DRIVES_SELECTED",
    "PROTECTION_SELECTED",
    "SERVERS_SELECTED",
    "ENCLOSURES_SELECTED",
    "NETWORK_SELECTED",
    "COMPLETE",
)


class ArchitectureStateError(RuntimeError):
    """Erreur de transition ou de validation de l'ArchitectureState H7."""


def _finite(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ArchitectureStateError(f"{field}: nombre requis.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ArchitectureStateError(f"{field}: nombre requis.") from error
    if not math.isfinite(number) or number < minimum:
        raise ArchitectureStateError(f"{field}: valeur finie >= {minimum} requise.")
    return number


def _int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ArchitectureStateError(f"{field}: entier >= {minimum} requis.")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ArchitectureStateError(f"{field}: entier >= {minimum} requis.") from error
    if number < minimum:
        raise ArchitectureStateError(f"{field}: entier >= {minimum} requis.")
    return number


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArchitectureStateError(f"{field}: objet JSON requis.")
    return value


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ArchitectureStateError(f"{field}: liste JSON requise.")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArchitectureStateError(f"{field}: chaîne non vide requise.")
    return value.strip()


def _stage_index(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except ValueError as error:
        raise ArchitectureStateError(f"Stage inconnu: {stage!r}.") from error


def _assert_stage(state: dict[str, Any], expected: str) -> None:
    actual = state.get("stage")
    if actual != expected:
        raise ArchitectureStateError(
            f"Transition interdite: stage={actual!r}, attendu={expected!r}."
        )


def _append_trace(
    state: dict[str, Any],
    transition: str,
    details: dict[str, Any],
) -> None:
    trace = _list(state.get("trace"), field="state.trace")
    trace.append(
        {
            "sequence": len(trace) + 1,
            "transition": transition,
            "stage": state["stage"],
            "details": copy.deepcopy(details),
        }
    )


def _identity(candidate: dict[str, Any]) -> dict[str, Any]:
    identity = _mapping(candidate.get("identity"), field="candidate.identity")
    drive_id = _string(identity.get("drive_id"), field="candidate.identity.drive_id")
    return {
        "drive_id": drive_id,
        "drive_name": identity.get("drive_name"),
        "manufacturer": identity.get("manufacturer"),
        "series": identity.get("series"),
        "media_type": identity.get("media_type"),
        "catalog_id": identity.get("catalog_id"),
        "model_number": identity.get("model_number"),
    }


def _resources(path: dict[str, Any], role: str) -> dict[str, int]:
    if path.get("role") != role or path.get("compatible") is not True:
        raise ArchitectureStateError(f"Chemin hardware {role} non compatible/incohérent.")
    raw = _mapping(path.get("minimum_resources"), field=f"{role}.minimum_resources")
    return {
        "physical_drive_count": _int(raw.get("physical_drive_count"), field=f"{role}.physical_drive_count", minimum=1),
        "server_count": _int(raw.get("server_count"), field=f"{role}.server_count", minimum=1),
        "controller_count": _int(raw.get("controller_count"), field=f"{role}.controller_count", minimum=1),
        "enclosure_count": _int(raw.get("enclosure_count"), field=f"{role}.enclosure_count", minimum=0),
        "network_adapter_count": _int(raw.get("network_adapter_count"), field=f"{role}.network_adapter_count", minimum=1),
    }


def new_full_architecture_state(*, handoff: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(handoff, dict):
        raise ArchitectureStateError("handoff doit être un objet JSON.")
    case_id = _string(handoff.get("case_id"), field="handoff.case_id")
    requirements = _mapping(handoff.get("requirements"), field="handoff.requirements")
    provenance = _mapping(handoff.get("ranking_provenance"), field="handoff.ranking_provenance")

    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "case_id": case_id,
        "stage": "EMPTY",
        "source": {
            "handoff_schema_version": handoff.get("schema_version"),
            "requested_top_k": handoff.get("requested_top_k"),
            "ranking_provenance": copy.deepcopy(provenance),
        },
        "requirements": copy.deepcopy(requirements),
        "selected": {
            "mdt_drive": None,
            "ost_drive": None,
            "mdt_protection": None,
            "ost_protection": None,
            "mdt_hardware_path": None,
            "ost_hardware_path": None,
        },
        "counts": {
            "mdt_physical_drives": 0,
            "ost_physical_drives": 0,
            "mdt_count": 0,
            "ost_count": 0,
            "mds_count": 0,
            "oss_count": 0,
            "mdt_controller_count": 0,
            "ost_controller_count": 0,
            "mdt_enclosure_count": 0,
            "ost_enclosure_count": 0,
            "mdt_network_adapter_count": 0,
            "ost_network_adapter_count": 0,
        },
        "performance": {
            "metadata_capacity_tib": 0.0,
            "mdt_read_iops": 0.0,
            "mdt_write_iops": 0.0,
            "ost_usable_capacity_tib": 0.0,
            "ost_read_bandwidth_gb_s": 0.0,
            "ost_write_bandwidth_gb_s": 0.0,
            "ost_total_bandwidth_gb_s": 0.0,
        },
        "cost_power": {
            "mdt_drive_cost_usd": 0.0,
            "ost_drive_cost_usd": 0.0,
            "mdt_hardware_cost_usd": 0.0,
            "ost_hardware_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "mdt_drive_power_w": 0.0,
            "ost_drive_power_w": 0.0,
            "mdt_hardware_power_w": 0.0,
            "ost_hardware_power_w": 0.0,
            "total_power_w": 0.0,
        },
        "validation": {
            "is_complete": False,
            "is_valid": False,
            "status": "PENDING_FULL_VALIDATOR",
            "violations": [],
        },
        "trace": [],
    }
    validate_full_architecture_state(state)
    return state


def apply_drive_selection(
    *,
    state: dict[str, Any],
    mdt_candidate: dict[str, Any],
    ost_candidate: dict[str, Any],
) -> dict[str, Any]:
    _assert_stage(state, "EMPTY")
    if mdt_candidate.get("role") != "MDT" or ost_candidate.get("role") != "OST":
        raise ArchitectureStateError("Les rôles drive doivent être MDT et OST.")
    result = copy.deepcopy(state)
    result["selected"]["mdt_drive"] = _identity(mdt_candidate)
    result["selected"]["ost_drive"] = _identity(ost_candidate)
    result["stage"] = "DRIVES_SELECTED"
    _append_trace(
        result,
        "select_drives",
        {
            "mdt_drive_id": result["selected"]["mdt_drive"]["drive_id"],
            "ost_drive_id": result["selected"]["ost_drive"]["drive_id"],
        },
    )
    validate_full_architecture_state(result)
    return result


def _check_protection(
    role: str,
    selected_drive: dict[str, Any],
    protection: dict[str, Any],
) -> None:
    if protection.get("role") != role:
        raise ArchitectureStateError(f"{role}: rôle protection incohérent.")
    if protection.get("drive_id") != selected_drive.get("drive_id"):
        raise ArchitectureStateError(f"{role}: protection associée à un autre drive.")
    satisfied = _mapping(protection.get("satisfied"), field=f"{role}.protection.satisfied")
    if not satisfied or any(value is not True for value in satisfied.values()):
        raise ArchitectureStateError(f"{role}: protection non satisfaite.")


def apply_protection_selection(
    *,
    state: dict[str, Any],
    mdt_protection: dict[str, Any],
    ost_protection: dict[str, Any],
) -> dict[str, Any]:
    _assert_stage(state, "DRIVES_SELECTED")
    selected = state["selected"]
    _check_protection("MDT", selected["mdt_drive"], mdt_protection)
    _check_protection("OST", selected["ost_drive"], ost_protection)

    result = copy.deepcopy(state)
    result["selected"]["mdt_protection"] = copy.deepcopy(mdt_protection)
    result["selected"]["ost_protection"] = copy.deepcopy(ost_protection)
    result["counts"]["mdt_physical_drives"] = _int(mdt_protection.get("physical_drive_count"), field="MDT.physical_drive_count", minimum=1)
    result["counts"]["ost_physical_drives"] = _int(ost_protection.get("physical_drive_count"), field="OST.physical_drive_count", minimum=1)

    mp = _mapping(mdt_protection.get("provided"), field="MDT.provided")
    op = _mapping(ost_protection.get("provided"), field="OST.provided")
    perf = result["performance"]
    perf["metadata_capacity_tib"] = _finite(mp.get("usable_capacity_tib"), field="MDT.usable_capacity_tib")
    perf["mdt_read_iops"] = _finite(mp.get("read_iops"), field="MDT.read_iops")
    perf["mdt_write_iops"] = _finite(mp.get("write_iops"), field="MDT.write_iops")
    perf["ost_usable_capacity_tib"] = _finite(op.get("usable_capacity_tib"), field="OST.usable_capacity_tib")
    perf["ost_read_bandwidth_gb_s"] = _finite(op.get("read_bandwidth_gb_s"), field="OST.read_bandwidth_gb_s")
    perf["ost_write_bandwidth_gb_s"] = _finite(op.get("write_bandwidth_gb_s"), field="OST.write_bandwidth_gb_s")
    perf["ost_total_bandwidth_gb_s"] = _finite(op.get("total_bandwidth_gb_s"), field="OST.total_bandwidth_gb_s")

    cp = result["cost_power"]
    cp["mdt_drive_cost_usd"] = _finite(mdt_protection.get("protected_drive_cost_usd"), field="MDT.protected_drive_cost_usd")
    cp["ost_drive_cost_usd"] = _finite(ost_protection.get("protected_drive_cost_usd"), field="OST.protected_drive_cost_usd")
    cp["mdt_drive_power_w"] = _finite(mdt_protection.get("protected_drive_power_w"), field="MDT.protected_drive_power_w")
    cp["ost_drive_power_w"] = _finite(ost_protection.get("protected_drive_power_w"), field="OST.protected_drive_power_w")

    result["stage"] = "PROTECTION_SELECTED"
    _append_trace(
        result,
        "select_protection",
        {
            "mdt_profile": mdt_protection.get("protection_profile_id"),
            "ost_profile": ost_protection.get("protection_profile_id"),
            "mdt_physical_drives": result["counts"]["mdt_physical_drives"],
            "ost_physical_drives": result["counts"]["ost_physical_drives"],
        },
    )
    validate_full_architecture_state(result)
    return result


def _check_path_against_protection(
    role: str,
    path: dict[str, Any],
    protection: dict[str, Any],
) -> dict[str, int]:
    resources = _resources(path, role)
    if path.get("drive_id") != protection.get("drive_id"):
        raise ArchitectureStateError(f"{role}: chemin associé à un autre drive.")
    if path.get("protection_profile_id") != protection.get("protection_profile_id"):
        raise ArchitectureStateError(f"{role}: chemin associé à une autre protection.")
    if resources["physical_drive_count"] != int(protection["physical_drive_count"]):
        raise ArchitectureStateError(f"{role}: nombre physique incohérent entre protection et path.")
    return resources


def apply_server_selection(
    *,
    state: dict[str, Any],
    mdt_path: dict[str, Any],
    ost_path: dict[str, Any],
) -> dict[str, Any]:
    _assert_stage(state, "PROTECTION_SELECTED")
    mdt_prot = state["selected"]["mdt_protection"]
    ost_prot = state["selected"]["ost_protection"]
    mr = _check_path_against_protection("MDT", mdt_path, mdt_prot)
    or_ = _check_path_against_protection("OST", ost_path, ost_prot)

    result = copy.deepcopy(state)
    result["selected"]["mdt_hardware_path"] = copy.deepcopy(mdt_path)
    result["selected"]["ost_hardware_path"] = copy.deepcopy(ost_path)
    result["counts"]["mds_count"] = mr["server_count"]
    result["counts"]["oss_count"] = or_["server_count"]
    # H7: 1 protection group = 1 target-group placeholder. H10 must validate/refine.
    result["counts"]["mdt_count"] = _int(mdt_prot.get("group_count"), field="MDT.group_count", minimum=1)
    result["counts"]["ost_count"] = _int(ost_prot.get("group_count"), field="OST.group_count", minimum=1)
    result["stage"] = "SERVERS_SELECTED"
    _append_trace(
        result,
        "select_servers",
        {
            "mds_server_id": mdt_path.get("server_id"),
            "oss_server_id": ost_path.get("server_id"),
            "mds_count": result["counts"]["mds_count"],
            "oss_count": result["counts"]["oss_count"],
        },
    )
    validate_full_architecture_state(result)
    return result


def apply_storage_fabric_selection(*, state: dict[str, Any]) -> dict[str, Any]:
    _assert_stage(state, "SERVERS_SELECTED")
    mdt_path = state["selected"]["mdt_hardware_path"]
    ost_path = state["selected"]["ost_hardware_path"]
    mr = _resources(mdt_path, "MDT")
    or_ = _resources(ost_path, "OST")

    result = copy.deepcopy(state)
    counts = result["counts"]
    counts["mdt_controller_count"] = mr["controller_count"]
    counts["ost_controller_count"] = or_["controller_count"]
    counts["mdt_enclosure_count"] = mr["enclosure_count"]
    counts["ost_enclosure_count"] = or_["enclosure_count"]
    result["stage"] = "ENCLOSURES_SELECTED"
    _append_trace(
        result,
        "select_storage_fabric",
        {
            "mdt_controller_id": mdt_path.get("controller_id"),
            "ost_controller_id": ost_path.get("controller_id"),
            "mdt_enclosure_id": mdt_path.get("enclosure_id"),
            "ost_enclosure_id": ost_path.get("enclosure_id"),
        },
    )
    validate_full_architecture_state(result)
    return result


def apply_network_ha_selection(*, state: dict[str, Any]) -> dict[str, Any]:
    _assert_stage(state, "ENCLOSURES_SELECTED")
    mdt_path = state["selected"]["mdt_hardware_path"]
    ost_path = state["selected"]["ost_hardware_path"]
    mr = _resources(mdt_path, "MDT")
    or_ = _resources(ost_path, "OST")

    result = copy.deepcopy(state)
    counts = result["counts"]
    counts["mdt_network_adapter_count"] = mr["network_adapter_count"]
    counts["ost_network_adapter_count"] = or_["network_adapter_count"]
    cp = result["cost_power"]
    cp["mdt_hardware_cost_usd"] = _finite(mdt_path.get("component_cost_lower_bound_usd"), field="MDT.hardware_cost")
    cp["ost_hardware_cost_usd"] = _finite(ost_path.get("component_cost_lower_bound_usd"), field="OST.hardware_cost")
    cp["mdt_hardware_power_w"] = _finite(mdt_path.get("component_power_lower_bound_w"), field="MDT.hardware_power")
    cp["ost_hardware_power_w"] = _finite(ost_path.get("component_power_lower_bound_w"), field="OST.hardware_power")
    result["stage"] = "NETWORK_SELECTED"
    _append_trace(
        result,
        "select_network_and_ha",
        {
            "mdt_network_id": mdt_path.get("network_id"),
            "ost_network_id": ost_path.get("network_id"),
            "mdt_ha_profile_id": mdt_path.get("ha_profile_id"),
            "ost_ha_profile_id": ost_path.get("ha_profile_id"),
        },
    )
    validate_full_architecture_state(result)
    return result


def finalize_architecture_state(*, state: dict[str, Any]) -> dict[str, Any]:
    _assert_stage(state, "NETWORK_SELECTED")
    result = copy.deepcopy(state)
    cp = result["cost_power"]
    cp["total_cost_usd"] = sum(
        _finite(cp[field], field=f"cost_power.{field}")
        for field in (
            "mdt_drive_cost_usd",
            "ost_drive_cost_usd",
            "mdt_hardware_cost_usd",
            "ost_hardware_cost_usd",
        )
    )
    cp["total_power_w"] = sum(
        _finite(cp[field], field=f"cost_power.{field}")
        for field in (
            "mdt_drive_power_w",
            "ost_drive_power_w",
            "mdt_hardware_power_w",
            "ost_hardware_power_w",
        )
    )
    result["stage"] = "COMPLETE"
    result["validation"].update(
        {
            "is_complete": True,
            "is_valid": False,
            "status": "PENDING_FULL_VALIDATOR",
        }
    )
    _append_trace(
        result,
        "finalize_state",
        {
            "total_cost_usd": cp["total_cost_usd"],
            "total_power_w": cp["total_power_w"],
        },
    )
    validate_full_architecture_state(result)
    return result


def build_complete_state_from_choices(
    *,
    handoff: dict[str, Any],
    mdt_candidate: dict[str, Any],
    ost_candidate: dict[str, Any],
    mdt_protection: dict[str, Any],
    ost_protection: dict[str, Any],
    mdt_path: dict[str, Any],
    ost_path: dict[str, Any],
) -> dict[str, Any]:
    state = new_full_architecture_state(handoff=handoff)
    state = apply_drive_selection(state=state, mdt_candidate=mdt_candidate, ost_candidate=ost_candidate)
    state = apply_protection_selection(state=state, mdt_protection=mdt_protection, ost_protection=ost_protection)
    state = apply_server_selection(state=state, mdt_path=mdt_path, ost_path=ost_path)
    state = apply_storage_fabric_selection(state=state)
    state = apply_network_ha_selection(state=state)
    return finalize_architecture_state(state=state)


def validate_full_architecture_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise ArchitectureStateError("state doit être un objet JSON.")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ArchitectureStateError("schema_version ArchitectureState incorrecte.")
    _string(state.get("case_id"), field="state.case_id")
    stage = _string(state.get("stage"), field="state.stage").upper()
    stage_index = _stage_index(stage)

    selected = _mapping(state.get("selected"), field="state.selected")
    counts = _mapping(state.get("counts"), field="state.counts")
    performance = _mapping(state.get("performance"), field="state.performance")
    cost_power = _mapping(state.get("cost_power"), field="state.cost_power")
    validation = _mapping(state.get("validation"), field="state.validation")
    trace = _list(state.get("trace"), field="state.trace")

    for field in (
        "mdt_drive",
        "ost_drive",
        "mdt_protection",
        "ost_protection",
        "mdt_hardware_path",
        "ost_hardware_path",
    ):
        if field not in selected:
            raise ArchitectureStateError(f"state.selected.{field}: champ manquant.")

    for field in (
        "mdt_physical_drives",
        "ost_physical_drives",
        "mdt_count",
        "ost_count",
        "mds_count",
        "oss_count",
        "mdt_controller_count",
        "ost_controller_count",
        "mdt_enclosure_count",
        "ost_enclosure_count",
        "mdt_network_adapter_count",
        "ost_network_adapter_count",
    ):
        _int(counts.get(field), field=f"state.counts.{field}", minimum=0)

    for field in (
        "metadata_capacity_tib",
        "mdt_read_iops",
        "mdt_write_iops",
        "ost_usable_capacity_tib",
        "ost_read_bandwidth_gb_s",
        "ost_write_bandwidth_gb_s",
        "ost_total_bandwidth_gb_s",
    ):
        _finite(performance.get(field), field=f"state.performance.{field}")

    for field in (
        "mdt_drive_cost_usd",
        "ost_drive_cost_usd",
        "mdt_hardware_cost_usd",
        "ost_hardware_cost_usd",
        "total_cost_usd",
        "mdt_drive_power_w",
        "ost_drive_power_w",
        "mdt_hardware_power_w",
        "ost_hardware_power_w",
        "total_power_w",
    ):
        _finite(cost_power.get(field), field=f"state.cost_power.{field}")

    if not isinstance(validation.get("is_complete"), bool):
        raise ArchitectureStateError("validation.is_complete doit être booléen.")
    if not isinstance(validation.get("is_valid"), bool):
        raise ArchitectureStateError("validation.is_valid doit être booléen.")
    _list(validation.get("violations"), field="validation.violations")

    if stage_index >= _stage_index("DRIVES_SELECTED"):
        if not isinstance(selected["mdt_drive"], dict) or not isinstance(selected["ost_drive"], dict):
            raise ArchitectureStateError("Drives MDT/OST absents après DRIVES_SELECTED.")
    if stage_index >= _stage_index("PROTECTION_SELECTED"):
        if not isinstance(selected["mdt_protection"], dict) or not isinstance(selected["ost_protection"], dict):
            raise ArchitectureStateError("Protections MDT/OST absentes.")
        if counts["mdt_physical_drives"] <= 0 or counts["ost_physical_drives"] <= 0:
            raise ArchitectureStateError("Nombres physiques MDT/OST invalides.")
    if stage_index >= _stage_index("SERVERS_SELECTED"):
        if not isinstance(selected["mdt_hardware_path"], dict) or not isinstance(selected["ost_hardware_path"], dict):
            raise ArchitectureStateError("Chemins hardware MDT/OST absents.")
        if counts["mds_count"] <= 0 or counts["oss_count"] <= 0:
            raise ArchitectureStateError("MDS/OSS counts invalides.")
    if stage == "COMPLETE":
        if validation["is_complete"] is not True:
            raise ArchitectureStateError("COMPLETE exige is_complete=True.")

        status = validation.get("status")

        if status == "PENDING_FULL_VALIDATOR":
            if validation["is_valid"] is not False:
                raise ArchitectureStateError(
                    "PENDING_FULL_VALIDATOR exige is_valid=False."
                )
        elif status == "VALIDATED":
            if validation["is_valid"] is not True:
                raise ArchitectureStateError(
                    "VALIDATED exige is_valid=True."
                )
            if validation.get("violations"):
                raise ArchitectureStateError(
                    "VALIDATED exige violations vide."
                )
        elif status == "INVALID":
            if validation["is_valid"] is not False:
                raise ArchitectureStateError(
                    "INVALID exige is_valid=False."
                )
            if not validation.get("violations"):
                raise ArchitectureStateError(
                    "INVALID exige au moins une violation."
                )
        else:
            raise ArchitectureStateError(
                "Statut COMPLETE non supporté: " + repr(status)
            )

    for expected, entry in enumerate(trace, start=1):
        if not isinstance(entry, dict):
            raise ArchitectureStateError("Chaque entrée trace doit être un objet.")
        if _int(entry.get("sequence"), field="trace.sequence", minimum=1) != expected:
            raise ArchitectureStateError("Séquence trace non contiguë.")
