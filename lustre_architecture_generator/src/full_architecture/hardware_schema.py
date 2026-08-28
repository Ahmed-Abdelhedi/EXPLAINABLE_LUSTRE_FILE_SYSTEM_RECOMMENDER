from __future__ import annotations

import math
from typing import Any


SCHEMA_VERSION = "1.0"

SERVER_ROLES = {"MDS", "OSS", "BOTH"}
CONTROLLER_TYPES = {"HBA", "RAID", "NVME_SWITCH"}
PROTOCOLS = {"SATA", "SAS", "NVME"}
FORM_FACTORS = {"FF_2_5", "FF_3_5", "FF_U2", "FF_U3", "FF_E1S", "FF_E3S"}
NETWORK_FABRICS = {"ETHERNET", "INFINIBAND"}
HA_MODES = {"NONE", "ACTIVE_PASSIVE", "ACTIVE_ACTIVE"}
ARCHITECTURE_STAGES = {
    "EMPTY",
    "DRIVES_SELECTED",
    "PROTECTION_SELECTED",
    "SERVERS_SELECTED",
    "ENCLOSURES_SELECTED",
    "NETWORK_SELECTED",
    "COMPLETE",
}


class HardwareSchemaError(ValueError):
    """Erreur de validation d'un objet de la Full Architecture Layer."""


def _require_dict(
    parent: dict[str, Any],
    key: str,
    context: str,
) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise HardwareSchemaError(
            f"{context}.{key}: objet JSON requis."
        )
    return value


def _require_list(
    parent: dict[str, Any],
    key: str,
    context: str,
) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise HardwareSchemaError(
            f"{context}.{key}: liste JSON requise."
        )
    return value


def _require_string(
    parent: dict[str, Any],
    key: str,
    context: str,
) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HardwareSchemaError(
            f"{context}.{key}: chaîne non vide requise."
        )
    return value.strip()


def _require_bool(
    parent: dict[str, Any],
    key: str,
    context: str,
) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise HardwareSchemaError(
            f"{context}.{key}: booléen requis."
        )
    return value


def _require_number(
    parent: dict[str, Any],
    key: str,
    context: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    value = parent.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise HardwareSchemaError(
            f"{context}.{key}: nombre fini requis."
        )

    number = float(value)

    if strictly_positive and number <= 0:
        raise HardwareSchemaError(
            f"{context}.{key}: valeur strictement positive requise."
        )

    if minimum is not None and number < minimum:
        raise HardwareSchemaError(
            f"{context}.{key}: valeur >= {minimum} requise."
        )

    return number


def _require_int(
    parent: dict[str, Any],
    key: str,
    context: str,
    *,
    minimum: int = 0,
) -> int:
    value = parent.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise HardwareSchemaError(
            f"{context}.{key}: entier requis."
        )

    if value < minimum:
        raise HardwareSchemaError(
            f"{context}.{key}: valeur >= {minimum} requise."
        )

    return value


def _validate_unique_strings(
    values: list[Any],
    *,
    allowed: set[str] | None,
    context: str,
) -> list[str]:
    result: list[str] = []

    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise HardwareSchemaError(
                f"{context}[{index}]: chaîne non vide requise."
            )

        normalized = value.strip().upper()

        if allowed is not None and normalized not in allowed:
            raise HardwareSchemaError(
                f"{context}[{index}]: valeur non supportée={normalized!r}."
            )

        if normalized in result:
            raise HardwareSchemaError(
                f"{context}: valeur dupliquée={normalized!r}."
            )

        result.append(normalized)

    if not result:
        raise HardwareSchemaError(
            f"{context}: au moins une valeur requise."
        )

    return result


def _validate_common_component(
    component: dict[str, Any],
    *,
    context: str,
) -> None:
    _require_string(component, "id", context)
    _require_string(component, "name", context)
    _require_string(component, "manufacturer", context)
    _require_number(
        component,
        "price_usd",
        context,
        minimum=0.0,
    )
    _require_number(
        component,
        "power_w",
        context,
        minimum=0.0,
    )


def validate_server_profile(
    server: dict[str, Any],
) -> None:
    context = "server"
    _validate_common_component(server, context=context)

    role = _require_string(
        server,
        "role",
        context,
    ).upper()

    if role not in SERVER_ROLES:
        raise HardwareSchemaError(
            f"{context}.role non supporté={role!r}."
        )

    _require_int(
        server,
        "cpu_cores",
        context,
        minimum=1,
    )
    _require_number(
        server,
        "memory_gib",
        context,
        strictly_positive=True,
    )
    _require_int(
        server,
        "pcie_slot_count",
        context,
        minimum=0,
    )
    _require_int(
        server,
        "pcie_lane_budget",
        context,
        minimum=0,
    )

    bay_capacity = _require_dict(
        server,
        "drive_bays",
        context,
    )

    for form_factor, count in bay_capacity.items():
        normalized = str(form_factor).upper()

        if normalized not in FORM_FACTORS:
            raise HardwareSchemaError(
                f"{context}.drive_bays: form factor non supporté={normalized!r}."
            )

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise HardwareSchemaError(
                f"{context}.drive_bays.{normalized}: entier >= 0 requis."
            )

    if not bay_capacity:
        raise HardwareSchemaError(
            f"{context}.drive_bays ne doit pas être vide."
        )

    _validate_unique_strings(
        _require_list(
            server,
            "native_drive_protocols",
            context,
        ),
        allowed=PROTOCOLS,
        context=f"{context}.native_drive_protocols",
    )

    _validate_unique_strings(
        _require_list(
            server,
            "network_interfaces",
            context,
        ),
        allowed=None,
        context=f"{context}.network_interfaces",
    )

    _require_bool(
        server,
        "supports_dual_psu",
        context,
    )


def validate_controller_profile(
    controller: dict[str, Any],
) -> None:
    context = "controller"
    _validate_common_component(
        controller,
        context=context,
    )

    controller_type = _require_string(
        controller,
        "controller_type",
        context,
    ).upper()

    if controller_type not in CONTROLLER_TYPES:
        raise HardwareSchemaError(
            f"{context}.controller_type non supporté={controller_type!r}."
        )

    _validate_unique_strings(
        _require_list(
            controller,
            "supported_protocols",
            context,
        ),
        allowed=PROTOCOLS,
        context=f"{context}.supported_protocols",
    )

    _require_int(
        controller,
        "port_count",
        context,
        minimum=1,
    )
    _require_number(
        controller,
        "max_aggregate_bandwidth_gb_s",
        context,
        strictly_positive=True,
    )
    _require_int(
        controller,
        "pcie_gen",
        context,
        minimum=1,
    )
    _require_int(
        controller,
        "pcie_lanes",
        context,
        minimum=1,
    )
    _require_bool(
        controller,
        "supports_multipath",
        context,
    )


def validate_enclosure_profile(
    enclosure: dict[str, Any],
) -> None:
    context = "enclosure"
    _validate_common_component(
        enclosure,
        context=context,
    )

    _validate_unique_strings(
        _require_list(
            enclosure,
            "supported_protocols",
            context,
        ),
        allowed=PROTOCOLS,
        context=f"{context}.supported_protocols",
    )

    _validate_unique_strings(
        _require_list(
            enclosure,
            "supported_form_factors",
            context,
        ),
        allowed=FORM_FACTORS,
        context=f"{context}.supported_form_factors",
    )

    _require_int(
        enclosure,
        "drive_bay_count",
        context,
        minimum=1,
    )
    _require_int(
        enclosure,
        "uplink_count",
        context,
        minimum=1,
    )
    _require_number(
        enclosure,
        "uplink_bandwidth_gb_s",
        context,
        strictly_positive=True,
    )
    _require_bool(
        enclosure,
        "supports_redundant_paths",
        context,
    )


def validate_network_profile(
    network: dict[str, Any],
) -> None:
    context = "network"
    _validate_common_component(
        network,
        context=context,
    )

    fabric = _require_string(
        network,
        "fabric",
        context,
    ).upper()

    if fabric not in NETWORK_FABRICS:
        raise HardwareSchemaError(
            f"{context}.fabric non supporté={fabric!r}."
        )

    _require_number(
        network,
        "link_speed_gbit_s",
        context,
        strictly_positive=True,
    )
    _require_int(
        network,
        "ports_per_adapter",
        context,
        minimum=1,
    )
    _require_number(
        network,
        "usable_efficiency",
        context,
        strictly_positive=True,
    )

    efficiency = float(
        network["usable_efficiency"]
    )
    if efficiency > 1.0:
        raise HardwareSchemaError(
            f"{context}.usable_efficiency doit être <= 1."
        )

    _require_bool(
        network,
        "supports_redundant_fabric",
        context,
    )


def validate_protection_profile(
    profile: dict[str, Any],
) -> None:
    context = "protection"

    _require_string(
        profile,
        "id",
        context,
    )
    _require_string(
        profile,
        "name",
        context,
    )

    level = _require_string(
        profile,
        "raid_level",
        context,
    ).upper()

    if level not in {
        "RAID1",
        "RAID10",
        "RAID5",
        "RAID6",
    }:
        raise HardwareSchemaError(
            f"{context}.raid_level non supporté={level!r}."
        )

    minimum_drives = _require_int(
        profile,
        "minimum_drives_per_group",
        context,
        minimum=2,
    )
    data_drives = _require_int(
        profile,
        "data_drives_per_group",
        context,
        minimum=1,
    )
    parity_drives = _require_int(
        profile,
        "parity_drives_per_group",
        context,
        minimum=0,
    )
    mirror_copies = _require_int(
        profile,
        "mirror_copies",
        context,
        minimum=1,
    )

    if level == "RAID1":
        if minimum_drives != 2 or data_drives != 1 or mirror_copies != 2:
            raise HardwareSchemaError(
                "RAID1 exige minimum=2, data=1, mirror_copies=2."
            )
        if parity_drives != 0:
            raise HardwareSchemaError(
                "RAID1 ne doit pas utiliser de parity_drives."
            )

    elif level == "RAID10":
        if minimum_drives < 4 or minimum_drives % 2 != 0:
            raise HardwareSchemaError(
                "RAID10 exige au moins 4 drives et un nombre pair."
            )
        if data_drives * mirror_copies != minimum_drives:
            raise HardwareSchemaError(
                "RAID10: data_drives * mirror_copies doit égaler le groupe."
            )
        if parity_drives != 0:
            raise HardwareSchemaError(
                "RAID10 ne doit pas utiliser de parity_drives."
            )

    elif level == "RAID5":
        if minimum_drives < 3 or parity_drives != 1:
            raise HardwareSchemaError(
                "RAID5 exige >=3 drives et 1 parité."
            )
        if data_drives + parity_drives != minimum_drives:
            raise HardwareSchemaError(
                "RAID5: data + parity doit égaler le groupe."
            )

    elif level == "RAID6":
        if minimum_drives < 4 or parity_drives != 2:
            raise HardwareSchemaError(
                "RAID6 exige >=4 drives et 2 parités."
            )
        if data_drives + parity_drives != minimum_drives:
            raise HardwareSchemaError(
                "RAID6: data + parity doit égaler le groupe."
            )

    _require_number(
        profile,
        "read_efficiency",
        context,
        strictly_positive=True,
    )
    _require_number(
        profile,
        "write_efficiency",
        context,
        strictly_positive=True,
    )
    _require_number(
        profile,
        "capacity_efficiency",
        context,
        strictly_positive=True,
    )

    for key in (
        "read_efficiency",
        "write_efficiency",
        "capacity_efficiency",
    ):
        if float(profile[key]) > 1.0:
            raise HardwareSchemaError(
                f"{context}.{key} doit être <= 1."
            )

    _require_int(
        profile,
        "fault_tolerance_drives_per_group",
        context,
        minimum=1,
    )


def validate_ha_profile(
    profile: dict[str, Any],
) -> None:
    context = "ha"

    _require_string(
        profile,
        "id",
        context,
    )
    _require_string(
        profile,
        "name",
        context,
    )

    mode = _require_string(
        profile,
        "mode",
        context,
    ).upper()

    if mode not in HA_MODES:
        raise HardwareSchemaError(
            f"{context}.mode non supporté={mode!r}."
        )

    _require_int(
        profile,
        "minimum_nodes_per_role",
        context,
        minimum=1,
    )
    _require_bool(
        profile,
        "requires_shared_storage",
        context,
    )
    _require_bool(
        profile,
        "requires_redundant_network",
        context,
    )

    if mode == "NONE":
        if profile["minimum_nodes_per_role"] != 1:
            raise HardwareSchemaError(
                "HA NONE exige minimum_nodes_per_role=1."
            )

    if mode != "NONE" and profile["minimum_nodes_per_role"] < 2:
        raise HardwareSchemaError(
            "Un mode HA actif exige au moins 2 nodes par rôle."
        )


def new_architecture_state(
    *,
    case_id: str,
) -> dict[str, Any]:
    if not isinstance(case_id, str) or not case_id.strip():
        raise HardwareSchemaError(
            "case_id: chaîne non vide requise."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id.strip(),
        "stage": "EMPTY",
        "selected": {
            "mdt_drive": None,
            "ost_drive": None,
            "mdt_protection": None,
            "ost_protection": None,
            "mds_server": None,
            "oss_server": None,
            "controller": None,
            "enclosure": None,
            "network": None,
            "ha": None,
        },
        "counts": {
            "mdt_physical_drives": 0,
            "ost_physical_drives": 0,
            "mdt_count": 0,
            "ost_count": 0,
            "mds_count": 0,
            "oss_count": 0,
            "controller_count": 0,
            "enclosure_count": 0,
            "network_adapter_count": 0,
        },
        "totals": {
            "usable_capacity_tib": 0.0,
            "mdt_read_iops": 0.0,
            "mdt_write_iops": 0.0,
            "ost_read_bandwidth_gb_s": 0.0,
            "ost_write_bandwidth_gb_s": 0.0,
            "cost_usd": 0.0,
            "power_w": 0.0,
        },
        "validation": {
            "is_complete": False,
            "is_valid": False,
            "violations": [],
        },
        "trace": [],
    }


def validate_architecture_state(
    state: dict[str, Any],
) -> None:
    context = "architecture_state"

    if state.get("schema_version") != SCHEMA_VERSION:
        raise HardwareSchemaError(
            f"{context}.schema_version incorrecte."
        )

    _require_string(
        state,
        "case_id",
        context,
    )

    stage = _require_string(
        state,
        "stage",
        context,
    ).upper()

    if stage not in ARCHITECTURE_STAGES:
        raise HardwareSchemaError(
            f"{context}.stage non supporté={stage!r}."
        )

    _require_dict(
        state,
        "selected",
        context,
    )
    counts = _require_dict(
        state,
        "counts",
        context,
    )
    totals = _require_dict(
        state,
        "totals",
        context,
    )
    validation = _require_dict(
        state,
        "validation",
        context,
    )
    _require_list(
        state,
        "trace",
        context,
    )

    required_count_fields = {
        "mdt_physical_drives",
        "ost_physical_drives",
        "mdt_count",
        "ost_count",
        "mds_count",
        "oss_count",
        "controller_count",
        "enclosure_count",
        "network_adapter_count",
    }

    missing_counts = (
        required_count_fields
        - set(counts)
    )

    if missing_counts:
        raise HardwareSchemaError(
            f"{context}.counts: champs manquants={sorted(missing_counts)}."
        )

    for field in required_count_fields:
        _require_int(
            counts,
            field,
            f"{context}.counts",
            minimum=0,
        )

    required_total_fields = {
        "usable_capacity_tib",
        "mdt_read_iops",
        "mdt_write_iops",
        "ost_read_bandwidth_gb_s",
        "ost_write_bandwidth_gb_s",
        "cost_usd",
        "power_w",
    }

    missing_totals = (
        required_total_fields
        - set(totals)
    )

    if missing_totals:
        raise HardwareSchemaError(
            f"{context}.totals: champs manquants={sorted(missing_totals)}."
        )

    for field in required_total_fields:
        _require_number(
            totals,
            field,
            f"{context}.totals",
            minimum=0.0,
        )

    _require_bool(
        validation,
        "is_complete",
        f"{context}.validation",
    )
    _require_bool(
        validation,
        "is_valid",
        f"{context}.validation",
    )
    _require_list(
        validation,
        "violations",
        f"{context}.validation",
    )

    if (
        validation["is_valid"]
        and not validation["is_complete"]
    ):
        raise HardwareSchemaError(
            "Un ArchitectureState ne peut pas être validé avant d'être complet."
        )


def validate_hardware_catalog_bundle(
    bundle: dict[str, Any],
) -> None:
    if not isinstance(bundle, dict):
        raise HardwareSchemaError(
            "Le bundle hardware doit être un objet JSON."
        )

    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise HardwareSchemaError(
            "schema_version hardware incorrecte."
        )

    validators = {
        "servers": validate_server_profile,
        "controllers": validate_controller_profile,
        "enclosures": validate_enclosure_profile,
        "networks": validate_network_profile,
        "protection_profiles": validate_protection_profile,
        "ha_profiles": validate_ha_profile,
    }

    for section, validator in validators.items():
        values = bundle.get(section)

        if not isinstance(values, list) or not values:
            raise HardwareSchemaError(
                f"{section}: liste non vide requise."
            )

        seen_ids: set[str] = set()

        for index, value in enumerate(values):
            if not isinstance(value, dict):
                raise HardwareSchemaError(
                    f"{section}[{index}]: objet requis."
                )

            validator(value)

            component_id = value.get("id")
            if component_id in seen_ids:
                raise HardwareSchemaError(
                    f"{section}: id dupliqué={component_id!r}."
                )

            seen_ids.add(str(component_id))
