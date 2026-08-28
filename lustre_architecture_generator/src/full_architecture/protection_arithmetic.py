from __future__ import annotations

import math
from typing import Any

from .hardware_schema import (
    HardwareSchemaError,
    validate_protection_profile,
)


class ProtectionArithmeticError(RuntimeError):
    """Erreur de calcul post-RAID/protection."""


def _finite(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ProtectionArithmeticError(
            f"{field}: nombre requis."
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ProtectionArithmeticError(
            f"{field}: nombre requis."
        ) from error

    if not math.isfinite(number):
        raise ProtectionArithmeticError(
            f"{field}: nombre fini requis."
        )

    if minimum is not None and number < minimum:
        raise ProtectionArithmeticError(
            f"{field}: valeur >= {minimum} requise."
        )

    return number


def _positive_int(
    value: Any,
    *,
    field: str,
) -> int:
    if isinstance(value, bool):
        raise ProtectionArithmeticError(
            f"{field}: entier > 0 requis."
        )

    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ProtectionArithmeticError(
            f"{field}: entier > 0 requis."
        ) from error

    if number <= 0:
        raise ProtectionArithmeticError(
            f"{field}: entier > 0 requis."
        )

    return number


def ceil_ratio(
    required: float,
    available: float,
) -> int:
    if required <= 0:
        return 0

    if available <= 0:
        raise ProtectionArithmeticError(
            "La ressource disponible doit être > 0."
        )

    return int(
        math.ceil(
            required / available
        )
    )


def _validate_candidate_package(
    candidate: dict[str, Any],
    *,
    expected_role: str,
) -> None:
    if not isinstance(candidate, dict):
        raise ProtectionArithmeticError(
            "candidate doit être un objet JSON."
        )

    role = candidate.get("role")

    if role != expected_role:
        raise ProtectionArithmeticError(
            f"Role candidat={role!r}, attendu={expected_role!r}."
        )

    pre_raid = candidate.get("pre_raid")

    if not isinstance(pre_raid, dict):
        raise ProtectionArithmeticError(
            "candidate.pre_raid absent."
        )

    if pre_raid.get(
        "is_lower_bound_only"
    ) is not True:
        raise ProtectionArithmeticError(
            "Le nombre pré-RAID doit être explicitement une borne basse."
        )

    _positive_int(
        pre_raid.get(
            "minimum_drive_count"
        ),
        field="pre_raid.minimum_drive_count",
    )


def _validate_profile(
    profile: dict[str, Any],
) -> None:
    try:
        validate_protection_profile(
            profile
        )
    except HardwareSchemaError as error:
        raise ProtectionArithmeticError(
            f"Profil de protection invalide : {error}"
        ) from error


def _group_geometry(
    profile: dict[str, Any],
) -> dict[str, int | float | str]:
    _validate_profile(
        profile
    )

    return {
        "profile_id": str(
            profile["id"]
        ),
        "raid_level": str(
            profile[
                "raid_level"
            ]
        ).upper(),
        "group_size": _positive_int(
            profile[
                "minimum_drives_per_group"
            ],
            field=(
                "protection."
                "minimum_drives_per_group"
            ),
        ),
        "data_drives": _positive_int(
            profile[
                "data_drives_per_group"
            ],
            field=(
                "protection."
                "data_drives_per_group"
            ),
        ),
        "parity_drives": int(
            profile[
                "parity_drives_per_group"
            ]
        ),
        "mirror_copies": _positive_int(
            profile[
                "mirror_copies"
            ],
            field=(
                "protection."
                "mirror_copies"
            ),
        ),
        "read_efficiency": _finite(
            profile[
                "read_efficiency"
            ],
            field=(
                "protection."
                "read_efficiency"
            ),
            minimum=0.0,
        ),
        "write_efficiency": _finite(
            profile[
                "write_efficiency"
            ],
            field=(
                "protection."
                "write_efficiency"
            ),
            minimum=0.0,
        ),
        "capacity_efficiency": _finite(
            profile[
                "capacity_efficiency"
            ],
            field=(
                "protection."
                "capacity_efficiency"
            ),
            minimum=0.0,
        ),
        "fault_tolerance": int(
            profile[
                "fault_tolerance_drives_per_group"
            ]
        ),
    }


def _derive_per_drive(
    *,
    aggregate: float,
    raw_minimum_count: int,
    field: str,
) -> float:
    aggregate_value = _finite(
        aggregate,
        field=field,
        minimum=0.0,
    )

    if raw_minimum_count <= 0:
        raise ProtectionArithmeticError(
            "raw_minimum_count doit être > 0."
        )

    return (
        aggregate_value
        / raw_minimum_count
    )


def _common_cost_power(
    *,
    pre_raid: dict[str, Any],
    raw_minimum_count: int,
    physical_drive_count: int,
) -> dict[str, float]:
    raw_cost = _finite(
        pre_raid.get(
            "drive_level_cost_usd"
        ),
        field="pre_raid.drive_level_cost_usd",
        minimum=0.0,
    )

    raw_power = _finite(
        pre_raid.get(
            "drive_level_power_w"
        ),
        field="pre_raid.drive_level_power_w",
        minimum=0.0,
    )

    per_drive_cost = (
        raw_cost
        / raw_minimum_count
    )

    per_drive_power = (
        raw_power
        / raw_minimum_count
    )

    return {
        "per_drive_cost_usd": per_drive_cost,
        "per_drive_power_w": per_drive_power,
        "protected_drive_cost_usd": (
            per_drive_cost
            * physical_drive_count
        ),
        "protected_drive_power_w": (
            per_drive_power
            * physical_drive_count
        ),
    }


def calculate_mdt_protection(
    *,
    candidate: dict[str, Any],
    protection_profile: dict[str, Any],
    requirement: dict[str, Any],
) -> dict[str, Any]:
    """
    Transforme une borne pré-RAID MDT en dimensionnement physique protégé.

    H5 ne choisit PAS le profil RAID. Il calcule uniquement les conséquences
    d'un profil donné.
    """

    _validate_candidate_package(
        candidate,
        expected_role="MDT",
    )

    geometry = _group_geometry(
        protection_profile
    )

    pre_raid = candidate[
        "pre_raid"
    ]

    raw_minimum_count = (
        _positive_int(
            pre_raid.get(
                "minimum_drive_count"
            ),
            field=(
                "pre_raid."
                "minimum_drive_count"
            ),
        )
    )

    per_drive_capacity = (
        _derive_per_drive(
            aggregate=pre_raid.get(
                "provided_capacity_tib"
            ),
            raw_minimum_count=(
                raw_minimum_count
            ),
            field=(
                "pre_raid."
                "provided_capacity_tib"
            ),
        )
    )

    per_drive_read_iops = (
        _derive_per_drive(
            aggregate=pre_raid.get(
                "provided_read_iops"
            ),
            raw_minimum_count=(
                raw_minimum_count
            ),
            field=(
                "pre_raid."
                "provided_read_iops"
            ),
        )
    )

    per_drive_write_iops = (
        _derive_per_drive(
            aggregate=pre_raid.get(
                "provided_write_iops"
            ),
            raw_minimum_count=(
                raw_minimum_count
            ),
            field=(
                "pre_raid."
                "provided_write_iops"
            ),
        )
    )

    required_capacity = _finite(
        requirement.get(
            "required_metadata_capacity_tib"
        ),
        field=(
            "MDT."
            "required_metadata_capacity_tib"
        ),
        minimum=0.0,
    )

    required_read_iops = _finite(
        requirement.get(
            "required_read_iops"
        ),
        field=(
            "MDT."
            "required_read_iops"
        ),
        minimum=0.0,
    )

    required_write_iops = _finite(
        requirement.get(
            "required_write_iops"
        ),
        field=(
            "MDT."
            "required_write_iops"
        ),
        minimum=0.0,
    )

    group_size = int(
        geometry[
            "group_size"
        ]
    )

    data_drives = int(
        geometry[
            "data_drives"
        ]
    )

    read_efficiency = float(
        geometry[
            "read_efficiency"
        ]
    )

    write_efficiency = float(
        geometry[
            "write_efficiency"
        ]
    )

    capacity_per_group = (
        per_drive_capacity
        * data_drives
    )

    read_iops_per_group = (
        per_drive_read_iops
        * group_size
        * read_efficiency
    )

    write_iops_per_group = (
        per_drive_write_iops
        * data_drives
        * write_efficiency
    )

    groups_by_lower_bound = (
        int(
            math.ceil(
                raw_minimum_count
                / group_size
            )
        )
    )

    groups_by_capacity = ceil_ratio(
        required_capacity,
        capacity_per_group,
    )

    groups_by_read = ceil_ratio(
        required_read_iops,
        read_iops_per_group,
    )

    groups_by_write = ceil_ratio(
        required_write_iops,
        write_iops_per_group,
    )

    group_count = max(
        1,
        groups_by_lower_bound,
        groups_by_capacity,
        groups_by_read,
        groups_by_write,
    )

    physical_drive_count = (
        group_count
        * group_size
    )

    provided_capacity = (
        group_count
        * capacity_per_group
    )

    provided_read_iops = (
        group_count
        * read_iops_per_group
    )

    provided_write_iops = (
        group_count
        * write_iops_per_group
    )

    cost_power = (
        _common_cost_power(
            pre_raid=pre_raid,
            raw_minimum_count=(
                raw_minimum_count
            ),
            physical_drive_count=(
                physical_drive_count
            ),
        )
    )

    return {
        "role": "MDT",
        "drive_id": candidate[
            "identity"
        ][
            "drive_id"
        ],
        "protection_profile_id": (
            geometry[
                "profile_id"
            ]
        ),
        "raid_level": geometry[
            "raid_level"
        ],
        "raw_minimum_drive_count": (
            raw_minimum_count
        ),
        "group_count": group_count,
        "group_size": group_size,
        "data_drives_per_group": (
            data_drives
        ),
        "parity_drives_per_group": (
            geometry[
                "parity_drives"
            ]
        ),
        "mirror_copies": geometry[
            "mirror_copies"
        ],
        "physical_drive_count": (
            physical_drive_count
        ),
        "counts_basis": {
            "lower_bound": groups_by_lower_bound,
            "capacity": groups_by_capacity,
            "read_iops": groups_by_read,
            "write_iops": groups_by_write,
        },
        "per_drive": {
            "capacity_tib": (
                per_drive_capacity
            ),
            "read_iops": (
                per_drive_read_iops
            ),
            "write_iops": (
                per_drive_write_iops
            ),
        },
        "provided": {
            "usable_capacity_tib": (
                provided_capacity
            ),
            "read_iops": (
                provided_read_iops
            ),
            "write_iops": (
                provided_write_iops
            ),
        },
        "requirements": {
            "usable_capacity_tib": (
                required_capacity
            ),
            "read_iops": (
                required_read_iops
            ),
            "write_iops": (
                required_write_iops
            ),
        },
        "satisfied": {
            "capacity": (
                provided_capacity
                >= required_capacity
            ),
            "read_iops": (
                provided_read_iops
                >= required_read_iops
            ),
            "write_iops": (
                provided_write_iops
                >= required_write_iops
            ),
        },
        "fault_tolerance_drives_per_group": (
            geometry[
                "fault_tolerance"
            ]
        ),
        **cost_power,
    }


def calculate_ost_protection(
    *,
    candidate: dict[str, Any],
    protection_profile: dict[str, Any],
    requirement: dict[str, Any],
) -> dict[str, Any]:
    """
    Transforme une borne pré-RAID OST en dimensionnement physique protégé.

    Les champs historiques `_gbps` du sizing/ranking sont interprétés ici
    selon le contrat downstream déjà figé : GB/s.
    """

    _validate_candidate_package(
        candidate,
        expected_role="OST",
    )

    geometry = _group_geometry(
        protection_profile
    )

    pre_raid = candidate[
        "pre_raid"
    ]

    raw_minimum_count = (
        _positive_int(
            pre_raid.get(
                "minimum_drive_count"
            ),
            field=(
                "pre_raid."
                "minimum_drive_count"
            ),
        )
    )

    per_drive_capacity = (
        _derive_per_drive(
            aggregate=pre_raid.get(
                "provided_capacity_tib"
            ),
            raw_minimum_count=(
                raw_minimum_count
            ),
            field=(
                "pre_raid."
                "provided_capacity_tib"
            ),
        )
    )

    per_drive_read = (
        _derive_per_drive(
            aggregate=pre_raid.get(
                "provided_read_bandwidth_gb_s"
            ),
            raw_minimum_count=(
                raw_minimum_count
            ),
            field=(
                "pre_raid."
                "provided_read_bandwidth_gb_s"
            ),
        )
    )

    per_drive_write = (
        _derive_per_drive(
            aggregate=pre_raid.get(
                "provided_write_bandwidth_gb_s"
            ),
            raw_minimum_count=(
                raw_minimum_count
            ),
            field=(
                "pre_raid."
                "provided_write_bandwidth_gb_s"
            ),
        )
    )

    required_capacity = _finite(
        requirement.get(
            "required_usable_capacity_tib"
        ),
        field=(
            "OST."
            "required_usable_capacity_tib"
        ),
        minimum=0.0,
    )

    required_read = _finite(
        requirement.get(
            "required_read_bandwidth_gbps"
        ),
        field=(
            "OST."
            "required_read_bandwidth_gbps"
        ),
        minimum=0.0,
    )

    required_write = _finite(
        requirement.get(
            "required_write_bandwidth_gbps"
        ),
        field=(
            "OST."
            "required_write_bandwidth_gbps"
        ),
        minimum=0.0,
    )

    required_total = _finite(
        requirement.get(
            "required_total_bandwidth_gbps"
        ),
        field=(
            "OST."
            "required_total_bandwidth_gbps"
        ),
        minimum=0.0,
    )

    group_size = int(
        geometry[
            "group_size"
        ]
    )

    data_drives = int(
        geometry[
            "data_drives"
        ]
    )

    read_efficiency = float(
        geometry[
            "read_efficiency"
        ]
    )

    write_efficiency = float(
        geometry[
            "write_efficiency"
        ]
    )

    capacity_per_group = (
        per_drive_capacity
        * data_drives
    )

    read_per_group = (
        per_drive_read
        * group_size
        * read_efficiency
    )

    write_per_group = (
        per_drive_write
        * data_drives
        * write_efficiency
    )

    total_per_group = (
        read_per_group
        + write_per_group
    )

    groups_by_lower_bound = (
        int(
            math.ceil(
                raw_minimum_count
                / group_size
            )
        )
    )

    groups_by_capacity = ceil_ratio(
        required_capacity,
        capacity_per_group,
    )

    groups_by_read = ceil_ratio(
        required_read,
        read_per_group,
    )

    groups_by_write = ceil_ratio(
        required_write,
        write_per_group,
    )

    groups_by_total = ceil_ratio(
        required_total,
        total_per_group,
    )

    group_count = max(
        1,
        groups_by_lower_bound,
        groups_by_capacity,
        groups_by_read,
        groups_by_write,
        groups_by_total,
    )

    physical_drive_count = (
        group_count
        * group_size
    )

    provided_capacity = (
        group_count
        * capacity_per_group
    )

    provided_read = (
        group_count
        * read_per_group
    )

    provided_write = (
        group_count
        * write_per_group
    )

    provided_total = (
        provided_read
        + provided_write
    )

    cost_power = (
        _common_cost_power(
            pre_raid=pre_raid,
            raw_minimum_count=(
                raw_minimum_count
            ),
            physical_drive_count=(
                physical_drive_count
            ),
        )
    )

    return {
        "role": "OST",
        "drive_id": candidate[
            "identity"
        ][
            "drive_id"
        ],
        "protection_profile_id": (
            geometry[
                "profile_id"
            ]
        ),
        "raid_level": geometry[
            "raid_level"
        ],
        "raw_minimum_drive_count": (
            raw_minimum_count
        ),
        "group_count": group_count,
        "group_size": group_size,
        "data_drives_per_group": (
            data_drives
        ),
        "parity_drives_per_group": (
            geometry[
                "parity_drives"
            ]
        ),
        "mirror_copies": geometry[
            "mirror_copies"
        ],
        "physical_drive_count": (
            physical_drive_count
        ),
        "counts_basis": {
            "lower_bound": groups_by_lower_bound,
            "capacity": groups_by_capacity,
            "read_bandwidth": groups_by_read,
            "write_bandwidth": groups_by_write,
            "total_bandwidth": groups_by_total,
        },
        "per_drive": {
            "capacity_tib": (
                per_drive_capacity
            ),
            "read_bandwidth_gb_s": (
                per_drive_read
            ),
            "write_bandwidth_gb_s": (
                per_drive_write
            ),
        },
        "provided": {
            "usable_capacity_tib": (
                provided_capacity
            ),
            "read_bandwidth_gb_s": (
                provided_read
            ),
            "write_bandwidth_gb_s": (
                provided_write
            ),
            "total_bandwidth_gb_s": (
                provided_total
            ),
        },
        "requirements": {
            "usable_capacity_tib": (
                required_capacity
            ),
            "read_bandwidth_gb_s": (
                required_read
            ),
            "write_bandwidth_gb_s": (
                required_write
            ),
            "total_bandwidth_gb_s": (
                required_total
            ),
        },
        "satisfied": {
            "capacity": (
                provided_capacity
                >= required_capacity
            ),
            "read_bandwidth": (
                provided_read
                >= required_read
            ),
            "write_bandwidth": (
                provided_write
                >= required_write
            ),
            "total_bandwidth": (
                provided_total
                >= required_total
            ),
        },
        "fault_tolerance_drives_per_group": (
            geometry[
                "fault_tolerance"
            ]
        ),
        **cost_power,
    }


def assert_protection_result_valid(
    result: dict[str, Any],
) -> None:
    if not isinstance(
        result,
        dict,
    ):
        raise ProtectionArithmeticError(
            "Le résultat protection doit être un objet."
        )

    role = result.get(
        "role"
    )

    if role not in {
        "MDT",
        "OST",
    }:
        raise ProtectionArithmeticError(
            "Role de protection invalide."
        )

    raw_count = _positive_int(
        result.get(
            "raw_minimum_drive_count"
        ),
        field=(
            "result."
            "raw_minimum_drive_count"
        ),
    )

    physical_count = _positive_int(
        result.get(
            "physical_drive_count"
        ),
        field=(
            "result."
            "physical_drive_count"
        ),
    )

    group_count = _positive_int(
        result.get(
            "group_count"
        ),
        field=(
            "result."
            "group_count"
        ),
    )

    group_size = _positive_int(
        result.get(
            "group_size"
        ),
        field=(
            "result."
            "group_size"
        ),
    )

    if physical_count < raw_count:
        raise ProtectionArithmeticError(
            "Le nombre physique protégé ne peut pas être inférieur à la borne pré-RAID."
        )

    if physical_count != (
        group_count
        * group_size
    ):
        raise ProtectionArithmeticError(
            "physical_drive_count != group_count * group_size."
        )

    satisfied = result.get(
        "satisfied"
    )

    if not isinstance(
        satisfied,
        dict,
    ):
        raise ProtectionArithmeticError(
            "Bloc satisfied absent."
        )

    failing = [
        name
        for name, value
        in satisfied.items()
        if value is not True
    ]

    if failing:
        raise ProtectionArithmeticError(
            "Protection insuffisante : "
            + ", ".join(
                failing
            )
        )


def enumerate_candidate_protections(
    *,
    candidate: dict[str, Any],
    protection_profiles: list[dict[str, Any]],
    requirement: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Calcule toutes les variantes de protection pour un candidat.

    La sélection du meilleur profil n'appartient pas à H5.
    """

    if not isinstance(
        protection_profiles,
        list,
    ) or not protection_profiles:
        raise ProtectionArithmeticError(
            "protection_profiles doit être une liste non vide."
        )

    role = candidate.get(
        "role"
    )

    results: list[
        dict[str, Any]
    ] = []

    for profile in protection_profiles:
        if not isinstance(
            profile,
            dict,
        ):
            raise ProtectionArithmeticError(
                "Chaque protection profile doit être un objet."
            )

        if role == "MDT":
            result = calculate_mdt_protection(
                candidate=candidate,
                protection_profile=profile,
                requirement=requirement,
            )
        elif role == "OST":
            result = calculate_ost_protection(
                candidate=candidate,
                protection_profile=profile,
                requirement=requirement,
            )
        else:
            raise ProtectionArithmeticError(
                f"Role candidat non supporté={role!r}."
            )

        assert_protection_result_valid(
            result
        )

        results.append(
            result
        )

    return results
