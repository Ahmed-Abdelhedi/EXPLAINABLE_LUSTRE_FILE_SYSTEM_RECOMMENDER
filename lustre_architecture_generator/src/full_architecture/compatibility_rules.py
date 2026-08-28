from __future__ import annotations

import math
import re
from typing import Any


ATTACHMENT_MODES = {"DIRECT", "ENCLOSURE"}

FORM_FACTOR_ALIASES = {
    "FF_2_5": "FF_2_5",
    "2.5": "FF_2_5",
    "2.5IN": "FF_2_5",
    "2.5-INCH": "FF_2_5",
    "FF_3_5": "FF_3_5",
    "3.5": "FF_3_5",
    "3.5IN": "FF_3_5",
    "3.5-INCH": "FF_3_5",
    "FF_U2": "FF_U2",
    "U.2": "FF_U2",
    "U2": "FF_U2",
    "FF_U3": "FF_U3",
    "U.3": "FF_U3",
    "U3": "FF_U3",
    "FF_E1S": "FF_E1S",
    "E1.S": "FF_E1S",
    "E1S": "FF_E1S",
    "FF_E3S": "FF_E3S",
    "E3.S": "FF_E3S",
    "E3S": "FF_E3S",
}

TWO_POINT_FIVE_FAMILY = {
    "FF_2_5",
    "FF_U2",
    "FF_U3",
}


class CompatibilityRuleError(RuntimeError):
    """Erreur du moteur déterministe de compatibilité hardware H6."""


def _finite(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise CompatibilityRuleError(
            f"{field}: nombre requis."
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise CompatibilityRuleError(
            f"{field}: nombre requis."
        ) from error

    if not math.isfinite(number):
        raise CompatibilityRuleError(
            f"{field}: nombre fini requis."
        )

    if minimum is not None and number < minimum:
        raise CompatibilityRuleError(
            f"{field}: valeur >= {minimum} requise."
        )

    return number


def _positive_int(
    value: Any,
    *,
    field: str,
) -> int:
    if isinstance(value, bool):
        raise CompatibilityRuleError(
            f"{field}: entier > 0 requis."
        )

    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise CompatibilityRuleError(
            f"{field}: entier > 0 requis."
        ) from error

    if number <= 0:
        raise CompatibilityRuleError(
            f"{field}: entier > 0 requis."
        )

    return number


def _ceil_ratio(
    required: float,
    available: float,
) -> int:
    if required <= 0:
        return 0

    if available <= 0:
        raise CompatibilityRuleError(
            "La ressource disponible doit être > 0."
        )

    return int(
        math.ceil(
            required / available
        )
    )


def normalize_form_factor(
    value: Any,
) -> str:
    text = str(value).strip().upper()

    if not text:
        return ""

    return FORM_FACTOR_ALIASES.get(
        text,
        text,
    )


def normalize_pcie_generation(
    value: Any,
) -> int | None:
    """
    Accepte les représentations du catalogue drive telles que:
    5, "5", "GEN5", "PCIe Gen5", "PCIE5".
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value > 0 else None

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        integer = int(value)
        return integer if integer > 0 else None

    text = str(value).strip().upper()

    if not text:
        return None

    match = re.search(
        r"([1-9][0-9]*)",
        text,
    )

    if match is None:
        return None

    generation = int(
        match.group(1)
    )

    return (
        generation
        if generation > 0
        else None
    )


def form_factor_compatible(
    *,
    drive_form_factor: str,
    supported_form_factors: list[str] | set[str],
) -> bool:
    """
    Compatibilité mécanique H6.

    FF_U2 / FF_U3 / FF_2_5 forment la famille mécanique 2.5".
    Le protocole est validé séparément, donc cette règle ne rend jamais
    SATA/SAS/NVMe interchangeables.
    """

    drive_ff = normalize_form_factor(
        drive_form_factor
    )

    supported = {
        normalize_form_factor(
            value
        )
        for value in supported_form_factors
    }

    if drive_ff in supported:
        return True

    if (
        drive_ff in TWO_POINT_FIVE_FAMILY
        and supported.intersection(
            TWO_POINT_FIVE_FAMILY
        )
    ):
        return True

    return False


def server_role_compatible(
    *,
    server: dict[str, Any],
    role: str,
) -> bool:
    server_role = str(
        server.get(
            "role",
            "",
        )
    ).strip().upper()

    expected = str(
        role
    ).strip().upper()

    return (
        server_role == expected
        or server_role == "BOTH"
    )


def drive_controller_compatible(
    *,
    candidate: dict[str, Any],
    controller: dict[str, Any],
) -> bool:
    interface = candidate.get(
        "hardware_interface"
    )

    if not isinstance(
        interface,
        dict,
    ):
        return False

    protocol = str(
        interface.get(
            "protocol",
            "",
        )
    ).strip().upper()

    supported = {
        str(value).strip().upper()
        for value in controller.get(
            "supported_protocols",
            []
        )
    }

    if protocol not in supported:
        return False

    if protocol == "NVME":
        required_gen = normalize_pcie_generation(
            interface.get(
                "pcie_gen_required"
            )
        )

        controller_gen = normalize_pcie_generation(
            controller.get(
                "pcie_gen"
            )
        )

        if (
            required_gen is not None
            and (
                controller_gen is None
                or controller_gen
                < required_gen
            )
        ):
            return False

    return True


def drive_server_native_compatible(
    *,
    candidate: dict[str, Any],
    server: dict[str, Any],
) -> bool:
    interface = candidate.get(
        "hardware_interface"
    )

    if not isinstance(
        interface,
        dict,
    ):
        return False

    protocol = str(
        interface.get(
            "protocol",
            "",
        )
    ).strip().upper()

    protocols = {
        str(value).strip().upper()
        for value in server.get(
            "native_drive_protocols",
            []
        )
    }

    if protocol not in protocols:
        return False

    drive_bays = server.get(
        "drive_bays"
    )

    if not isinstance(
        drive_bays,
        dict,
    ):
        return False

    positive_bays = {
        normalize_form_factor(
            key
        )
        for key, count
        in drive_bays.items()
        if (
            isinstance(
                count,
                int,
            )
            and not isinstance(
                count,
                bool,
            )
            and count > 0
        )
    }

    return form_factor_compatible(
        drive_form_factor=(
            normalize_form_factor(
                interface.get(
                    "form_factor",
                    "",
                )
            )
        ),
        supported_form_factors=(
            positive_bays
        ),
    )


def drive_enclosure_compatible(
    *,
    candidate: dict[str, Any],
    enclosure: dict[str, Any],
) -> bool:
    interface = candidate.get(
        "hardware_interface"
    )

    if not isinstance(
        interface,
        dict,
    ):
        return False

    protocol = str(
        interface.get(
            "protocol",
            "",
        )
    ).strip().upper()

    supported_protocols = {
        str(value).strip().upper()
        for value in enclosure.get(
            "supported_protocols",
            []
        )
    }

    if protocol not in supported_protocols:
        return False

    return form_factor_compatible(
        drive_form_factor=(
            normalize_form_factor(
                interface.get(
                    "form_factor",
                    "",
                )
            )
        ),
        supported_form_factors={
            normalize_form_factor(
                value
            )
            for value in enclosure.get(
                "supported_form_factors",
                []
            )
        },
    )


def controller_server_compatible(
    *,
    controller: dict[str, Any],
    server: dict[str, Any],
) -> bool:
    try:
        slots = int(
            server.get(
                "pcie_slot_count",
                0,
            )
        )
        lane_budget = int(
            server.get(
                "pcie_lane_budget",
                0,
            )
        )
        controller_lanes = int(
            controller.get(
                "pcie_lanes",
                0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return False

    return (
        slots >= 2
        and lane_budget
        >= controller_lanes
        and controller_lanes > 0
    )


def network_usable_bandwidth_gb_s(
    network: dict[str, Any],
) -> float:
    line_rate_gbit_s = _finite(
        network.get(
            "link_speed_gbit_s"
        ),
        field=(
            "network."
            "link_speed_gbit_s"
        ),
        minimum=0.0,
    )

    ports = _positive_int(
        network.get(
            "ports_per_adapter"
        ),
        field=(
            "network."
            "ports_per_adapter"
        ),
    )

    efficiency = _finite(
        network.get(
            "usable_efficiency"
        ),
        field=(
            "network."
            "usable_efficiency"
        ),
        minimum=0.0,
    )

    if efficiency > 1.0:
        raise CompatibilityRuleError(
            "network.usable_efficiency doit être <= 1."
        )

    return (
        line_rate_gbit_s
        / 8.0
        * ports
        * efficiency
    )


def _server_bay_capacity(
    *,
    candidate: dict[str, Any],
    server: dict[str, Any],
) -> int:
    interface = candidate.get(
        "hardware_interface",
        {},
    )

    drive_ff = normalize_form_factor(
        interface.get(
            "form_factor",
            "",
        )
    )

    drive_bays = server.get(
        "drive_bays",
        {},
    )

    if not isinstance(
        drive_bays,
        dict,
    ):
        return 0

    total = 0

    for form_factor, count in drive_bays.items():
        if (
            isinstance(
                count,
                int,
            )
            and not isinstance(
                count,
                bool,
            )
            and count > 0
            and form_factor_compatible(
                drive_form_factor=drive_ff,
                supported_form_factors={
                    normalize_form_factor(
                        form_factor
                    )
                },
            )
        ):
            total += count

    return total


def _required_ost_bandwidth_gb_s(
    protection_result: dict[str, Any],
) -> float:
    if protection_result.get(
        "role"
    ) != "OST":
        return 0.0

    requirements = (
        protection_result.get(
            "requirements"
        )
    )

    if not isinstance(
        requirements,
        dict,
    ):
        raise CompatibilityRuleError(
            "Protection OST sans bloc requirements."
        )

    return max(
        _finite(
            requirements.get(
                "read_bandwidth_gb_s"
            ),
            field=(
                "requirements."
                "read_bandwidth_gb_s"
            ),
            minimum=0.0,
        ),
        _finite(
            requirements.get(
                "write_bandwidth_gb_s"
            ),
            field=(
                "requirements."
                "write_bandwidth_gb_s"
            ),
            minimum=0.0,
        ),
        _finite(
            requirements.get(
                "total_bandwidth_gb_s"
            ),
            field=(
                "requirements."
                "total_bandwidth_gb_s"
            ),
            minimum=0.0,
        ),
    )


def evaluate_hardware_path(
    *,
    candidate: dict[str, Any],
    protection_result: dict[str, Any],
    role: str,
    server: dict[str, Any],
    controller: dict[str, Any],
    network: dict[str, Any],
    ha_profile: dict[str, Any],
    ha_required: bool,
    enclosure: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Évalue un chemin hardware sans en faire une recommandation finale.

    Le résultat contient seulement une décision de compatibilité et des
    minimums de ressources nécessaires à la composition future.
    """

    expected_role = str(
        role
    ).strip().upper()

    if expected_role not in {
        "MDT",
        "OST",
    }:
        raise CompatibilityRuleError(
            f"role non supporté={role!r}."
        )

    if candidate.get(
        "role"
    ) != expected_role:
        raise CompatibilityRuleError(
            "Le role du candidat ne correspond pas au role demandé."
        )

    if protection_result.get(
        "role"
    ) != expected_role:
        raise CompatibilityRuleError(
            "Le role protection ne correspond pas au role demandé."
        )

    physical_drive_count = (
        _positive_int(
            protection_result.get(
                "physical_drive_count"
            ),
            field=(
                "protection."
                "physical_drive_count"
            ),
        )
    )

    violations: list[
        str
    ] = []

    if not server_role_compatible(
        server=server,
        role=(
            "MDS"
            if expected_role == "MDT"
            else "OSS"
        ),
    ):
        violations.append(
            "server_role_incompatible"
        )

    if not drive_controller_compatible(
        candidate=candidate,
        controller=controller,
    ):
        violations.append(
            "drive_controller_incompatible"
        )

    if not controller_server_compatible(
        controller=controller,
        server=server,
    ):
        violations.append(
            "controller_server_pcie_incompatible"
        )

    mode = (
        "ENCLOSURE"
        if enclosure is not None
        else "DIRECT"
    )

    ha_mode = str(
        ha_profile.get(
            "mode",
            "",
        )
    ).strip().upper()

    if ha_required and ha_mode == "NONE":
        violations.append(
            "ha_required_but_disabled"
        )

    if ha_required:
        if not bool(
            server.get(
                "supports_dual_psu",
                False,
            )
        ):
            violations.append(
                "ha_requires_dual_psu"
            )

        if not bool(
            controller.get(
                "supports_multipath",
                False,
            )
        ):
            violations.append(
                "ha_requires_controller_multipath"
            )

        if not bool(
            network.get(
                "supports_redundant_fabric",
                False,
            )
        ):
            violations.append(
                "ha_requires_redundant_network"
            )

    requires_shared_storage = bool(
        ha_profile.get(
            "requires_shared_storage",
            False,
        )
    )

    if (
        ha_required
        and requires_shared_storage
        and enclosure is None
    ):
        violations.append(
            "ha_shared_storage_requires_enclosure"
        )

    if enclosure is None:
        if not drive_server_native_compatible(
            candidate=candidate,
            server=server,
        ):
            violations.append(
                "drive_server_direct_incompatible"
            )
    else:
        if not drive_enclosure_compatible(
            candidate=candidate,
            enclosure=enclosure,
        ):
            violations.append(
                "drive_enclosure_incompatible"
            )

        if (
            ha_required
            and not bool(
                enclosure.get(
                    "supports_redundant_paths",
                    False,
                )
            )
        ):
            violations.append(
                "ha_requires_redundant_enclosure_paths"
            )

    required_bandwidth = (
        _required_ost_bandwidth_gb_s(
            protection_result
        )
    )

    controller_ports = (
        _positive_int(
            controller.get(
                "port_count"
            ),
            field=(
                "controller."
                "port_count"
            ),
        )
    )

    controller_bandwidth = (
        _finite(
            controller.get(
                "max_aggregate_bandwidth_gb_s"
            ),
            field=(
                "controller."
                "max_aggregate_bandwidth_gb_s"
            ),
            minimum=0.0,
        )
    )

    controller_lanes = (
        _positive_int(
            controller.get(
                "pcie_lanes"
            ),
            field=(
                "controller."
                "pcie_lanes"
            ),
        )
    )

    server_slots = (
        _positive_int(
            server.get(
                "pcie_slot_count"
            ),
            field=(
                "server."
                "pcie_slot_count"
            ),
        )
    )

    lane_budget = (
        _positive_int(
            server.get(
                "pcie_lane_budget"
            ),
            field=(
                "server."
                "pcie_lane_budget"
            ),
        )
    )

    network_bandwidth = (
        network_usable_bandwidth_gb_s(
            network
        )
    )

    if network_bandwidth <= 0:
        violations.append(
            "network_bandwidth_zero"
        )

    minimum_nodes = max(
        1,
        int(
            ha_profile.get(
                "minimum_nodes_per_role",
                1,
            )
        )
        if (
            ha_required
            or ha_mode != "NONE"
        )
        else 1,
    )

    enclosure_count = 0

    if enclosure is None:
        bay_capacity = (
            _server_bay_capacity(
                candidate=candidate,
                server=server,
            )
        )

        if bay_capacity <= 0:
            violations.append(
                "no_compatible_direct_bay"
            )
            servers_by_bays = (
                physical_drive_count
            )
        else:
            servers_by_bays = (
                _ceil_ratio(
                    physical_drive_count,
                    bay_capacity,
                )
            )

        controller_count = max(
            1,
            _ceil_ratio(
                physical_drive_count,
                controller_ports,
            ),
            _ceil_ratio(
                required_bandwidth,
                controller_bandwidth,
            ),
        )

        server_count = max(
            minimum_nodes,
            servers_by_bays,
        )

    else:
        bay_count = _positive_int(
            enclosure.get(
                "drive_bay_count"
            ),
            field=(
                "enclosure."
                "drive_bay_count"
            ),
        )

        enclosure_uplinks = _positive_int(
            enclosure.get(
                "uplink_count"
            ),
            field=(
                "enclosure."
                "uplink_count"
            ),
        )

        enclosure_uplink_bw = _finite(
            enclosure.get(
                "uplink_bandwidth_gb_s"
            ),
            field=(
                "enclosure."
                "uplink_bandwidth_gb_s"
            ),
            minimum=0.0,
        )

        enclosure_bandwidth = (
            enclosure_uplinks
            * enclosure_uplink_bw
        )

        enclosure_count = max(
            1,
            _ceil_ratio(
                physical_drive_count,
                bay_count,
            ),
            _ceil_ratio(
                required_bandwidth,
                enclosure_bandwidth,
            ),
        )

        host_paths_per_enclosure = (
            2
            if (
                ha_required
                or bool(
                    ha_profile.get(
                        "requires_redundant_network",
                        False,
                    )
                )
            )
            else 1
        )

        required_host_ports = (
            enclosure_count
            * host_paths_per_enclosure
        )

        controller_count = max(
            1,
            _ceil_ratio(
                required_host_ports,
                controller_ports,
            ),
            _ceil_ratio(
                required_bandwidth,
                controller_bandwidth,
            ),
        )

        server_count = (
            minimum_nodes
        )

    if (
        server_slots <= 1
        or lane_budget
        < controller_lanes
    ):
        violations.append(
            "insufficient_server_pcie_resources"
        )
        controllers_per_server = 0
    else:
        controllers_by_slots = (
            server_slots
            - 1
        )

        controllers_by_lanes = (
            lane_budget
            // controller_lanes
        )

        controllers_per_server = min(
            controllers_by_slots,
            controllers_by_lanes,
        )

        if controllers_per_server <= 0:
            violations.append(
                "controller_cannot_fit_in_server"
            )
        else:
            server_count = max(
                server_count,
                _ceil_ratio(
                    controller_count,
                    controllers_per_server,
                ),
            )

    if (
        ha_required
        and enclosure is not None
    ):
        controller_count = max(
            controller_count,
            server_count,
        )

        if controllers_per_server > 0:
            server_count = max(
                server_count,
                _ceil_ratio(
                    controller_count,
                    controllers_per_server,
                ),
            )

    network_adapter_count = max(
        server_count,
        _ceil_ratio(
            required_bandwidth,
            network_bandwidth,
        ),
    )

    for _ in range(8):
        previous_server_count = (
            server_count
        )

        server_count = max(
            server_count,
            minimum_nodes,
            _ceil_ratio(
                controller_count
                + network_adapter_count,
                server_slots,
            ),
        )

        network_adapter_count = max(
            network_adapter_count,
            server_count,
            _ceil_ratio(
                required_bandwidth,
                network_bandwidth,
            ),
        )

        if (
            ha_required
            and enclosure is not None
        ):
            controller_count = max(
                controller_count,
                server_count,
            )

        if (
            server_count
            == previous_server_count
        ):
            break

    total_controller_bandwidth = (
        controller_count
        * controller_bandwidth
    )

    total_network_bandwidth = (
        network_adapter_count
        * network_bandwidth
    )

    if (
        required_bandwidth
        > total_controller_bandwidth
        + 1e-12
    ):
        violations.append(
            "controller_bandwidth_insufficient"
        )

    if (
        required_bandwidth
        > total_network_bandwidth
        + 1e-12
    ):
        violations.append(
            "network_bandwidth_insufficient"
        )

    component_cost_lower_bound = (
        server_count
        * _finite(
            server.get(
                "price_usd"
            ),
            field=(
                "server."
                "price_usd"
            ),
            minimum=0.0,
        )
        + controller_count
        * _finite(
            controller.get(
                "price_usd"
            ),
            field=(
                "controller."
                "price_usd"
            ),
            minimum=0.0,
        )
        + network_adapter_count
        * _finite(
            network.get(
                "price_usd"
            ),
            field=(
                "network."
                "price_usd"
            ),
            minimum=0.0,
        )
    )

    component_power_lower_bound = (
        server_count
        * _finite(
            server.get(
                "power_w"
            ),
            field=(
                "server."
                "power_w"
            ),
            minimum=0.0,
        )
        + controller_count
        * _finite(
            controller.get(
                "power_w"
            ),
            field=(
                "controller."
                "power_w"
            ),
            minimum=0.0,
        )
        + network_adapter_count
        * _finite(
            network.get(
                "power_w"
            ),
            field=(
                "network."
                "power_w"
            ),
            minimum=0.0,
        )
    )

    if enclosure is not None:
        component_cost_lower_bound += (
            enclosure_count
            * _finite(
                enclosure.get(
                    "price_usd"
                ),
                field=(
                    "enclosure."
                    "price_usd"
                ),
                minimum=0.0,
            )
        )

        component_power_lower_bound += (
            enclosure_count
            * _finite(
                enclosure.get(
                    "power_w"
                ),
                field=(
                    "enclosure."
                    "power_w"
                ),
                minimum=0.0,
            )
        )

    return {
        "compatible": (
            len(
                violations
            )
            == 0
        ),
        "role": expected_role,
        "attachment_mode": mode,
        "drive_id": candidate.get(
            "identity",
            {},
        ).get(
            "drive_id"
        ),
        "protection_profile_id": (
            protection_result.get(
                "protection_profile_id"
            )
        ),
        "server_id": server.get(
            "id"
        ),
        "controller_id": (
            controller.get(
                "id"
            )
        ),
        "enclosure_id": (
            enclosure.get(
                "id"
            )
            if enclosure
            is not None
            else None
        ),
        "network_id": network.get(
            "id"
        ),
        "ha_profile_id": (
            ha_profile.get(
                "id"
            )
        ),
        "minimum_resources": {
            "physical_drive_count": (
                physical_drive_count
            ),
            "server_count": (
                server_count
            ),
            "controller_count": (
                controller_count
            ),
            "enclosure_count": (
                enclosure_count
            ),
            "network_adapter_count": (
                network_adapter_count
            ),
        },
        "bandwidth": {
            "required_gb_s": (
                required_bandwidth
            ),
            "controller_total_gb_s": (
                total_controller_bandwidth
            ),
            "network_total_gb_s": (
                total_network_bandwidth
            ),
        },
        "component_cost_lower_bound_usd": (
            component_cost_lower_bound
        ),
        "component_power_lower_bound_w": (
            component_power_lower_bound
        ),
        "violations": violations,
    }


def find_compatible_hardware_paths(
    *,
    candidate: dict[str, Any],
    protection_result: dict[str, Any],
    role: str,
    hardware_catalog: dict[str, Any],
    ha_required: bool,
    max_paths: int = 20,
) -> list[dict[str, Any]]:
    """
    Énumère des chemins compatibles de façon déterministe.

    H6 ne score pas et ne choisit pas la meilleure architecture.
    """

    if max_paths <= 0:
        raise ValueError(
            "max_paths doit être > 0."
        )

    servers = hardware_catalog.get(
        "servers"
    )
    controllers = hardware_catalog.get(
        "controllers"
    )
    enclosures = hardware_catalog.get(
        "enclosures"
    )
    networks = hardware_catalog.get(
        "networks"
    )
    ha_profiles = hardware_catalog.get(
        "ha_profiles"
    )

    for name, value in (
        ("servers", servers),
        ("controllers", controllers),
        ("enclosures", enclosures),
        ("networks", networks),
        ("ha_profiles", ha_profiles),
    ):
        if not isinstance(
            value,
            list,
        ) or not value:
            raise CompatibilityRuleError(
                f"hardware_catalog.{name}: liste non vide requise."
            )

    paths: list[
        dict[str, Any]
    ] = []

    expected_server_role = (
        "MDS"
        if str(
            role
        ).upper()
        == "MDT"
        else "OSS"
    )

    ordered_ha = sorted(
        ha_profiles,
        key=lambda item: (
            0
            if (
                (
                    ha_required
                    and str(
                        item.get(
                            "mode",
                            "",
                        )
                    ).upper()
                    != "NONE"
                )
                or (
                    not ha_required
                    and str(
                        item.get(
                            "mode",
                            "",
                        )
                    ).upper()
                    == "NONE"
                )
            )
            else 1,
            str(
                item.get(
                    "id",
                    "",
                )
            ),
        ),
    )

    for server in sorted(
        servers,
        key=lambda item: str(
            item.get(
                "id",
                "",
            )
        ),
    ):
        if not server_role_compatible(
            server=server,
            role=expected_server_role,
        ):
            continue

        for controller in sorted(
            controllers,
            key=lambda item: str(
                item.get(
                    "id",
                    "",
                )
            ),
        ):
            if not drive_controller_compatible(
                candidate=candidate,
                controller=controller,
            ):
                continue

            for network in sorted(
                networks,
                key=lambda item: str(
                    item.get(
                        "id",
                        "",
                    )
                ),
            ):
                for ha_profile in ordered_ha:
                    modes: list[
                        dict[str, Any]
                        | None
                    ] = [
                        None
                    ]

                    modes.extend(
                        sorted(
                            [
                                enclosure
                                for enclosure
                                in enclosures
                                if drive_enclosure_compatible(
                                    candidate=candidate,
                                    enclosure=enclosure,
                                )
                            ],
                            key=lambda item: str(
                                item.get(
                                    "id",
                                    "",
                                )
                            ),
                        )
                    )

                    for enclosure in modes:
                        result = (
                            evaluate_hardware_path(
                                candidate=candidate,
                                protection_result=(
                                    protection_result
                                ),
                                role=role,
                                server=server,
                                controller=(
                                    controller
                                ),
                                network=network,
                                ha_profile=(
                                    ha_profile
                                ),
                                ha_required=(
                                    ha_required
                                ),
                                enclosure=(
                                    enclosure
                                ),
                            )
                        )

                        if result[
                            "compatible"
                        ]:
                            paths.append(
                                result
                            )

                            if (
                                len(
                                    paths
                                )
                                >= max_paths
                            ):
                                return paths

    return paths
