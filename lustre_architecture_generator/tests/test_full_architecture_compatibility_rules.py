from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from full_architecture.compatibility_rules import (  # noqa: E402
    drive_controller_compatible,
    drive_enclosure_compatible,
    evaluate_hardware_path,
    find_compatible_hardware_paths,
    form_factor_compatible,
    network_usable_bandwidth_gb_s,
    normalize_form_factor,
    normalize_pcie_generation,
    server_role_compatible,
)


def candidate(
    *,
    role: str = "OST",
    protocol: str = "SAS",
    form_factor: str = "FF_3_5",
    pcie_gen: Any = None,
) -> dict[str, Any]:
    return {
        "role": role,
        "identity": {
            "drive_id": "DRV_TEST",
        },
        "hardware_interface": {
            "protocol": protocol,
            "form_factor": form_factor,
            "pcie_gen_required": (
                pcie_gen
                if pcie_gen is not None
                else (
                    4
                    if protocol == "NVME"
                    else None
                )
            ),
            "pcie_lanes_required": (
                4
                if protocol == "NVME"
                else None
            ),
        },
    }


def protection(
    *,
    role: str = "OST",
) -> dict[str, Any]:
    if role == "MDT":
        requirements = {
            "usable_capacity_tib": 1.0,
            "read_iops": 100000.0,
            "write_iops": 50000.0,
        }
    else:
        requirements = {
            "usable_capacity_tib": 100.0,
            "read_bandwidth_gb_s": 5.0,
            "write_bandwidth_gb_s": 4.0,
            "total_bandwidth_gb_s": 9.0,
        }

    return {
        "role": role,
        "protection_profile_id": "PROT_TEST",
        "physical_drive_count": 20,
        "requirements": requirements,
    }


def server(
    *,
    role: str = "OSS",
) -> dict[str, Any]:
    return {
        "id": "SERVER_TEST",
        "role": role,
        "pcie_slot_count": 8,
        "pcie_lane_budget": 128,
        "drive_bays": {
            "FF_3_5": 12,
            "FF_2_5": 24,
            "FF_U2": 8,
            "FF_E1S": 8,
            "FF_E3S": 8,
        },
        "native_drive_protocols": [
            "SAS",
            "SATA",
            "NVME",
        ],
        "network_interfaces": [
            "PCIe NIC/HCA",
        ],
        "supports_dual_psu": True,
        "price_usd": 7000.0,
        "power_w": 600.0,
    }


def controller(
    *,
    protocol: str = "SAS",
    pcie_gen: int = 4,
) -> dict[str, Any]:
    return {
        "id": "CTRL_TEST",
        "supported_protocols": [
            protocol
        ],
        "port_count": 16,
        "max_aggregate_bandwidth_gb_s": 63.0,
        "pcie_gen": pcie_gen,
        "pcie_lanes": 16 if protocol == "NVME" else 8,
        "supports_multipath": True,
        "price_usd": 500.0,
        "power_w": 25.0,
    }


def enclosure(
    *,
    protocol: str = "SAS",
    form_factor: str = "FF_3_5",
) -> dict[str, Any]:
    return {
        "id": "ENC_TEST",
        "supported_protocols": [
            protocol
        ],
        "supported_form_factors": [
            form_factor
        ],
        "drive_bay_count": 60,
        "uplink_count": 4,
        "uplink_bandwidth_gb_s": 64.0,
        "supports_redundant_paths": True,
        "price_usd": 8000.0,
        "power_w": 800.0,
    }


def network() -> dict[str, Any]:
    return {
        "id": "NET_TEST",
        "link_speed_gbit_s": 100.0,
        "ports_per_adapter": 2,
        "usable_efficiency": 0.90,
        "supports_redundant_fabric": True,
        "price_usd": 800.0,
        "power_w": 30.0,
    }


def ha(
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "id": "HA_NONE",
            "mode": "NONE",
            "minimum_nodes_per_role": 1,
            "requires_shared_storage": False,
            "requires_redundant_network": False,
        }

    return {
        "id": "HA_ACTIVE_PASSIVE",
        "mode": "ACTIVE_PASSIVE",
        "minimum_nodes_per_role": 2,
        "requires_shared_storage": True,
        "requires_redundant_network": True,
    }


def hardware_catalog() -> dict[str, Any]:
    return {
        "servers": [
            server()
        ],
        "controllers": [
            controller()
        ],
        "enclosures": [
            enclosure()
        ],
        "networks": [
            network()
        ],
        "ha_profiles": [
            ha(
                enabled=False
            ),
            ha(
                enabled=True
            ),
        ],
    }


def test_two_point_five_family_is_compatible() -> None:
    assert form_factor_compatible(
        drive_form_factor="FF_U2",
        supported_form_factors={
            "FF_2_5"
        },
    )


def test_raw_u2_alias_is_normalized() -> None:
    assert normalize_form_factor(
        "U.2"
    ) == "FF_U2"

    assert form_factor_compatible(
        drive_form_factor="U.2",
        supported_form_factors={
            "FF_U2"
        },
    )


def test_raw_e1s_and_e3s_aliases_are_normalized() -> None:
    assert normalize_form_factor(
        "E1.S"
    ) == "FF_E1S"
    assert normalize_form_factor(
        "E3.S"
    ) == "FF_E3S"


def test_gen5_string_is_normalized() -> None:
    assert normalize_pcie_generation(
        "GEN5"
    ) == 5
    assert normalize_pcie_generation(
        "PCIe Gen5"
    ) == 5


def test_server_role_both_is_compatible() -> None:
    item = server(
        role="BOTH"
    )

    assert server_role_compatible(
        server=item,
        role="MDS",
    )
    assert server_role_compatible(
        server=item,
        role="OSS",
    )


def test_wrong_drive_controller_protocol_is_rejected() -> None:
    assert not drive_controller_compatible(
        candidate=candidate(
            protocol="NVME",
            form_factor="U.2",
        ),
        controller=controller(
            protocol="SAS"
        ),
    )


def test_gen5_drive_requires_gen5_controller() -> None:
    gen5_drive = candidate(
        protocol="NVME",
        form_factor="U.2",
        pcie_gen="GEN5",
    )

    assert not drive_controller_compatible(
        candidate=gen5_drive,
        controller=controller(
            protocol="NVME",
            pcie_gen=4,
        ),
    )

    assert drive_controller_compatible(
        candidate=gen5_drive,
        controller=controller(
            protocol="NVME",
            pcie_gen=5,
        ),
    )


def test_drive_enclosure_protocol_and_form_factor() -> None:
    assert drive_enclosure_compatible(
        candidate=candidate(),
        enclosure=enclosure(),
    )

    assert not drive_enclosure_compatible(
        candidate=candidate(),
        enclosure=enclosure(
            protocol="NVME",
            form_factor="FF_U2",
        ),
    )


def test_e1s_drive_matches_e1s_enclosure() -> None:
    assert drive_enclosure_compatible(
        candidate=candidate(
            protocol="NVME",
            form_factor="E1.S",
            pcie_gen="GEN5",
        ),
        enclosure=enclosure(
            protocol="NVME",
            form_factor="FF_E1S",
        ),
    )


def test_network_conversion_gbit_to_gb() -> None:
    assert (
        network_usable_bandwidth_gb_s(
            network()
        )
        == 22.5
    )


def test_non_ha_direct_path_is_compatible() -> None:
    result = evaluate_hardware_path(
        candidate=candidate(),
        protection_result=protection(),
        role="OST",
        server=server(),
        controller=controller(),
        network=network(),
        ha_profile=ha(
            enabled=False
        ),
        ha_required=False,
        enclosure=None,
    )

    assert result["compatible"] is True
    assert (
        result[
            "attachment_mode"
        ]
        == "DIRECT"
    )


def test_required_ha_rejects_direct_shared_storage() -> None:
    result = evaluate_hardware_path(
        candidate=candidate(),
        protection_result=protection(),
        role="OST",
        server=server(),
        controller=controller(),
        network=network(),
        ha_profile=ha(
            enabled=True
        ),
        ha_required=True,
        enclosure=None,
    )

    assert result["compatible"] is False
    assert (
        "ha_shared_storage_requires_enclosure"
        in result["violations"]
    )


def test_required_ha_accepts_redundant_enclosure_path() -> None:
    result = evaluate_hardware_path(
        candidate=candidate(),
        protection_result=protection(),
        role="OST",
        server=server(),
        controller=controller(),
        network=network(),
        ha_profile=ha(
            enabled=True
        ),
        ha_required=True,
        enclosure=enclosure(),
    )

    assert result["compatible"] is True
    assert (
        result[
            "minimum_resources"
        ][
            "server_count"
        ]
        >= 2
    )


def test_insufficient_controller_pcie_is_rejected() -> None:
    small_server = server()
    small_server[
        "pcie_lane_budget"
    ] = 4

    result = evaluate_hardware_path(
        candidate=candidate(),
        protection_result=protection(),
        role="OST",
        server=small_server,
        controller=controller(),
        network=network(),
        ha_profile=ha(
            enabled=False
        ),
        ha_required=False,
        enclosure=None,
    )

    assert result["compatible"] is False
    assert (
        "controller_server_pcie_incompatible"
        in result[
            "violations"
        ]
    )


def test_find_compatible_paths_returns_paths() -> None:
    paths = find_compatible_hardware_paths(
        candidate=candidate(),
        protection_result=protection(),
        role="OST",
        hardware_catalog=(
            hardware_catalog()
        ),
        ha_required=False,
        max_paths=5,
    )

    assert paths
    assert all(
        item["compatible"]
        for item in paths
    )


def test_find_compatible_paths_honors_max_paths() -> None:
    data = hardware_catalog()
    data["networks"].append(
        {
            **network(),
            "id": "NET_TEST_2",
        }
    )

    paths = find_compatible_hardware_paths(
        candidate=candidate(),
        protection_result=protection(),
        role="OST",
        hardware_catalog=data,
        ha_required=False,
        max_paths=1,
    )

    assert len(paths) == 1


def test_compatible_path_is_deterministic() -> None:
    first = find_compatible_hardware_paths(
        candidate=candidate(),
        protection_result=protection(),
        role="OST",
        hardware_catalog=(
            hardware_catalog()
        ),
        ha_required=True,
        max_paths=3,
    )

    second = find_compatible_hardware_paths(
        candidate=candidate(),
        protection_result=protection(),
        role="OST",
        hardware_catalog=(
            copy.deepcopy(
                hardware_catalog()
            )
        ),
        ha_required=True,
        max_paths=3,
    )

    signature = lambda paths: [
        (
            item[
                "server_id"
            ],
            item[
                "controller_id"
            ],
            item[
                "enclosure_id"
            ],
            item[
                "network_id"
            ],
            item[
                "ha_profile_id"
            ],
            item[
                "attachment_mode"
            ],
        )
        for item in paths
    ]

    assert signature(first) == signature(second)


def test_gen5_e1s_ha_path_is_compatible() -> None:
    data = {
        "servers": [
            server(
                role="MDS"
            )
        ],
        "controllers": [
            controller(
                protocol="NVME",
                pcie_gen=5,
            )
        ],
        "enclosures": [
            enclosure(
                protocol="NVME",
                form_factor="FF_E1S",
            )
        ],
        "networks": [
            network()
        ],
        "ha_profiles": [
            ha(
                enabled=True
            )
        ],
    }

    paths = find_compatible_hardware_paths(
        candidate=candidate(
            role="MDT",
            protocol="NVME",
            form_factor="E1.S",
            pcie_gen="GEN5",
        ),
        protection_result=protection(
            role="MDT"
        ),
        role="MDT",
        hardware_catalog=data,
        ha_required=True,
        max_paths=1,
    )

    assert len(paths) == 1
