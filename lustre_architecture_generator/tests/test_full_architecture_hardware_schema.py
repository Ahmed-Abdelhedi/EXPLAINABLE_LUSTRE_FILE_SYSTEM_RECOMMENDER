from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from full_architecture.hardware_schema import (  # noqa: E402
    HardwareSchemaError,
    new_architecture_state,
    validate_architecture_state,
    validate_controller_profile,
    validate_enclosure_profile,
    validate_ha_profile,
    validate_hardware_catalog_bundle,
    validate_network_profile,
    validate_protection_profile,
    validate_server_profile,
)


def server() -> dict[str, Any]:
    return {
        "id": "SRV_MDS_001",
        "name": "Synthetic MDS",
        "manufacturer": "Synthetic",
        "role": "MDS",
        "cpu_cores": 32,
        "memory_gib": 256,
        "pcie_slot_count": 8,
        "pcie_lane_budget": 128,
        "drive_bays": {
            "FF_2_5": 24,
            "FF_U2": 8,
        },
        "native_drive_protocols": [
            "SATA",
            "SAS",
            "NVME",
        ],
        "network_interfaces": [
            "PCIe NIC",
        ],
        "supports_dual_psu": True,
        "price_usd": 6000,
        "power_w": 500,
    }


def controller() -> dict[str, Any]:
    return {
        "id": "CTRL_001",
        "name": "Synthetic HBA",
        "manufacturer": "Synthetic",
        "controller_type": "HBA",
        "supported_protocols": [
            "SATA",
            "SAS",
        ],
        "port_count": 16,
        "max_aggregate_bandwidth_gb_s": 12.0,
        "pcie_gen": 4,
        "pcie_lanes": 8,
        "supports_multipath": True,
        "price_usd": 500,
        "power_w": 25,
    }


def enclosure() -> dict[str, Any]:
    return {
        "id": "ENC_001",
        "name": "Synthetic JBOD",
        "manufacturer": "Synthetic",
        "supported_protocols": [
            "SAS",
            "SATA",
        ],
        "supported_form_factors": [
            "FF_3_5",
            "FF_2_5",
        ],
        "drive_bay_count": 60,
        "uplink_count": 4,
        "uplink_bandwidth_gb_s": 12.0,
        "supports_redundant_paths": True,
        "price_usd": 7000,
        "power_w": 700,
    }


def network() -> dict[str, Any]:
    return {
        "id": "NET_001",
        "name": "Synthetic 200G IB",
        "manufacturer": "Synthetic",
        "fabric": "INFINIBAND",
        "link_speed_gbit_s": 200.0,
        "ports_per_adapter": 2,
        "usable_efficiency": 0.92,
        "supports_redundant_fabric": True,
        "price_usd": 1200,
        "power_w": 30,
    }


def protection() -> dict[str, Any]:
    return {
        "id": "RAID6_8D2P",
        "name": "RAID6 8+2",
        "raid_level": "RAID6",
        "minimum_drives_per_group": 10,
        "data_drives_per_group": 8,
        "parity_drives_per_group": 2,
        "mirror_copies": 1,
        "read_efficiency": 0.95,
        "write_efficiency": 0.72,
        "capacity_efficiency": 0.8,
        "fault_tolerance_drives_per_group": 2,
    }


def ha() -> dict[str, Any]:
    return {
        "id": "HA_ACTIVE_PASSIVE",
        "name": "Active/passive HA",
        "mode": "ACTIVE_PASSIVE",
        "minimum_nodes_per_role": 2,
        "requires_shared_storage": True,
        "requires_redundant_network": True,
    }


def bundle() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "servers": [server()],
        "controllers": [controller()],
        "enclosures": [enclosure()],
        "networks": [network()],
        "protection_profiles": [protection()],
        "ha_profiles": [ha()],
    }


def test_valid_server_schema() -> None:
    validate_server_profile(server())


def test_valid_controller_schema() -> None:
    validate_controller_profile(controller())


def test_valid_enclosure_schema() -> None:
    validate_enclosure_profile(enclosure())


def test_valid_network_schema() -> None:
    validate_network_profile(network())


def test_valid_protection_schema() -> None:
    validate_protection_profile(protection())


def test_valid_ha_schema() -> None:
    validate_ha_profile(ha())


def test_valid_catalog_bundle() -> None:
    validate_hardware_catalog_bundle(bundle())


def test_invalid_network_efficiency_rejected() -> None:
    item = network()
    item["usable_efficiency"] = 1.2

    with pytest.raises(
        HardwareSchemaError,
        match="usable_efficiency",
    ):
        validate_network_profile(item)


def test_invalid_raid6_arithmetic_rejected() -> None:
    item = protection()
    item["parity_drives_per_group"] = 1

    with pytest.raises(
        HardwareSchemaError,
        match="RAID6",
    ):
        validate_protection_profile(item)


def test_duplicate_catalog_id_rejected() -> None:
    data = bundle()
    duplicate = copy.deepcopy(
        data["servers"][0]
    )
    data["servers"].append(
        duplicate
    )

    with pytest.raises(
        HardwareSchemaError,
        match="dupliqué",
    ):
        validate_hardware_catalog_bundle(data)


def test_new_architecture_state_is_valid_and_empty() -> None:
    state = new_architecture_state(
        case_id="REQ_TEST_001"
    )

    validate_architecture_state(state)

    assert state["stage"] == "EMPTY"
    assert state["validation"]["is_complete"] is False
    assert state["validation"]["is_valid"] is False


def test_state_cannot_be_valid_before_complete() -> None:
    state = new_architecture_state(
        case_id="REQ_TEST_002"
    )
    state["validation"]["is_valid"] = True

    with pytest.raises(
        HardwareSchemaError,
        match="complet",
    ):
        validate_architecture_state(state)
