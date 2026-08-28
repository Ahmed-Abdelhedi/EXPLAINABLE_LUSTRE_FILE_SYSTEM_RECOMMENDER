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


from full_architecture.full_architecture_generator import (  # noqa: E402
    FullArchitectureGeneratorError,
    architecture_id,
    architecture_signature,
    enumerate_role_options,
    generate_full_architectures,
    iter_full_architectures,
)


def raid1() -> dict[str, Any]:
    return {
        "id": "PROT_RAID1_2",
        "name": "RAID1 1+1",
        "raid_level": "RAID1",
        "minimum_drives_per_group": 2,
        "data_drives_per_group": 1,
        "parity_drives_per_group": 0,
        "mirror_copies": 2,
        "read_efficiency": 0.95,
        "write_efficiency": 0.90,
        "capacity_efficiency": 0.50,
        "fault_tolerance_drives_per_group": 1,
    }


def raid6() -> dict[str, Any]:
    return {
        "id": "PROT_RAID6_10",
        "name": "RAID6 8+2",
        "raid_level": "RAID6",
        "minimum_drives_per_group": 10,
        "data_drives_per_group": 8,
        "parity_drives_per_group": 2,
        "mirror_copies": 1,
        "read_efficiency": 0.92,
        "write_efficiency": 0.68,
        "capacity_efficiency": 0.80,
        "fault_tolerance_drives_per_group": 2,
    }


def mdt_candidate(drive_id: str = "MDT_A", rank: int = 1) -> dict[str, Any]:
    return {
        "role": "MDT",
        "selection_rank": rank,
        "identity": {
            "drive_id": drive_id,
            "drive_name": drive_id,
            "manufacturer": "Synthetic",
            "series": "Test",
            "media_type": "SSD",
            "catalog_id": f"CAT_{drive_id}",
            "model_number": drive_id,
        },
        "hardware_interface": {
            "protocol": "SAS",
            "form_factor": "FF_2_5",
            "pcie_gen_required": None,
            "pcie_lanes_required": None,
        },
        "ranking": {
            "ml_rank": rank,
            "ml_score": 10.0 - rank,
            "selection_reasons": ["ml_top_k"],
        },
        "pre_raid": {
            "minimum_drive_count": 2,
            "is_lower_bound_only": True,
            "provided_capacity_tib": 4.0,
            "provided_read_iops": 300000.0,
            "provided_write_iops": 240000.0,
            "drive_level_cost_usd": 900.0,
            "drive_level_power_w": 30.0,
        },
    }


def ost_candidate(drive_id: str = "OST_A", rank: int = 1) -> dict[str, Any]:
    return {
        "role": "OST",
        "selection_rank": rank,
        "identity": {
            "drive_id": drive_id,
            "drive_name": drive_id,
            "manufacturer": "Synthetic",
            "series": "Test",
            "media_type": "SSD",
            "catalog_id": f"CAT_{drive_id}",
            "model_number": drive_id,
        },
        "hardware_interface": {
            "protocol": "SAS",
            "form_factor": "FF_2_5",
            "pcie_gen_required": None,
            "pcie_lanes_required": None,
        },
        "ranking": {
            "ml_rank": rank,
            "ml_score": 10.0 - rank,
            "selection_reasons": ["ml_top_k"],
        },
        "pre_raid": {
            "minimum_drive_count": 4,
            "is_lower_bound_only": True,
            "provided_capacity_tib": 80.0,
            "provided_read_bandwidth_gb_s": 8.0,
            "provided_write_bandwidth_gb_s": 6.0,
            "provided_total_bandwidth_gb_s": 14.0,
            "drive_level_cost_usd": 2400.0,
            "drive_level_power_w": 120.0,
        },
    }


def server(server_id: str, role: str) -> dict[str, Any]:
    return {
        "id": server_id,
        "role": role,
        "pcie_slot_count": 8,
        "pcie_lane_budget": 128,
        "drive_bays": {"FF_2_5": 24},
        "native_drive_protocols": ["SAS"],
        "network_interfaces": ["PCIe NIC/HCA"],
        "supports_dual_psu": True,
        "price_usd": 7000.0,
        "power_w": 600.0,
    }


def controller() -> dict[str, Any]:
    return {
        "id": "CTRL_SAS",
        "supported_protocols": ["SAS"],
        "port_count": 16,
        "max_aggregate_bandwidth_gb_s": 24.0,
        "pcie_gen": 4,
        "pcie_lanes": 8,
        "supports_multipath": True,
        "price_usd": 500.0,
        "power_w": 25.0,
    }


def enclosure() -> dict[str, Any]:
    return {
        "id": "ENC_SAS_24",
        "supported_protocols": ["SAS"],
        "supported_form_factors": ["FF_2_5"],
        "drive_bay_count": 24,
        "uplink_count": 4,
        "uplink_bandwidth_gb_s": 12.0,
        "supports_redundant_paths": True,
        "price_usd": 8000.0,
        "power_w": 800.0,
    }


def network() -> dict[str, Any]:
    return {
        "id": "NET_100G",
        "link_speed_gbit_s": 100.0,
        "ports_per_adapter": 2,
        "usable_efficiency": 0.90,
        "supports_redundant_fabric": True,
        "price_usd": 800.0,
        "power_w": 30.0,
    }


def ha_none() -> dict[str, Any]:
    return {
        "id": "HA_NONE",
        "mode": "NONE",
        "minimum_nodes_per_role": 1,
        "requires_shared_storage": False,
        "requires_redundant_network": False,
    }


def hardware_catalog() -> dict[str, Any]:
    return {
        "servers": [
            server("SRV_MDS", "MDS"),
            server("SRV_OSS", "OSS"),
        ],
        "controllers": [controller()],
        "enclosures": [enclosure()],
        "networks": [network()],
        "protection_profiles": [raid1(), raid6()],
        "ha_profiles": [ha_none()],
    }


def handoff(*, two_ost: bool = False) -> dict[str, Any]:
    osts = [ost_candidate()]
    if two_ost:
        osts.append(ost_candidate("OST_B", 2))

    return {
        "schema_version": "1.0",
        "case_id": "REQ_H8_TEST",
        "requested_top_k": 10,
        "requirements": {
            "MDT_requirement": {
                "required_metadata_capacity_tib": 1.0,
                "required_read_iops": 100000.0,
                "required_write_iops": 80000.0,
            },
            "OST_requirement": {
                "required_usable_capacity_tib": 40.0,
                "required_read_bandwidth_gbps": 3.0,
                "required_write_bandwidth_gbps": 2.0,
                "required_total_bandwidth_gbps": 5.0,
            },
            "constraints": {
                "ha_required": False,
                "max_budget_usd": 100000.0,
                "max_power_w": 10000.0,
            },
            "preferences": {},
        },
        "ranking_provenance": {
            "mdt": {"model_family": "LightGBM", "model_seed": 168},
            "ost": {"model_family": "LightGBM", "model_seed": 84},
        },
        "mdt_candidates": [mdt_candidate()],
        "ost_candidates": osts,
    }


def test_enumerate_mdt_role_options_returns_compatible_options() -> None:
    options = enumerate_role_options(
        handoff=handoff(),
        hardware_catalog=hardware_catalog(),
        role="MDT",
        max_paths_per_variant=2,
    )
    assert options
    assert all(option["hardware_path"]["compatible"] for option in options)


def test_enumerate_role_options_respects_role_limit() -> None:
    options = enumerate_role_options(
        handoff=handoff(),
        hardware_catalog=hardware_catalog(),
        role="OST",
        max_paths_per_variant=2,
        max_role_options=1,
    )
    assert len(options) == 1


def test_enumerate_role_options_is_deterministic() -> None:
    first = enumerate_role_options(
        handoff=handoff(two_ost=True),
        hardware_catalog=hardware_catalog(),
        role="OST",
        max_paths_per_variant=2,
    )
    second = enumerate_role_options(
        handoff=copy.deepcopy(handoff(two_ost=True)),
        hardware_catalog=copy.deepcopy(hardware_catalog()),
        role="OST",
        max_paths_per_variant=2,
    )
    assert first == second


def test_generation_returns_complete_states() -> None:
    result = generate_full_architectures(
        handoff=handoff(),
        hardware_catalog=hardware_catalog(),
        max_paths_per_variant=1,
        max_role_options_per_role=2,
    )
    assert result["architectures"]
    assert all(
        record["state"]["stage"] == "COMPLETE"
        for record in result["architectures"]
    )


def test_generated_states_remain_pending_full_validator() -> None:
    result = generate_full_architectures(
        handoff=handoff(),
        hardware_catalog=hardware_catalog(),
        max_paths_per_variant=1,
        max_role_options_per_role=2,
    )
    assert all(
        record["state"]["validation"]["status"]
        == "PENDING_FULL_VALIDATOR"
        for record in result["architectures"]
    )


def test_generation_contract_has_no_beam_or_score() -> None:
    result = generate_full_architectures(
        handoff=handoff(),
        hardware_catalog=hardware_catalog(),
        max_paths_per_variant=1,
        max_role_options_per_role=2,
    )
    assert result["generation_contract"]["beam_search_applied"] is False
    assert result["generation_contract"]["architecture_score_applied"] is False


def test_generation_count_is_cartesian_product() -> None:
    result = generate_full_architectures(
        handoff=handoff(two_ost=True),
        hardware_catalog=hardware_catalog(),
        max_paths_per_variant=1,
        max_role_options_per_role=3,
    )
    assert result["summary"]["generated_architecture_count"] == (
        result["summary"]["mdt_role_options"]
        * result["summary"]["ost_role_options"]
    )


def test_max_architectures_truncates_explicitly() -> None:
    result = generate_full_architectures(
        handoff=handoff(two_ost=True),
        hardware_catalog=hardware_catalog(),
        max_paths_per_variant=2,
        max_role_options_per_role=4,
        max_architectures=3,
    )
    assert result["summary"]["generated_architecture_count"] == 3
    assert result["summary"]["truncated_by_max_architectures"] is True


def test_architecture_ids_are_unique() -> None:
    result = generate_full_architectures(
        handoff=handoff(two_ost=True),
        hardware_catalog=hardware_catalog(),
        max_paths_per_variant=2,
        max_role_options_per_role=4,
    )
    ids = [record["architecture_id"] for record in result["architectures"]]
    assert len(ids) == len(set(ids))


def test_architecture_id_is_stable() -> None:
    result = generate_full_architectures(
        handoff=handoff(),
        hardware_catalog=hardware_catalog(),
        max_paths_per_variant=1,
        max_role_options_per_role=1,
    )
    state = result["architectures"][0]["state"]
    assert architecture_id(state) == architecture_id(copy.deepcopy(state))


def test_architecture_signature_changes_with_ost_drive() -> None:
    first = generate_full_architectures(
        handoff=handoff(),
        hardware_catalog=hardware_catalog(),
        max_paths_per_variant=1,
        max_role_options_per_role=1,
    )["architectures"][0]["state"]

    second_handoff = handoff()
    second_handoff["ost_candidates"] = [ost_candidate("OST_B", 1)]
    second = generate_full_architectures(
        handoff=second_handoff,
        hardware_catalog=hardware_catalog(),
        max_paths_per_variant=1,
        max_role_options_per_role=1,
    )["architectures"][0]["state"]

    assert architecture_signature(first) != architecture_signature(second)


def test_iter_full_architectures_is_lazy_and_yields_records() -> None:
    iterator = iter_full_architectures(
        handoff=handoff(two_ost=True),
        hardware_catalog=hardware_catalog(),
        max_paths_per_variant=1,
        max_role_options_per_role=2,
    )
    first = next(iterator)
    assert first["architecture_id"].startswith("ARCH_REQ_H8_TEST_")
    assert first["state"]["stage"] == "COMPLETE"


def test_role_instances_are_explicitly_isolated() -> None:
    result = generate_full_architectures(
        handoff=handoff(),
        hardware_catalog=hardware_catalog(),
        max_paths_per_variant=1,
        max_role_options_per_role=1,
    )
    record = result["architectures"][0]
    assert record["generation_semantics"]["role_instances_are_isolated"] is True


def test_invalid_path_limit_is_rejected() -> None:
    with pytest.raises(FullArchitectureGeneratorError, match="max_paths_per_variant"):
        generate_full_architectures(
            handoff=handoff(),
            hardware_catalog=hardware_catalog(),
            max_paths_per_variant=0,
        )


def test_empty_ost_candidates_are_rejected() -> None:
    broken = handoff()
    broken["ost_candidates"] = []
    with pytest.raises(FullArchitectureGeneratorError, match="Aucun candidat OST"):
        enumerate_role_options(
            handoff=broken,
            hardware_catalog=hardware_catalog(),
            role="OST",
            max_paths_per_variant=1,
        )
