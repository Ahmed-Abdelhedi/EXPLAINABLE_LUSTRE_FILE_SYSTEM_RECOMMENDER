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


from full_architecture.architecture_state import (  # noqa: E402
    ArchitectureStateError,
    validate_full_architecture_state,
)
from full_architecture.full_architecture_generator import (  # noqa: E402
    generate_full_architectures,
)
from full_architecture.full_architecture_validator import (  # noqa: E402
    VALIDATOR_POLICY_ID,
    assert_full_validation_result_valid,
    validate_complete_architecture,
    validate_generated_architectures,
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


def candidate(role: str, drive_id: str) -> dict[str, Any]:
    if role == "MDT":
        pre_raid = {
            "minimum_drive_count": 2,
            "is_lower_bound_only": True,
            "provided_capacity_tib": 4.0,
            "provided_read_iops": 300000.0,
            "provided_write_iops": 240000.0,
            "drive_level_cost_usd": 900.0,
            "drive_level_power_w": 30.0,
        }
    else:
        pre_raid = {
            "minimum_drive_count": 4,
            "is_lower_bound_only": True,
            "provided_capacity_tib": 80.0,
            "provided_read_bandwidth_gb_s": 8.0,
            "provided_write_bandwidth_gb_s": 6.0,
            "provided_total_bandwidth_gb_s": 14.0,
            "drive_level_cost_usd": 2400.0,
            "drive_level_power_w": 120.0,
        }

    return {
        "role": role,
        "selection_rank": 1,
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
        "reliability": {
            "endurance_dwpd": 3.0,
            "mtbf_hours": 2500000,
            "warranty_years": 5,
            "workload_rating_tb_per_year": None,
        },
        "ranking": {
            "ml_rank": 1,
            "ml_score": 9.0,
            "selection_reasons": ["ml_top_k"],
        },
        "pre_raid": pre_raid,
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


def hardware_catalog() -> dict[str, Any]:
    return {
        "servers": [
            server("SRV_MDS", "MDS"),
            server("SRV_OSS", "OSS"),
        ],
        "controllers": [
            {
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
        ],
        "enclosures": [
            {
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
        ],
        "networks": [
            {
                "id": "NET_100G",
                "link_speed_gbit_s": 100.0,
                "ports_per_adapter": 2,
                "usable_efficiency": 0.90,
                "supports_redundant_fabric": True,
                "price_usd": 800.0,
                "power_w": 30.0,
            }
        ],
        "protection_profiles": [raid1(), raid6()],
        "ha_profiles": [
            {
                "id": "HA_NONE",
                "mode": "NONE",
                "minimum_nodes_per_role": 1,
                "requires_shared_storage": False,
                "requires_redundant_network": False,
            }
        ],
    }


def handoff(
    *,
    max_budget: float = 100000.0,
    max_power: float = 10000.0,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "case_id": "REQ_H10_TEST",
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
                "max_budget_usd": max_budget,
                "max_power_w": max_power,
            },
            "preferences": {
                "performance_priority": 0.4,
                "cost_priority": 0.2,
                "power_priority": 0.2,
                "reliability_priority": 0.2,
            },
        },
        "ranking_provenance": {
            "mdt": {"model_family": "LightGBM", "model_seed": 168},
            "ost": {"model_family": "LightGBM", "model_seed": 84},
        },
        "mdt_candidates": [candidate("MDT", "MDT_A")],
        "ost_candidates": [candidate("OST", "OST_A")],
    }


def generated(
    *,
    h: dict[str, Any] | None = None,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    h = h or handoff()
    catalog = catalog or hardware_catalog()

    return generate_full_architectures(
        handoff=h,
        hardware_catalog=catalog,
        max_paths_per_variant=1,
        max_role_options_per_role=2,
        max_architectures=4,
    )


def first_architecture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    h = handoff()
    catalog = hardware_catalog()
    g = generated(h=h, catalog=catalog)
    return copy.deepcopy(g["architectures"][0]), h, catalog


def codes(result: dict[str, Any]) -> set[str]:
    return {str(item["code"]) for item in result["violations"]}


def test_valid_generated_architecture_is_accepted() -> None:
    architecture, h, catalog = first_architecture()
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert result["valid"] is True
    assert result["decision"] == "VALID"
    assert result["violations"] == []


def test_validated_state_is_marked_validated() -> None:
    architecture, h, catalog = first_architecture()
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    validated = result["validated_state"]
    assert validated["validation"]["status"] == "VALIDATED"
    assert validated["validation"]["is_valid"] is True
    validate_full_architecture_state(validated)


def test_budget_exceeded_is_invalid() -> None:
    architecture, _, catalog = first_architecture()
    low_budget = handoff(max_budget=1.0)
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=low_budget,
        hardware_catalog=catalog,
    )
    assert result["valid"] is False
    assert "budget_exceeded" in codes(result)


def test_power_exceeded_is_invalid() -> None:
    architecture, _, catalog = first_architecture()
    low_power = handoff(max_power=1.0)
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=low_power,
        hardware_catalog=catalog,
    )
    assert result["valid"] is False
    assert "power_exceeded" in codes(result)


def test_tampered_physical_count_is_detected() -> None:
    architecture, h, catalog = first_architecture()
    architecture["state"]["counts"]["ost_physical_drives"] += 1
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert "state_count_mismatch" in codes(result)


def test_tampered_performance_is_detected() -> None:
    architecture, h, catalog = first_architecture()
    architecture["state"]["performance"]["ost_read_bandwidth_gb_s"] *= 2.0
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert "state_performance_mismatch" in codes(result)


def test_tampered_total_cost_is_detected() -> None:
    architecture, h, catalog = first_architecture()
    architecture["state"]["cost_power"]["total_cost_usd"] += 100.0
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert "state_cost_power_mismatch" in codes(result)


def test_tampered_protection_group_count_is_detected() -> None:
    architecture, h, catalog = first_architecture()
    architecture["state"]["selected"]["mdt_protection"]["group_count"] += 1
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert "mdt_protection_mismatch" in codes(result)


def test_unknown_controller_is_invalid_not_runtime_crash() -> None:
    architecture, h, catalog = first_architecture()
    architecture["state"]["selected"]["ost_hardware_path"][
        "controller_id"
    ] = "CTRL_UNKNOWN"
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert result["valid"] is False
    assert "architecture_recompute_failed" in codes(result)


def test_incompatible_controller_protocol_is_detected() -> None:
    architecture, h, catalog = first_architecture()
    broken_catalog = copy.deepcopy(catalog)
    broken_catalog["controllers"][0]["supported_protocols"] = ["NVME"]
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=broken_catalog,
    )
    assert result["valid"] is False
    assert "ost_hardware_path_incompatible" in codes(result) or (
        "mdt_hardware_path_incompatible" in codes(result)
    )


def test_wrong_selected_drive_is_integrity_failure() -> None:
    architecture, h, catalog = first_architecture()
    architecture["state"]["selected"]["mdt_drive"]["drive_id"] = "OTHER"
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert "architecture_recompute_failed" in codes(result)


def test_batch_validation_counts_all_architectures() -> None:
    h = handoff()
    catalog = hardware_catalog()
    g = generated(h=h, catalog=catalog)
    result = validate_generated_architectures(
        generation_result=g,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert result["summary"]["architecture_count"] == len(g["architectures"])


def test_batch_validation_has_valid_architecture() -> None:
    h = handoff()
    catalog = hardware_catalog()
    result = validate_generated_architectures(
        generation_result=generated(h=h, catalog=catalog),
        handoff=h,
        hardware_catalog=catalog,
    )
    assert result["summary"]["has_valid_architecture"] is True


def test_batch_can_be_valid_execution_with_zero_valid_architectures() -> None:
    h = handoff(max_budget=1.0)
    catalog = hardware_catalog()
    g = generated(h=h, catalog=catalog)
    result = validate_generated_architectures(
        generation_result=g,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert result["summary"]["valid_architecture_count"] == 0
    assert result["summary"]["has_valid_architecture"] is False
    assert_full_validation_result_valid(result)


def test_h10_is_deterministic() -> None:
    h = handoff()
    catalog = hardware_catalog()
    g = generated(h=h, catalog=catalog)
    first = validate_generated_architectures(
        generation_result=g,
        handoff=h,
        hardware_catalog=catalog,
    )
    second = validate_generated_architectures(
        generation_result=copy.deepcopy(g),
        handoff=copy.deepcopy(h),
        hardware_catalog=copy.deepcopy(catalog),
    )
    assert first == second


def test_h10_does_not_mutate_input_architecture() -> None:
    architecture, h, catalog = first_architecture()
    before = copy.deepcopy(architecture)
    validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert architecture == before


def test_h10_does_not_require_scoring() -> None:
    architecture, h, catalog = first_architecture()
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert result["scoring_required"] is False


def test_h10_does_not_apply_beam_search() -> None:
    architecture, h, catalog = first_architecture()
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert result["beam_search_applied"] is False


def test_validator_policy_is_explicit() -> None:
    architecture, h, catalog = first_architecture()
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=h,
        hardware_catalog=catalog,
    )
    assert result["validator_policy_id"] == VALIDATOR_POLICY_ID


def test_invalid_terminal_state_semantics_are_supported() -> None:
    architecture, h, catalog = first_architecture()
    result = validate_complete_architecture(
        architecture=architecture,
        handoff=handoff(max_budget=1.0),
        hardware_catalog=catalog,
    )
    validated = result["validated_state"]
    assert validated["validation"]["status"] == "INVALID"
    assert validated["validation"]["is_valid"] is False
    assert validated["validation"]["violations"]
    validate_full_architecture_state(validated)


def test_inconsistent_validated_terminal_state_is_rejected() -> None:
    architecture, _, _ = first_architecture()
    state = copy.deepcopy(architecture["state"])
    state["validation"]["status"] = "VALIDATED"
    state["validation"]["is_valid"] = False

    with pytest.raises(ArchitectureStateError, match="VALIDATED exige"):
        validate_full_architecture_state(state)


def test_batch_result_validator_accepts_valid_contract() -> None:
    h = handoff()
    catalog = hardware_catalog()
    result = validate_generated_architectures(
        generation_result=generated(h=h, catalog=catalog),
        handoff=h,
        hardware_catalog=catalog,
    )
    assert_full_validation_result_valid(result)
