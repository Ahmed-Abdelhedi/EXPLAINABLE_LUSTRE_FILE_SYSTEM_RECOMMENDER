from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from full_architecture.protection_arithmetic import (  # noqa: E402
    ProtectionArithmeticError,
    assert_protection_result_valid,
    calculate_mdt_protection,
    calculate_ost_protection,
    enumerate_candidate_protections,
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


def mdt_candidate() -> dict[str, Any]:
    return {
        "role": "MDT",
        "identity": {
            "drive_id": "MDT_A",
        },
        "pre_raid": {
            "minimum_drive_count": 3,
            "is_lower_bound_only": True,
            "provided_capacity_tib": 3.0,
            "provided_read_iops": 300000.0,
            "provided_write_iops": 240000.0,
            "drive_level_cost_usd": 900.0,
            "drive_level_power_w": 30.0,
        },
    }


def ost_candidate() -> dict[str, Any]:
    return {
        "role": "OST",
        "identity": {
            "drive_id": "OST_A",
        },
        "pre_raid": {
            "minimum_drive_count": 14,
            "is_lower_bound_only": True,
            "provided_capacity_tib": 140.0,
            "provided_read_bandwidth_gb_s": 7.0,
            "provided_write_bandwidth_gb_s": 5.6,
            "provided_total_bandwidth_gb_s": 12.6,
            "drive_level_cost_usd": 2800.0,
            "drive_level_power_w": 140.0,
        },
    }


def test_mdt_raid1_expands_to_whole_groups() -> None:
    result = calculate_mdt_protection(
        candidate=mdt_candidate(),
        protection_profile=raid1(),
        requirement={
            "required_metadata_capacity_tib": 1.0,
            "required_read_iops": 100000.0,
            "required_write_iops": 80000.0,
        },
    )

    assert result["group_count"] == 2
    assert result["physical_drive_count"] == 4
    assert result["physical_drive_count"] >= 3


def test_mdt_raid1_capacity_is_mirrored() -> None:
    result = calculate_mdt_protection(
        candidate=mdt_candidate(),
        protection_profile=raid1(),
        requirement={
            "required_metadata_capacity_tib": 2.0,
            "required_read_iops": 100000.0,
            "required_write_iops": 80000.0,
        },
    )

    assert result["provided"]["usable_capacity_tib"] == pytest.approx(2.0)
    assert result["satisfied"]["capacity"] is True


def test_ost_raid6_rounds_to_complete_groups() -> None:
    result = calculate_ost_protection(
        candidate=ost_candidate(),
        protection_profile=raid6(),
        requirement={
            "required_usable_capacity_tib": 100.0,
            "required_read_bandwidth_gbps": 5.0,
            "required_write_bandwidth_gbps": 4.0,
            "required_total_bandwidth_gbps": 9.0,
        },
    )

    assert result["physical_drive_count"] % 10 == 0
    assert result["physical_drive_count"] >= 14


def test_ost_raid6_can_be_capacity_driven() -> None:
    result = calculate_ost_protection(
        candidate=ost_candidate(),
        protection_profile=raid6(),
        requirement={
            "required_usable_capacity_tib": 150.0,
            "required_read_bandwidth_gbps": 1.0,
            "required_write_bandwidth_gbps": 1.0,
            "required_total_bandwidth_gbps": 2.0,
        },
    )

    assert result["counts_basis"]["capacity"] == 2
    assert result["group_count"] >= 2


def test_ost_write_penalty_is_applied() -> None:
    result = calculate_ost_protection(
        candidate=ost_candidate(),
        protection_profile=raid6(),
        requirement={
            "required_usable_capacity_tib": 20.0,
            "required_read_bandwidth_gbps": 1.0,
            "required_write_bandwidth_gbps": 4.0,
            "required_total_bandwidth_gbps": 5.0,
        },
    )

    per_drive_write = 5.6 / 14.0
    expected_group_write = (
        per_drive_write
        * 8
        * 0.68
    )

    assert (
        result["provided"]["write_bandwidth_gb_s"]
        >= expected_group_write
    )


def test_protected_cost_and_power_scale_with_physical_count() -> None:
    result = calculate_mdt_protection(
        candidate=mdt_candidate(),
        protection_profile=raid1(),
        requirement={
            "required_metadata_capacity_tib": 1.0,
            "required_read_iops": 100000.0,
            "required_write_iops": 80000.0,
        },
    )

    assert result["per_drive_cost_usd"] == pytest.approx(300.0)
    assert result["per_drive_power_w"] == pytest.approx(10.0)
    assert result["protected_drive_cost_usd"] == pytest.approx(1200.0)
    assert result["protected_drive_power_w"] == pytest.approx(40.0)


def test_result_validator_accepts_valid_result() -> None:
    result = calculate_mdt_protection(
        candidate=mdt_candidate(),
        protection_profile=raid1(),
        requirement={
            "required_metadata_capacity_tib": 1.0,
            "required_read_iops": 100000.0,
            "required_write_iops": 80000.0,
        },
    )

    assert_protection_result_valid(result)


def test_result_validator_rejects_count_below_raw_lower_bound() -> None:
    result = calculate_mdt_protection(
        candidate=mdt_candidate(),
        protection_profile=raid1(),
        requirement={
            "required_metadata_capacity_tib": 1.0,
            "required_read_iops": 100000.0,
            "required_write_iops": 80000.0,
        },
    )

    result["physical_drive_count"] = 2

    with pytest.raises(
        ProtectionArithmeticError,
        match="borne pré-RAID",
    ):
        assert_protection_result_valid(result)


def test_enumeration_keeps_all_profiles() -> None:
    results = enumerate_candidate_protections(
        candidate=ost_candidate(),
        protection_profiles=[
            raid1(),
            raid6(),
        ],
        requirement={
            "required_usable_capacity_tib": 50.0,
            "required_read_bandwidth_gbps": 2.0,
            "required_write_bandwidth_gbps": 2.0,
            "required_total_bandwidth_gbps": 4.0,
        },
    )

    assert len(results) == 2
    assert {
        result["raid_level"]
        for result in results
    } == {
        "RAID1",
        "RAID6",
    }


def test_wrong_role_is_rejected() -> None:
    candidate = ost_candidate()

    with pytest.raises(
        ProtectionArithmeticError,
        match="Role",
    ):
        calculate_mdt_protection(
            candidate=candidate,
            protection_profile=raid1(),
            requirement={
                "required_metadata_capacity_tib": 1.0,
                "required_read_iops": 1.0,
                "required_write_iops": 1.0,
            },
        )
