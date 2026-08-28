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
    apply_drive_selection,
    apply_protection_selection,
    apply_server_selection,
    apply_storage_fabric_selection,
    build_complete_state_from_choices,
    new_full_architecture_state,
    validate_full_architecture_state,
)


def handoff() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "case_id": "REQ_STATE_001",
        "requested_top_k": 10,
        "requirements": {
            "MDT_requirement": {},
            "OST_requirement": {},
            "constraints": {
                "ha_required": True,
                "max_budget_usd": 100000.0,
                "max_power_w": 10000.0,
            },
            "preferences": {},
        },
        "ranking_provenance": {
            "mdt": {"model_family": "LightGBM", "model_seed": 168},
            "ost": {"model_family": "LightGBM", "model_seed": 84},
        },
    }


def candidate(role: str, drive_id: str) -> dict[str, Any]:
    return {
        "role": role,
        "identity": {
            "drive_id": drive_id,
            "drive_name": drive_id,
            "manufacturer": "Synthetic",
            "series": "Test",
            "media_type": "SSD",
            "catalog_id": f"CAT_{drive_id}",
            "model_number": drive_id,
        },
    }


def protection(role: str, drive_id: str, profile_id: str) -> dict[str, Any]:
    if role == "MDT":
        provided = {
            "usable_capacity_tib": 2.0,
            "read_iops": 300000.0,
            "write_iops": 200000.0,
        }
        satisfied = {"capacity": True, "read_iops": True, "write_iops": True}
        group_count = 2
        physical_count = 4
        cost = 1200.0
        power = 40.0
    else:
        provided = {
            "usable_capacity_tib": 100.0,
            "read_bandwidth_gb_s": 12.0,
            "write_bandwidth_gb_s": 8.0,
            "total_bandwidth_gb_s": 20.0,
        }
        satisfied = {
            "capacity": True,
            "read_bandwidth": True,
            "write_bandwidth": True,
            "total_bandwidth": True,
        }
        group_count = 2
        physical_count = 20
        cost = 4000.0
        power = 200.0

    return {
        "role": role,
        "drive_id": drive_id,
        "protection_profile_id": profile_id,
        "group_count": group_count,
        "physical_drive_count": physical_count,
        "provided": provided,
        "satisfied": satisfied,
        "protected_drive_cost_usd": cost,
        "protected_drive_power_w": power,
    }


def path(role: str, drive_id: str, profile_id: str) -> dict[str, Any]:
    is_mdt = role == "MDT"
    return {
        "compatible": True,
        "role": role,
        "attachment_mode": "ENCLOSURE",
        "drive_id": drive_id,
        "protection_profile_id": profile_id,
        "server_id": "MDS_01" if is_mdt else "OSS_01",
        "controller_id": "CTRL_MDT" if is_mdt else "CTRL_OST",
        "enclosure_id": "ENC_MDT" if is_mdt else "ENC_OST",
        "network_id": "NET_01",
        "ha_profile_id": "HA_ACTIVE_PASSIVE",
        "minimum_resources": {
            "physical_drive_count": 4 if is_mdt else 20,
            "server_count": 2,
            "controller_count": 2,
            "enclosure_count": 1,
            "network_adapter_count": 2,
        },
        "component_cost_lower_bound_usd": 15000.0 if is_mdt else 24000.0,
        "component_power_lower_bound_w": 1300.0 if is_mdt else 2100.0,
    }


def choices() -> dict[str, Any]:
    return {
        "mdt_candidate": candidate("MDT", "MDT_A"),
        "ost_candidate": candidate("OST", "OST_A"),
        "mdt_protection": protection("MDT", "MDT_A", "PROT_RAID1_2"),
        "ost_protection": protection("OST", "OST_A", "PROT_RAID6_10"),
        "mdt_path": path("MDT", "MDT_A", "PROT_RAID1_2"),
        "ost_path": path("OST", "OST_A", "PROT_RAID6_10"),
    }


def build_complete() -> dict[str, Any]:
    c = choices()
    return build_complete_state_from_choices(handoff=handoff(), **c)


def test_new_state_is_empty() -> None:
    state = new_full_architecture_state(handoff=handoff())
    assert state["stage"] == "EMPTY"
    assert state["trace"] == []
    validate_full_architecture_state(state)


def test_drive_transition_is_immutable() -> None:
    initial = new_full_architecture_state(handoff=handoff())
    result = apply_drive_selection(
        state=initial,
        mdt_candidate=candidate("MDT", "MDT_A"),
        ost_candidate=candidate("OST", "OST_A"),
    )
    assert initial["stage"] == "EMPTY"
    assert result["stage"] == "DRIVES_SELECTED"


def test_wrong_transition_order_is_rejected() -> None:
    initial = new_full_architecture_state(handoff=handoff())
    with pytest.raises(ArchitectureStateError, match="Transition interdite"):
        apply_storage_fabric_selection(state=initial)


def test_protection_must_match_selected_drive() -> None:
    state = new_full_architecture_state(handoff=handoff())
    state = apply_drive_selection(
        state=state,
        mdt_candidate=candidate("MDT", "MDT_A"),
        ost_candidate=candidate("OST", "OST_A"),
    )
    with pytest.raises(ArchitectureStateError, match="autre drive"):
        apply_protection_selection(
            state=state,
            mdt_protection=protection("MDT", "OTHER", "PROT_RAID1_2"),
            ost_protection=protection("OST", "OST_A", "PROT_RAID6_10"),
        )


def test_protection_transition_sets_physical_counts() -> None:
    c = choices()
    state = new_full_architecture_state(handoff=handoff())
    state = apply_drive_selection(
        state=state,
        mdt_candidate=c["mdt_candidate"],
        ost_candidate=c["ost_candidate"],
    )
    state = apply_protection_selection(
        state=state,
        mdt_protection=c["mdt_protection"],
        ost_protection=c["ost_protection"],
    )
    assert state["counts"]["mdt_physical_drives"] == 4
    assert state["counts"]["ost_physical_drives"] == 20


def test_server_transition_sets_mds_and_oss_counts() -> None:
    c = choices()
    state = new_full_architecture_state(handoff=handoff())
    state = apply_drive_selection(state=state, mdt_candidate=c["mdt_candidate"], ost_candidate=c["ost_candidate"])
    state = apply_protection_selection(state=state, mdt_protection=c["mdt_protection"], ost_protection=c["ost_protection"])
    state = apply_server_selection(state=state, mdt_path=c["mdt_path"], ost_path=c["ost_path"])
    assert state["counts"]["mds_count"] == 2
    assert state["counts"]["oss_count"] == 2


def test_incompatible_path_is_rejected() -> None:
    c = choices()
    c["mdt_path"]["compatible"] = False
    state = new_full_architecture_state(handoff=handoff())
    state = apply_drive_selection(state=state, mdt_candidate=c["mdt_candidate"], ost_candidate=c["ost_candidate"])
    state = apply_protection_selection(state=state, mdt_protection=c["mdt_protection"], ost_protection=c["ost_protection"])
    with pytest.raises(ArchitectureStateError, match="non compatible"):
        apply_server_selection(state=state, mdt_path=c["mdt_path"], ost_path=c["ost_path"])


def test_complete_state_sets_controller_and_enclosure_counts() -> None:
    state = build_complete()
    assert state["counts"]["mdt_controller_count"] == 2
    assert state["counts"]["ost_enclosure_count"] == 1


def test_complete_state_sets_network_adapter_counts() -> None:
    state = build_complete()
    assert state["counts"]["mdt_network_adapter_count"] == 2
    assert state["counts"]["ost_network_adapter_count"] == 2


def test_complete_state_has_total_cost_and_power() -> None:
    state = build_complete()
    assert state["stage"] == "COMPLETE"
    assert state["cost_power"]["total_cost_usd"] == pytest.approx(44200.0)
    assert state["cost_power"]["total_power_w"] == pytest.approx(3640.0)


def test_complete_state_remains_pending_full_validator() -> None:
    state = build_complete()
    assert state["validation"]["is_complete"] is True
    assert state["validation"]["is_valid"] is False
    assert state["validation"]["status"] == "PENDING_FULL_VALIDATOR"


def test_trace_is_contiguous_and_has_six_transitions() -> None:
    state = build_complete()
    assert [item["sequence"] for item in state["trace"]] == [1, 2, 3, 4, 5, 6]


def test_build_complete_is_deterministic() -> None:
    assert build_complete() == build_complete()


def test_validation_rejects_broken_trace_sequence() -> None:
    state = build_complete()
    broken = copy.deepcopy(state)
    broken["trace"][2]["sequence"] = 99
    with pytest.raises(ArchitectureStateError, match="non contiguë"):
        validate_full_architecture_state(broken)
