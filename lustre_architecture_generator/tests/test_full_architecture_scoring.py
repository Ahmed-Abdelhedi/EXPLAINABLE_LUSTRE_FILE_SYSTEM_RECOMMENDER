from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from full_architecture.architecture_scoring import (  # noqa: E402
    ArchitectureScoringError,
    assert_scoring_result_valid,
    normalize_preference_weights,
    score_generated_architectures,
)


def candidate(
    role: str,
    drive_id: str,
    media_type: str,
    mtbf: float,
    warranty: float,
    endurance: float | None,
    workload_rating: float | None,
) -> dict[str, Any]:
    return {
        "role": role,
        "identity": {
            "drive_id": drive_id,
            "media_type": media_type,
        },
        "reliability": {
            "endurance_dwpd": endurance,
            "mtbf_hours": mtbf,
            "warranty_years": warranty,
            "workload_rating_tb_per_year": workload_rating,
        },
    }


def handoff(
    *,
    performance_priority: float = 0.4,
    cost_priority: float = 0.2,
    power_priority: float = 0.2,
    reliability_priority: float = 0.2,
    max_budget: float = 100000.0,
    max_power: float = 10000.0,
) -> dict[str, Any]:
    return {
        "case_id": "REQ_SCORE_001",
        "requirements": {
            "MDT_requirement": {
                "required_metadata_capacity_tib": 1.0,
                "required_read_iops": 1000.0,
                "required_write_iops": 500.0,
            },
            "OST_requirement": {
                "required_usable_capacity_tib": 100.0,
                "required_read_bandwidth_gbps": 10.0,
                "required_write_bandwidth_gbps": 5.0,
                "required_total_bandwidth_gbps": 15.0,
            },
            "constraints": {
                "ha_required": True,
                "max_budget_usd": max_budget,
                "max_power_w": max_power,
            },
            "preferences": {
                "performance_priority": performance_priority,
                "cost_priority": cost_priority,
                "power_priority": power_priority,
                "reliability_priority": reliability_priority,
            },
        },
        "mdt_candidates": [
            candidate(
                "MDT",
                "MDT_A",
                "SSD",
                2000000,
                5,
                1.0,
                None,
            ),
            candidate(
                "MDT",
                "MDT_B",
                "SSD",
                2500000,
                5,
                3.0,
                None,
            ),
        ],
        "ost_candidates": [
            candidate(
                "OST",
                "OST_A",
                "HDD",
                2000000,
                5,
                None,
                300.0,
            ),
            candidate(
                "OST",
                "OST_B",
                "HDD",
                2500000,
                5,
                None,
                550.0,
            ),
        ],
    }


def state(
    *,
    mdt_drive: str,
    ost_drive: str,
    cost: float,
    power: float,
    perf_factor: float,
    mdt_ft: int = 1,
    ost_ft: int = 2,
    ha: bool = True,
) -> dict[str, Any]:
    ha_id = "HA_ACTIVE_PASSIVE" if ha else "HA_NONE"

    return {
        "schema_version": "1.1",
        "case_id": "REQ_SCORE_001",
        "stage": "COMPLETE",
        "source": {},
        "requirements": {},
        "selected": {
            "mdt_drive": {"drive_id": mdt_drive},
            "ost_drive": {"drive_id": ost_drive},
            "mdt_protection": {
                "drive_id": mdt_drive,
                "protection_profile_id": "PROT_MDT",
                "fault_tolerance_drives_per_group": mdt_ft,
            },
            "ost_protection": {
                "drive_id": ost_drive,
                "protection_profile_id": "PROT_OST",
                "fault_tolerance_drives_per_group": ost_ft,
            },
            "mdt_hardware_path": {
                "compatible": True,
                "role": "MDT",
                "drive_id": mdt_drive,
                "protection_profile_id": "PROT_MDT",
                "ha_profile_id": ha_id,
            },
            "ost_hardware_path": {
                "compatible": True,
                "role": "OST",
                "drive_id": ost_drive,
                "protection_profile_id": "PROT_OST",
                "ha_profile_id": ha_id,
            },
        },
        "counts": {
            "mdt_physical_drives": 2,
            "ost_physical_drives": 20,
            "mdt_count": 1,
            "ost_count": 2,
            "mds_count": 2,
            "oss_count": 2,
            "mdt_controller_count": 2,
            "ost_controller_count": 2,
            "mdt_enclosure_count": 1,
            "ost_enclosure_count": 1,
            "mdt_network_adapter_count": 2,
            "ost_network_adapter_count": 2,
        },
        "performance": {
            "metadata_capacity_tib": 1.0 * perf_factor,
            "mdt_read_iops": 1000.0 * perf_factor,
            "mdt_write_iops": 500.0 * perf_factor,
            "ost_usable_capacity_tib": 100.0 * perf_factor,
            "ost_read_bandwidth_gb_s": 10.0 * perf_factor,
            "ost_write_bandwidth_gb_s": 5.0 * perf_factor,
            "ost_total_bandwidth_gb_s": 15.0 * perf_factor,
        },
        "cost_power": {
            "mdt_drive_cost_usd": cost * 0.1,
            "ost_drive_cost_usd": cost * 0.4,
            "mdt_hardware_cost_usd": cost * 0.2,
            "ost_hardware_cost_usd": cost * 0.3,
            "total_cost_usd": cost,
            "mdt_drive_power_w": power * 0.1,
            "ost_drive_power_w": power * 0.4,
            "mdt_hardware_power_w": power * 0.2,
            "ost_hardware_power_w": power * 0.3,
            "total_power_w": power,
        },
        "validation": {
            "is_complete": True,
            "is_valid": False,
            "status": "PENDING_FULL_VALIDATOR",
            "violations": [],
        },
        "trace": [],
    }


def generation() -> dict[str, Any]:
    return {
        "case_id": "REQ_SCORE_001",
        "architectures": [
            {
                "architecture_id": "ARCH_A",
                "case_id": "REQ_SCORE_001",
                "generation_index": 1,
                "state": state(
                    mdt_drive="MDT_A",
                    ost_drive="OST_A",
                    cost=40000.0,
                    power=4000.0,
                    perf_factor=1.2,
                    mdt_ft=1,
                    ost_ft=1,
                    ha=False,
                ),
            },
            {
                "architecture_id": "ARCH_B",
                "case_id": "REQ_SCORE_001",
                "generation_index": 2,
                "state": state(
                    mdt_drive="MDT_B",
                    ost_drive="OST_B",
                    cost=60000.0,
                    power=5000.0,
                    perf_factor=2.0,
                    mdt_ft=2,
                    ost_ft=2,
                    ha=True,
                ),
            },
        ],
    }


def test_preference_weights_are_normalized() -> None:
    info = normalize_preference_weights(
        {
            "performance_priority": 4.0,
            "cost_priority": 2.0,
            "power_priority": 2.0,
            "reliability_priority": 2.0,
        }
    )

    assert info["normalized_sum"] == pytest.approx(1.0)
    assert info["normalized"]["performance_priority"] == pytest.approx(0.4)


def test_zero_preference_sum_is_rejected() -> None:
    with pytest.raises(
        ArchitectureScoringError,
        match="somme des poids",
    ):
        normalize_preference_weights(
            {
                "performance_priority": 0.0,
                "cost_priority": 0.0,
                "power_priority": 0.0,
                "reliability_priority": 0.0,
            }
        )


def test_scoring_returns_one_record_per_architecture() -> None:
    result = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(),
    )

    assert result["summary"]["architecture_count"] == 2
    assert len(result["architectures"]) == 2


def test_scores_are_bounded() -> None:
    result = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(),
    )

    for record in result["architectures"]:
        assert 0.0 <= record["score"] <= 1.0
        for component in record["components"].values():
            assert 0.0 <= component <= 1.0


def test_high_performance_weight_prefers_high_headroom() -> None:
    result = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(
            performance_priority=1.0,
            cost_priority=0.0,
            power_priority=0.0,
            reliability_priority=0.0,
        ),
    )

    assert result["architectures"][0]["architecture_id"] == "ARCH_B"


def test_high_cost_weight_prefers_lower_cost() -> None:
    result = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(
            performance_priority=0.0,
            cost_priority=1.0,
            power_priority=0.0,
            reliability_priority=0.0,
        ),
    )

    assert result["architectures"][0]["architecture_id"] == "ARCH_A"


def test_high_power_weight_prefers_lower_power() -> None:
    result = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(
            performance_priority=0.0,
            cost_priority=0.0,
            power_priority=1.0,
            reliability_priority=0.0,
        ),
    )

    assert result["architectures"][0]["architecture_id"] == "ARCH_A"


def test_high_reliability_weight_prefers_stronger_proxy() -> None:
    result = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(
            performance_priority=0.0,
            cost_priority=0.0,
            power_priority=0.0,
            reliability_priority=1.0,
        ),
    )

    assert result["architectures"][0]["architecture_id"] == "ARCH_B"


def test_over_budget_architecture_is_flagged_but_not_validated_here() -> None:
    h = handoff(
        performance_priority=1.0,
        cost_priority=0.0,
        power_priority=0.0,
        reliability_priority=0.0,
        max_budget=50000.0,
    )

    result = score_generated_architectures(
        generation_result=generation(),
        handoff=h,
    )

    by_id = {
        row["architecture_id"]: row
        for row in result["architectures"]
    }

    assert by_id["ARCH_B"]["pre_h10_hard_snapshot_passed"] is False
    assert (
        by_id["ARCH_B"]["hard_constraint_snapshot"]["budget"]["satisfied"]
        is False
    )
    assert by_id["ARCH_B"]["full_validator_applied"] is False


def test_over_power_architecture_is_ineligible() -> None:
    h = handoff(max_power=4500.0)

    result = score_generated_architectures(
        generation_result=generation(),
        handoff=h,
    )

    by_id = {
        row["architecture_id"]: row
        for row in result["architectures"]
    }

    assert by_id["ARCH_B"]["pre_h10_hard_snapshot_passed"] is False


def test_underperforming_architecture_is_ineligible() -> None:
    g = generation()
    g["architectures"][0]["state"]["performance"]["mdt_read_iops"] = 900.0

    result = score_generated_architectures(
        generation_result=g,
        handoff=handoff(),
    )

    by_id = {
        row["architecture_id"]: row
        for row in result["architectures"]
    }

    assert by_id["ARCH_A"]["pre_h10_hard_snapshot_passed"] is False


def test_h9_preserves_pending_full_validator_status() -> None:
    result = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(),
    )

    for row in result["architectures"]:
        assert (
            row["state_validation_status_preserved"]
            == "PENDING_FULL_VALIDATOR"
        )
        assert row["full_validator_applied"] is False


def test_no_beam_search_is_applied() -> None:
    result = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(),
    )

    assert result["scoring_policy"]["beam_search_applied"] is False
    assert all(
        row["beam_search_applied"] is False
        for row in result["architectures"]
    )


def test_scoring_does_not_mutate_generation_result() -> None:
    g = generation()
    before = copy.deepcopy(g)

    score_generated_architectures(
        generation_result=g,
        handoff=handoff(),
    )

    assert g == before


def test_scoring_is_deterministic() -> None:
    first = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(),
    )
    second = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(),
    )

    assert first == second


def test_validator_accepts_valid_result() -> None:
    result = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(),
    )

    assert_scoring_result_valid(result)


def test_validator_rejects_score_above_one() -> None:
    result = score_generated_architectures(
        generation_result=generation(),
        handoff=handoff(),
    )
    result["architectures"][0]["score"] = 1.1

    with pytest.raises(
        ArchitectureScoringError,
        match="score H9 > 1",
    ):
        assert_scoring_result_valid(result)
