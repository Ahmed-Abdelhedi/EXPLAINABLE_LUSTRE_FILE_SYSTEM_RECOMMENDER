from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "full_architecture"
    / "handoff_contract.py"
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load_module(
    "_test_full_architecture_handoff",
    MODULE_PATH,
)


def make_drive(drive_id: str) -> dict[str, Any]:
    return {
        "drive_id": drive_id,
        "name": f"Drive {drive_id}",
        "manufacturer": "Synthetic",
        "series": "Test",
        "catalog_id": f"CAT_{drive_id}",
        "model_number": f"MODEL_{drive_id}",
        "media_type": "SSD",
        "protocol": "NVME",
        "drive_form_factor_standard": "FF_U2",
        "pcie_gen_required": 4,
        "pcie_lanes_required": 4,
        "endurance_dwpd_numeric": 3.0,
        "mtbf_hours": 2500000,
        "warranty_years": 5,
        "workload_rating_tb_per_year": None,
        "latency_class": "very_low",
        "quality_status": "verified",
    }


def make_architecture() -> dict[str, Any]:
    return {
        "case_id": "REQ_TEST_001",
        "MDT_requirement": {
            "required_metadata_capacity_tib": 1.0,
            "required_total_iops": 200000,
            "required_read_iops": 120000,
            "required_write_iops": 80000,
            "latency_requirement": "low",
            "endurance_requirement": "medium",
            "reliability_requirement": "high",
        },
        "OST_requirement": {
            "required_usable_capacity_tib": 8.0,
            "required_read_bandwidth_gbps": 4.0,
            "required_write_bandwidth_gbps": 2.0,
            "required_total_bandwidth_gbps": 6.0,
            "reliability_requirement": "high",
        },
        "constraints": {
            "ha_required": True,
            "max_budget_usd": 100000.0,
            "max_power_w": 10000.0,
        },
        "preferences": {
            "performance_priority": 0.4,
            "cost_priority": 0.2,
            "power_priority": 0.1,
            "reliability_priority": 0.3,
        },
        "workload_analysis": {"workload_class": "balanced"},
        "role_analysis": {},
    }


def make_mdt(drive_id: str, rank: int) -> dict[str, Any]:
    return {
        "drive_id": drive_id,
        "drive_name": f"Drive {drive_id}",
        "manufacturer": "Synthetic",
        "series": "Test",
        "media_type": "SSD",
        "protocol": "NVME",
        "ml_score": 10.0 - rank,
        "ml_rank": rank,
        "raw_minimum_drive_count": 1,
        "raw_provided_capacity_tib": 4.0,
        "raw_provided_read_iops": 500000.0,
        "raw_provided_write_iops": 300000.0,
        "raw_drive_cost_usd": 500.0,
        "raw_drive_power_w": 15.0,
    }


def make_ost(drive_id: str, rank: int) -> dict[str, Any]:
    return {
        "drive_id": drive_id,
        "drive_name": f"Drive {drive_id}",
        "manufacturer": "Synthetic",
        "series": "Test",
        "media_type": "SSD",
        "protocol": "NVME",
        "ml_score": 10.0 - rank,
        "ml_rank": rank,
        "raw_minimum_drive_count": 2,
        "raw_provided_capacity_tib": 8.0,
        "raw_provided_read_bandwidth_gbps": 14.0,
        "raw_provided_write_bandwidth_gbps": 10.0,
        "raw_provided_total_bandwidth_gbps": 24.0,
        "raw_drive_cost_usd": 1000.0,
        "raw_drive_power_w": 30.0,
        "selection_reasons": ["global_ml_top"],
        "diversified_rank": rank,
    }


def make_valid_handoff() -> dict[str, Any]:
    arch = make_architecture()
    catalog = [
        make_drive("MDT_A"),
        make_drive("MDT_B"),
        make_drive("OST_A"),
        make_drive("OST_B"),
    ]

    mdt_items = [
        make_mdt("MDT_A", 1),
        make_mdt("MDT_B", 2),
    ]
    ost_items = [
        make_ost("OST_A", 1),
        make_ost("OST_B", 2),
    ]

    mdt_result = {
        "case_id": arch["case_id"],
        "model_family": "LightGBM",
        "model_type": "LGBMRanker",
        "model_seed": 168,
        "feature_count": 49,
        "feasible_candidate_count": 2,
        "ranked_candidates": mdt_items,
    }
    ost_result = {
        "case_id": arch["case_id"],
        "model_family": "LightGBM",
        "model_type": "LGBMRanker",
        "model_seed": 84,
        "feature_count": 52,
        "feasible_candidate_count": 2,
        "ranked_candidates": ost_items,
    }
    diversified = {
        "case_id": arch["case_id"],
        "requested_top_k": 2,
        "selected_count": 2,
        "global_top_count": 2,
        "diversification_pool_size": 2,
        "maximum_specialized_ml_rank": None,
        "media_distribution": {"SSD": 2},
        "diversified_candidates": ost_items,
    }

    return contract.assemble_architecture_handoff(
        arch,
        catalog,
        mdt_result,
        ost_result,
        diversified,
        2,
    )

def test_valid_handoff() -> None:
    handoff = make_valid_handoff()
    assert contract.validate_architecture_handoff(handoff) == []


def test_pre_raid_counts_are_lower_bounds() -> None:
    handoff = make_valid_handoff()

    for item in handoff["mdt_candidates"] + handoff["ost_candidates"]:
        assert item["pre_raid"]["is_lower_bound_only"] is True
        assert item["pre_raid"]["minimum_drive_count"] > 0


def test_ost_unit_contract_is_gb_per_second() -> None:
    handoff = make_valid_handoff()
    assert handoff["unit_contract"]["throughput_rate"] == "GB/s"
    assert (
        handoff["ost_candidates"][0]["pre_raid"][
            "provided_read_bandwidth_gb_s"
        ]
        == pytest.approx(14.0)
    )


def test_catalog_enrichment() -> None:
    candidate = make_valid_handoff()["mdt_candidates"][0]

    assert candidate["hardware_interface"]["form_factor"] == "FF_U2"
    assert candidate["hardware_interface"]["pcie_lanes_required"] == 4
    assert candidate["reliability"]["mtbf_hours"] == 2500000


def test_duplicate_candidate_is_rejected() -> None:
    handoff = make_valid_handoff()
    duplicate = copy.deepcopy(handoff["mdt_candidates"][0])
    duplicate["selection_rank"] = 3

    handoff["mdt_candidates"].append(duplicate)
    handoff["actual_top_k"]["mdt"] = 3
    handoff["requested_top_k"] = 3

    errors = contract.validate_architecture_handoff(handoff)
    assert any("dupliqué" in error for error in errors)


def test_final_hardware_is_forbidden() -> None:
    handoff = make_valid_handoff()
    handoff["mdt_candidates"][0]["raid_level"] = "RAID1"

    errors = contract.validate_architecture_handoff(handoff)
    assert any("hardware finales" in error for error in errors)


def test_unsatisfied_evidence_is_rejected() -> None:
    handoff = make_valid_handoff()
    handoff["ost_candidates"][0][
        "deterministic_filter_evidence"
    ]["capacity"]["satisfied"] = False

    errors = contract.validate_architecture_handoff(handoff)
    assert any("capacity" in error for error in errors)


def test_missing_catalog_drive_is_rejected() -> None:
    arch = make_architecture()

    with pytest.raises(contract.ArchitectureHandoffError):
        contract.assemble_architecture_handoff(
            arch,
            [make_drive("OST_A")],
            {
                "case_id": arch["case_id"],
                "ranked_candidates": [make_mdt("MISSING", 1)],
            },
            {
                "case_id": arch["case_id"],
                "ranked_candidates": [make_ost("OST_A", 1)],
            },
            {
                "case_id": arch["case_id"],
                "diversified_candidates": [make_ost("OST_A", 1)],
            },
            1,
        )

def test_pcie_gen_string_is_preserved_as_generation_number() -> None:
    drive = make_drive("MDT_GEN5")
    drive["pcie_gen_required"] = "GEN5"

    package = contract.build_mdt_candidate_package(
        make_mdt("MDT_GEN5", 1),
        drive,
        make_architecture()["MDT_requirement"],
        make_architecture()["constraints"],
        1,
    )

    assert (
        package["hardware_interface"]["pcie_gen_required"]
        == 5
    )


def test_pcie_gen_verbose_string_is_preserved() -> None:
    drive = make_drive("OST_GEN5")
    drive["pcie_gen_required"] = "PCIe Gen5"

    package = contract.build_ost_candidate_package(
        make_ost("OST_GEN5", 1),
        drive,
        make_architecture()["OST_requirement"],
        make_architecture()["constraints"],
        1,
    )

    assert (
        package["hardware_interface"]["pcie_gen_required"]
        == 5
    )
