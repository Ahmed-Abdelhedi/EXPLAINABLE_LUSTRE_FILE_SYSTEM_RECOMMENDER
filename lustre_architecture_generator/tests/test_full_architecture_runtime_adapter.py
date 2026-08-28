from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from full_architecture.runtime_adapter import (  # noqa: E402
    ArchitectureRuntimeAdapterError,
    build_runtime_handoff,
)


def architecture() -> dict[str, Any]:
    return {
        "case_id": "REQ_RUNTIME_001",
        "MDT_requirement": {
            "required_metadata_capacity_tib": 1.0,
            "required_total_iops": 100000,
            "required_read_iops": 60000,
            "required_write_iops": 40000,
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
        "workload_analysis": {},
        "role_analysis": {},
    }


def catalog() -> list[dict[str, Any]]:
    common = {
        "manufacturer": "Synthetic",
        "series": "Test",
        "pcie_gen_required": None,
        "pcie_lanes_required": None,
        "endurance_dwpd_numeric": 3.0,
        "mtbf_hours": 2500000,
        "warranty_years": 5,
        "workload_rating_tb_per_year": None,
        "latency_class": "low",
        "quality_status": "verified",
    }

    return [
        {
            **common,
            "drive_id": "MDT_A",
            "name": "MDT A",
            "catalog_id": "CAT_MDT_A",
            "model_number": "MODEL_MDT_A",
            "media_type": "SSD",
            "protocol": "NVME",
            "drive_form_factor_standard": "FF_U2",
            "pcie_gen_required": 4,
            "pcie_lanes_required": 4,
        },
        {
            **common,
            "drive_id": "OST_A",
            "name": "OST A",
            "catalog_id": "CAT_OST_A",
            "model_number": "MODEL_OST_A",
            "media_type": "HDD",
            "protocol": "SAS",
            "drive_form_factor_standard": "FF_3_5",
        },
    ]


def fake_mdt(*, architecture: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    del catalog
    return {
        "case_id": architecture["case_id"],
        "model_family": "LightGBM",
        "model_type": "LGBMRanker",
        "model_seed": 168,
        "feature_count": 49,
        "feasible_candidate_count": 1,
        "ranked_candidates": [
            {
                "drive_id": "MDT_A",
                "drive_name": "MDT A",
                "manufacturer": "Synthetic",
                "series": "Test",
                "media_type": "SSD",
                "protocol": "NVME",
                "ml_score": 0.9,
                "ml_rank": 1,
                "raw_minimum_drive_count": 1,
                "raw_provided_capacity_tib": 4.0,
                "raw_provided_read_iops": 500000.0,
                "raw_provided_write_iops": 300000.0,
                "raw_drive_cost_usd": 500.0,
                "raw_drive_power_w": 15.0,
            }
        ],
    }


def fake_ost(*, architecture: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    del catalog
    return {
        "case_id": architecture["case_id"],
        "model_family": "LightGBM",
        "model_type": "LGBMRanker",
        "model_seed": 84,
        "feature_count": 52,
        "feasible_candidate_count": 1,
        "ranked_candidates": [
            {
                "drive_id": "OST_A",
                "drive_name": "OST A",
                "manufacturer": "Synthetic",
                "series": "Test",
                "media_type": "HDD",
                "protocol": "SAS",
                "capacity_tib": 4.0,
                "seq_read_mb_s": 300.0,
                "seq_write_mb_s": 280.0,
                "price_usd": 200.0,
                "power_w": 8.0,
                "ml_score": 0.8,
                "ml_rank": 1,
                "raw_minimum_drive_count": 20,
                "raw_provided_capacity_tib": 80.0,
                "raw_provided_read_bandwidth_gbps": 6.0,
                "raw_provided_write_bandwidth_gbps": 5.6,
                "raw_provided_total_bandwidth_gbps": 11.6,
                "raw_drive_cost_usd": 4000.0,
                "raw_drive_power_w": 160.0,
            }
        ],
    }


def fake_topk(
    *,
    ranking_result: dict[str, Any],
    top_k: int,
    global_top_count: int,
    diversification_multiplier: int,
    minimum_diversification_pool_size: int,
) -> dict[str, Any]:
    del global_top_count, diversification_multiplier, minimum_diversification_pool_size

    candidate = dict(ranking_result["ranked_candidates"][0])
    candidate["selection_reasons"] = ["global_ml_top"]
    candidate["diversified_rank"] = 1

    return {
        "case_id": ranking_result["case_id"],
        "source_candidate_count": 1,
        "requested_top_k": top_k,
        "selected_count": 1,
        "global_top_count": 1,
        "diversification_pool_size": 1,
        "maximum_specialized_ml_rank": None,
        "media_distribution": {"HDD": 1},
        "diversified_candidates": [candidate],
    }


def build() -> dict[str, Any]:
    return build_runtime_handoff(
        architecture=architecture(),
        catalog=catalog(),
        top_k=10,
        mdt_ranker=fake_mdt,
        ost_ranker=fake_ost,
        ost_topk_selector=fake_topk,
    )


def test_runtime_adapter_builds_handoff() -> None:
    handoff = build()
    assert handoff["case_id"] == "REQ_RUNTIME_001"
    assert handoff["actual_top_k"] == {"mdt": 1, "ost": 1}


def test_runtime_adapter_preserves_model_provenance() -> None:
    handoff = build()
    assert handoff["ranking_provenance"]["mdt"]["model_seed"] == 168
    assert handoff["ranking_provenance"]["ost"]["model_seed"] == 84


def test_runtime_adapter_preserves_no_beam_invariant() -> None:
    handoff = build()
    assert handoff["contract_invariants"]["beam_search_not_applied"] is True
    assert handoff["contract_invariants"]["no_final_hardware_selected"] is True


def test_runtime_adapter_passes_topk_parameters() -> None:
    captured: dict[str, Any] = {}

    def selector(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return fake_topk(**kwargs)

    build_runtime_handoff(
        architecture=architecture(),
        catalog=catalog(),
        top_k=7,
        global_top_count=3,
        diversification_multiplier=5,
        minimum_diversification_pool_size=35,
        mdt_ranker=fake_mdt,
        ost_ranker=fake_ost,
        ost_topk_selector=selector,
    )

    assert captured["top_k"] == 7
    assert captured["global_top_count"] == 3
    assert captured["diversification_multiplier"] == 5
    assert captured["minimum_diversification_pool_size"] == 35


def test_runtime_adapter_rejects_wrong_case_id() -> None:
    def bad_mdt(**kwargs: Any) -> dict[str, Any]:
        result = fake_mdt(**kwargs)
        result["case_id"] = "WRONG"
        return result

    with pytest.raises(ArchitectureRuntimeAdapterError, match="case_id"):
        build_runtime_handoff(
            architecture=architecture(),
            catalog=catalog(),
            mdt_ranker=bad_mdt,
            ost_ranker=fake_ost,
            ost_topk_selector=fake_topk,
        )
