from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import full_architecture.feasibility_coverage as coverage  # noqa: E402
from full_architecture.feasibility_coverage import (  # noqa: E402
    FeasibilityCoverageError,
    analyze_case_full_path_domain,
    hardware_path_upper_bound,
    pareto_frontier_options,
    unresolved_case_ids_from_h10b,
)


def option(
    cost: float,
    power: float,
    *,
    marker: str,
    path_index: int = 1,
) -> dict[str, Any]:
    return {
        "candidate": {
            "identity": {
                "drive_id": marker,
            },
        },
        "protection": {
            "protected_drive_cost_usd": cost * 0.4,
            "protected_drive_power_w": power * 0.4,
        },
        "hardware_path": {
            "component_cost_lower_bound_usd": cost * 0.6,
            "component_power_lower_bound_w": power * 0.6,
        },
        "provenance": {
            "hardware_path_index": path_index,
        },
        "marker": marker,
    }


def catalog() -> dict[str, Any]:
    return {
        "servers": [{"id": "S1"}, {"id": "S2"}],
        "controllers": [{"id": "C1"}],
        "networks": [{"id": "N1"}, {"id": "N2"}],
        "ha_profiles": [{"id": "H1"}, {"id": "H2"}],
        "enclosures": [{"id": "E1"}, {"id": "E2"}],
    }


def handoff() -> dict[str, Any]:
    return {
        "case_id": "REQ_PATH_001",
        "requirements": {
            "constraints": {
                "max_budget_usd": 100.0,
                "max_power_w": 100.0,
            },
        },
    }


def test_hardware_path_upper_bound_matches_h6_loop_domain() -> None:
    # 2 servers × 1 controller × 2 networks × 2 HA × (2 enclosures + DIRECT)
    assert hardware_path_upper_bound(catalog()) == 24


def test_hardware_path_upper_bound_requires_non_empty_lists() -> None:
    bad = catalog()
    bad["servers"] = []

    with pytest.raises(
        FeasibilityCoverageError,
        match="liste non vide",
    ):
        hardware_path_upper_bound(bad)


def test_pareto_frontier_removes_dominated_option() -> None:
    items = [
        option(10, 10, marker="A"),
        option(12, 12, marker="B"),
        option(8, 15, marker="C"),
    ]

    frontier = pareto_frontier_options(items)

    assert [item["marker"] for item in frontier] == ["C", "A"]


def test_pareto_frontier_keeps_cost_power_tradeoff() -> None:
    items = [
        option(5, 20, marker="CHEAP"),
        option(20, 5, marker="LOW_POWER"),
    ]

    frontier = pareto_frontier_options(items)

    assert {
        item["marker"]
        for item in frontier
    } == {
        "CHEAP",
        "LOW_POWER",
    }


def test_pareto_frontier_deduplicates_equal_resource_points() -> None:
    items = [
        option(10, 10, marker="FIRST"),
        option(10, 10, marker="SECOND"),
    ]

    frontier = pareto_frontier_options(items)

    assert len(frontier) == 1
    assert frontier[0]["marker"] == "FIRST"


def test_pareto_frontier_rejects_empty_input() -> None:
    with pytest.raises(
        FeasibilityCoverageError,
        match="Aucune option",
    ):
        pareto_frontier_options([])


def test_unresolved_case_ids_from_h10b_filters_recovered_cases() -> None:
    result = unresolved_case_ids_from_h10b(
        {
            "cases": [
                {
                    "case_id": "A",
                    "status": "OK",
                    "recovered_valid_architecture": True,
                },
                {
                    "case_id": "B",
                    "status": "OK",
                    "recovered_valid_architecture": False,
                },
                {
                    "case_id": "C",
                    "status": "FAILED",
                    "recovered_valid_architecture": False,
                },
            ]
        }
    )

    assert result == ["B"]


def test_full_path_analysis_uses_catalog_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_caps: list[int] = []

    def fake_enumerate_role_options(**kwargs: Any) -> list[dict[str, Any]]:
        observed_caps.append(
            kwargs["max_paths_per_variant"]
        )
        return [
            option(
                40,
                40,
                marker=kwargs["role"],
                path_index=3,
            )
        ]

    def fake_confirm(**kwargs: Any) -> dict[str, Any]:
        return {
            "valid": True,
            "architecture_id": "ARCH_OK",
            "decision": "VALID",
        }

    monkeypatch.setattr(
        coverage,
        "enumerate_role_options",
        fake_enumerate_role_options,
    )
    monkeypatch.setattr(
        coverage,
        "_confirm_pair_with_h10",
        fake_confirm,
    )

    result = analyze_case_full_path_domain(
        handoff=handoff(),
        hardware_catalog=catalog(),
    )

    assert observed_caps == [24, 24]
    assert result["recovered_valid_architecture"] is True


def test_full_path_analysis_records_path_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_enumerate_role_options(**kwargs: Any) -> list[dict[str, Any]]:
        path_index = 7 if kwargs["role"] == "MDT" else 9
        return [
            option(
                40,
                40,
                marker=kwargs["role"],
                path_index=path_index,
            )
        ]

    monkeypatch.setattr(
        coverage,
        "enumerate_role_options",
        fake_enumerate_role_options,
    )
    monkeypatch.setattr(
        coverage,
        "_confirm_pair_with_h10",
        lambda **kwargs: {
            "valid": True,
            "architecture_id": "ARCH_OK",
            "decision": "VALID",
        },
    )

    result = analyze_case_full_path_domain(
        handoff=handoff(),
        hardware_catalog=catalog(),
    )

    assert (
        result["selected_path_indexes"][
            "mdt_hardware_path_index"
        ]
        == 7
    )
    assert (
        result["selected_path_indexes"][
            "ost_hardware_path_index"
        ]
        == 9
    )


def test_full_path_analysis_can_remain_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_enumerate_role_options(**kwargs: Any) -> list[dict[str, Any]]:
        return [
            option(
                80,
                80,
                marker=kwargs["role"],
                path_index=4,
            )
        ]

    monkeypatch.setattr(
        coverage,
        "enumerate_role_options",
        fake_enumerate_role_options,
    )

    result = analyze_case_full_path_domain(
        handoff=handoff(),
        hardware_catalog=catalog(),
    )

    assert result["recovered_valid_architecture"] is False
    assert result["global_infeasibility_claimed"] is False
    assert (
        result["coverage_interpretation"]
        == (
            "NO_FEASIBLE_PAIR_WITHIN_CURRENT_TOPK_AND_"
            "FULL_REFERENCE_HARDWARE_PATH_DOMAIN"
        )
    )


def test_full_path_analysis_keeps_topk_uncertainty_when_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coverage,
        "enumerate_role_options",
        lambda **kwargs: [
            option(
                80,
                80,
                marker=kwargs["role"],
            )
        ],
    )

    result = analyze_case_full_path_domain(
        handoff=handoff(),
        hardware_catalog=catalog(),
    )

    assert (
        result["remaining_uncertainty"]
        == "candidate Top-K truncation and reference-catalog scope"
    )


def test_full_path_analysis_does_not_use_scoring_or_beam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        coverage,
        "enumerate_role_options",
        lambda **kwargs: [
            option(
                40,
                40,
                marker=kwargs["role"],
                path_index=5,
            )
        ],
    )
    monkeypatch.setattr(
        coverage,
        "_confirm_pair_with_h10",
        lambda **kwargs: {
            "valid": True,
            "architecture_id": "ARCH_OK",
            "decision": "VALID",
        },
    )

    result = analyze_case_full_path_domain(
        handoff=handoff(),
        hardware_catalog=catalog(),
    )

    assert result["domain"]["beam_search_applied"] is False
    assert result["domain"]["architecture_scoring_required"] is False
