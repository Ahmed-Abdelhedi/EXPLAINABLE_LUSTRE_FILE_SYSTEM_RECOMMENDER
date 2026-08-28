from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from full_architecture.feasibility_coverage import (  # noqa: E402
    FeasibilityCoverageError,
    classify_no_feasible_pair,
    find_first_cost_power_feasible_pair,
    role_option_cost_power,
    unresolved_case_ids_from_h10,
)


def option(cost: float, power: float) -> dict:
    return {
        "protection": {
            "protected_drive_cost_usd": cost * 0.4,
            "protected_drive_power_w": power * 0.4,
        },
        "hardware_path": {
            "component_cost_lower_bound_usd": cost * 0.6,
            "component_power_lower_bound_w": power * 0.6,
        },
    }


def test_role_option_cost_power_is_additive() -> None:
    value = role_option_cost_power(option(1000.0, 100.0))

    assert value["cost_usd"] == pytest.approx(1000.0)
    assert value["power_w"] == pytest.approx(100.0)


def test_first_feasible_pair_uses_deterministic_order() -> None:
    result = find_first_cost_power_feasible_pair(
        mdt_options=[option(10, 10), option(1, 1)],
        ost_options=[option(10, 10), option(1, 1)],
        maximum_budget_usd=11,
        maximum_power_w=11,
    )

    assert result["found"] is True
    assert result["mdt_option_index"] == 1
    assert result["ost_option_index"] == 2


def test_budget_lower_bound_classification() -> None:
    value = classify_no_feasible_pair(
        minimum_total_cost_usd=101,
        maximum_budget_usd=100,
        minimum_total_power_w=50,
        maximum_power_w=100,
        joint_pair_exists=False,
    )

    assert value == "BUDGET_LOWER_BOUND_EXCEEDS"


def test_power_lower_bound_classification() -> None:
    value = classify_no_feasible_pair(
        minimum_total_cost_usd=50,
        maximum_budget_usd=100,
        minimum_total_power_w=101,
        maximum_power_w=100,
        joint_pair_exists=False,
    )

    assert value == "POWER_LOWER_BOUND_EXCEEDS"


def test_both_lower_bounds_classification() -> None:
    value = classify_no_feasible_pair(
        minimum_total_cost_usd=101,
        maximum_budget_usd=100,
        minimum_total_power_w=101,
        maximum_power_w=100,
        joint_pair_exists=False,
    )

    assert value == "BUDGET_AND_POWER_LOWER_BOUNDS_EXCEED"


def test_joint_conflict_classification() -> None:
    value = classify_no_feasible_pair(
        minimum_total_cost_usd=90,
        maximum_budget_usd=100,
        minimum_total_power_w=90,
        maximum_power_w=100,
        joint_pair_exists=False,
    )

    assert value == "JOINT_BUDGET_POWER_CONFLICT"


def test_no_pair_reports_exact_potential_count() -> None:
    result = find_first_cost_power_feasible_pair(
        mdt_options=[option(95, 5), option(5, 95)],
        ost_options=[option(95, 5), option(5, 95)],
        maximum_budget_usd=99,
        maximum_power_w=99,
    )

    assert result["found"] is False
    assert result["potential_pair_count"] == 4
    assert result["pairs_examined"] == 4
    assert result["classification"] == "JOINT_BUDGET_POWER_CONFLICT"


def test_pair_search_reports_independent_lower_bounds() -> None:
    result = find_first_cost_power_feasible_pair(
        mdt_options=[option(90, 5), option(5, 90)],
        ost_options=[option(90, 5), option(5, 90)],
        maximum_budget_usd=50,
        maximum_power_w=50,
    )

    assert result["minimum_total_cost_usd"] == pytest.approx(10.0)
    assert result["minimum_total_power_w"] == pytest.approx(10.0)


def test_unresolved_case_ids_only_include_ok_without_valid() -> None:
    baseline = {
        "cases": [
            {
                "case_id": "A",
                "status": "OK",
                "has_valid_architecture": True,
            },
            {
                "case_id": "B",
                "status": "OK",
                "has_valid_architecture": False,
            },
            {
                "case_id": "C",
                "status": "FAILED",
                "has_valid_architecture": False,
            },
        ]
    }

    assert unresolved_case_ids_from_h10(baseline) == ["B"]


def test_empty_role_options_are_rejected() -> None:
    with pytest.raises(
        FeasibilityCoverageError,
        match="Aucune option MDT",
    ):
        find_first_cost_power_feasible_pair(
            mdt_options=[],
            ost_options=[option(1, 1)],
            maximum_budget_usd=10,
            maximum_power_w=10,
        )


def test_negative_limit_is_rejected() -> None:
    with pytest.raises(
        FeasibilityCoverageError,
        match="valeur >= 0",
    ):
        find_first_cost_power_feasible_pair(
            mdt_options=[option(1, 1)],
            ost_options=[option(1, 1)],
            maximum_budget_usd=-1,
            maximum_power_w=10,
        )
