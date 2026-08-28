import math

import pytest

from e2e_pipeline.requirement_to_sizing_adapter import (
    RequirementToSizingAdapterError,
    adapt_requirement_to_sizing_case,
)


def _requirement():
    return {
        "requested_usable_capacity_tib": 100,
        "client_count": 64,
        "average_file_size_gb": 20,
        "max_file_size_gb": 50,
        "total_file_count": 20,
        "read_write_ratio": {"read_percent": 20.0, "write_percent": 80.0},
        "access_type": "sequential",
        "target_read_gbps": 20,
        "target_write_gbps": 22,
        "ha_required": True,
        "max_budget_usd": 5000,
        "max_power_w": 20_000_000,
        "annual_growth_percent": 20,
        "planning_horizon_years": 3,
        "cost_priority": "HIGH",
        "power_priority": "LOW",
        "reliability_priority": "HIGH",
        "performance_priority": "HIGH",
        "preference_weights": {
            "cost": 0.5865384615384616,
            "power": 0.06730769230769233,
            "performance": 0.1923076923076923,
            "reliability": 0.15384615384615383,
        },
    }


def test_adapter_maps_bwm_weights_to_frozen_sizing_contract():
    case = adapt_requirement_to_sizing_case(_requirement())

    assert case["case_id"].startswith("ONLINE_")
    assert case["cost_priority"] == pytest.approx(0.5865384615384616)
    assert case["power_priority"] == pytest.approx(0.06730769230769233)
    assert case["performance_priority"] == pytest.approx(0.1923076923076923)
    assert case["reliability_priority"] == pytest.approx(0.15384615384615383)
    assert case["planning_horizon_years"] == 3
    assert case["read_write_ratio"] == {
        "read_percent": 20.0,
        "write_percent": 80.0,
    }
    assert math.isclose(
        case["cost_priority"]
        + case["power_priority"]
        + case["performance_priority"]
        + case["reliability_priority"],
        1.0,
        abs_tol=1e-12,
    )


def test_adapter_is_deterministic_and_does_not_mutate_requirement():
    requirement = _requirement()
    original_labels = {
        key: requirement[key]
        for key in (
            "cost_priority",
            "power_priority",
            "performance_priority",
            "reliability_priority",
        )
    }

    first = adapt_requirement_to_sizing_case(requirement)
    second = adapt_requirement_to_sizing_case(requirement)

    assert first == second
    assert {
        key: requirement[key]
        for key in original_labels
    } == original_labels


def test_adapter_rejects_invalid_weight_sum():
    requirement = _requirement()
    requirement["preference_weights"]["cost"] = 0.1

    with pytest.raises(RequirementToSizingAdapterError, match="somme"):
        adapt_requirement_to_sizing_case(requirement)


def test_adapter_rejects_missing_planning_horizon():
    requirement = _requirement()
    requirement.pop("planning_horizon_years")

    with pytest.raises(RequirementToSizingAdapterError, match="planning_horizon_years"):
        adapt_requirement_to_sizing_case(requirement)
