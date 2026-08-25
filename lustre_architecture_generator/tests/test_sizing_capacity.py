from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
CONFIG_FILE = BASE_DIR / "config" / "architecture_rules.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workload_analyzer import (  # noqa: E402
    WorkloadAnalysisError,
    analyze_workload,
    validate_config,
)


def load_config() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = json.load(file)
    validate_config(config)
    return config


def make_case(**changes):
    case = {
        "case_id": "SIZING_TEST",
        "requested_usable_capacity_tib": 100,
        "client_count": 100,
        "average_file_size_gb": 2.0,
        "max_file_size_gb": 20.0,
        "total_file_count": 1_000_000,
        "read_write_ratio": {
            "read_percent": 70,
            "write_percent": 30,
        },
        "access_type": "mixed",
        "target_read_gbps": 20,
        "target_write_gbps": 10,
        "ha_required": True,
        "max_budget_usd": 200000,
        "max_power_w": 20000,
        "annual_growth_percent": 20,
        "planning_horizon_years": 3,
        "performance_priority": 0.4,
        "cost_priority": 0.2,
        "power_priority": 0.15,
        "reliability_priority": 0.25,
    }
    case.update(changes)
    return case


def test_capacity_formula_three_year_horizon_is_216_tib():
    config = load_config()
    result = analyze_workload(
        make_case(planning_horizon_years=3),
        config,
    )

    planning = result["capacity_planning"]

    assert math.isclose(planning["annual_growth_factor"], 1.2, abs_tol=1e-9)
    assert math.isclose(planning["growth_factor"], 1.728, abs_tol=1e-9)
    assert math.isclose(
        planning["planned_usable_capacity_tib"],
        216.0,
        abs_tol=1e-6,
    )


def test_explicit_horizon_is_preserved_and_traced():
    config = load_config()
    result = analyze_workload(
        make_case(planning_horizon_years=4),
        config,
    )

    assert result["source_requirement"]["planning_horizon_years"] == 4
    assert result["capacity_planning"]["planning_horizon_years"] == 4
    assert result["trace"]["planning_horizon_source"] == "input"


def test_missing_horizon_is_rejected_after_s10_freeze():
    config = load_config()
    case = make_case()
    case.pop("planning_horizon_years")

    with pytest.raises(WorkloadAnalysisError, match="planning_horizon_years.*obligatoire"):
        analyze_workload(case, config)


@pytest.mark.parametrize("invalid_horizon", [0, -1, -5.0])
def test_non_positive_horizon_is_rejected(invalid_horizon):
    config = load_config()

    with pytest.raises(WorkloadAnalysisError):
        analyze_workload(
            make_case(planning_horizon_years=invalid_horizon),
            config,
        )


def test_boolean_horizon_is_rejected():
    config = load_config()

    with pytest.raises(WorkloadAnalysisError):
        analyze_workload(
            make_case(planning_horizon_years=True),
            config,
        )


def test_longer_horizon_increases_planned_capacity_when_growth_is_positive():
    config = load_config()

    one_year = analyze_workload(
        make_case(planning_horizon_years=1),
        config,
    )
    five_years = analyze_workload(
        make_case(planning_horizon_years=5),
        config,
    )

    assert (
        five_years["capacity_planning"]["planned_usable_capacity_tib"]
        > one_year["capacity_planning"]["planned_usable_capacity_tib"]
    )


def test_zero_growth_is_invariant_to_horizon():
    config = load_config()

    one_year = analyze_workload(
        make_case(
            annual_growth_percent=0,
            planning_horizon_years=1,
        ),
        config,
    )
    ten_years = analyze_workload(
        make_case(
            annual_growth_percent=0,
            planning_horizon_years=10,
        ),
        config,
    )

    assert math.isclose(
        one_year["capacity_planning"]["planned_usable_capacity_tib"],
        ten_years["capacity_planning"]["planned_usable_capacity_tib"],
        abs_tol=1e-6,
    )


from validate_workload_analysis import (  # noqa: E402
    validate_dataset_structure,
    validate_monotonicity,
)


def test_validator_accepts_explicit_multi_year_capacity():
    config = load_config()
    source = make_case(planning_horizon_years=3)
    result = analyze_workload(source, config)

    errors = validate_dataset_structure(
        [source],
        [result],
        config,
    )

    assert errors == []


def test_validator_rejects_source_without_explicit_horizon():
    config = load_config()
    source = make_case()
    result = analyze_workload(source, config)
    legacy_source = dict(source)
    legacy_source.pop("planning_horizon_years")

    errors = validate_dataset_structure(
        [legacy_source],
        [result],
        config,
    )

    assert any("planning_horizon_years est obligatoire" in error for error in errors)


def test_validator_monotonicity_includes_planning_horizon():
    config = load_config()

    assert validate_monotonicity(config) == []
