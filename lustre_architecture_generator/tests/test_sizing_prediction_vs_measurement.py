from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
SIZING_DIR = BASE_DIR / "evaluation" / "sizing"
if str(SIZING_DIR) not in sys.path:
    sys.path.insert(0, str(SIZING_DIR))

from prediction_vs_measurement import build_report  # noqa: E402


@pytest.fixture(scope="module")
def report() -> dict:
    return build_report()


def test_s8_uses_directional_not_absolute_fit(report: dict):
    assert report["comparison_policy"]["absolute_fit_allowed"] is False
    assert report["s8_decision"]["modify_config_now"] is False
    assert report["s8_decision"]["next_stage"] == "S9_sensitivity_analysis"


def test_metadata_client_scaling_is_positive_but_sublinear(report: dict):
    rows = [
        row
        for row in report["metadata_scaling"]
        if row["clients"] in {2, 4}
    ]

    assert rows
    for row in rows:
        assert row["observed_relative_scale"] > 1.0
        assert row["observed_relative_scale"] < row["model_relative_scale"]


def test_median_is_retained_for_ior_write_outliers(report: dict):
    rows = {
        (row["metric"], row["clients"]): row
        for row in report["ior_scaling"]
    }

    two_nodes = rows[("write_mib_s", 2)]
    four_nodes = rows[("write_mib_s", 4)]

    assert two_nodes["measured_median_mib_s"] > two_nodes["measured_mean_mib_s"]
    assert four_nodes["measured_median_mib_s"] > four_nodes["measured_mean_mib_s"]


def test_mixed_write_contention_is_close_to_current_ost_headroom(report: dict):
    mixed = report["mixed_workload_impact"]["ior"]["write"]
    configured = report["configured_ost_bandwidth_safety_factor"]

    assert mixed["change_percent"] < 0
    assert 1.15 < mixed["implied_recovery_headroom_factor"] < 1.30
    assert configured == pytest.approx(1.25)
    assert configured >= mixed["implied_recovery_headroom_factor"]


def test_ior_access_pattern_is_not_used_to_validate_mdt_access_factors(report: dict):
    access_ids = {
        "MDT_RANDOM_ACCESS_MULTIPLIER",
        "MDT_MIXED_ACCESS_MULTIPLIER",
        "MDT_SEQUENTIAL_ACCESS_MULTIPLIER",
    }
    rows = {
        row["id"]: row
        for row in report["assumption_coverage"]
        if row["id"] in access_ids
    }

    assert set(rows) == access_ids
    assert all("IOR pattern effect cannot validate an MDT multiplier." == row["warning"] for row in rows.values())


def test_m6c_failure_is_rejected_as_measurement(report: dict):
    random_case = report["access_pattern_impact"]["fully_random_overlapping"]

    assert random_case["status"] == "FAILED"
    assert random_case["accepted_as_measurement"] is False
    assert random_case["failure"]["signal"] == 8


def test_every_registry_assumption_has_s8_coverage(report: dict):
    registry = json.loads(
        (SIZING_DIR / "sizing_assumptions.json").read_text(encoding="utf-8")
    )
    expected = {item["id"] for item in registry["assumptions"]}
    actual = {item["id"] for item in report["assumption_coverage"]}

    assert actual == expected
    assert all(
        row["final_status"] == "PENDING_SENSITIVITY"
        for row in report["assumption_coverage"]
    )
