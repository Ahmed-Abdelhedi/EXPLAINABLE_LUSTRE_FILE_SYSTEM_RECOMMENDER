from __future__ import annotations

import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
SIZING_DIR = BASE_DIR / "evaluation" / "sizing"
if str(SIZING_DIR) not in sys.path:
    sys.path.insert(0, str(SIZING_DIR))

from sensitivity_analysis import build_report  # noqa: E402


@pytest.fixture(scope="module")
def report() -> dict:
    return build_report()


def by_id(report: dict, assumption_id: str) -> dict:
    return next(
        item for item in report["assumptions"]
        if item["id"] == assumption_id
    )


def variant(assumption: dict, label: str) -> dict:
    return next(item for item in assumption["variants"] if item["label"] == label)


def test_sensitivity_covers_all_registered_assumptions(report: dict):
    assert report["case_count"] == 1200
    assert len(report["assumptions"]) == 23
    assert len({item["id"] for item in report["assumptions"]}) == 23


def test_base_iops_is_linear_in_mdt_requirement(report: dict):
    assumption = by_id(report, "MDT_BASE_IOPS_PER_CLIENT")
    plus_20 = variant(assumption, "+20%")
    change = plus_20["summary"]["numeric"]["mdt_iops"]["median_change_percent"]
    assert change == pytest.approx(20.0, abs=0.1)


def test_ost_bandwidth_factor_is_linear(report: dict):
    assumption = by_id(report, "OST_BANDWIDTH_SAFETY_FACTOR")
    plus_20 = variant(assumption, "+20%")
    read_change = plus_20["summary"]["numeric"]["ost_read_bandwidth"]["median_change_percent"]
    write_change = plus_20["summary"]["numeric"]["ost_write_bandwidth"]["median_change_percent"]
    assert read_change == pytest.approx(20.0, abs=0.01)
    assert write_change == pytest.approx(20.0, abs=0.01)


def test_fill_ratio_07_increases_capacity_vs_08(report: dict):
    assumption = by_id(report, "CAPACITY_FILL_RATIO_DEFAULT")
    value_07 = variant(assumption, "value=0.7")
    change = value_07["summary"]["numeric"]["planned_capacity_tib"]["median_change_percent"]
    assert change == pytest.approx((0.8 / 0.7 - 1.0) * 100.0, abs=0.01)


def test_score_weight_perturbations_are_renormalized(report: dict):
    assumption = by_id(report, "WORKLOAD_METADATA_FILE_COUNT_WEIGHT")
    assert assumption["weight_renormalized"] is True
    assert all(item["summary"]["workload_type_flip_percent"] >= 0 for item in assumption["variants"])


def test_sensitivity_levels_are_valid(report: dict):
    allowed = {"LOW", "MEDIUM", "HIGH"}
    assert all(item["sensitivity_level"] in allowed for item in report["assumptions"])
