from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
SIZING_DIR = BASE_DIR / "evaluation" / "sizing"
if str(SIZING_DIR) not in sys.path:
    sys.path.insert(0, str(SIZING_DIR))

from finalize_sizing_assumptions import build_decisions  # noqa: E402


@pytest.fixture(scope="module")
def report() -> dict:
    return build_decisions()


def by_id(report: dict, assumption_id: str) -> dict:
    return next(row for row in report["assumptions"] if row["id"] == assumption_id)


def test_all_23_assumptions_have_a_final_status(report: dict):
    assert len(report["assumptions"]) == 23
    assert sum(report["status_counts"].values()) == 23
    assert report["status_counts"]["NEEDS_REVISION"] == 0
    assert all(row["status"] != "TO_VALIDATE" for row in report["assumptions"])


def test_ost_bandwidth_factor_is_calibrated_to_125(report: dict):
    row = by_id(report, "OST_BANDWIDTH_SAFETY_FACTOR")
    calibration = report["calibration"]

    assert row["status"] == "CALIBRATED"
    assert row["initial_value"] == pytest.approx(1.2)
    assert row["final_value"] == pytest.approx(1.25)
    assert calibration["implied_recovery_headroom_factor"] < 1.25
    assert calibration["relative_change_from_old_percent"] == pytest.approx(4.1667, abs=0.001)


def test_ost_capacity_factor_avoids_double_margin(report: dict):
    row = by_id(report, "OST_CAPACITY_SAFETY_FACTOR")

    assert row["status"] == "SUPPORTED"
    assert row["final_value"] == pytest.approx(1.0)
    assert "double-counting" in row["decision_reason"]


def test_unidentified_mdt_access_factors_remain_policy_choices(report: dict):
    for assumption_id in (
        "MDT_RANDOM_ACCESS_MULTIPLIER",
        "MDT_MIXED_ACCESS_MULTIPLIER",
        "MDT_SEQUENTIAL_ACCESS_MULTIPLIER",
    ):
        row = by_id(report, assumption_id)
        assert row["status"] == "POLICY_CHOICE"


def test_registry_and_decision_artifact_are_consistent(report: dict):
    registry = json.loads(
        (SIZING_DIR / "sizing_assumptions.json").read_text(encoding="utf-8")
    )
    registry_by_id = {row["id"]: row for row in registry["assumptions"]}

    for decision in report["assumptions"]:
        registered = registry_by_id[decision["id"]]
        assert registered["status"] == decision["status"]
        assert registered["final_value"] == decision["final_value"]
