from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
CONFIG_FILE = BASE_DIR / "config" / "architecture_rules.json"
REGISTRY_FILE = BASE_DIR / "evaluation" / "sizing" / "sizing_assumptions.json"
MANIFEST_FILE = BASE_DIR / "evaluation" / "sizing" / "sizing_freeze_manifest.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workload_analyzer import WorkloadAnalysisError, analyze_workload, validate_config  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_case() -> dict:
    return {
        "case_id": "FREEZE_CASE",
        "requested_usable_capacity_tib": 100,
        "client_count": 10,
        "average_file_size_gb": 2.0,
        "max_file_size_gb": 10.0,
        "total_file_count": 100000,
        "read_write_ratio": {"read_percent": 70, "write_percent": 30},
        "access_type": "mixed",
        "target_read_gbps": 10,
        "target_write_gbps": 5,
        "ha_required": True,
        "max_budget_usd": 100000,
        "max_power_w": 10000,
        "annual_growth_percent": 10,
        "planning_horizon_years": 3,
        "performance_priority": 0.4,
        "cost_priority": 0.2,
        "power_priority": 0.15,
        "reliability_priority": 0.25,
    }


def test_freeze_config_is_v2_and_has_no_legacy_horizon_key():
    config = load_json(CONFIG_FILE)
    assert config["version"] == "2.0"
    assert "legacy_default_planning_horizon_years" not in config["capacity_planning"]
    validate_config(config)


def test_deprecated_legacy_horizon_config_key_is_rejected():
    config = load_json(CONFIG_FILE)
    config["capacity_planning"]["legacy_default_planning_horizon_years"] = 1.0
    with pytest.raises(WorkloadAnalysisError, match="Configuration obsolète"):
        validate_config(config)


def test_missing_planning_horizon_has_no_production_fallback():
    config = load_json(CONFIG_FILE)
    case = make_case()
    case.pop("planning_horizon_years")
    with pytest.raises(WorkloadAnalysisError, match="aucun fallback"):
        analyze_workload(case, config)


def test_explicit_horizon_trace_is_input_only():
    config = load_json(CONFIG_FILE)
    result = analyze_workload(make_case(), config)
    assert result["trace"]["planning_horizon_source"] == "input"
    assert result["trace"]["analyzer_version"] == "3.0"


def test_all_23_assumptions_are_finalized():
    registry = load_json(REGISTRY_FILE)
    statuses = [row["status"] for row in registry["assumptions"]]
    assert len(statuses) == 23
    assert statuses.count("SUPPORTED") == 1
    assert statuses.count("CALIBRATED") == 1
    assert statuses.count("POLICY_CHOICE") == 21
    assert statuses.count("NEEDS_REVISION") == 0
    assert "TO_VALIDATE" not in statuses


def test_frozen_ost_factors_match_final_decisions():
    config = load_json(CONFIG_FILE)
    assert config["ost_estimation"]["bandwidth_safety_factor"] == pytest.approx(1.25)
    assert config["ost_estimation"]["capacity_safety_factor"] == pytest.approx(1.0)


def test_freeze_manifest_declares_strict_contract_and_boundary():
    manifest = load_json(MANIFEST_FILE)
    assert manifest["status"] == "FROZEN"
    assert manifest["config_version"] == "2.0"
    assert manifest["input_contract"]["planning_horizon_years"]["required"] is True
    assert manifest["input_contract"]["planning_horizon_years"]["fallback_allowed"] is False
    assert manifest["assumption_status_counts"]["TO_VALIDATE"] == 0
    assert "drive ranking" in manifest["downstream_boundary"]
