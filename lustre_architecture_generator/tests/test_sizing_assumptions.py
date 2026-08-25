from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY = BASE_DIR / "evaluation" / "sizing" / "sizing_assumptions.json"
CONFIG = BASE_DIR / "config" / "architecture_rules.json"


FINAL_STATUSES = {
    "SUPPORTED",
    "CALIBRATED",
    "POLICY_CHOICE",
    "NEEDS_REVISION",
}


def load_registry() -> dict:
    with REGISTRY.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_config() -> dict:
    with CONFIG.open("r", encoding="utf-8") as file:
        return json.load(file)


def value_at_path(config: dict, dotted_path: str):
    value = config
    for part in dotted_path.split("."):
        value = value[part]
    return value


def test_registry_has_unique_ids_and_config_paths():
    data = load_registry()
    assumptions = data["assumptions"]

    ids = [item["id"] for item in assumptions]
    paths = [item["config_path"] for item in assumptions]

    assert len(ids) == len(set(ids))
    assert len(paths) == len(set(paths))


def test_registry_is_finalized_with_no_to_validate_status():
    data = load_registry()

    assert data["registry_version"] == "2.0"
    assert data["finalization"]["all_assumptions_finalized"] is True
    assert data["assumptions"]
    assert all(item["status"] in FINAL_STATUSES for item in data["assumptions"])
    assert all(item["status"] != "TO_VALIDATE" for item in data["assumptions"])


def test_registry_final_values_match_configuration():
    registry = load_registry()
    config = load_config()

    for item in registry["assumptions"]:
        expected = value_at_path(config, item["config_path"])
        assert item["current_value"] == expected
        assert item["final_value"] == expected


def test_registry_contains_core_sizing_assumptions():
    data = load_registry()
    paths = {item["config_path"] for item in data["assumptions"]}

    required = {
        "capacity_planning.default_target_fill_ratio",
        "workload_classification.dominance_margin",
        "mdt_estimation.base_iops_per_client",
        "mdt_estimation.iops_safety_factor",
        "mdt_estimation.metadata_bytes_per_file",
        "mdt_estimation.metadata_capacity_safety_factor",
        "ost_estimation.bandwidth_safety_factor",
        "ost_estimation.capacity_safety_factor",
    }

    assert required <= paths


def test_every_assumption_has_evidence_and_final_decision_metadata():
    data = load_registry()

    for item in data["assumptions"]:
        assert item["evidence_type_required"]
        assert item["formula"]
        assert item["role"]
        assert item["sensitivity_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert item["toubkal_evidence_level"]
        assert item["decision_reason"]
