from __future__ import annotations

import json
import math
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
FREEZE_MANIFEST = (
    BASE_DIR
    / "evaluation"
    / "sizing"
    / "sizing_freeze_manifest.json"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ost_candidate_generator as ost_generator  # noqa: E402


def test_frozen_contract_defines_legacy_gbps_values_as_gb_per_second() -> None:
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    notes = "\n".join(manifest.get("known_semantic_debt", []))

    assert "_gbps" in notes
    assert "GB/s" in notes


def test_catalogue_mb_s_is_converted_to_legacy_gb_per_second_values() -> None:
    assert math.isclose(
        ost_generator.MB_S_TO_LEGACY_BANDWIDTH_UNIT,
        0.001,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert ost_generator.LEGACY_BANDWIDTH_VALUE_UNIT == "GB/s"


def test_ost_candidate_count_uses_same_bandwidth_unit_as_frozen_sizing() -> None:
    drive = {
        "drive_id": "DRV_TEST",
        "name": "Test Drive",
        "manufacturer": "Test",
        "series": "Test",
        "media_type": "SSD",
        "protocol": "NVMe",
        "drive_form_factor_standard": "U.2",
        "ost_eligible": True,
        "capacity_tib": 100.0,
        "seq_read_mb_s": 1000.0,
        "seq_write_mb_s": 500.0,
        "mtbf_hours": 3_000_000,
        "price_en_dollars": 100.0,
        "power_consumption_en_w": 10.0,
    }
    requirement = {
        "required_usable_capacity_tib": 10.0,
        "required_read_bandwidth_gbps": 2.1,
        "required_write_bandwidth_gbps": 0.9,
        "required_total_bandwidth_gbps": 3.0,
        "reliability_requirement": "medium",
        "access_pattern": "sequential",
        "file_size_class": "large_files",
    }
    constraints = {
        "max_budget_usd": 10_000.0,
        "max_power_w": 1_000.0,
    }
    preferences = {
        "performance_priority": 0.25,
        "cost_priority": 0.25,
        "power_priority": 0.25,
        "reliability_priority": 0.25,
    }

    candidate, rejection_reasons = ost_generator.evaluate_drive(
        drive,
        requirement,
        constraints,
        preferences,
    )

    assert rejection_reasons == []
    assert candidate["drive_read_bandwidth_gbps"] == 1.0
    assert candidate["drive_write_bandwidth_gbps"] == 0.5
    assert candidate["count_by_read_bandwidth"] == 3
    assert candidate["count_by_write_bandwidth"] == 2
    assert candidate["raw_minimum_drive_count"] == 3
    assert candidate["raw_provided_read_bandwidth_gbps"] == 3.0
    assert candidate["raw_provided_write_bandwidth_gbps"] == 1.5
