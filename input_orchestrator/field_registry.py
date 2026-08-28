from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class FieldSpec:
    name: str
    domain: str
    priority: int
    question: str
    expected_answer_type: str
    skippable: bool = True


FIELD_SPECS: Dict[str, FieldSpec] = {
    "requested_usable_capacity_tib": FieldSpec(
        "requested_usable_capacity_tib", "quantity", 10,
        "What usable storage capacity do you need, in TiB?",
        "quantity_capacity", False,
    ),
    "client_count": FieldSpec(
        "client_count", "quantity", 20,
        "Approximately how many clients will access the storage system?",
        "positive_integer", False,
    ),
    "average_file_size_gb": FieldSpec(
        "average_file_size_gb", "quantity", 70,
        "What is the approximate average file size?",
        "quantity_size", True,
    ),
    "max_file_size_gb": FieldSpec(
        "max_file_size_gb", "quantity", 71,
        "What is the approximate maximum file size?",
        "quantity_size", True,
    ),
    "total_file_count": FieldSpec(
        "total_file_count", "quantity", 72,
        "Approximately how many files do you expect?",
        "positive_integer", True,
    ),
    "read_write_ratio": FieldSpec(
        "read_write_ratio", "quantity", 55,
        "What is the approximate read/write ratio?",
        "ratio", True,
    ),
    "access_type": FieldSpec(
        "access_type", "categorical", 30,
        "Is the I/O access pattern mainly sequential, random, or mixed?",
        "access_type", False,
    ),
    "target_read_gbps": FieldSpec(
        "target_read_gbps", "quantity", 50,
        "Do you have a target aggregate read throughput?",
        "throughput", True,
    ),
    "target_write_gbps": FieldSpec(
        "target_write_gbps", "quantity", 51,
        "Do you have a target aggregate write throughput?",
        "throughput", True,
    ),
    "ha_required": FieldSpec(
        "ha_required", "categorical", 40,
        "Is high availability required for this system?",
        "yes_no", False,
    ),
    "max_budget_usd": FieldSpec(
        "max_budget_usd", "quantity", 60,
        "Do you have a maximum hardware budget?",
        "currency", True,
    ),
    "max_power_w": FieldSpec(
        "max_power_w", "quantity", 61,
        "Do you have a maximum power limit?",
        "power", True,
    ),
    "annual_growth_percent": FieldSpec(
        "annual_growth_percent", "quantity", 80,
        "What annual data growth rate should be planned for, if any?",
        "percent_or_none", True,
    ),
    "planning_horizon_years": FieldSpec(
        "planning_horizon_years", "quantity", 81,
        "Over how many years should the growth be planned?",
        "positive_integer_years", True,
    ),
    "cost_priority": FieldSpec(
        "cost_priority", "preference", 90,
        "How important is cost in the final design?",
        "preference", True,
    ),
    "power_priority": FieldSpec(
        "power_priority", "preference", 91,
        "How important is power efficiency in the final design?",
        "preference", True,
    ),
    "reliability_priority": FieldSpec(
        "reliability_priority", "preference", 92,
        "How important is reliability in the final design?",
        "preference", True,
    ),
    "performance_priority": FieldSpec(
        "performance_priority", "preference", 93,
        "How important is performance in the final design?",
        "preference", True,
    ),
}

CORE_REQUIRED_FIELDS: Tuple[str, ...] = (
    "requested_usable_capacity_tib",
    "client_count",
    "access_type",
    "ha_required",
)

PREFERENCE_FIELDS: Tuple[str, ...] = (
    "cost_priority",
    "power_priority",
    "reliability_priority",
    "performance_priority",
)

WEIGHT_FIELD = "preference_weights"
