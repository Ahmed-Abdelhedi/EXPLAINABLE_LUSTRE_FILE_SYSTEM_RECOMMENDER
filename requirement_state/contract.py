from __future__ import annotations

from typing import Final, Tuple


QUANTITY_FIELDS: Final[Tuple[str, ...]] = (
    "requested_usable_capacity_tib",
    "client_count",
    "average_file_size_gb",
    "max_file_size_gb",
    "total_file_count",
    "read_write_ratio",
    "target_read_gbps",
    "target_write_gbps",
    "max_budget_usd",
    "max_power_w",
    "annual_growth_percent",
    "planning_horizon_years",
)

CATEGORICAL_FIELDS: Final[Tuple[str, ...]] = (
    "access_type",
    "ha_required",
)

PREFERENCE_LABEL_FIELDS: Final[Tuple[str, ...]] = (
    "cost_priority",
    "power_priority",
    "reliability_priority",
    "performance_priority",
)

REQUIREMENT_FIELDS: Final[Tuple[str, ...]] = (
    "requested_usable_capacity_tib",
    "client_count",
    "average_file_size_gb",
    "max_file_size_gb",
    "total_file_count",
    "read_write_ratio",
    "access_type",
    "target_read_gbps",
    "target_write_gbps",
    "ha_required",
    "max_budget_usd",
    "max_power_w",
    "annual_growth_percent",
    "planning_horizon_years",
    "cost_priority",
    "power_priority",
    "reliability_priority",
    "performance_priority",
)

PREFERENCE_WEIGHT_DIMENSIONS: Final[Tuple[str, ...]] = (
    "cost",
    "power",
    "performance",
    "reliability",
)

ACCESS_TYPES: Final[Tuple[str, ...]] = (
    "sequential",
    "random",
    "mixed",
)

PREFERENCE_LEVELS: Final[Tuple[str, ...]] = (
    "NO_SIGNAL",
    "VERY_LOW",
    "LOW",
    "MEDIUM",
    "HIGH",
    "VERY_HIGH",
)
