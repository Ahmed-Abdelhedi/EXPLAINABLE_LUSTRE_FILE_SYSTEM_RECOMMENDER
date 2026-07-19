from __future__ import annotations

from typing import Any, Optional, Tuple

from models import ParamName


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_number(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)

    return round(float(value), 6)


def normalize_unit_value(
    field: ParamName,
    value: Any,
    unit: Optional[str],
) -> Tuple[Any, Optional[str]]:
    """
    Convertit les unités vers les unités cibles du système.

    Exemples :
    - 500 MB -> 0.5 GB
    - 8 kW -> 8000 W
    - 15 TB -> 15 TiB dans cette version MVP
    """

    if not is_number(value):
        return value, unit

    unit_norm = (unit or "").strip().lower()
    value_float = float(value)

    if field in {
        ParamName.average_file_size_gb,
        ParamName.max_file_size_gb,
    }:
        if unit_norm in {"mb", "mib"}:
            return normalize_number(value_float / 1000), "GB"

        if unit_norm in {"gb", "gib"}:
            return normalize_number(value_float), "GB"

        return normalize_number(value_float), "GB"

    if field == ParamName.max_power_w:
        if unit_norm in {"kw", "kilowatt", "kilowatts"}:
            return normalize_number(value_float * 1000), "W"

        return normalize_number(value_float), "W"

    if field == ParamName.requested_usable_capacity_tib:
        return normalize_number(value_float), "TiB"

    if field in {
        ParamName.target_read_gbps,
        ParamName.target_write_gbps,
    }:
        return normalize_number(value_float), "GB/s"

    if field == ParamName.max_budget_usd:
        return normalize_number(value_float), "USD"

    if field == ParamName.annual_growth_percent:
        return normalize_number(value_float), "%"

    if field in {
        ParamName.client_count,
        ParamName.total_file_count,
    }:
        return int(value_float), None

    return normalize_number(value_float), unit