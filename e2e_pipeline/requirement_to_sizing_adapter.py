"""Contract adapter from the canonical Requirement JSON to frozen S10 sizing.

The Requirement layer keeps qualitative labels such as ``HIGH`` and also
stores formal BWM weights in ``preference_weights``.  The frozen S10 sizing
contract, however, expects the four ``*_priority`` fields to be numeric and to
sum to one.  This adapter is the only place where that contract translation is
performed.

No sizing formula is implemented here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any


ADAPTER_VERSION = "1.0"

RAW_FIELDS = (
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
)

WEIGHT_TO_SIZING_FIELD = {
    "cost": "cost_priority",
    "power": "power_priority",
    "performance": "performance_priority",
    "reliability": "reliability_priority",
}


class RequirementToSizingAdapterError(ValueError):
    """Raised when the final Requirement cannot satisfy the frozen S10 input."""


def _finite_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequirementToSizingAdapterError(
            f"{field}: une valeur numérique est requise."
        )

    number = float(value)
    if not math.isfinite(number):
        raise RequirementToSizingAdapterError(
            f"{field}: une valeur numérique finie est requise."
        )

    if minimum is not None and number < minimum:
        raise RequirementToSizingAdapterError(
            f"{field}: la valeur doit être >= {minimum}."
        )

    if strictly_positive and number <= 0:
        raise RequirementToSizingAdapterError(
            f"{field}: la valeur doit être > 0."
        )

    return number


def _deterministic_case_id(requirement: dict[str, Any]) -> str:
    existing = requirement.get("case_id")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()

    canonical = json.dumps(
        requirement,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:16].upper()
    return f"ONLINE_{digest}"


def _validate_raw_contract(requirement: dict[str, Any]) -> None:
    missing = [field for field in RAW_FIELDS if field not in requirement]
    if missing:
        raise RequirementToSizingAdapterError(
            "Champs Requirement manquants pour S10: " + ", ".join(missing)
        )

    for field in (
        "requested_usable_capacity_tib",
        "average_file_size_gb",
        "max_file_size_gb",
        "target_read_gbps",
        "target_write_gbps",
        "max_budget_usd",
        "max_power_w",
        "annual_growth_percent",
    ):
        _finite_number(requirement[field], field=field, minimum=0.0)

    _finite_number(
        requirement["client_count"],
        field="client_count",
        strictly_positive=True,
    )
    _finite_number(
        requirement["total_file_count"],
        field="total_file_count",
        strictly_positive=True,
    )
    _finite_number(
        requirement["planning_horizon_years"],
        field="planning_horizon_years",
        strictly_positive=True,
    )

    if not isinstance(requirement["ha_required"], bool):
        raise RequirementToSizingAdapterError(
            "ha_required: un booléen est requis."
        )

    access_type = requirement["access_type"]
    if not isinstance(access_type, str) or access_type.strip().lower() not in {
        "random",
        "mixed",
        "sequential",
    }:
        raise RequirementToSizingAdapterError(
            "access_type: valeur autorisée = random, mixed ou sequential."
        )

    ratio = requirement["read_write_ratio"]
    if not isinstance(ratio, dict):
        raise RequirementToSizingAdapterError(
            "read_write_ratio: un objet {read_percent, write_percent} est requis."
        )

    read_percent = _finite_number(
        ratio.get("read_percent"),
        field="read_write_ratio.read_percent",
        minimum=0.0,
    )
    write_percent = _finite_number(
        ratio.get("write_percent"),
        field="read_write_ratio.write_percent",
        minimum=0.0,
    )
    if not math.isclose(read_percent + write_percent, 100.0, abs_tol=0.5):
        raise RequirementToSizingAdapterError(
            "read_write_ratio: read_percent + write_percent doit être proche de 100."
        )

    if float(requirement["max_file_size_gb"]) < float(
        requirement["average_file_size_gb"]
    ):
        raise RequirementToSizingAdapterError(
            "max_file_size_gb doit être >= average_file_size_gb."
        )


def _validated_weights(requirement: dict[str, Any]) -> dict[str, float]:
    raw_weights = requirement.get("preference_weights")
    if not isinstance(raw_weights, dict):
        raise RequirementToSizingAdapterError(
            "preference_weights: objet BWM numérique requis."
        )

    missing = [key for key in WEIGHT_TO_SIZING_FIELD if key not in raw_weights]
    if missing:
        raise RequirementToSizingAdapterError(
            "Poids BWM manquants: " + ", ".join(missing)
        )

    weights = {
        key: _finite_number(
            raw_weights[key],
            field=f"preference_weights.{key}",
            minimum=0.0,
        )
        for key in WEIGHT_TO_SIZING_FIELD
    }

    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=0.01):
        raise RequirementToSizingAdapterError(
            "La somme des preference_weights doit être proche de 1.0; "
            f"somme actuelle={total:.12f}."
        )

    # Preserve the BWM ratios while eliminating tiny floating-point drift.
    if total <= 0.0:
        raise RequirementToSizingAdapterError(
            "La somme des preference_weights doit être > 0."
        )

    return {key: value / total for key, value in weights.items()}


def adapt_requirement_to_sizing_case(
    requirement: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact single-case input expected by the frozen S10 sizing.

    Qualitative labels from the Requirement State are deliberately not copied
    into the four S10 ``*_priority`` fields.  Those fields receive only the
    normalized formal BWM weights.
    """

    if not isinstance(requirement, dict):
        raise RequirementToSizingAdapterError(
            "Le Requirement final doit être un objet JSON."
        )

    _validate_raw_contract(requirement)
    weights = _validated_weights(requirement)

    case: dict[str, Any] = {
        "case_id": _deterministic_case_id(requirement),
    }

    for field in RAW_FIELDS:
        case[field] = copy.deepcopy(requirement[field])

    case["access_type"] = str(case["access_type"]).strip().lower()

    for weight_key, sizing_field in WEIGHT_TO_SIZING_FIELD.items():
        case[sizing_field] = weights[weight_key]

    return case
