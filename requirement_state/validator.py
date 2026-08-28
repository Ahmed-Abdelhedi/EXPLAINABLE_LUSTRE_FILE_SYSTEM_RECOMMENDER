from __future__ import annotations

import math
from typing import Dict, List, Optional, TypeGuard

from .contract import ACCESS_TYPES, PREFERENCE_LEVELS, REQUIREMENT_FIELDS
from .models import FinalRequirementState, RequirementFieldStatus
from .validation_models import ValidationIssue, ValidationSeverity


CORE_REQUIRED_FIELDS = (
    "requested_usable_capacity_tib",
    "client_count",
    "access_type",
    "ha_required",
)

POSITIVE_NUMERIC_FIELDS = (
    "requested_usable_capacity_tib",
    "average_file_size_gb",
    "max_file_size_gb",
    "max_budget_usd",
    "max_power_w",
)

NONNEGATIVE_NUMERIC_FIELDS = (
    "target_read_gbps",
    "target_write_gbps",
    "annual_growth_percent",
)

POSITIVE_INTEGER_FIELDS = (
    "client_count",
    "total_file_count",
)

PREFERENCE_LABEL_FIELDS = (
    "cost_priority",
    "power_priority",
    "reliability_priority",
    "performance_priority",
)

WEIGHT_TOLERANCE = 1e-8


def _finite_number(
    value: object,
) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _issue(code, message, *, field=None, details=None):
    return ValidationIssue(
        code=code,
        message=message,
        field=field,
        severity=ValidationSeverity.ERROR,
        details=details,
    )


class DeterministicRequirementValidator:
    """Final deterministic acceptance boundary before sizing."""

    def validate(self, state: FinalRequirementState) -> FinalRequirementState:
        issues: List[ValidationIssue] = []
        followups: List[str] = []

        issues += self._validate_collection(state, followups)
        issues += self._validate_scalars(state)
        issues += self._validate_cross_fields(state)
        issues += self._validate_read_write_ratio(state)
        issues += self._validate_categorical(state)
        issues += self._validate_preferences(state)
        issues += self._validate_weights(state)

        state.validation_issues = [item.to_dict() for item in issues]
        state.follow_up_questions = list(dict.fromkeys(followups))
        state.ready_for_sizing = not any(
            item.severity == ValidationSeverity.ERROR
            for item in issues
        )
        return state

    def _validate_collection(self, state, followups):
        issues = []

        for field_name in REQUIREMENT_FIELDS:
            trace = state.field_traces.get(field_name)

            if trace is None:
                issues.append(_issue(
                    "MISSING_FIELD_TRACE",
                    f"Missing trace for {field_name}.",
                    field=field_name,
                ))
                followups.append(self._followup(field_name))
                continue

            if trace.status in {
                RequirementFieldStatus.MISSING,
                RequirementFieldStatus.PARTIAL,
                RequirementFieldStatus.UNRESOLVED,
                RequirementFieldStatus.CONFLICT,
            }:
                issues.append(_issue(
                    "FIELD_NOT_RESOLVED",
                    f"{field_name} is still {trace.status.value}.",
                    field=field_name,
                    details={"status": trace.status.value},
                ))
                followups.append(self._followup(field_name))

            if (
                field_name in CORE_REQUIRED_FIELDS
                and trace.status != RequirementFieldStatus.VERIFIED
            ):
                issues.append(_issue(
                    "REQUIRED_FIELD_NOT_VERIFIED",
                    f"{field_name} must be VERIFIED.",
                    field=field_name,
                ))
                followups.append(self._followup(field_name))

        return issues

    def _validate_scalars(self, state):
        issues = []

        for field_name in POSITIVE_NUMERIC_FIELDS:
            value = getattr(state, field_name)
            if value is None:
                continue
            if not _finite_number(value):
                issues.append(_issue(
                    "NON_FINITE_NUMBER",
                    f"{field_name} must be finite.",
                    field=field_name,
                ))
            elif float(value) <= 0:
                issues.append(_issue(
                    "NON_POSITIVE_VALUE",
                    f"{field_name} must be > 0.",
                    field=field_name,
                ))

        for field_name in NONNEGATIVE_NUMERIC_FIELDS:
            value = getattr(state, field_name)
            if value is None:
                continue
            if not _finite_number(value):
                issues.append(_issue(
                    "NON_FINITE_NUMBER",
                    f"{field_name} must be finite.",
                    field=field_name,
                ))
            elif float(value) < 0:
                issues.append(_issue(
                    "NEGATIVE_VALUE",
                    f"{field_name} must be >= 0.",
                    field=field_name,
                ))

        for field_name in POSITIVE_INTEGER_FIELDS:
            value = getattr(state, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                issues.append(_issue(
                    "INVALID_POSITIVE_INTEGER",
                    f"{field_name} must be a positive integer.",
                    field=field_name,
                ))

        horizon = state.planning_horizon_years
        if horizon is not None:
            if (
                isinstance(horizon, bool)
                or not isinstance(horizon, int)
                or horizon <= 0
            ):
                issues.append(_issue(
                    "INVALID_PLANNING_HORIZON",
                    "planning_horizon_years must be a positive integer.",
                    field="planning_horizon_years",
                ))

        return issues

    def _validate_cross_fields(self, state):
        issues = []
        avg = state.average_file_size_gb
        maximum = state.max_file_size_gb

        if (
            avg is not None
            and maximum is not None
            and _finite_number(avg)
            and _finite_number(maximum)
            and float(avg) > float(maximum)
        ):
            issues.append(_issue(
                "AVERAGE_FILE_SIZE_EXCEEDS_MAX",
                "average_file_size_gb cannot exceed max_file_size_gb.",
                field="average_file_size_gb",
                details={
                    "average_file_size_gb": avg,
                    "max_file_size_gb": maximum,
                },
            ))

        growth = state.annual_growth_percent
        if (
            growth is not None
            and _finite_number(growth)
            and float(growth) > 0
            and state.planning_horizon_years is None
        ):
            issues.append(_issue(
                "GROWTH_REQUIRES_HORIZON",
                "Non-zero growth requires planning_horizon_years.",
                field="planning_horizon_years",
                details={"annual_growth_percent": growth},
            ))

        return issues

    def _validate_read_write_ratio(self, state):
        issues = []
        ratio = state.read_write_ratio

        if ratio is None:
            return issues

        if not isinstance(ratio, dict):
            issues.append(_issue(
                "INVALID_READ_WRITE_RATIO_STRUCTURE",
                (
                    "read_write_ratio must be an object with "
                    "read_percent and write_percent."
                ),
                field="read_write_ratio",
                details={"value": ratio},
            ))
            return issues

        if (
            "read_percent" not in ratio
            or "write_percent" not in ratio
        ):
            issues.append(_issue(
                "INVALID_READ_WRITE_RATIO_STRUCTURE",
                (
                    "read_write_ratio requires read_percent "
                    "and write_percent."
                ),
                field="read_write_ratio",
                details={"value": ratio},
            ))
            return issues

        read_value = ratio.get("read_percent")
        write_value = ratio.get("write_percent")

        if (
            not _finite_number(read_value)
            or not _finite_number(write_value)
        ):
            issues.append(_issue(
                "INVALID_READ_WRITE_RATIO_VALUE",
                "Read/write percentages must be finite numbers.",
                field="read_write_ratio",
                details={"value": ratio},
            ))
            return issues

        read_value = float(read_value)
        write_value = float(write_value)

        if (
            read_value < 0
            or read_value > 100
            or write_value < 0
            or write_value > 100
        ):
            issues.append(_issue(
                "INVALID_READ_WRITE_RATIO_VALUE",
                "Read/write percentages must each be between 0 and 100.",
                field="read_write_ratio",
                details={"value": ratio},
            ))

        total = read_value + write_value
        if not math.isclose(
            total,
            100.0,
            abs_tol=1e-6,
        ):
            issues.append(_issue(
                "READ_WRITE_RATIO_NOT_NORMALIZED",
                "read_percent + write_percent must equal 100.",
                field="read_write_ratio",
                details={
                    "read_percent": read_value,
                    "write_percent": write_value,
                    "sum": total,
                },
            ))

        return issues

    def _validate_categorical(self, state):
        issues = []

        if state.access_type is not None and state.access_type not in ACCESS_TYPES:
            issues.append(_issue(
                "INVALID_ACCESS_TYPE",
                "access_type must be sequential, random, or mixed.",
                field="access_type",
                details={"value": state.access_type},
            ))

        if state.ha_required is not None and not isinstance(state.ha_required, bool):
            issues.append(_issue(
                "INVALID_HA_VALUE",
                "ha_required must be boolean.",
                field="ha_required",
                details={"value": state.ha_required},
            ))

        return issues

    def _validate_preferences(self, state):
        issues = []

        for field_name in PREFERENCE_LABEL_FIELDS:
            value = getattr(state, field_name)
            if value is None:
                continue
            if value not in PREFERENCE_LEVELS:
                issues.append(_issue(
                    "INVALID_PREFERENCE_LEVEL",
                    f"Invalid value for {field_name}: {value!r}.",
                    field=field_name,
                ))

        return issues

    def _validate_weights(self, state):
        issues = []
        active = set(self._active_dimensions(state))
        weights = state.preference_weights

        if weights is None:
            if active:
                issues.append(_issue(
                    "ACTIVE_PREFERENCES_REQUIRE_WEIGHTS",
                    "Active preferences exist but preference_weights are missing.",
                    field="preference_weights",
                    details={"active_dimensions": sorted(active)},
                ))
            return issues

        values = weights.values_dict()

        for dimension, value in values.items():
            if not _finite_number(value):
                issues.append(_issue(
                    "NON_FINITE_PREFERENCE_WEIGHT",
                    f"{dimension} weight must be finite.",
                    field="preference_weights",
                    details={"dimension": dimension, "value": value},
                ))
            elif float(value) < 0:
                issues.append(_issue(
                    "NEGATIVE_PREFERENCE_WEIGHT",
                    f"{dimension} weight must be >= 0.",
                    field="preference_weights",
                    details={"dimension": dimension, "value": value},
                ))

        if all(_finite_number(v) for v in values.values()):
            total = float(sum(values.values()))
            if abs(total - 1.0) > WEIGHT_TOLERANCE:
                issues.append(_issue(
                    "PREFERENCE_WEIGHTS_NOT_NORMALIZED",
                    "Preference weights must sum to 1.",
                    field="preference_weights",
                    details={"sum": total, "tolerance": WEIGHT_TOLERANCE},
                ))

            for dimension, value in values.items():
                if (
                    dimension not in active
                    and abs(float(value)) > WEIGHT_TOLERANCE
                ):
                    issues.append(_issue(
                        "INACTIVE_PREFERENCE_HAS_NONZERO_WEIGHT",
                        f"Inactive {dimension} preference must have zero weight.",
                        field="preference_weights",
                        details={"dimension": dimension, "weight": value},
                    ))

        if (
            weights.consistency_status is not None
            and weights.consistency_status != "PASS"
        ):
            issues.append(_issue(
                "BWM_CONSISTENCY_NOT_PASS",
                "BWM consistency must be PASS.",
                field="preference_weights",
                details={"consistency_status": weights.consistency_status},
            ))

        return issues

    @staticmethod
    def _active_dimensions(state):
        mapping = {
            "cost": state.cost_priority,
            "power": state.power_priority,
            "performance": state.performance_priority,
            "reliability": state.reliability_priority,
        }
        return [
            dimension
            for dimension, level in mapping.items()
            if level is not None and level != "NO_SIGNAL"
        ]

    @staticmethod
    def _followup(field_name):
        prompts: Dict[str, str] = {
            "requested_usable_capacity_tib":
                "What usable capacity do you need?",
            "client_count":
                "How many clients will access the system?",
            "access_type":
                "Is the workload sequential, random, or mixed?",
            "ha_required":
                "Is high availability required?",
            "planning_horizon_years":
                "Over how many years should growth be planned?",
        }
        return prompts.get(
            field_name,
            f"Please clarify {field_name}.",
        )
