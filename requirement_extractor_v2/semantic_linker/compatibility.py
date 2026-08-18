from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

from ..models import QuantityDimension, SemanticRole
from .labels import (
    FIELD_LABELS,
    ROLE_LABELS,
    SemanticField,
)


# =====================================================================
# DIMENSION -> FIELD COMPATIBILITY
# =====================================================================
#
# These rules use deterministic evidence already produced by the
# QuantityScanner. They do NOT decide the semantic field themselves.
#
# Example:
#     800 W -> dimension=POWER
#
# The Semantic Linker is therefore allowed to consider:
#     max_power_w
#     UNRESOLVED
#
# but not:
#     client_count
#     max_budget_usd
#     ...
#
# UNKNOWN intentionally keeps the complete quantitative field space
# available because unitless quantities such as "200 hosts" require
# semantic interpretation from the Transformer.
# =====================================================================


_ALL_RESOLVED_FIELDS: FrozenSet[SemanticField] = frozenset(
    field
    for field in FIELD_LABELS
    if field is not SemanticField.UNRESOLVED
)


ALLOWED_FIELDS_BY_DIMENSION: Dict[
    QuantityDimension,
    FrozenSet[SemanticField],
] = {
    QuantityDimension.CAPACITY: frozenset(
        {
            SemanticField.REQUESTED_USABLE_CAPACITY_TIB,
            SemanticField.UNRESOLVED,
        }
    ),
    QuantityDimension.FILE_SIZE: frozenset(
        {
            SemanticField.AVERAGE_FILE_SIZE_GB,
            SemanticField.MAX_FILE_SIZE_GB,
            SemanticField.UNRESOLVED,
        }
    ),
    QuantityDimension.THROUGHPUT: frozenset(
        {
            SemanticField.TARGET_READ_GBPS,
            SemanticField.TARGET_WRITE_GBPS,
            SemanticField.UNRESOLVED,
        }
    ),
    QuantityDimension.POWER: frozenset(
        {
            SemanticField.MAX_POWER_W,
            SemanticField.UNRESOLVED,
        }
    ),
    QuantityDimension.MONEY: frozenset(
        {
            SemanticField.MAX_BUDGET_USD,
            SemanticField.UNRESOLVED,
        }
    ),
    QuantityDimension.PERCENT: frozenset(
        {
            SemanticField.READ_WRITE_RATIO,
            SemanticField.ANNUAL_GROWTH_PERCENT,
            SemanticField.UNRESOLVED,
        }
    ),
    QuantityDimension.COUNT: frozenset(
        {
            SemanticField.CLIENT_COUNT,
            SemanticField.TOTAL_FILE_COUNT,
            SemanticField.UNRESOLVED,
        }
    ),
    QuantityDimension.UNKNOWN: frozenset(
        set(_ALL_RESOLVED_FIELDS)
        | {SemanticField.UNRESOLVED}
    ),
}


# =====================================================================
# FIELD -> ROLE COMPATIBILITY
# =====================================================================
#
# The role describes the business function of the quantity, not merely
# a surface word such as "expected".
#
# Examples:
#     "around 250 hosts will mount the filesystem"
#         -> CLIENT_COUNT + TOTAL_COUNT
#
#     "we expect to need 500 TiB"
#         -> REQUESTED_USABLE_CAPACITY_TIB + TARGET
#
#     "we currently have 300 TiB"
#         -> REQUESTED_USABLE_CAPACITY_TIB + CURRENT_VALUE
#
# UNSPECIFIED is allowed for a resolved field when the semantic field is
# identifiable but the exact role is not reliable enough.
#
# UNRESOLVED is special: it is compatible ONLY with UNSPECIFIED.
# =====================================================================


ALLOWED_ROLES_BY_FIELD: Dict[
    SemanticField,
    FrozenSet[SemanticRole],
] = {
    SemanticField.REQUESTED_USABLE_CAPACITY_TIB: frozenset(
        {
            SemanticRole.TARGET,
            SemanticRole.CURRENT_VALUE,
            SemanticRole.UNSPECIFIED,
        }
    ),
    SemanticField.CLIENT_COUNT: frozenset(
        {
            SemanticRole.TOTAL_COUNT,
            SemanticRole.UNSPECIFIED,
        }
    ),
    SemanticField.AVERAGE_FILE_SIZE_GB: frozenset(
        {
            SemanticRole.AVERAGE_VALUE,
            SemanticRole.UNSPECIFIED,
        }
    ),
    SemanticField.MAX_FILE_SIZE_GB: frozenset(
        {
            SemanticRole.MAXIMUM_LIMIT,
            SemanticRole.UNSPECIFIED,
        }
    ),
    SemanticField.TOTAL_FILE_COUNT: frozenset(
        {
            SemanticRole.TOTAL_COUNT,
            SemanticRole.UNSPECIFIED,
        }
    ),
    SemanticField.READ_WRITE_RATIO: frozenset(
        {
            SemanticRole.RATIO_COMPONENT,
            SemanticRole.UNSPECIFIED,
        }
    ),
    SemanticField.TARGET_READ_GBPS: frozenset(
        {
            SemanticRole.TARGET,
            SemanticRole.CURRENT_VALUE,
            SemanticRole.MINIMUM_LIMIT,
            SemanticRole.UNSPECIFIED,
        }
    ),
    SemanticField.TARGET_WRITE_GBPS: frozenset(
        {
            SemanticRole.TARGET,
            SemanticRole.CURRENT_VALUE,
            SemanticRole.MINIMUM_LIMIT,
            SemanticRole.UNSPECIFIED,
        }
    ),
    SemanticField.MAX_BUDGET_USD: frozenset(
        {
            SemanticRole.MAXIMUM_LIMIT,
            SemanticRole.CURRENT_VALUE,
            SemanticRole.UNSPECIFIED,
        }
    ),
    SemanticField.MAX_POWER_W: frozenset(
        {
            SemanticRole.MAXIMUM_LIMIT,
            SemanticRole.CURRENT_VALUE,
            SemanticRole.EXPECTED_VALUE,
            SemanticRole.UNSPECIFIED,
        }
    ),
    SemanticField.ANNUAL_GROWTH_PERCENT: frozenset(
        {
            SemanticRole.GROWTH_RATE,
            SemanticRole.UNSPECIFIED,
        }
    ),
    SemanticField.UNRESOLVED: frozenset(
        {
            SemanticRole.UNSPECIFIED,
        }
    ),
}


# =====================================================================
# PUBLIC HELPERS
# =====================================================================


def allowed_fields_for_dimension(
    dimension: QuantityDimension,
) -> Tuple[SemanticField, ...]:
    """
    Return allowed FIELD labels in the same stable order as FIELD_LABELS.
    """
    allowed = ALLOWED_FIELDS_BY_DIMENSION[dimension]
    return tuple(
        field
        for field in FIELD_LABELS
        if field in allowed
    )


def allowed_roles_for_field(
    field: SemanticField,
) -> Tuple[SemanticRole, ...]:
    """
    Return allowed ROLE labels in the same stable order as ROLE_LABELS.
    """
    allowed = ALLOWED_ROLES_BY_FIELD[field]
    return tuple(
        role
        for role in ROLE_LABELS
        if role in allowed
    )


def is_field_allowed_for_dimension(
    dimension: QuantityDimension,
    field: SemanticField,
) -> bool:
    return field in ALLOWED_FIELDS_BY_DIMENSION[dimension]


def is_role_allowed_for_field(
    field: SemanticField,
    role: SemanticRole,
) -> bool:
    return role in ALLOWED_ROLES_BY_FIELD[field]


def is_valid_field_role_pair(
    field: SemanticField,
    role: SemanticRole,
) -> bool:
    return is_role_allowed_for_field(field, role)


def build_field_mask(
    dimension: QuantityDimension,
) -> Tuple[bool, ...]:
    """
    Boolean mask aligned exactly with FIELD_LABELS.

    Later, the Transformer inference code can convert this mask to a
    tensor and set forbidden FIELD logits to -inf before softmax.
    """
    allowed = ALLOWED_FIELDS_BY_DIMENSION[dimension]
    return tuple(
        field in allowed
        for field in FIELD_LABELS
    )


def build_role_mask(
    field: SemanticField,
) -> Tuple[bool, ...]:
    """
    Boolean mask aligned exactly with ROLE_LABELS.

    Later, the Transformer inference code can convert this mask to a
    tensor and set forbidden ROLE logits to -inf before softmax.
    """
    allowed = ALLOWED_ROLES_BY_FIELD[field]
    return tuple(
        role in allowed
        for role in ROLE_LABELS
    )