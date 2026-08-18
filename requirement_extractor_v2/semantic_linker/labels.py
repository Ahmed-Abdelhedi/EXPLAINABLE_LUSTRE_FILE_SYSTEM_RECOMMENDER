from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

from ..models import ParamName, SemanticRole

class SemanticField(str, Enum):
    REQUESTED_USABLE_CAPACITY_TIB = ParamName.requested_usable_capacity_tib.value
    CLIENT_COUNT = ParamName.client_count.value
    AVERAGE_FILE_SIZE_GB = ParamName.average_file_size_gb.value
    MAX_FILE_SIZE_GB = ParamName.max_file_size_gb.value
    TOTAL_FILE_COUNT = ParamName.total_file_count.value
    READ_WRITE_RATIO = ParamName.read_write_ratio.value
    TARGET_READ_GBPS = ParamName.target_read_gbps.value
    TARGET_WRITE_GBPS = ParamName.target_write_gbps.value
    MAX_BUDGET_USD = ParamName.max_budget_usd.value
    MAX_POWER_W = ParamName.max_power_w.value
    ANNUAL_GROWTH_PERCENT = ParamName.annual_growth_percent.value
    UNRESOLVED = "__UNRESOLVED__"

    @property
    def is_unresolved(self) -> bool:
        return self is SemanticField.UNRESOLVED

    def to_param_name(self) -> Optional[ParamName]:
        if self is SemanticField.UNRESOLVED:
            return None
        return ParamName(self.value)

    @classmethod
    def from_param_name(cls, field: ParamName) -> "SemanticField":
        try:
            result = cls(field.value)
        except ValueError as exc:
            raise ValueError(
                f"{field.value!r} is not a quantitative Semantic Linker field"
            ) from exc
        if result is cls.UNRESOLVED:
            raise ValueError("UNRESOLVED is not a Requirement Contract field")
        return result


FIELD_LABELS: Tuple[SemanticField, ...] = (
    SemanticField.REQUESTED_USABLE_CAPACITY_TIB,
    SemanticField.CLIENT_COUNT,
    SemanticField.AVERAGE_FILE_SIZE_GB,
    SemanticField.MAX_FILE_SIZE_GB,
    SemanticField.TOTAL_FILE_COUNT,
    SemanticField.READ_WRITE_RATIO,
    SemanticField.TARGET_READ_GBPS,
    SemanticField.TARGET_WRITE_GBPS,
    SemanticField.MAX_BUDGET_USD,
    SemanticField.MAX_POWER_W,
    SemanticField.ANNUAL_GROWTH_PERCENT,
    SemanticField.UNRESOLVED,
)

FIELD_TO_ID: Dict[SemanticField, int] = {
    label: index for index, label in enumerate(FIELD_LABELS)
}
ID_TO_FIELD: Dict[int, SemanticField] = {
    index: label for index, label in enumerate(FIELD_LABELS)
}

ROLE_LABELS: Tuple[SemanticRole, ...] = (
    SemanticRole.MAXIMUM_LIMIT,
    SemanticRole.MINIMUM_LIMIT,
    SemanticRole.TARGET,
    SemanticRole.CURRENT_VALUE,
    SemanticRole.EXPECTED_VALUE,
    SemanticRole.AVERAGE_VALUE,
    SemanticRole.TOTAL_COUNT,
    SemanticRole.RATIO_COMPONENT,
    SemanticRole.GROWTH_RATE,
    SemanticRole.UNSPECIFIED,
)

ROLE_TO_ID: Dict[SemanticRole, int] = {
    role: index for index, role in enumerate(ROLE_LABELS)
}
ID_TO_ROLE: Dict[int, SemanticRole] = {
    index: role for index, role in enumerate(ROLE_LABELS)
}

NUM_FIELD_LABELS = len(FIELD_LABELS)
NUM_ROLE_LABELS = len(ROLE_LABELS)


def field_id(label: SemanticField) -> int:
    return FIELD_TO_ID[label]


def field_from_id(index: int) -> SemanticField:
    try:
        return ID_TO_FIELD[index]
    except KeyError as exc:
        raise ValueError(f"Invalid FIELD label id: {index}") from exc


def role_id(role: SemanticRole) -> int:
    return ROLE_TO_ID[role]


def role_from_id(index: int) -> SemanticRole:
    try:
        return ID_TO_ROLE[index]
    except KeyError as exc:
        raise ValueError(f"Invalid ROLE label id: {index}") from exc