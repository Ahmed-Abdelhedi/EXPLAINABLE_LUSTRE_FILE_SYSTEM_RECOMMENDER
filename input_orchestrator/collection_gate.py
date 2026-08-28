from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .field_registry import FIELD_SPECS
from .models import FieldState
from .policies import OrchestrationPolicy
from .session_state import WorkingSessionState


@dataclass(frozen=True)
class CollectionGapReport:
    """
    Blocking state before READY_FOR_FINAL_VALIDATION.

    A Requirement field is considered answered when:
      - REQUIRED field: VERIFIED only
      - OPTIONAL field: VERIFIED or explicitly DECLINED
      - CONDITIONAL horizon with non-zero growth: VERIFIED only

    Therefore an absent/partial/unresolved/conflicting field can never be
    silently bypassed.
    """
    blocking_fields: List[str]
    missing_fields: List[str]
    unresolved_fields: List[str]
    declined_but_required_fields: List[str]
    conditional_required_fields: List[str]

    @property
    def complete(self) -> bool:
        return not self.blocking_fields

    def to_dict(self) -> dict:
        return {
            "complete": self.complete,
            "blocking_fields": list(self.blocking_fields),
            "missing_fields": list(self.missing_fields),
            "unresolved_fields": list(self.unresolved_fields),
            "declined_but_required_fields": list(
                self.declined_but_required_fields
            ),
            "conditional_required_fields": list(
                self.conditional_required_fields
            ),
        }


def collection_gap_report(
    session: WorkingSessionState,
    policy: OrchestrationPolicy,
) -> CollectionGapReport:
    required = set(policy.core_required_fields)

    growth = session.get("annual_growth_percent")
    growth_requires_horizon = (
        policy.require_horizon_when_growth_present
        and growth.state == FieldState.VERIFIED
        and growth.value not in {None, 0, 0.0}
    )

    missing: List[str] = []
    unresolved: List[str] = []
    declined_required: List[str] = []
    conditional_required: List[str] = []
    blocking: List[str] = []

    for field_name in FIELD_SPECS:
        record = session.get(field_name)

        is_required = field_name in required
        is_conditional_required = (
            field_name == "planning_horizon_years"
            and growth_requires_horizon
        )

        if is_conditional_required:
            conditional_required.append(field_name)

        if record.state == FieldState.VERIFIED:
            continue

        # Optional field explicitly declined = user answered the question.
        if (
            record.state == FieldState.DECLINED
            and not is_required
            and not is_conditional_required
        ):
            continue

        blocking.append(field_name)

        if record.state == FieldState.MISSING:
            missing.append(field_name)
        elif record.state in {
            FieldState.UNRESOLVED,
            FieldState.PARTIAL,
            FieldState.CONFLICT,
        }:
            unresolved.append(field_name)
        elif record.state == FieldState.DECLINED:
            declined_required.append(field_name)

    return CollectionGapReport(
        blocking_fields=blocking,
        missing_fields=missing,
        unresolved_fields=unresolved,
        declined_but_required_fields=declined_required,
        conditional_required_fields=conditional_required,
    )
