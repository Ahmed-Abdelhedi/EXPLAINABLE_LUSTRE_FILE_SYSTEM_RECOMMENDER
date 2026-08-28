from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .field_registry import CORE_REQUIRED_FIELDS


@dataclass(frozen=True)
class OrchestrationPolicy:
    core_required_fields: Tuple[str, ...] = CORE_REQUIRED_FIELDS

    # Production rule:
    # every Requirement field must be either VERIFIED or explicitly answered
    # as no-constraint/declined before final validation.
    ask_optional_fields: bool = True

    one_question_per_turn: bool = True
    require_horizon_when_growth_present: bool = True

    # Kept only for narrow wiring/smoke compatibility.
    # MUST remain False in the real production conversation.
    ready_when_core_complete_and_no_conflict: bool = False


DEFAULT_POLICY = OrchestrationPolicy()
STRICT_PRODUCTION_POLICY = DEFAULT_POLICY
