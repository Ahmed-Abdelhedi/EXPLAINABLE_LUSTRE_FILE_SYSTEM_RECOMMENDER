from __future__ import annotations

from typing import Optional

from .field_registry import FIELD_SPECS
from .models import FieldState, PendingQuestion
from .policies import OrchestrationPolicy
from .session_state import WorkingSessionState


class QuestionPlanner:
    def __init__(self, policy: OrchestrationPolicy) -> None:
        self.policy = policy

    def next_question(
        self,
        session: WorkingSessionState,
        *,
        question_id: str,
        message_id: str,
    ) -> Optional[PendingQuestion]:
        growth = session.get("annual_growth_percent")
        horizon = session.get("planning_horizon_years")

        # CONDITIONAL REQUIRED:
        # if growth > 0, horizon must be VERIFIED.
        # DECLINED is NOT sufficient and the question is asked again.
        if (
            self.policy.require_horizon_when_growth_present
            and growth.state == FieldState.VERIFIED
            and growth.value not in {None, 0, 0.0}
            and horizon.state != FieldState.VERIFIED
        ):
            spec = FIELD_SPECS["planning_horizon_years"]
            return PendingQuestion(
                question_id=question_id,
                target_field=spec.name,
                question=spec.question,
                expected_answer_type=spec.expected_answer_type,
                created_after_message_id=message_id,
                context={
                    "annual_growth_percent": growth.value,
                    "required_reason": "NON_ZERO_GROWTH_REQUIRES_HORIZON",
                },
            )

        candidates = []
        required = set(self.policy.core_required_fields)

        for name, spec in FIELD_SPECS.items():
            record = session.get(name)

            # Required fields are complete only when VERIFIED.
            if name in required:
                if record.state == FieldState.VERIFIED:
                    continue
            else:
                # Optional fields are complete after a real value OR an
                # explicit user no-constraint/skip answer.
                if record.state in {
                    FieldState.VERIFIED,
                    FieldState.DECLINED,
                }:
                    continue

            if not self.policy.ask_optional_fields and spec.skippable:
                continue

            candidates.append(spec)

        if not candidates:
            return None

        spec = min(
            candidates,
            key=lambda item: item.priority,
        )

        context = {}

        if spec.name in required:
            context["required_reason"] = "CORE_REQUIRED_FIELD"

        return PendingQuestion(
            question_id=question_id,
            target_field=spec.name,
            question=spec.question,
            expected_answer_type=spec.expected_answer_type,
            created_after_message_id=message_id,
            context=context,
        )
