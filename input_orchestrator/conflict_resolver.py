from __future__ import annotations

from typing import Optional

from .models import Conflict, PendingQuestion


class ConflictResolver:
    def question_for(
        self,
        conflict: Conflict,
        *,
        question_id: str,
        message_id: str,
    ) -> PendingQuestion:
        return PendingQuestion(
            question_id=question_id,
            target_field=conflict.field,
            question=(
                f"I have two different values for {conflict.field}: "
                f"{conflict.previous_value!r} and {conflict.new_value!r}. "
                "Which value should be kept?"
            ),
            expected_answer_type="conflict_choice",
            created_after_message_id=message_id,
            context={
                "previous_value": conflict.previous_value,
                "new_value": conflict.new_value,
            },
        )
