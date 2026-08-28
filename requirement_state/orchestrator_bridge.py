from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .finalizer import RequirementStateFinalizer
from .json_io import canonical_json_string
from .models import FinalRequirementState


EXPECTED_READY_STATE = "READY_FOR_FINAL_VALIDATION"


@dataclass(frozen=True)
class FinalRequirementOutput:
    state: FinalRequirementState
    canonical_json: str

    def canonical_dict(self) -> Dict[str, Any]:
        return self.state.to_canonical_json_dict()

    def full_state_dict(self) -> Dict[str, Any]:
        return self.state.to_dict(
            include_traceability=True
        )


class OrchestratorRequirementBridge:
    """
    Step 6.2C production boundary:

        InputOrchestrator session
            -> RequirementStateFinalizer
            -> deterministic validation
            -> canonical Requirement JSON

    The bridge never mutates extractor outputs and never invents missing values.
    """

    def __init__(
        self,
        *,
        finalizer: RequirementStateFinalizer | None = None,
    ) -> None:
        self.finalizer = (
            finalizer
            if finalizer is not None
            else RequirementStateFinalizer()
        )

    def finalize_session(
        self,
        session,
        *,
        require_orchestrator_ready: bool = True,
    ) -> FinalRequirementState:
        if require_orchestrator_ready:
            state_obj = getattr(
                session,
                "conversation_state",
                None,
            )
            state_value = getattr(
                state_obj,
                "value",
                state_obj,
            )

            if state_value != EXPECTED_READY_STATE:
                raise RuntimeError(
                    "Cannot finalize Requirement State before the "
                    "Input Orchestrator reaches "
                    "READY_FOR_FINAL_VALIDATION."
                )

            if getattr(
                session,
                "pending_question",
                None,
            ) is not None:
                raise RuntimeError(
                    "Cannot finalize while a clarification/BWM "
                    "question is still pending."
                )

        return self.finalizer.from_session(
            session
        )

    def export_ready_session(
        self,
        session,
    ) -> FinalRequirementOutput:
        state = self.finalize_session(
            session,
            require_orchestrator_ready=True,
        )

        if not state.ready_for_sizing:
            issue_codes = [
                issue.get("code")
                for issue in state.validation_issues
            ]

            raise RuntimeError(
                "Final Requirement State failed deterministic "
                "validation and cannot be exported for sizing. "
                f"Issues: {issue_codes}"
            )

        return FinalRequirementOutput(
            state=state,
            canonical_json=canonical_json_string(
                state
            ),
        )
