from __future__ import annotations

from .builder import RequirementStateBuilder
from .validator import DeterministicRequirementValidator


class RequirementStateFinalizer:
    """Build + deterministic validation from an orchestrator session."""

    def __init__(self, *, builder=None, validator=None) -> None:
        self.builder = builder or RequirementStateBuilder()
        self.validator = validator or DeterministicRequirementValidator()

    def from_session(self, session):
        dialogue = getattr(session, "bwm_dialogue", {}) or {}
        result = dialogue.get("last_result") or {}
        consistency = result.get("consistency") or {}

        bwm_metadata = {
            "method": result.get("method"),
            "xi_star": result.get("xi_star"),
            "consistency_status": (
                result.get("ordinal_consistency")
                or consistency.get("status")
            ),
            "source": result.get("source"),
        }

        state = self.builder.from_session(
            session,
            bwm_metadata=bwm_metadata,
        )
        return self.validator.validate(state)
