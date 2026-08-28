from __future__ import annotations

from .orchestrator_bridge import (
    FinalRequirementOutput,
    OrchestratorRequirementBridge,
)


def finalize_orchestrator_session(
    session,
) -> FinalRequirementOutput:
    """
    Public Step 6.2C production helper.

    Call only after the conversation response reports:
        ready_for_final_validation == True
    """
    return OrchestratorRequirementBridge().export_ready_session(
        session
    )
