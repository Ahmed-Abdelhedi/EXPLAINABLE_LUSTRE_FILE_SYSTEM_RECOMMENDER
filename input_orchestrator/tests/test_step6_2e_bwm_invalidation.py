from input_orchestrator.models import (
    Evidence,
    FieldObservation,
    FieldState,
)
from input_orchestrator.result_merger import ResultMerger
from input_orchestrator.session_state import WorkingSessionState


def test_preference_change_invalidates_stale_bwm():
    session = WorkingSessionState("S")

    session.bwm_dialogue["answers"] = {
        "B2O:reliability:performance": 3
    }
    session.bwm_dialogue["last_status"] = "WEIGHTS_READY"

    weights = session.get("preference_weights")
    weights.value = {
        "cost": 0.0,
        "power": 0.0,
        "performance": 0.25,
        "reliability": 0.75,
    }
    weights.state = FieldState.VERIFIED

    result = ResultMerger().merge(
        session,
        [
            FieldObservation(
                field="reliability_priority",
                value="VERY_HIGH",
                state=FieldState.VERIFIED,
                source="TEST",
                evidence=Evidence(
                    text="very high reliability"
                ),
                message_id="M1",
            )
        ],
    )

    assert "reliability_priority" in result.updated_fields
    assert session.bwm_dialogue["answers"] == {}
    assert session.bwm_dialogue["last_status"] is None
    assert (
        session.get("preference_weights").state
        == FieldState.MISSING
    )
