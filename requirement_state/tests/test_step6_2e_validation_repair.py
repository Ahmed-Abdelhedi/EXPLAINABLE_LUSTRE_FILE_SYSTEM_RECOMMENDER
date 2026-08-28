from input_orchestrator.models import (
    ConversationState,
    FieldState,
)
from input_orchestrator.session_state import WorkingSessionState
from requirement_state import FinalRequirementState
from requirement_state.validation_repair import (
    prepare_validation_repair,
)


def test_average_greater_than_max_returns_to_targeted_clarification():
    session = WorkingSessionState("S")

    avg = session.get("average_file_size_gb")
    avg.value = 20
    avg.state = FieldState.VERIFIED

    maximum = session.get("max_file_size_gb")
    maximum.value = 1
    maximum.state = FieldState.VERIFIED

    state = FinalRequirementState()
    state.validation_issues = [
        {
            "code": "AVERAGE_FILE_SIZE_EXCEEDS_MAX",
            "field": "average_file_size_gb",
            "message": "average cannot exceed max",
            "details": {
                "average_file_size_gb": 20,
                "max_file_size_gb": 1,
            },
        }
    ]

    action = prepare_validation_repair(
        state=state,
        session=session,
    )

    assert action.mode == "ASK_FIELD"
    assert action.target_field == "max_file_size_gb"
    assert (
        session.conversation_state
        == ConversationState.WAITING_FOR_ANSWER
    )
    assert session.pending_question is not None
    assert (
        session.pending_question.target_field
        == "max_file_size_gb"
    )
    assert (
        session.pending_question.context[
            "validation_repair"
        ]
        is True
    )
