from input_orchestrator.context_answers import contextual_observation
from input_orchestrator.models import FieldState, PendingQuestion


def test_short_preference_level_uses_pending_context():
    pending = PendingQuestion(
        question_id="Q1",
        target_field="performance_priority",
        question="How important is performance?",
        expected_answer_type="preference",
        created_after_message_id="M0",
    )

    out = contextual_observation(
        "very high",
        pending_question=pending,
        message_id="M1",
    )

    assert len(out) == 1
    assert out[0].value == "VERY_HIGH"
    assert out[0].state == FieldState.VERIFIED


def test_conflict_choice_latest_selects_new_value():
    pending = PendingQuestion(
        question_id="Q1",
        target_field="access_type",
        question="Which value?",
        expected_answer_type="conflict_choice",
        created_after_message_id="M0",
        context={
            "previous_value": "sequential",
            "new_value": "random",
        },
    )

    out = contextual_observation(
        "latest",
        pending_question=pending,
        message_id="M1",
    )

    assert len(out) == 1
    assert out[0].value == "random"
    assert out[0].explicit_correction is True
