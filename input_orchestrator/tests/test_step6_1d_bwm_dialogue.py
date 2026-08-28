from dataclasses import dataclass
from enum import Enum

from input_orchestrator.bwm_coordinator import BWMCoordinator
from input_orchestrator.models import (
    FieldState,
    PendingQuestion,
)
from input_orchestrator.session_state import WorkingSessionState


def test_bwm_numeric_answer_is_stored():
    coordinator = BWMCoordinator(
        weighting_layer=object(),
        enabled=True,
    )
    session = WorkingSessionState("S1")

    pending = PendingQuestion(
        question_id="Q1",
        target_field="preference_weights",
        question="How much more important?",
        expected_answer_type="bwm_judgment",
        created_after_message_id="M1",
        context={
            "comparison_id":
                "B2O:reliability:performance",
        },
    )

    consumed = coordinator.consume_pending_answer(
        text="3",
        session=session,
        pending_question=pending,
    )

    assert consumed is True
    assert (
        session.bwm_dialogue["answers"][
            "B2O:reliability:performance"
        ]
        == 3
    )


def test_bwm_rejects_out_of_range_or_non_integer():
    coordinator = BWMCoordinator(
        weighting_layer=object(),
        enabled=True,
    )
    session = WorkingSessionState("S2")

    pending = PendingQuestion(
        question_id="Q1",
        target_field="preference_weights",
        question="Score?",
        expected_answer_type="bwm_judgment",
        created_after_message_id="M1",
        context={"comparison_id": "B2O:a:b"},
    )

    assert coordinator.consume_pending_answer(
        text="10",
        session=session,
        pending_question=pending,
    ) is False

    assert session.bwm_dialogue["answers"] == {}
    assert session.bwm_dialogue["last_input_error"]


def test_best_dimension_parser_supports_french_alias():
    coordinator = BWMCoordinator(
        weighting_layer=object(),
        enabled=True,
    )
    session = WorkingSessionState("S3")

    pending = PendingQuestion(
        question_id="Q1",
        target_field="preference_weights",
        question="Best?",
        expected_answer_type="bwm_best",
        created_after_message_id="M1",
        context={
            "active_dimensions": [
                "cost",
                "reliability",
            ]
        },
    )

    assert coordinator.consume_pending_answer(
        text="fiabilité",
        session=session,
        pending_question=pending,
    ) is True

    assert (
        session.bwm_dialogue["explicit_best"]
        == "reliability"
    )


def test_single_active_confirmation_is_explicit():
    coordinator = BWMCoordinator(
        weighting_layer=object(),
        enabled=True,
    )
    session = WorkingSessionState("S4")

    pending = PendingQuestion(
        question_id="Q1",
        target_field="preference_weights",
        question="Confirm?",
        expected_answer_type="bwm_single_confirmation",
        created_after_message_id="M1",
        context={},
    )

    assert coordinator.consume_pending_answer(
        text="yes",
        session=session,
        pending_question=pending,
    ) is True

    assert (
        session.bwm_dialogue[
            "single_active_confirmed"
        ]
        is True
    )
