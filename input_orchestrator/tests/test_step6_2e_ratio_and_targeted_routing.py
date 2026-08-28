from dataclasses import dataclass

from input_orchestrator.context_answers import contextual_observation
from input_orchestrator.models import (
    Evidence,
    FieldObservation,
    FieldState,
    PendingQuestion,
)
from input_orchestrator.orchestrator import InputOrchestrator
from input_orchestrator.ratio_parser import parse_read_write_ratio
from input_orchestrator.router import ExtractionRouter


def _pending(field, expected):
    return PendingQuestion(
        question_id="Q1",
        target_field=field,
        question="question",
        expected_answer_type=expected,
        created_after_message_id="M1",
        context={},
    )


def test_pending_20_80_ratio_becomes_canonical_structure():
    assert parse_read_write_ratio(
        "20/80",
        pending_ratio_question=True,
    ) == {
        "read_percent": 20.0,
        "write_percent": 80.0,
    }


def test_ratio_4_to_1_is_normalized_to_percentages():
    assert parse_read_write_ratio(
        "4:1",
        pending_ratio_question=True,
    ) == {
        "read_percent": 80.0,
        "write_percent": 20.0,
    }


def test_labeled_ratio_is_supported():
    assert parse_read_write_ratio(
        "20 read 80 write",
    ) == {
        "read_percent": 20.0,
        "write_percent": 80.0,
    }


def test_not_important_maps_to_low_preference():
    obs = contextual_observation(
        "not important",
        pending_question=_pending(
            "power_priority",
            "preference",
        ),
        message_id="M2",
    )

    assert len(obs) == 1
    assert obs[0].field == "power_priority"
    assert obs[0].value == "LOW"
    assert obs[0].state == FieldState.VERIFIED


class CountingExtractor:
    domain = "fake"

    def __init__(self):
        self.calls = 0

    def extract(
        self,
        text,
        *,
        message_id,
        pending_question=None,
    ):
        self.calls += 1
        return [
            FieldObservation(
                field="ha_required",
                value=False,
                state=FieldState.VERIFIED,
                source="SHOULD_NOT_RUN",
                evidence=Evidence(text=text),
                message_id=message_id,
            )
        ]


def test_short_contextual_preference_answer_skips_unrelated_extractors():
    extractor = CountingExtractor()
    orchestrator = InputOrchestrator(
        router=ExtractionRouter(
            [extractor]
        )
    )
    session = orchestrator.new_session("S")

    session.pending_question = _pending(
        "power_priority",
        "preference",
    )

    response = orchestrator.handle_message(
        "not important",
        session,
    )

    assert extractor.calls == 0
    assert session.get("power_priority").value == "LOW"
    assert session.get("ha_required").state != FieldState.VERIFIED
