from input_orchestrator.multi_turn_regression_smoke import (
    OPTIONAL_SKIP_FIELDS,
    _answer_for_pending,
)
from input_orchestrator.models import PendingQuestion


def _pending(field, expected, context=None):
    return PendingQuestion(
        question_id="Q",
        target_field=field,
        question="q",
        expected_answer_type=expected,
        created_after_message_id="M",
        context=context or {},
    )


def test_horizon_answer_is_contextual_integer():
    assert _answer_for_pending(
        _pending(
            "planning_horizon_years",
            "positive_integer_years",
        )
    ) == "3"


def test_optional_fields_have_explicit_skip_policy():
    assert "max_budget_usd" in OPTIONAL_SKIP_FIELDS
    assert "target_read_gbps" in OPTIONAL_SKIP_FIELDS
    assert "cost_priority" in OPTIONAL_SKIP_FIELDS


def test_bwm_answer_is_explicit_1_to_9_judgment():
    answer = _answer_for_pending(
        _pending(
            "preference_weights",
            "bwm_judgment",
            {
                "comparison_id":
                    "B2O:reliability:performance"
            },
        )
    )

    assert answer == "3"
