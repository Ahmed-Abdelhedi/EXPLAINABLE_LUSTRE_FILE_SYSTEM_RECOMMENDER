from input_orchestrator.models import PendingQuestion
from input_orchestrator.orchestrator import InputOrchestrator


def test_exact_bwm_answer_skips_normal_extractors():
    pending = PendingQuestion(
        question_id="Q1",
        target_field="preference_weights",
        question="BWM score?",
        expected_answer_type="bwm_judgment",
        created_after_message_id="M1",
        context={
            "comparison_id": "B2O:reliability:performance",
        },
    )

    assert InputOrchestrator._is_pure_bwm_control_turn(
        text="3",
        previous_pending=pending,
        bwm_answer_consumed=True,
        raw_collection_complete=True,
    ) is True


def test_rich_bwm_answer_is_not_suppressed():
    pending = PendingQuestion(
        question_id="Q1",
        target_field="preference_weights",
        question="BWM score?",
        expected_answer_type="bwm_judgment",
        created_after_message_id="M1",
        context={
            "comparison_id": "B2O:reliability:performance",
        },
    )

    # The strict BWM parser would not consume this rich answer.
    assert InputOrchestrator._is_pure_bwm_control_turn(
        text="3, and write throughput is 8 GB/s",
        previous_pending=pending,
        bwm_answer_consumed=False,
        raw_collection_complete=True,
    ) is False


def test_continue_after_complete_collection_skips_extractors():
    assert InputOrchestrator._is_pure_bwm_control_turn(
        text="continue",
        previous_pending=None,
        bwm_answer_consumed=False,
        raw_collection_complete=True,
    ) is True


def test_continue_before_complete_collection_is_still_routed():
    assert InputOrchestrator._is_pure_bwm_control_turn(
        text="continue",
        previous_pending=None,
        bwm_answer_consumed=False,
        raw_collection_complete=False,
    ) is False
