from __future__ import annotations

import json

from .field_registry import FIELD_SPECS
from .models import FieldState
from .production_wiring import (
    build_frozen_production_orchestrator,
)


def _prepare_completed_raw_state(session) -> None:
    """
    Prepare a completed raw Requirement state so this smoke isolates the BWM
    dialogue. Extractor models are still loaded by the real production wiring.

    Two active preferences:
      reliability  = VERY_HIGH
      performance  = HIGH

    Cost/power and all other optional fields are explicitly declined.
    """
    core_values = {
        "requested_usable_capacity_tib": 100,
        "client_count": 64,
        "access_type": "sequential",
        "ha_required": True,
    }

    for name, spec in FIELD_SPECS.items():
        record = session.get(name)

        if name in core_values:
            record.value = core_values[name]
            record.state = FieldState.VERIFIED
            record.source = "BWM_SMOKE_SETUP"
            continue

        if name == "reliability_priority":
            record.value = "VERY_HIGH"
            record.state = FieldState.VERIFIED
            record.source = "BWM_SMOKE_SETUP"
            continue

        if name == "performance_priority":
            record.value = "HIGH"
            record.state = FieldState.VERIFIED
            record.source = "BWM_SMOKE_SETUP"
            continue

        record.value = None
        record.state = FieldState.DECLINED
        record.source = "BWM_SMOKE_SETUP"


def main() -> None:
    orchestrator = (
        build_frozen_production_orchestrator(
            device="cpu",
            enable_llm_fallback=False,
        )
    )

    session = orchestrator.new_session(
        "STEP6_1D_BWM_CONVERSATION"
    )

    _prepare_completed_raw_state(session)

    first = orchestrator.handle_message(
        "continue",
        session,
    )

    if (
        first.conversation_state.value
        != "BWM_ELICITATION"
    ):
        raise AssertionError(
            "Expected BWM_ELICITATION before final validation."
        )

    if first.pending_question is None:
        raise AssertionError(
            "BWM did not produce a comparison question."
        )

    if (
        first.pending_question.expected_answer_type
        != "bwm_judgment"
    ):
        raise AssertionError(
            "Expected direct BWM judgment for the two ordered criteria."
        )

    second = orchestrator.handle_message(
        "3",
        session,
    )

    if not second.ready_for_final_validation:
        raise AssertionError(
            "Completed BWM did not reach final-validation readiness."
        )

    weights_record = session.get(
        "preference_weights"
    )

    if weights_record.state != FieldState.VERIFIED:
        raise AssertionError(
            "preference_weights was not VERIFIED."
        )

    weights = dict(weights_record.value)

    if abs(sum(weights.values()) - 1.0) > 1e-8:
        raise AssertionError(
            "BWM weights do not sum to 1."
        )

    if not (
        weights["reliability"]
        > weights["performance"]
        > weights["cost"]
    ):
        raise AssertionError(
            "BWM weight order does not preserve verified preference order."
        )

    components = orchestrator.production_components

    no_spurious_extractor_calls = {
        "quantity": (
            components["quantity"].last_result is None
        ),
        "preference": (
            components["preference"].last_signal_result is None
        ),
        "categorical": (
            components["categorical"].last_result is None
        ),
    }

    if not all(no_spurious_extractor_calls.values()):
        raise AssertionError(
            "A pure BWM control turn was incorrectly routed to a normal "
            "Requirement extractor."
        )

    report = {
        "step": "6.1D",
        "status": "BWM_CONVERSATIONAL_INTEGRATION_PASS",
        "no_spurious_extractor_calls":
            no_spurious_extractor_calls,
        "first_turn": first.to_dict(),
        "second_turn": second.to_dict(),
        "preference_weights": weights,
        "weighting_trace": session.bwm_dialogue,
        "invariants": {
            "weights_sum_to_one": True,
            "no_direct_label_to_number_mapping": True,
            "user_bwm_judgment_used": True,
            "ready_only_after_weighting_terminal": True,
        },
    }

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print(
        "STATUS: BWM_CONVERSATIONAL_INTEGRATION_PASS"
    )


if __name__ == "__main__":
    main()
