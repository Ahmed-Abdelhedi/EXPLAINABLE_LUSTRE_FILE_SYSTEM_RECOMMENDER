from __future__ import annotations

import json

from input_orchestrator.multi_turn_regression_smoke import (
    _answer_for_pending,
)
from input_orchestrator.production_wiring import (
    build_frozen_production_orchestrator,
)

from .production import finalize_orchestrator_session


def main() -> None:
    orchestrator = (
        build_frozen_production_orchestrator(
            device="cpu",
            enable_llm_fallback=False,
        )
    )

    session = orchestrator.new_session(
        "STEP6_2C_REAL_FINAL_JSON"
    )

    initial = (
        "We need 100 TiB of usable capacity for 64 clients. "
        "High availability is mandatory. "
        "The workload is sequential. "
        "Expected annual growth is 20 percent. "
        "Reliability is absolutely critical and "
        "performance is very important."
    )

    response = orchestrator.handle_message(
        initial,
        session,
    )

    turn_count = 1

    while not response.ready_for_final_validation:
        if response.pending_question is None:
            raise AssertionError(
                "Conversation is not ready but no pending question exists."
            )

        answer = _answer_for_pending(
            response.pending_question
        )

        response = orchestrator.handle_message(
            answer,
            session,
        )
        turn_count += 1

        if turn_count > 40:
            raise AssertionError(
                "Conversation exceeded 40 turns."
            )

    output = finalize_orchestrator_session(
        session
    )

    state = output.state
    canonical = json.loads(
        output.canonical_json
    )

    if not state.ready_for_sizing:
        raise AssertionError(
            "Final validated state is not ready_for_sizing."
        )

    if state.validation_issues:
        raise AssertionError(
            "Final validated state contains issues."
        )

    if canonical[
        "requested_usable_capacity_tib"
    ] != 100:
        raise AssertionError(
            "Canonical capacity mismatch."
        )

    if canonical[
        "planning_horizon_years"
    ] != 3:
        raise AssertionError(
            "Canonical planning horizon mismatch."
        )

    weights = canonical[
        "preference_weights"
    ]

    if weights is None:
        raise AssertionError(
            "Canonical preference_weights are missing."
        )

    if abs(
        sum(weights.values()) - 1.0
    ) > 1e-8:
        raise AssertionError(
            "Canonical preference weights do not sum to 1."
        )

    report = {
        "step": "6.2C",
        "status": (
            "REAL_ORCHESTRATOR_TO_FINAL_JSON_PASS"
        ),
        "turn_count": turn_count,
        "orchestrator_state":
            session.conversation_state.value,
        "ready_for_sizing":
            state.ready_for_sizing,
        "validation_issues":
            state.validation_issues,
        "canonical_requirement_json":
            canonical,
        "bwm_trace":
            session.bwm_dialogue.get(
                "last_result"
            ),
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
        "STATUS: "
        "REAL_ORCHESTRATOR_TO_FINAL_JSON_PASS"
    )


if __name__ == "__main__":
    main()
