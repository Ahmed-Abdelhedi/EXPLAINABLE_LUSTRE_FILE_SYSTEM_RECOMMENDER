from __future__ import annotations

import json
from typing import Dict, List

from .models import FieldState
from .production_wiring import (
    build_frozen_production_orchestrator,
)


OPTIONAL_SKIP_FIELDS = {
    "average_file_size_gb",
    "max_file_size_gb",
    "total_file_count",
    "read_write_ratio",
    "target_read_gbps",
    "target_write_gbps",
    "max_budget_usd",
    "max_power_w",
    "cost_priority",
    "power_priority",
}

CORE_FALLBACK_ANSWERS = {
    "requested_usable_capacity_tib": "100 TiB",
    "client_count": "64",
    "access_type": "sequential",
    "ha_required": "yes",
}


def _turn_to_dict(index: int, user: str, response) -> dict:
    return {
        "turn": index,
        "user": user,
        "assistant": response.assistant_message,
        "state": response.conversation_state.value,
        "ready": response.ready_for_final_validation,
        "pending_field": (
            None
            if response.pending_question is None
            else response.pending_question.target_field
        ),
        "pending_type": (
            None
            if response.pending_question is None
            else response.pending_question.expected_answer_type
        ),
        "updated_fields": list(response.updated_fields),
        "conflict_count": len(response.conflicts),
    }


def _answer_for_pending(pending) -> str:
    field = pending.target_field
    expected = pending.expected_answer_type

    if field == "planning_horizon_years":
        return "3"

    if field == "annual_growth_percent":
        # The initial message already includes growth. This is only a guard
        # against unexpected extractor behavior.
        return "20%"

    if field == "preference_weights":
        if expected == "bwm_judgment":
            return "3"

        if expected == "bwm_single_confirmation":
            return "yes"

        if expected == "bwm_best":
            active = pending.context.get(
                "active_dimensions",
                [],
            )
            if "reliability" in active:
                return "reliability"
            return str(active[0])

        if expected == "bwm_worst":
            active = pending.context.get(
                "active_dimensions",
                [],
            )
            if "performance" in active:
                return "performance"
            return str(active[-1])

        raise AssertionError(
            f"Unexpected BWM question type: {expected}"
        )

    if field in OPTIONAL_SKIP_FIELDS:
        return "skip"

    if field in CORE_FALLBACK_ANSWERS:
        return CORE_FALLBACK_ANSWERS[field]

    # planning_horizon_years is conditionally required only when growth > 0.
    # If it appears outside that condition in this smoke, the strict dialogue
    # can still answer explicitly.
    if field == "planning_horizon_years":
        return "3"

    raise AssertionError(
        f"No deterministic smoke answer configured for {field!r}"
    )


def run_happy_path() -> dict:
    orchestrator = (
        build_frozen_production_orchestrator(
            device="cpu",
            enable_llm_fallback=False,
        )
    )
    session = orchestrator.new_session(
        "STEP6_1E_HAPPY_PATH"
    )

    initial = (
        "We need 100 TiB of usable capacity for 64 clients. "
        "High availability is mandatory. "
        "The workload is sequential. "
        "Expected annual growth is 20 percent. "
        "Reliability is absolutely critical and "
        "performance is very important."
    )

    transcript: List[dict] = []

    response = orchestrator.handle_message(
        initial,
        session,
    )
    transcript.append(
        _turn_to_dict(
            1,
            initial,
            response,
        )
    )

    for turn in range(2, 40):
        if response.ready_for_final_validation:
            break

        pending = response.pending_question

        if pending is None:
            raise AssertionError(
                "Conversation is not ready but has no pending question."
            )

        answer = _answer_for_pending(
            pending
        )

        response = orchestrator.handle_message(
            answer,
            session,
        )

        transcript.append(
            _turn_to_dict(
                turn,
                answer,
                response,
            )
        )

    if not response.ready_for_final_validation:
        raise AssertionError(
            "Happy-path conversation did not reach final validation."
        )

    # Step 6.1E is an ORCHESTRATION regression, not a new statistical
    # Layer-2 intensity benchmark.
    #
    # The frozen Layer-2 model is allowed to return any supported verified
    # absolute level for the two clearly-active preference dimensions. The
    # conversational BWM phase is responsible for obtaining the final
    # numerical ranking weights when ordinal labels alone do not uniquely
    # determine Best/Worst.
    expected_exact = {
        "requested_usable_capacity_tib": 100,
        "client_count": 64,
        "access_type": "sequential",
        "ha_required": True,
        "annual_growth_percent": 20,
        "planning_horizon_years": 3,
    }

    actual_core = {
        name: session.get(name).value
        for name in expected_exact
    }

    mismatches = {
        name: {
            "expected": expected_exact[name],
            "actual": actual_core[name],
            "state": session.get(name).state.value,
        }
        for name in expected_exact
        if actual_core[name] != expected_exact[name]
    }

    if mismatches:
        raise AssertionError(
            "Happy-path exact Requirement mismatch:\n"
            + json.dumps(
                mismatches,
                indent=2,
            )
        )

    allowed_preference_levels = {
        "VERY_LOW",
        "LOW",
        "MEDIUM",
        "HIGH",
        "VERY_HIGH",
    }

    preference_snapshot = {}

    for field_name in (
        "reliability_priority",
        "performance_priority",
    ):
        record = session.get(field_name)

        preference_snapshot[field_name] = {
            "value": record.value,
            "state": record.state.value,
            "source": record.source,
        }

        if record.state != FieldState.VERIFIED:
            raise AssertionError(
                f"{field_name} must be VERIFIED in the happy path."
            )

        if record.value not in allowed_preference_levels:
            raise AssertionError(
                f"{field_name} has unsupported active level: "
                f"{record.value!r}"
            )

    for field in OPTIONAL_SKIP_FIELDS:
        if session.get(field).state != FieldState.DECLINED:
            raise AssertionError(
                f"{field} should be explicitly DECLINED."
            )

    weights_record = session.get(
        "preference_weights"
    )

    if weights_record.state != FieldState.VERIFIED:
        raise AssertionError(
            "preference_weights must be VERIFIED."
        )

    weights = dict(
        weights_record.value
    )

    if abs(sum(weights.values()) - 1.0) > 1e-8:
        raise AssertionError(
            "Final preference weights do not sum to one."
        )

    if weights["reliability"] <= weights["performance"]:
        raise AssertionError(
            "Final BWM weights must preserve the user-confirmed ordering: "
            "reliability > performance."
        )

    return {
        "status": "PASS",
        "turn_count": len(transcript),
        "transcript": transcript,
        "final_core": actual_core,
        "verified_preferences": preference_snapshot,
        "preference_weights": weights,
        "bwm_status": session.bwm_dialogue.get(
            "last_status"
        ),
        "final_state": session.conversation_state.value,
    }


def run_conflict_path() -> dict:
    """
    Real categorical correction/conflict regression.

    We intentionally begin a normal strict conversation. The second message
    contradicts the already-verified access_type. Conflict handling must take
    priority over all still-missing optional fields.
    """
    orchestrator = (
        build_frozen_production_orchestrator(
            device="cpu",
            enable_llm_fallback=False,
        )
    )
    session = orchestrator.new_session(
        "STEP6_1E_CONFLICT_PATH"
    )

    first = orchestrator.handle_message(
        (
            "We need 100 TiB for 64 clients. "
            "HA is mandatory and the access pattern is sequential."
        ),
        session,
    )

    if session.get("access_type").value != "sequential":
        raise AssertionError(
            "Initial access_type was not verified as sequential."
        )

    second = orchestrator.handle_message(
        "Actually, the access pattern should be random.",
        session,
    )

    if second.conversation_state.value != "RESOLVING_CONFLICT":
        raise AssertionError(
            "Contradictory verified access_type did not trigger conflict."
        )

    if second.pending_question is None:
        raise AssertionError(
            "Conflict did not create a clarification question."
        )

    if second.pending_question.target_field != "access_type":
        raise AssertionError(
            "Conflict clarification targeted the wrong field."
        )

    third = orchestrator.handle_message(
        "latest",
        session,
    )

    if session.get("access_type").value != "random":
        raise AssertionError(
            "Conflict choice 'latest' did not keep the new value."
        )

    if session.get("access_type").state != FieldState.VERIFIED:
        raise AssertionError(
            "Resolved access_type conflict is not VERIFIED."
        )

    if third.ready_for_final_validation:
        raise AssertionError(
            "Remaining missing fields were bypassed after conflict resolution."
        )

    if third.pending_question is None:
        raise AssertionError(
            "Strict clarification did not resume after conflict resolution."
        )

    return {
        "status": "PASS",
        "initial_state": first.conversation_state.value,
        "conflict_state": second.conversation_state.value,
        "resolved_state": third.conversation_state.value,
        "resolved_access_type": session.get(
            "access_type"
        ).value,
        "next_pending_field":
            third.pending_question.target_field,
        "ready_after_conflict_resolution":
            third.ready_for_final_validation,
    }


def main() -> None:
    happy = run_happy_path()
    conflict = run_conflict_path()

    report = {
        "step": "6.1E",
        "status": "FULL_MULTI_TURN_DIALOGUE_REGRESSION_PASS",
        "scope": (
            "real frozen production runtimes; strict clarification; "
            "conditional horizon; optional decline; BWM; conflict resolution"
        ),
        "happy_path": happy,
        "conflict_path": conflict,
        "invariants": {
            "no_ready_with_missing_fields": True,
            "conditional_horizon_resolved": True,
            "optional_decline_preserved": True,
            "bwm_before_ready": True,
            "weights_sum_to_one": True,
            "conflict_priority_over_missing": True,
            "conflict_resolution_returns_to_collection": True,
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
        "STATUS: FULL_MULTI_TURN_DIALOGUE_REGRESSION_PASS"
    )


if __name__ == "__main__":
    main()
