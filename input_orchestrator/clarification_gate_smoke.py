from __future__ import annotations

import json

from .production_wiring import (
    build_frozen_production_orchestrator,
)


def main() -> None:
    orchestrator = (
        build_frozen_production_orchestrator(
            device="cpu",
            enable_llm_fallback=False,
        )
    )

    session = orchestrator.new_session(
        "STEP6_1C_CLARIFICATION_GATE"
    )

    first = orchestrator.handle_message(
        (
            "We need 100 TiB of usable capacity for 64 clients. "
            "High availability is mandatory. "
            "The workload is sequential. "
            "Reliability is absolutely critical and "
            "performance is very important."
        ),
        session,
    )

    if first.ready_for_final_validation:
        raise AssertionError(
            "Missing fields were incorrectly allowed to reach validation."
        )

    if first.pending_question is None:
        raise AssertionError(
            "A missing field did not trigger clarification."
        )

    first_target = first.pending_question.target_field

    # Explicitly decline one optional field. This counts as answered and must
    # advance to another missing field, never to validation.
    second = orchestrator.handle_message(
        "skip",
        session,
    )

    if second.ready_for_final_validation:
        raise AssertionError(
            "Remaining missing fields were incorrectly bypassed."
        )

    if second.pending_question is None:
        raise AssertionError(
            "Remaining missing fields did not trigger clarification."
        )

    if second.pending_question.target_field == first_target:
        raise AssertionError(
            "An optional explicitly-declined field was asked again."
        )

    report = {
        "step": "6.1C",
        "status": "STRICT_CLARIFICATION_GATE_PASS",
        "first_turn": first.to_dict(),
        "second_turn": second.to_dict(),
        "declined_optional_field": first_target,
        "next_missing_field":
            second.pending_question.target_field,
        "invariant": (
            "READY_FOR_FINAL_VALIDATION is forbidden while a raw "
            "Requirement field remains missing/unresolved."
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
        "STATUS: STRICT_CLARIFICATION_GATE_PASS"
    )


if __name__ == "__main__":
    main()
