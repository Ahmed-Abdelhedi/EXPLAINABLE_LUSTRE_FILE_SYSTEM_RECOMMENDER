from __future__ import annotations

from requirement_extractor_v2.conversation_scope_resolver import (
    ConversationScopeResolver,
)

from requirement_extractor_v2.models import (
    ParamName,
    ScopeIntent,
)


def check(
    resolver,
    text,
    expected,
    **kwargs,
):
    result = resolver.resolve(
        text,
        **kwargs,
    )

    assert result.intent == expected, (
        f"\nTEXT: {text!r}\n"
        f"EXPECTED: {expected.value}\n"
        f"GOT: {result.intent.value}\n"
        f"RESULT: {result.to_dict()}"
    )

    print(
        f"[PASS] {expected.value:<30} | {text}"
    )


def main():

    resolver = ConversationScopeResolver()

    # ================================================================
    # 1. Previously false CORRECTION
    # ================================================================

    check(
        resolver,
        "Set it to 50.",
        ScopeIntent.NEW_REQUIREMENT,
    )

    # ================================================================
    # 2. Real English correction remains correction
    # ================================================================

    check(
        resolver,
        "Actually, change the maximum power to 800 W.",
        ScopeIntent.CORRECTION,
    )

    check(
        resolver,
        "Use 800 W instead.",
        ScopeIntent.CORRECTION,
    )

    # ================================================================
    # 3. Real French correction remains correction
    # ================================================================

    check(
        resolver,
        "Finalement, mets la puissance maximale à 800 W.",
        ScopeIntent.CORRECTION,
    )

    # ================================================================
    # 4. Movie recommendation — English
    # ================================================================

    check(
        resolver,
        "Can you recommend a movie?",
        ScopeIntent.OUT_OF_SCOPE,
    )

    check(
        resolver,
        "Give me a movie recommendation.",
        ScopeIntent.OUT_OF_SCOPE,
    )

    # ================================================================
    # 5. Movie recommendation — French
    # ================================================================

    check(
        resolver,
        "Peux-tu recommander un film ?",
        ScopeIntent.OUT_OF_SCOPE,
    )

    # ================================================================
    # 6. Football score — both lexical orders
    # ================================================================

    check(
        resolver,
        "What is the football match score?",
        ScopeIntent.OUT_OF_SCOPE,
    )

    check(
        resolver,
        "What is the score of the football match?",
        ScopeIntent.OUT_OF_SCOPE,
    )

    check(
        resolver,
        "Quel est le score du match de football ?",
        ScopeIntent.OUT_OF_SCOPE,
    )

    # ================================================================
    # 7. Normal HPC requirement must remain in scope
    # ================================================================

    check(
        resolver,
        "Maximum power is 800 W.",
        ScopeIntent.NEW_REQUIREMENT,
    )

    check(
        resolver,
        "Need 500 TiB usable storage for 200 clients.",
        ScopeIntent.NEW_REQUIREMENT,
    )

    # ================================================================
    # 8. Contextual numeric answer
    # ================================================================

    result = resolver.resolve(
        "200",
        previous_question_field=
            ParamName.max_power_w,
        requested_unit="W",
        previous_question=
            "What is the maximum power in watts?",
    )

    assert (
        result.intent
        ==
        ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
    )

    assert (
        result.target_field
        ==
        ParamName.max_power_w
    )

    assert result.inherited_unit == "W"

    print(
        "[PASS] ANSWER_TO_PREVIOUS_QUESTION | 200"
    )

    # ================================================================
    # 9. Rich message while clarification is active
    # ================================================================

    check(
        resolver,
        "We need 500 TiB usable storage for 200 clients.",
        ScopeIntent.NEW_REQUIREMENT,
        previous_question_field=
            ParamName.max_power_w,
        requested_unit="W",
    )

    print()
    print(
        "ConversationScopeResolver regression: "
        "ALL TESTS PASSED"
    )


if __name__ == "__main__":
    main()