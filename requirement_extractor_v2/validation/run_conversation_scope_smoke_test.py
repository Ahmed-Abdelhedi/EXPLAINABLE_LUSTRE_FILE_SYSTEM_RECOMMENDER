from requirement_extractor_v2.conversation_scope_resolver import (
    ConversationScopeResolver,
)

from requirement_extractor_v2.models import (
    ParamName,
    ScopeIntent,
)


def show_case(
    resolver,
    title,
    text,
    *,
    previous_question_field=None,
    requested_unit=None,
    previous_question=None,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print("USER:", text)

    result = resolver.resolve(
        user_text=text,
        previous_question_field=previous_question_field,
        requested_unit=requested_unit,
        previous_question=previous_question,
    )

    print(result.to_dict())

    return result


def main():

    resolver = ConversationScopeResolver()

    # ================================================================
    # CASE 1 — normal new requirement
    # ================================================================

    result = show_case(
        resolver,
        "CASE 1 — NEW REQUIREMENT",
        "Need 500 TiB usable storage for 200 clients.",
    )

    assert (
        result.intent
        == ScopeIntent.NEW_REQUIREMENT
    )

    # ================================================================
    # CASE 2 — bare answer to previous power question
    # ================================================================

    result = show_case(
        resolver,
        "CASE 2 — ANSWER TO PREVIOUS QUESTION",
        "200",
        previous_question_field=ParamName.max_power_w,
        requested_unit="W",
        previous_question=(
            "What is the maximum power in watts?"
        ),
    )

    assert (
        result.intent
        == ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
    )

    assert (
        result.target_field
        == ParamName.max_power_w
    )

    assert result.inherited_unit == "W"

    # ================================================================
    # CASE 3 — answer already contains explicit unit
    # ================================================================

    result = show_case(
        resolver,
        "CASE 3 — ANSWER WITH EXPLICIT UNIT",
        "15 kW",
        previous_question_field=ParamName.max_power_w,
        requested_unit="W",
        previous_question=(
            "What is the maximum power in watts?"
        ),
    )

    assert (
        result.intent
        == ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
    )

    assert (
        result.target_field
        == ParamName.max_power_w
    )

    # Do not inherit W because user explicitly supplied kW.
    assert result.inherited_unit is None

    # ================================================================
    # CASE 4 — correction
    # ================================================================

    result = show_case(
        resolver,
        "CASE 4 — CORRECTION",
        "Actually, change the maximum power to 12 kW.",
    )

    assert (
        result.intent
        == ScopeIntent.CORRECTION
    )

    # ================================================================
    # CASE 5 — out of scope
    # ================================================================

    result = show_case(
        resolver,
        "CASE 5 — OUT OF SCOPE",
        "What is the weather today?",
    )

    assert (
        result.intent
        == ScopeIntent.OUT_OF_SCOPE
    )

    # ================================================================
    # CASE 6 — rich requirement must NOT be bound to previous question
    # ================================================================

    result = show_case(
        resolver,
        "CASE 6 — RICH MESSAGE WITH ACTIVE QUESTION",
        (
            "Need 500 TiB usable storage "
            "for 200 clients."
        ),
        previous_question_field=ParamName.max_power_w,
        requested_unit="W",
        previous_question=(
            "What is the maximum power?"
        ),
    )

    assert (
        result.intent
        == ScopeIntent.NEW_REQUIREMENT
    )

    # ================================================================
    # CASE 7 — HA short answer
    # ================================================================

    result = show_case(
        resolver,
        "CASE 7 — HA YES/NO ANSWER",
        "yes",
        previous_question_field=ParamName.ha_required,
        previous_question=(
            "Is high availability required?"
        ),
    )

    assert (
        result.intent
        == ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
    )

    assert (
        result.target_field
        == ParamName.ha_required
    )

    # ================================================================
    # CASE 8 — access type short answer
    # ================================================================

    result = show_case(
        resolver,
        "CASE 8 — ACCESS TYPE ANSWER",
        "mixed",
        previous_question_field=ParamName.access_type,
        previous_question=(
            "What access pattern do you expect?"
        ),
    )

    assert (
        result.intent
        == ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
    )

    assert (
        result.target_field
        == ParamName.access_type
    )

    print()
    print("=" * 70)
    print("CONVERSATION SCOPE RESOLVER: ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()