from __future__ import annotations

from requirement_extractor_v2.deterministic_verifier import (
    DeterministicVerifier,
)
from requirement_extractor_v2.models import (
    ParamName,
    ScopeIntent,
    VerificationStatus,
)
from requirement_extractor_v2.quantity_scanner import (
    QuantityScanner,
)
from requirement_extractor_v2.selective_cascade import (
    SelectiveCascadeResult,
)
from requirement_extractor_v2.verified_pipeline import (
    VerifiedRequirementPipeline,
)


class MinimalCascade:
    """
    Lightweight cascade used only to isolate scope integration.

    Contextual-answer and out-of-scope cases must never call resolve().
    """

    def __init__(self):
        self.scanner = QuantityScanner()
        self.resolve_call_count = 0

    def resolve(
        self,
        text: str,
        previous_question=None,
    ):
        self.resolve_call_count += 1

        return SelectiveCascadeResult(
            text=text,
            quantities=[],
            links=[],
            unresolved_quantity_ids=[],
            traces={},
        )

    def info(self):
        return {
            "test_double": True,
            "resolve_call_count":
                self.resolve_call_count,
        }


def assert_verified(
    result,
    *,
    field,
    value,
    unit,
):
    assert (
        result.scope is not None
    )

    assert (
        result.scope.intent
        == ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
    )

    assert len(
        result.decisions
    ) >= 1

    verified = [
        decision
        for decision
        in result.decisions
        if decision.status
        == VerificationStatus.VERIFIED
    ]

    assert verified, (
        result.to_dict()
    )

    decision = verified[0]

    assert (
        decision.field == field
    ), result.to_dict()

    assert (
        abs(
            float(decision.value)
            - float(value)
        )
        < 1e-9
    ), result.to_dict()

    assert (
        decision.unit == unit
    ), result.to_dict()

    trace = (
        result.cascade.traces[
            decision.quantity_id
        ]
    )

    assert (
        trace.final_resolver
        == "conversation_scope"
    )

    assert (
        trace.semantic_attempted
        is False
    )

    assert (
        trace.llm_attempted
        is False
    )


def main():

    cascade = MinimalCascade()

    pipeline = (
        VerifiedRequirementPipeline(
            cascade=cascade,
            verifier=(
                DeterministicVerifier()
            ),
        )
    )

    # ================================================================
    # 1. Bare client count
    # ================================================================

    result = pipeline.process(
        "320",
        previous_question=(
            "How many clients will access Lustre?"
        ),
        previous_question_field=(
            ParamName.client_count
        ),
    )

    assert_verified(
        result,
        field=ParamName.client_count,
        value=320,
        unit=None,
    )

    print(
        "[PASS] contextual client_count 320"
    )

    # ================================================================
    # 2. Bare total file count
    # ================================================================

    result = pipeline.process(
        "9000000",
        previous_question=(
            "How many files will be stored?"
        ),
        previous_question_field=(
            ParamName.total_file_count
        ),
    )

    assert_verified(
        result,
        field=ParamName.total_file_count,
        value=9000000,
        unit=None,
    )

    print(
        "[PASS] contextual total_file_count 9000000"
    )

    # ================================================================
    # 3. Unitless power answer inherits W
    # ================================================================

    result = pipeline.process(
        "200",
        previous_question=(
            "What is the maximum power in watts?"
        ),
        previous_question_field=(
            ParamName.max_power_w
        ),
        requested_unit="W",
    )

    assert_verified(
        result,
        field=ParamName.max_power_w,
        value=200,
        unit="W",
    )

    print(
        "[PASS] contextual max_power_w 200 -> 200 W"
    )

    # ================================================================
    # 4. Explicit kW must override inherited W and normalize
    # ================================================================

    result = pipeline.process(
        "15 kW",
        previous_question=(
            "What is the maximum power in watts?"
        ),
        previous_question_field=(
            ParamName.max_power_w
        ),
        requested_unit="W",
    )

    assert_verified(
        result,
        field=ParamName.max_power_w,
        value=15000,
        unit="W",
    )

    print(
        "[PASS] contextual max_power_w 15 kW -> 15000 W"
    )

    # ================================================================
    # 5. Maximum file size
    # ================================================================

    result = pipeline.process(
        "80 GB",
        previous_question=(
            "What is the maximum file size?"
        ),
        previous_question_field=(
            ParamName.max_file_size_gb
        ),
    )

    assert_verified(
        result,
        field=ParamName.max_file_size_gb,
        value=80,
        unit="GB",
    )

    print(
        "[PASS] contextual max_file_size_gb 80 GB"
    )

    # ================================================================
    # 6. Write throughput
    # ================================================================

    result = pipeline.process(
        "42 GB/s",
        previous_question=(
            "What write throughput do you need?"
        ),
        previous_question_field=(
            ParamName.target_write_gbps
        ),
    )

    assert_verified(
        result,
        field=ParamName.target_write_gbps,
        value=42,
        unit="GB/s",
    )

    print(
        "[PASS] contextual target_write_gbps 42 GB/s"
    )

    # ================================================================
    # 7. Unitless capacity inherits TiB
    # ================================================================

    result = pipeline.process(
        "500",
        previous_question=(
            "What usable capacity do you need?"
        ),
        previous_question_field=(
            ParamName.requested_usable_capacity_tib
        ),
        requested_unit="TiB",
    )

    assert_verified(
        result,
        field=(
            ParamName.requested_usable_capacity_tib
        ),
        value=500,
        unit="TiB",
    )

    print(
        "[PASS] contextual capacity 500 -> 500 TiB"
    )

    # ================================================================
    # 8. Ratio 70/30
    # ================================================================

    result = pipeline.process(
        "70/30",
        previous_question=(
            "What is the read/write ratio?"
        ),
        previous_question_field=(
            ParamName.read_write_ratio
        ),
        requested_unit="%",
    )

    assert (
        result.scope is not None
        and
        result.scope.intent
        == ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
    )

    ratio_verified = [
        d
        for d in result.decisions
        if (
            d.status
            == VerificationStatus.VERIFIED
            and
            d.field
            == ParamName.read_write_ratio
        )
    ]

    assert len(
        ratio_verified
    ) == 2, result.to_dict()

    ratio_values = sorted(
        float(d.value)
        for d in ratio_verified
    )

    assert ratio_values == [
        30.0,
        70.0,
    ], result.to_dict()

    assert all(
        d.unit == "%"
        for d in ratio_verified
    )

    assert all(
        result.cascade.traces[
            d.quantity_id
        ].llm_attempted
        is False
        for d in ratio_verified
    )

    print(
        "[PASS] contextual read_write_ratio 70/30"
    )

    # ================================================================
    # 9. Wrong explicit physical unit must NOT be silently inherited
    # ================================================================

    result = pipeline.process(
        "15 GB",
        previous_question=(
            "What is the maximum power?"
        ),
        previous_question_field=(
            ParamName.max_power_w
        ),
        requested_unit="W",
    )

    assert (
        result.scope is not None
        and
        result.scope.intent
        == ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
    )

    assert result.invalid, (
        result.to_dict()
    )

    assert not result.verified, (
        result.to_dict()
    )

    print(
        "[PASS] wrong explicit unit rejected"
    )

    # ================================================================
    # 10. OUT_OF_SCOPE must stop before cascade
    # ================================================================

    before = (
        cascade.resolve_call_count
    )

    result = pipeline.process(
        "What is the weather today?"
    )

    after = (
        cascade.resolve_call_count
    )

    assert (
        result.scope is not None
        and
        result.scope.intent
        == ScopeIntent.OUT_OF_SCOPE
    )

    assert before == after

    assert (
        result.decisions == []
    )

    assert (
        result.cascade.quantities == []
    )

    print(
        "[PASS] out-of-scope stops extraction"
    )

    # ================================================================
    # 11. Rich message is NOT force-bound to active power question
    # ================================================================

    before = (
        cascade.resolve_call_count
    )

    result = pipeline.process(
        "Need 500 TiB usable storage for 200 clients.",
        previous_question=(
            "What is the maximum power?"
        ),
        previous_question_field=(
            ParamName.max_power_w
        ),
        requested_unit="W",
    )

    after = (
        cascade.resolve_call_count
    )

    assert (
        result.scope is not None
        and
        result.scope.intent
        == ScopeIntent.NEW_REQUIREMENT
    )

    assert (
        after == before + 1
    )

    print(
        "[PASS] rich message continues through normal cascade"
    )

    # ================================================================
    # 12. Non-quantitative short answer is scoped but not converted here
    # ================================================================

    result = pipeline.process(
        "yes",
        previous_question=(
            "Is high availability required?"
        ),
        previous_question_field=(
            ParamName.ha_required
        ),
    )

    assert (
        result.scope is not None
        and
        result.scope.intent
        == ScopeIntent.ANSWER_TO_PREVIOUS_QUESTION
        and
        result.scope.target_field
        == ParamName.ha_required
    )

    assert (
        result.decisions == []
    )

    print(
        "[PASS] HA answer scoped; quantitative pipeline does not invent value"
    )

    print()
    print("=" * 76)
    print(
        "VERIFIED PIPELINE + CONVERSATION SCOPE: ALL TESTS PASSED"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()