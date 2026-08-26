from preference_extractor.evaluation.layer2_hybrid_guard import (
    GUARD_VERSION,
    Layer2DeterministicSemanticGuard,
)


GUARD = (
    Layer2DeterministicSemanticGuard()
)


def resolve(
    text,
    dimension,
):
    result = GUARD.resolve_dimension(
        text=text,
        dimension=dimension,
    )

    assert result is not None

    return result


def test_guard_version():
    assert GUARD_VERSION


def test_very_low_not_no_signal():
    result = resolve(
        "High availability can be largely ignored.",
        "reliability",
    )

    assert result.status == "RESOLVED"
    assert result.level == "VERY_LOW"


def test_low_anchor():
    result = resolve(
        "Keeping procurement cost down is weakly prioritized.",
        "cost",
    )

    assert result.level == "LOW"


def test_medium_anchor():
    result = resolve(
        "Keeping wattage modest matters, though it is not dominant.",
        "power",
    )

    assert result.level == "MEDIUM"


def test_high_anchor_technical_low_is_not_low_priority():
    result = resolve(
        "Keeping wattage low carries strong importance.",
        "power",
    )

    assert result.level == "HIGH"


def test_very_high_anchor():
    result = resolve(
        "Nothing matters more than continuity of service.",
        "reliability",
    )

    assert result.level == "VERY_HIGH"


def test_pure_comparison_is_relative_only():
    result = resolve(
        "Performance is more important than cost.",
        "performance",
    )

    assert result.status == "RELATIVE_ONLY"
    assert result.level is None


def test_lower_side_of_comparison_is_relative_only():
    result = resolve(
        "Performance is more important than cost.",
        "cost",
    )

    assert result.status == "RELATIVE_ONLY"


def test_directive_comparison_is_relative_only():
    result = resolve(
        "Can you prioritize performance over cost for our design?",
        "cost",
    )

    assert result.status == "RELATIVE_ONLY"


def test_chain_order_is_relative_only():
    text = (
        "Preference order: throughput first, "
        "then electricity consumption, then fault tolerance."
    )

    for dimension in (
        "performance",
        "power",
        "reliability",
    ):
        result = resolve(
            text,
            dimension,
        )

        assert result.status == "RELATIVE_ONLY"


def test_hard_negative_throughput_is_no_signal():
    result = resolve(
        (
            "The API documentation shows a priority field beside "
            "performance metrics; the real requirement here is only "
            "83 GB/s read throughput and 119 GB/s write throughput."
        ),
        "performance",
    )

    assert result.status == "NO_SIGNAL"
    assert result.level is None


def test_absolute_wins_for_same_dimension_over_comparison():
    result = resolve(
        (
            "Reliability carries strong importance; "
            "reliability outranks cost."
        ),
        "reliability",
    )

    assert result.status == "RESOLVED"
    assert result.level == "HIGH"


def test_comparison_only_other_dimension_stays_relative():
    result = resolve(
        (
            "Reliability carries strong importance; "
            "reliability outranks cost."
        ),
        "cost",
    )

    assert result.status == "RELATIVE_ONLY"


def test_latest_correction_wins():
    result = resolve(
        (
            "Initially, reducing cooling and power load receives "
            "a low priority. Correction: keeping wattage modest "
            "has a meaningful but balanced role."
        ),
        "power",
    )

    assert result.level == "MEDIUM"


def test_unknown_wording_abstains():
    result = GUARD.resolve_dimension(
        text=(
            "We will discuss storage economics later."
        ),
        dimension="cost",
    )

    assert result is None
