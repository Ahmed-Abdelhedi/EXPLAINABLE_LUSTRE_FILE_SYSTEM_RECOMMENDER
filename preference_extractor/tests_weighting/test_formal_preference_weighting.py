from __future__ import annotations

import math

import pytest

from preference_extractor.layer2.labels import (
    PreferenceDimension,
    PreferenceLevel,
    ResolutionSource,
    ResolutionStatus,
)
from preference_extractor.layer2.schemas import (
    DimensionPreferenceResult,
    PreferenceExtractionResult,
    PreferenceRelation,
)
from preference_extractor.weighting.best_worst_selector import (
    select_best_worst,
)
from preference_extractor.weighting.active_set import build_active_set
from preference_extractor.weighting.bwm_solver import solve_linear_bwm
from preference_extractor.weighting.elicitation import (
    b2o_id,
    build_bwm_vectors,
    build_questions,
    o2w_id,
)
from preference_extractor.weighting.models import WeightingStatus
from preference_extractor.weighting.runtime import (
    FormalPreferenceWeightingLayer,
)


def make_extraction(
    *,
    cost=("NO_SIGNAL", None),
    power=("NO_SIGNAL", None),
    performance=("NO_SIGNAL", None),
    reliability=("NO_SIGNAL", None),
    relations=None,
):
    raw = {
        PreferenceDimension.COST: cost,
        PreferenceDimension.POWER: power,
        PreferenceDimension.PERFORMANCE: performance,
        PreferenceDimension.RELIABILITY: reliability,
    }

    dimensions = {}

    for dimension, (status_name, level_name) in raw.items():
        status = ResolutionStatus(status_name)
        level = (
            PreferenceLevel(level_name)
            if level_name is not None
            else None
        )

        dimensions[dimension] = DimensionPreferenceResult(
            dimension=dimension,
            status=status,
            source=ResolutionSource.TRANSFORMER,
            level=level,
        )

    return PreferenceExtractionResult(
        text="test",
        dimensions=dimensions,
        relations=list(relations or []),
    )


def example_three_active():
    return make_extraction(
        cost=("RESOLVED", "LOW"),
        power=("NO_SIGNAL", None),
        performance=("RESOLVED", "HIGH"),
        reliability=("RESOLVED", "VERY_HIGH"),
    )


def example_answers():
    return {
        b2o_id(
            PreferenceDimension.RELIABILITY,
            PreferenceDimension.COST,
        ): 5,
        b2o_id(
            PreferenceDimension.RELIABILITY,
            PreferenceDimension.PERFORMANCE,
        ): 3,
        o2w_id(
            PreferenceDimension.PERFORMANCE,
            PreferenceDimension.COST,
        ): 2,
    }


def test_no_active_does_not_invent_equal_weights():
    result = FormalPreferenceWeightingLayer().run(
        make_extraction()
    )

    assert result.status == WeightingStatus.NO_ACTIVE_PREFERENCE
    assert sum(result.all_four_weights().values()) == 0.0


def test_unresolved_blocks_weight_generation():
    result = FormalPreferenceWeightingLayer().run(
        make_extraction(
            performance=("UNRESOLVED", None)
        )
    )

    assert result.status == WeightingStatus.BLOCKED_UNRESOLVED
    assert sum(result.all_four_weights().values()) == 0.0


def test_single_high_becomes_one():
    result = FormalPreferenceWeightingLayer().run(
        make_extraction(
            reliability=("RESOLVED", "HIGH")
        )
    )

    assert result.status == WeightingStatus.WEIGHTS_READY
    assert result.all_four_weights()["reliability"] == 1.0
    assert sum(result.all_four_weights().values()) == 1.0


def test_single_low_requires_confirmation():
    extraction = make_extraction(
        cost=("RESOLVED", "LOW")
    )
    layer = FormalPreferenceWeightingLayer()

    blocked = layer.run(extraction)
    assert (
        blocked.status
        == WeightingStatus.NEEDS_SINGLE_CRITERION_CONFIRMATION
    )

    confirmed = layer.run(
        extraction,
        single_active_confirmed=True,
    )
    assert confirmed.status == WeightingStatus.WEIGHTS_READY
    assert confirmed.all_four_weights()["cost"] == 1.0


def test_best_worst_selected_from_ordinal_labels():
    active = build_active_set(
        example_three_active()
    )
    selection = select_best_worst(active)

    assert selection.status == "READY"
    assert selection.best == PreferenceDimension.RELIABILITY
    assert selection.worst == PreferenceDimension.COST


def test_tie_requires_best_worst_clarification():
    extraction = make_extraction(
        cost=("RESOLVED", "HIGH"),
        performance=("RESOLVED", "HIGH"),
        reliability=("RESOLVED", "LOW"),
    )

    result = FormalPreferenceWeightingLayer().run(extraction)

    assert result.status == WeightingStatus.NEEDS_BEST_WORST


def test_relative_relation_can_disambiguate_tie():
    relation = PreferenceRelation(
        higher=PreferenceDimension.PERFORMANCE,
        lower=PreferenceDimension.COST,
        evidence="performance over cost",
    )

    extraction = make_extraction(
        cost=("RELATIVE_ONLY", None),
        performance=("RELATIVE_ONLY", None),
        relations=[relation],
    )

    result = FormalPreferenceWeightingLayer().run(extraction)

    assert result.status == WeightingStatus.NEEDS_BWM_COMPARISONS
    assert result.best == PreferenceDimension.PERFORMANCE
    assert result.worst == PreferenceDimension.COST


def test_cycle_is_rejected():
    relations = [
        PreferenceRelation(
            higher=PreferenceDimension.COST,
            lower=PreferenceDimension.PERFORMANCE,
            evidence="cost > perf",
        ),
        PreferenceRelation(
            higher=PreferenceDimension.PERFORMANCE,
            lower=PreferenceDimension.COST,
            evidence="perf > cost",
        ),
    ]

    extraction = make_extraction(
        cost=("RELATIVE_ONLY", None),
        performance=("RELATIVE_ONLY", None),
        relations=relations,
    )

    result = FormalPreferenceWeightingLayer().run(extraction)

    assert result.status == WeightingStatus.INCONSISTENT_PREFERENCES
    assert "PREFERENCE_ORDER_CYCLE" in result.violations


def test_bwm_question_count_is_2n_minus_3():
    questions = build_questions(
        active_dimensions=[
            PreferenceDimension.COST,
            PreferenceDimension.PERFORMANCE,
            PreferenceDimension.RELIABILITY,
        ],
        best=PreferenceDimension.RELIABILITY,
        worst=PreferenceDimension.COST,
    )

    assert len(questions) == 3


def test_incomplete_answers_request_only_missing_comparisons():
    layer = FormalPreferenceWeightingLayer()
    answers = {
        b2o_id(
            PreferenceDimension.RELIABILITY,
            PreferenceDimension.COST,
        ): 5,
    }

    result = layer.run(
        example_three_active(),
        bwm_answers=answers,
    )

    assert result.status == WeightingStatus.NEEDS_BWM_COMPARISONS
    assert len(result.missing_questions) == 2


def test_invalid_bwm_value_is_rejected():
    answers = example_answers()
    key = next(iter(answers))
    answers[key] = 10

    result = FormalPreferenceWeightingLayer().run(
        example_three_active(),
        bwm_answers=answers,
    )

    assert result.status == WeightingStatus.INVALID_BWM_JUDGMENTS


def test_linear_bwm_reference_example_exact():
    active = [
        PreferenceDimension.RELIABILITY,
        PreferenceDimension.PERFORMANCE,
        PreferenceDimension.COST,
    ]

    answers = example_answers()

    best_to_others, others_to_worst = build_bwm_vectors(
        active_dimensions=active,
        best=PreferenceDimension.RELIABILITY,
        worst=PreferenceDimension.COST,
        answers=answers,
    )

    solution = solve_linear_bwm(
        active_dimensions=active,
        best=PreferenceDimension.RELIABILITY,
        worst=PreferenceDimension.COST,
        best_to_others=best_to_others,
        others_to_worst=others_to_worst,
    )

    assert math.isclose(
        solution.weights[PreferenceDimension.RELIABILITY],
        0.650,
        abs_tol=1e-8,
    )
    assert math.isclose(
        solution.weights[PreferenceDimension.PERFORMANCE],
        0.225,
        abs_tol=1e-8,
    )
    assert math.isclose(
        solution.weights[PreferenceDimension.COST],
        0.125,
        abs_tol=1e-8,
    )
    assert math.isclose(
        solution.xi_star,
        0.025,
        abs_tol=1e-8,
    )


def test_full_runtime_reference_example():
    result = FormalPreferenceWeightingLayer().run(
        example_three_active(),
        bwm_answers=example_answers(),
    )

    assert result.status == WeightingStatus.WEIGHTS_READY
    weights = result.all_four_weights()

    assert math.isclose(weights["reliability"], 0.650, abs_tol=1e-8)
    assert math.isclose(weights["performance"], 0.225, abs_tol=1e-8)
    assert math.isclose(weights["cost"], 0.125, abs_tol=1e-8)
    assert weights["power"] == 0.0
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-10)
    assert result.consistency.status.value == "PASS"


def test_ordinal_contradiction_is_not_silently_accepted():
    # Layer 2 says reliability > performance > cost, but user BWM judgments
    # can be made contradictory. The post-solver gate must reject the vector.
    answers = {
        b2o_id(
            PreferenceDimension.RELIABILITY,
            PreferenceDimension.COST,
        ): 1,
        b2o_id(
            PreferenceDimension.RELIABILITY,
            PreferenceDimension.PERFORMANCE,
        ): 1,
        o2w_id(
            PreferenceDimension.PERFORMANCE,
            PreferenceDimension.COST,
        ): 1,
    }

    result = FormalPreferenceWeightingLayer().run(
        example_three_active(),
        bwm_answers=answers,
    )

    assert result.status == WeightingStatus.INCONSISTENT_PREFERENCES
    assert any(
        violation.startswith("ORDINAL_ORDER_VIOLATION:")
        for violation in result.violations
    )


def test_no_default_xi_threshold_is_invented():
    layer = FormalPreferenceWeightingLayer(max_xi=None)
    result = layer.run(
        example_three_active(),
        bwm_answers=example_answers(),
    )

    assert result.status == WeightingStatus.WEIGHTS_READY
    assert result.consistency.deviation_threshold is None


def test_configured_xi_threshold_can_gate_solution():
    layer = FormalPreferenceWeightingLayer(max_xi=0.01)
    result = layer.run(
        example_three_active(),
        bwm_answers=example_answers(),
    )

    assert result.status == WeightingStatus.INCONSISTENT_PREFERENCES
    assert any(
        violation.startswith(
            "BWM_DEVIATION_ABOVE_CONFIGURED_THRESHOLD:"
        )
        for violation in result.violations
    )
