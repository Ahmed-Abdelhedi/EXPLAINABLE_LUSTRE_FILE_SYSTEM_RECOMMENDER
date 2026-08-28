from requirement_state import (
    DeterministicRequirementValidator,
    FinalRequirementState,
    PreferenceWeights,
)


def _codes(state):
    DeterministicRequirementValidator().validate(state)
    return {item["code"] for item in state.validation_issues}


def test_inactive_dimension_must_have_zero_weight():
    state = FinalRequirementState(
        reliability_priority="HIGH",
        performance_priority="HIGH",
        preference_weights=PreferenceWeights(
            cost=0.1,
            power=0.0,
            performance=0.2,
            reliability=0.7,
            consistency_status="PASS",
        ),
    )
    assert "INACTIVE_PREFERENCE_HAS_NONZERO_WEIGHT" in _codes(state)


def test_weights_must_sum_to_one():
    state = FinalRequirementState(
        reliability_priority="HIGH",
        performance_priority="HIGH",
        preference_weights=PreferenceWeights(
            cost=0.0,
            power=0.0,
            performance=0.2,
            reliability=0.7,
            consistency_status="PASS",
        ),
    )
    assert "PREFERENCE_WEIGHTS_NOT_NORMALIZED" in _codes(state)


def test_active_preferences_require_weights():
    state = FinalRequirementState(
        reliability_priority="HIGH",
        preference_weights=None,
    )
    assert "ACTIVE_PREFERENCES_REQUIRE_WEIGHTS" in _codes(state)
