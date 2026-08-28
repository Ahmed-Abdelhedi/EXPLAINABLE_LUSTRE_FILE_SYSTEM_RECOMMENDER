from requirement_state import (
    FinalRequirementState,
    PreferenceWeights,
    REQUIREMENT_FIELDS,
)


def test_contract_has_18_requirement_fields():
    assert len(REQUIREMENT_FIELDS) == 18
    assert len(set(REQUIREMENT_FIELDS)) == 18


def test_preference_weights_are_separate_from_priority_labels():
    state = FinalRequirementState(
        reliability_priority="VERY_HIGH",
        performance_priority="HIGH",
        preference_weights=PreferenceWeights(
            cost=0.0,
            power=0.0,
            performance=0.25,
            reliability=0.75,
            method="LINEAR_BWM",
        ),
    )

    payload = state.to_canonical_json_dict()

    assert payload["reliability_priority"] == "VERY_HIGH"
    assert payload["performance_priority"] == "HIGH"
    assert payload["preference_weights"] == {
        "cost": 0.0,
        "power": 0.0,
        "performance": 0.25,
        "reliability": 0.75,
    }


def test_canonical_payload_contains_exact_contract_plus_weights():
    state = FinalRequirementState()
    payload = state.to_canonical_json_dict()

    assert list(payload.keys()) == [
        *REQUIREMENT_FIELDS,
        "preference_weights",
    ]
