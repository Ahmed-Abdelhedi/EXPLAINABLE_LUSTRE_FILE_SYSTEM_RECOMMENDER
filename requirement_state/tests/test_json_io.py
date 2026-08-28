import json

from requirement_state import (
    FinalRequirementState,
    PreferenceWeights,
    canonical_json_string,
)


def test_canonical_json_is_strict_json():
    state = FinalRequirementState(
        requested_usable_capacity_tib=100,
        client_count=64,
        access_type="sequential",
        ha_required=True,
        preference_weights=PreferenceWeights(
            cost=0.0,
            power=0.0,
            performance=0.25,
            reliability=0.75,
        ),
    )

    raw = canonical_json_string(state)
    payload = json.loads(raw)

    assert payload["requested_usable_capacity_tib"] == 100
    assert payload["ha_required"] is True
    assert payload["preference_weights"]["reliability"] == 0.75
