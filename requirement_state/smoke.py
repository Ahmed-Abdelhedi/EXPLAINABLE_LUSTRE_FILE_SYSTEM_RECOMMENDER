from __future__ import annotations

import json

from .json_io import canonical_json_string
from .models import (
    FinalRequirementState,
    PreferenceWeights,
)


def main() -> None:
    state = FinalRequirementState(
        requested_usable_capacity_tib=100,
        client_count=64,
        access_type="sequential",
        ha_required=True,
        annual_growth_percent=20,
        planning_horizon_years=3,
        reliability_priority="HIGH",
        performance_priority="HIGH",
        preference_weights=PreferenceWeights(
            cost=0.0,
            power=0.0,
            performance=0.25,
            reliability=0.75,
            method="LINEAR_BWM",
            xi_star=0.0,
            consistency_status="PASS",
            source="USER_ELICITED_BWM",
        ),
        ready_for_sizing=False,
    )

    canonical = json.loads(
        canonical_json_string(state)
    )

    if canonical["preference_weights"] != {
        "cost": 0.0,
        "power": 0.0,
        "performance": 0.25,
        "reliability": 0.75,
    }:
        raise AssertionError(
            "Canonical preference_weights contract mismatch."
        )

    print(
        json.dumps(
            canonical,
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print(
        "STATUS: FINAL_REQUIREMENT_STATE_SCHEMA_PASS"
    )


if __name__ == "__main__":
    main()
