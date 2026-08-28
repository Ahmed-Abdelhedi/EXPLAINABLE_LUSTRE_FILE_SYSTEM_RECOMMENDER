from dataclasses import dataclass
from enum import Enum

from requirement_state import (
    RequirementStateBuilder,
)


class S(str, Enum):
    VERIFIED = "VERIFIED"
    DECLINED = "DECLINED"
    MISSING = "MISSING"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class R:
    state: S
    value: object = None
    source: str | None = None
    evidence: str | None = None
    confidence: float | None = None
    message_id: str | None = None
    revision: int = 0


class FakeSession:
    def __init__(self):
        self.records = {}

    def get(self, name):
        if name not in self.records:
            self.records[name] = R(S.MISSING)
        return self.records[name]


def test_builder_copies_verified_only_to_canonical_values():
    session = FakeSession()

    session.records["requested_usable_capacity_tib"] = R(
        S.VERIFIED,
        value=100,
        source="QUANTITY_VERIFIER",
        evidence="100 TiB",
    )
    session.records["client_count"] = R(
        S.DECLINED,
        value=None,
    )
    session.records["access_type"] = R(
        S.UNRESOLVED,
        value=None,
    )

    state = RequirementStateBuilder().from_session(
        session
    )

    assert state.requested_usable_capacity_tib == 100
    assert state.client_count is None
    assert state.access_type is None
    assert "access_type" in state.unresolved_fields


def test_builder_preserves_bwm_metadata():
    session = FakeSession()

    session.records["preference_weights"] = R(
        S.VERIFIED,
        value={
            "cost": 0.0,
            "power": 0.0,
            "performance": 0.25,
            "reliability": 0.75,
        },
        source="LINEAR_BWM",
    )

    state = RequirementStateBuilder().from_session(
        session,
        bwm_metadata={
            "method": "LINEAR_BWM",
            "xi_star": 0.0,
            "consistency_status": "PASS",
            "source": "USER_ELICITED_BWM",
        },
    )

    assert state.preference_weights is not None
    assert state.preference_weights.method == "LINEAR_BWM"
    assert state.preference_weights.xi_star == 0.0
    assert (
        state.preference_weights.consistency_status
        == "PASS"
    )
    assert (
        state.preference_weights.source
        == "USER_ELICITED_BWM"
    )
