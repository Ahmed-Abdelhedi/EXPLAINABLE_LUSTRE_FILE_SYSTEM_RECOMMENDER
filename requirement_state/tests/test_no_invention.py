from dataclasses import dataclass
from enum import Enum

from requirement_state import RequirementStateBuilder


class S(str, Enum):
    VERIFIED = "VERIFIED"
    DECLINED = "DECLINED"
    MISSING = "MISSING"


@dataclass
class R:
    state: S
    value: object = None
    source: str | None = None
    evidence: str | None = None
    confidence: float | None = None
    message_id: str | None = None
    revision: int = 0


class Session:
    def __init__(self):
        self.data = {}

    def get(self, field):
        return self.data.setdefault(field, R(S.DECLINED))


def test_declined_optional_field_stays_null():
    session = Session()
    session.data["max_budget_usd"] = R(S.DECLINED, None)

    state = RequirementStateBuilder().from_session(session)

    assert state.max_budget_usd is None
