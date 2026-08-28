from dataclasses import dataclass
from enum import Enum

from requirement_state import RequirementStateFinalizer


class S(str, Enum):
    VERIFIED = "VERIFIED"
    DECLINED = "DECLINED"


@dataclass
class R:
    state: S
    value: object = None
    source: str | None = None
    evidence: str | None = None
    confidence: float | None = None
    message_id: str | None = None
    revision: int = 1


class Session:
    def __init__(self):
        self.data = {}

        verified = {
            "requested_usable_capacity_tib": 100,
            "client_count": 64,
            "access_type": "mixed",
            "ha_required": True,
            "annual_growth_percent": 0,
            "reliability_priority": "HIGH",
            "preference_weights": {
                "cost": 0.0,
                "power": 0.0,
                "performance": 0.0,
                "reliability": 1.0,
            },
        }

        for field, value in verified.items():
            self.data[field] = R(S.VERIFIED, value, "TEST")

        for field in (
            "average_file_size_gb",
            "max_file_size_gb",
            "total_file_count",
            "read_write_ratio",
            "target_read_gbps",
            "target_write_gbps",
            "max_budget_usd",
            "max_power_w",
            "planning_horizon_years",
            "cost_priority",
            "power_priority",
            "performance_priority",
        ):
            self.data[field] = R(S.DECLINED, None, "USER_DECLINED")

        self.bwm_dialogue = {
            "last_result": {
                "method": "SINGLE_ACTIVE",
                "xi_star": 0.0,
                "ordinal_consistency": "PASS",
                "source": "CONFIRMED_SINGLE_ACTIVE_CRITERION",
            }
        }

    def get(self, field):
        return self.data[field]


def test_valid_complete_session_is_ready_for_sizing():
    state = RequirementStateFinalizer().from_session(Session())

    assert state.validation_issues == []
    assert state.ready_for_sizing is True
