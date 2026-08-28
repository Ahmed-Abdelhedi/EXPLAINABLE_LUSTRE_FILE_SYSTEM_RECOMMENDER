from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from .finalizer import RequirementStateFinalizer


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


class FakeSession:
    def __init__(self) -> None:
        self.records = {}

        verified = {
            "requested_usable_capacity_tib": 100,
            "client_count": 64,
            "access_type": "sequential",
            "ha_required": True,
            "annual_growth_percent": 20,
            "planning_horizon_years": 3,
            "reliability_priority": "HIGH",
            "performance_priority": "HIGH",
            "preference_weights": {
                "cost": 0.0,
                "power": 0.0,
                "performance": 0.25,
                "reliability": 0.75,
            },
        }

        for field_name, value in verified.items():
            self.records[field_name] = R(
                S.VERIFIED,
                value=value,
                source=(
                    "LINEAR_BWM"
                    if field_name == "preference_weights"
                    else "SMOKE_VERIFIED"
                ),
            )

        for field_name in (
            "average_file_size_gb",
            "max_file_size_gb",
            "total_file_count",
            "read_write_ratio",
            "target_read_gbps",
            "target_write_gbps",
            "max_budget_usd",
            "max_power_w",
            "cost_priority",
            "power_priority",
        ):
            self.records[field_name] = R(
                S.DECLINED,
                source="USER_DECLINED",
            )

        self.bwm_dialogue = {
            "last_result": {
                "status": "WEIGHTS_READY",
                "method": "LINEAR_BWM",
                "xi_star": 0.0,
                "ordinal_consistency": "PASS",
                "source": "USER_ELICITED_BWM",
            }
        }

    def get(self, name):
        return self.records[name]


def main() -> None:
    state = RequirementStateFinalizer().from_session(
        FakeSession()
    )

    if not state.ready_for_sizing:
        raise AssertionError(
            "Valid completed Requirement State was not accepted."
        )

    if state.validation_issues:
        raise AssertionError(
            "Valid state produced validation issues."
        )

    print(
        json.dumps(
            state.to_dict(include_traceability=False),
            indent=2,
            ensure_ascii=False,
        )
    )
    print()
    print("STATUS: DETERMINISTIC_FINAL_VALIDATION_PASS")


if __name__ == "__main__":
    main()
