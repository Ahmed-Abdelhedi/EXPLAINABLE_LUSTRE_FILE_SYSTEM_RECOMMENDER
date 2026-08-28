from dataclasses import dataclass
from enum import Enum

import pytest

from requirement_state import (
    OrchestratorRequirementBridge,
)


class Conversation(str, Enum):
    WAITING = "WAITING_FOR_ANSWER"
    READY = "READY_FOR_FINAL_VALIDATION"


class FieldState(str, Enum):
    VERIFIED = "VERIFIED"
    DECLINED = "DECLINED"


@dataclass
class Record:
    state: FieldState
    value: object = None
    source: str | None = None
    evidence: str | None = None
    confidence: float | None = None
    message_id: str | None = None
    revision: int = 1


class FakeSession:
    def __init__(self, ready=True):
        self.conversation_state = (
            Conversation.READY
            if ready
            else Conversation.WAITING
        )
        self.pending_question = None
        self.records = {}

        verified = {
            "requested_usable_capacity_tib": 100,
            "client_count": 64,
            "access_type": "sequential",
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
            self.records[field] = Record(
                FieldState.VERIFIED,
                value,
                "TEST",
            )

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
            self.records[field] = Record(
                FieldState.DECLINED,
                None,
                "USER_DECLINED",
            )

        self.bwm_dialogue = {
            "last_result": {
                "method": "SINGLE_ACTIVE",
                "xi_star": 0.0,
                "ordinal_consistency": "PASS",
                "source":
                    "CONFIRMED_SINGLE_ACTIVE_CRITERION",
            }
        }

    def get(self, field):
        return self.records[field]


def test_bridge_rejects_session_before_orchestrator_ready():
    bridge = OrchestratorRequirementBridge()

    with pytest.raises(
        RuntimeError,
        match="READY_FOR_FINAL_VALIDATION",
    ):
        bridge.export_ready_session(
            FakeSession(ready=False)
        )


def test_bridge_exports_only_valid_ready_state():
    output = (
        OrchestratorRequirementBridge()
        .export_ready_session(
            FakeSession(ready=True)
        )
    )

    assert output.state.ready_for_sizing is True
    assert output.state.validation_issues == []
    assert (
        output.canonical_dict()[
            "requested_usable_capacity_tib"
        ]
        == 100
    )
    assert (
        output.canonical_dict()[
            "preference_weights"
        ]["reliability"]
        == 1.0
    )
