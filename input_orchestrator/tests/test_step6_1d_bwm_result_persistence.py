import sys
import types
from dataclasses import dataclass
from enum import Enum

from input_orchestrator.bwm_coordinator import BWMCoordinator
from input_orchestrator.models import FieldState
from input_orchestrator.session_state import WorkingSessionState


class FakeDimension(str, Enum):
    COST = "cost"
    POWER = "power"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"


class FakeWeightingStatus(str, Enum):
    NO_ACTIVE_PREFERENCE = "NO_ACTIVE_PREFERENCE"
    BLOCKED_UNRESOLVED = "BLOCKED_UNRESOLVED"
    NEEDS_SINGLE_CRITERION_CONFIRMATION = "NEEDS_SINGLE_CRITERION_CONFIRMATION"
    NEEDS_BEST_WORST = "NEEDS_BEST_WORST"
    NEEDS_BWM_COMPARISONS = "NEEDS_BWM_COMPARISONS"
    INVALID_BWM_JUDGMENTS = "INVALID_BWM_JUDGMENTS"
    INCONSISTENT_PREFERENCES = "INCONSISTENT_PREFERENCES"
    WEIGHTS_READY = "WEIGHTS_READY"


class FakeConsistencyStatus(str, Enum):
    PASS = "PASS"


class FakeConsistency:
    status = FakeConsistencyStatus.PASS


@dataclass
class FakeReadyResult:
    status = FakeWeightingStatus.WEIGHTS_READY
    method = "LINEAR_BWM"
    xi_star = 0.0
    consistency = FakeConsistency()
    active_dimensions = [
        FakeDimension.RELIABILITY,
        FakeDimension.PERFORMANCE,
    ]
    best = FakeDimension.RELIABILITY
    worst = FakeDimension.PERFORMANCE
    missing_questions = ()
    violations = ()

    def all_four_weights(self):
        return {
            "cost": 0.0,
            "power": 0.0,
            "performance": 0.25,
            "reliability": 0.75,
        }

    def to_dict(self):
        return {
            "status": self.status.value,
            "weights": self.all_four_weights(),
        }


class FakeLayer:
    def run(self, extraction, **kwargs):
        return FakeReadyResult()


class FakeCoordinator(BWMCoordinator):
    def _build_extraction(self, session):
        return object()


def _install_fake_modules(monkeypatch):
    pref = types.ModuleType("preference_extractor")
    layer2 = types.ModuleType("preference_extractor.layer2")
    labels = types.ModuleType(
        "preference_extractor.layer2.labels"
    )
    weighting = types.ModuleType(
        "preference_extractor.weighting"
    )
    models = types.ModuleType(
        "preference_extractor.weighting.models"
    )

    setattr(labels, "PreferenceDimension", FakeDimension)
    setattr(models, "WeightingStatus", FakeWeightingStatus)

    monkeypatch.setitem(
        sys.modules,
        "preference_extractor",
        pref,
    )
    monkeypatch.setitem(
        sys.modules,
        "preference_extractor.layer2",
        layer2,
    )
    monkeypatch.setitem(
        sys.modules,
        "preference_extractor.layer2.labels",
        labels,
    )
    monkeypatch.setitem(
        sys.modules,
        "preference_extractor.weighting",
        weighting,
    )
    monkeypatch.setitem(
        sys.modules,
        "preference_extractor.weighting.models",
        models,
    )


def test_weights_ready_are_persisted_and_normalized(monkeypatch):
    _install_fake_modules(monkeypatch)

    session = WorkingSessionState("S")
    coordinator = FakeCoordinator(
        weighting_layer=FakeLayer(),
        enabled=True,
    )

    action = coordinator.evaluate(
        session,
        message_id="M1",
    )

    assert action.complete is True
    assert action.status == "WEIGHTS_READY"

    record = session.get("preference_weights")

    assert record.state == FieldState.VERIFIED
    assert abs(sum(record.value.values()) - 1.0) < 1e-12
    assert record.value["reliability"] == 0.75
