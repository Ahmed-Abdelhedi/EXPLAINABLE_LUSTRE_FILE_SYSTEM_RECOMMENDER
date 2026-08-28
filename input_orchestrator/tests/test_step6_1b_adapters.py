from dataclasses import dataclass
from enum import Enum

from input_orchestrator.models import (
    FieldState,
    PendingQuestion,
)
from input_orchestrator.production_adapters import (
    CategoricalProductionAdapter,
    PreferenceProductionAdapter,
    QuantityProductionAdapter,
)
from input_orchestrator.result_merger import ResultMerger
from input_orchestrator.session_state import WorkingSessionState


class E(str, Enum):
    VERIFIED = "VERIFIED"
    CORRECTION = "CORRECTION"
    RESOLVED = "RESOLVED"


@dataclass
class FakeField:
    value: str


@dataclass
class FakeDecision:
    field: FakeField
    value: object
    status: E = E.VERIFIED
    unit: str | None = None
    evidence: str = ""
    reasons: list | None = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class FakeScope:
    intent = E.CORRECTION


class FakeQuantityResult:
    def __init__(self):
        self.decisions = [
            FakeDecision(
                FakeField("client_count"),
                128,
                evidence="128 clients",
            )
        ]
        self.scope = FakeScope()


class FakeQuantityPipeline:
    def process(self, text, **kwargs):
        self.kwargs = kwargs
        return FakeQuantityResult()


def test_quantity_adapter_passes_pending_context_and_correction(monkeypatch):
    import sys
    import types

    fake_package = types.ModuleType("requirement_extractor_v2")
    fake_models = types.ModuleType("requirement_extractor_v2.models")

    class FakeParamName:
        def __init__(self, value):
            self.value = value

    setattr(fake_models, "ParamName", FakeParamName)
    monkeypatch.setitem(sys.modules, "requirement_extractor_v2", fake_package)
    monkeypatch.setitem(sys.modules, "requirement_extractor_v2.models", fake_models)

    pipeline = FakeQuantityPipeline()
    adapter = QuantityProductionAdapter(pipeline=pipeline)

    pending = PendingQuestion(
        question_id="Q1",
        target_field="client_count",
        question="How many clients?",
        expected_answer_type="positive_integer",
        created_after_message_id="M0",
    )

    out = adapter.extract(
        "128",
        message_id="M1",
        pending_question=pending,
    )

    assert len(out) == 1
    assert out[0].field == "client_count"
    assert out[0].value == 128
    assert out[0].state == FieldState.VERIFIED
    assert out[0].explicit_correction is True
    assert pipeline.kwargs["previous_question"] == "How many clients?"


class SignalResult:
    has_preference_signal = True


class FakeSignal:
    def predict(self, text):
        return SignalResult()


class D(str, Enum):
    COST = "cost"
    PERFORMANCE = "performance"


class S(str, Enum):
    RESOLVED = "RESOLVED"
    RELATIVE_ONLY = "RELATIVE_ONLY"


class Src(str, Enum):
    MODEL = "TRANSFORMER"


class Level(str, Enum):
    LOW = "LOW"


@dataclass
class DimResult:
    status: S
    source: Src = Src.MODEL
    level: object = None
    presence_probability: float = 0.99
    intensity_confidence: float = 0.95
    evidence: str = "cost is secondary"


class Relation:
    def to_dict(self):
        return {
            "higher": "performance",
            "lower": "cost",
            "relation_type": "MORE_IMPORTANT_THAN",
            "evidence": "performance is more important than cost",
        }


class Layer2Result:
    dimensions = {
        D.COST: DimResult(
            status=S.RESOLVED,
            level=Level.LOW,
        ),
        D.PERFORMANCE: DimResult(
            status=S.RELATIVE_ONLY,
            level=None,
        ),
    }
    relations = [Relation()]


class FakeLayer2:
    def extract(self, text):
        return Layer2Result()


def test_preference_absolute_and_relation_are_separate():
    adapter = PreferenceProductionAdapter(
        signal_detector=FakeSignal(),
        layer2_provider=lambda: FakeLayer2(),
    )

    out = adapter.extract("x", message_id="M1")

    session = WorkingSessionState("S")
    ResultMerger().merge(session, out)

    assert session.get("cost_priority").value == "LOW"
    assert len(session.preference_relations) == 1
    assert session.preference_relations[0]["higher"] == "performance"


class FakeCategoricalResult:
    def to_dict(self):
        return {
            "ha_required": {
                "status": "VERIFIED",
                "value": True,
                "source": "SEMANTIC_MODEL",
                "evidence": "HA mandatory",
                "semantic_confidence": 0.999,
            },
            "access_type": {
                "status": "NO_EVIDENCE",
                "value": None,
                "source": "SEMANTIC_MODEL",
            },
        }


class FakeCategorical:
    def extract(self, text, **kwargs):
        return FakeCategoricalResult()


def test_no_evidence_never_becomes_false_or_null_overwrite():
    adapter = CategoricalProductionAdapter(FakeCategorical())

    out = adapter.extract(
        "HA mandatory",
        message_id="M1",
    )

    assert len(out) == 1
    assert out[0].field == "ha_required"
    assert out[0].value is True
